//! Batched dp4a GEMV path for `quant_matmul_batch`.
//!
//! Split out of `impl_ops.rs` to stay under the `cuda/*.rs` 400-line limit.
//! Quantizes the shared activation to Q8_1 once, then reuses it across every
//! weight in the batch instead of re-quantizing per weight.

use crate::error::{Error, Result};
use crate::quant::traits::QuantMatmulOps;
use crate::quant::{QuantFormat, QuantTensor};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::super::kernels::{
    self, GEMV_Q2_K_MODULE, GEMV_Q3_K_MODULE, GEMV_Q5_K_MODULE, QUANT_GEMV_MODULE,
};
use super::helpers::quantize_activation_q8_1;

/// Batched quantized matmul: dp4a GEMV when every weight qualifies, else
/// falls back to calling `quant_matmul` per weight.
pub(super) fn quant_matmul_batch_impl(
    client: &CudaClient,
    activation: &Tensor<CudaRuntime>,
    weights: &[&QuantTensor<CudaRuntime>],
) -> Result<Vec<Tensor<CudaRuntime>>> {
    if weights.is_empty() {
        return Ok(vec![]);
    }

    if activation.dtype() != DType::F32 {
        return Err(Error::QuantError {
            reason: format!(
                "quant_matmul_batch activation must be F32, got {:?}",
                activation.dtype()
            ),
        });
    }

    let a_shape = activation.shape();
    if a_shape.is_empty() {
        return Err(Error::QuantError {
            reason: "quant_matmul_batch activation must be at least 1D".into(),
        });
    }
    let k = a_shape[a_shape.len() - 1];
    let m = a_shape.iter().product::<usize>() / k;
    let act_contig = activation.contiguous()?;

    // Check if all weights support dp4a (Q4_K, Q6_K, Q8_0, Q5_K, Q3_K, Q2_K)
    let all_dp4a = weights.iter().all(|w| {
        w.format().is_ok_and(|f| {
            matches!(
                f,
                QuantFormat::Q4K
                    | QuantFormat::Q6K
                    | QuantFormat::Q8_0
                    | QuantFormat::Q5K
                    | QuantFormat::Q3K
                    | QuantFormat::Q2K
            )
        })
    });
    let use_dp4a = all_dp4a && m <= 4 && k.is_multiple_of(32);

    if use_dp4a {
        // Quantize activation to Q8_1 ONCE, reuse for all weights
        let q8_buf = quantize_activation_q8_1(client, &act_contig, m, k)?;
        let q8_ptr = q8_buf.ptr();
        let device_index = activation.device().id();

        let m_u32 = m as u32;
        let k_u32 = k as u32;

        // Pre-load modules for all formats that might appear
        let module_main =
            kernels::get_or_load_module(client.context(), device_index, QUANT_GEMV_MODULE)?;
        let func_q4k = kernels::get_kernel_function(&module_main, "quant_gemv_q4_k_q8_1_mwr")?;
        let func_q6k = kernels::get_kernel_function(&module_main, "quant_gemv_q6_k_q8_1_mwr")?;
        let func_q8_0 = kernels::get_kernel_function(&module_main, "quant_gemv_q8_0_q8_1_mwr")?;

        // Lazily load per-format modules only if needed
        let has_q5k = weights
            .iter()
            .any(|w| w.format().is_ok_and(|f| f == QuantFormat::Q5K));
        let has_q3k = weights
            .iter()
            .any(|w| w.format().is_ok_and(|f| f == QuantFormat::Q3K));
        let has_q2k = weights
            .iter()
            .any(|w| w.format().is_ok_and(|f| f == QuantFormat::Q2K));

        let func_q5k = if has_q5k {
            let m = kernels::get_or_load_module(client.context(), device_index, GEMV_Q5_K_MODULE)?;
            Some(kernels::get_kernel_function(
                &m,
                "quant_gemv_q5_k_q8_1_mwr",
            )?)
        } else {
            None
        };
        let func_q3k = if has_q3k {
            let m = kernels::get_or_load_module(client.context(), device_index, GEMV_Q3_K_MODULE)?;
            Some(kernels::get_kernel_function(
                &m,
                "quant_gemv_q3_k_q8_1_mwr",
            )?)
        } else {
            None
        };
        let func_q2k = if has_q2k {
            let m = kernels::get_or_load_module(client.context(), device_index, GEMV_Q2_K_MODULE)?;
            Some(kernels::get_kernel_function(
                &m,
                "quant_gemv_q2_k_q8_1_mwr",
            )?)
        } else {
            None
        };

        let mut results = Vec::with_capacity(weights.len());
        for w in weights {
            let w_shape = w.shape();
            if w_shape.len() != 2 || w_shape[1] != k {
                return Err(Error::QuantError {
                    reason: format!(
                        "quant_matmul_batch weight shape mismatch: {:?}, expected [N, {}]",
                        w_shape, k
                    ),
                });
            }
            let n = w_shape[0];
            let n_u32 = n as u32;

            let func = match w.format()? {
                QuantFormat::Q4K => &func_q4k,
                QuantFormat::Q6K => &func_q6k,
                QuantFormat::Q8_0 => &func_q8_0,
                QuantFormat::Q5K => func_q5k.as_ref().ok_or_else(|| Error::QuantError {
                    reason: "Q5K GEMV module failed to load".into(),
                })?,
                QuantFormat::Q3K => func_q3k.as_ref().ok_or_else(|| Error::QuantError {
                    reason: "Q3K GEMV module failed to load".into(),
                })?,
                QuantFormat::Q2K => func_q2k.as_ref().ok_or_else(|| Error::QuantError {
                    reason: "Q2K GEMV module failed to load".into(),
                })?,
                _ => unreachable!(),
            };

            let mut out_shape = a_shape[..a_shape.len() - 1].to_vec();
            out_shape.push(n);
            let output = Tensor::<CudaRuntime>::empty(&out_shape, DType::F32, activation.device())?;
            let output_ptr = output.ptr();
            let weight_ptr = w.storage().ptr();

            // MWR: one output column per block, 128 threads (4 warps)
            let cfg = LaunchConfig {
                grid_dim: (n_u32, m_u32, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                let mut builder = client.stream().launch_builder(func);
                builder.arg(&q8_ptr);
                builder.arg(&weight_ptr);
                builder.arg(&output_ptr);
                builder.arg(&m_u32);
                builder.arg(&k_u32);
                builder.arg(&n_u32);
                builder.launch(cfg).map_err(|e| Error::QuantError {
                    reason: format!("CUDA dp4a mr batch launch failed: {:?}", e),
                })?;
            }

            results.push(output);
        }
        Ok(results)
    } else {
        // Fallback: call quant_matmul individually
        weights
            .iter()
            .map(|w| client.quant_matmul(activation, w))
            .collect()
    }
}
