//! impl QuantMatmulOps<CudaRuntime> for CudaClient

use crate::error::{Error, Result};
use crate::quant::traits::QuantMatmulOps;
use crate::quant::{QuantFormat, QuantTensor};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::super::int4_gemm as int4_dispatch;
use super::super::kernels::{
    self, GEMV_Q2_K_MODULE, GEMV_Q3_K_MODULE, GEMV_Q5_K_MODULE, QUANT_GEMV_MODULE,
};
use super::fallback::{quant_matmul_via_dequant, quant_swiglu_via_dequant};
use super::format_dispatch::{dispatch_gemv, dispatch_matmul};
use super::helpers::{quantize_activation_q8_1, validate_input_cuda};

impl QuantMatmulOps<CudaRuntime> for CudaClient {
    fn int4_gemm(
        &self,
        input: &Tensor<CudaRuntime>,
        qweight: &Tensor<CudaRuntime>,
        scales: &Tensor<CudaRuntime>,
        zeros: &Tensor<CudaRuntime>,
        group_size: usize,
    ) -> Result<Tensor<CudaRuntime>> {
        let (m, k) = validate_input_cuda(input)?;
        let n = qweight.shape()[1] * 8;
        let act_contig = input.contiguous();

        let mut out_shape = input.shape()[..input.shape().len() - 1].to_vec();
        out_shape.push(n);
        let output = Tensor::<CudaRuntime>::empty(&out_shape, DType::F32, input.device());
        int4_dispatch::launch_int4_gemm(
            self,
            &act_contig,
            qweight,
            scales,
            zeros,
            &output,
            m as u32,
            k as u32,
            n as u32,
            group_size as u32,
        )?;
        Ok(output)
    }

    fn int4_gemm_gptq(
        &self,
        input: &Tensor<CudaRuntime>,
        qweight: &Tensor<CudaRuntime>,
        qzeros: &Tensor<CudaRuntime>,
        scales: &Tensor<CudaRuntime>,
        g_idx: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        let (m, k) = validate_input_cuda(input)?;
        let n = qweight.shape()[1];
        let act_contig = input.contiguous();

        let mut out_shape = input.shape()[..input.shape().len() - 1].to_vec();
        out_shape.push(n);
        let output = Tensor::<CudaRuntime>::empty(&out_shape, DType::F32, input.device());
        int4_dispatch::launch_int4_gemm_gptq(
            self,
            &act_contig,
            qweight,
            qzeros,
            scales,
            g_idx,
            &output,
            m as u32,
            k as u32,
            n as u32,
        )?;
        Ok(output)
    }

    fn marlin_gemm(
        &self,
        input: &Tensor<CudaRuntime>,
        weight: &Tensor<CudaRuntime>,
        scales: &Tensor<CudaRuntime>,
        zeros: &Tensor<CudaRuntime>,
        group_size: usize,
    ) -> Result<Tensor<CudaRuntime>> {
        let (m, k) = validate_input_cuda(input)?;
        let n = weight.shape()[1];
        let act_contig = input.contiguous();

        let mut out_shape = input.shape()[..input.shape().len() - 1].to_vec();
        out_shape.push(n);
        let output = Tensor::<CudaRuntime>::empty(&out_shape, DType::F32, input.device());
        int4_dispatch::launch_marlin_gemm(
            self,
            &act_contig,
            weight,
            scales,
            zeros,
            &output,
            m as u32,
            k as u32,
            n as u32,
            group_size as u32,
        )?;
        Ok(output)
    }

    fn quant_matmul(
        &self,
        activation: &Tensor<CudaRuntime>,
        weight: &QuantTensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        if activation.dtype() != DType::F32 {
            return Err(Error::QuantError {
                reason: format!(
                    "quant_matmul activation must be F32, got {:?}",
                    activation.dtype()
                ),
            });
        }

        let w_shape = weight.shape();
        if w_shape.len() != 2 {
            return Err(Error::QuantError {
                reason: format!("quant_matmul weight must be 2D [N, K], got {:?}", w_shape),
            });
        }
        let n = w_shape[0];
        let k = w_shape[1];

        let a_shape = activation.shape();
        if a_shape.is_empty() {
            return Err(Error::QuantError {
                reason: "quant_matmul activation must be at least 1D".into(),
            });
        }
        let a_k = a_shape[a_shape.len() - 1];
        if a_k != k {
            return Err(Error::QuantError {
                reason: format!(
                    "quant_matmul dimension mismatch: activation K={}, weight K={}",
                    a_k, k
                ),
            });
        }

        let m = a_shape.iter().product::<usize>() / k;
        let act_contig = activation.contiguous();

        let mut out_shape = a_shape[..a_shape.len() - 1].to_vec();
        out_shape.push(n);
        let output = Tensor::<CudaRuntime>::empty(&out_shape, DType::F32, activation.device());
        let output_ptr = output.ptr();

        if m <= 64 {
            match dispatch_gemv(self, &act_contig, weight, output_ptr, m, k, n)? {
                Some(()) => {}
                None => return quant_matmul_via_dequant(self, activation, weight),
            }
        } else {
            match dispatch_matmul(self, &act_contig, weight, output_ptr, m, k, n)? {
                Some(()) => {}
                None => return quant_matmul_via_dequant(self, activation, weight),
            }
        }

        Ok(output)
    }

    fn quant_matmul_batch(
        &self,
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
        let act_contig = activation.contiguous();

        // Check if all weights support dp4a (Q4_K, Q6_K, Q8_0, Q5_K, Q3_K, Q2_K)
        let all_dp4a = weights.iter().all(|w| {
            matches!(
                w.format(),
                QuantFormat::Q4K
                    | QuantFormat::Q6K
                    | QuantFormat::Q8_0
                    | QuantFormat::Q5K
                    | QuantFormat::Q3K
                    | QuantFormat::Q2K
            )
        });
        let use_dp4a = all_dp4a && m <= 4 && k % 32 == 0;

        if use_dp4a {
            // Quantize activation to Q8_1 ONCE, reuse for all weights
            let q8_buf = quantize_activation_q8_1(self, &act_contig, m, k)?;
            let q8_ptr = q8_buf.ptr();
            let device_index = activation.device().id();

            let m_u32 = m as u32;
            let k_u32 = k as u32;

            // Pre-load modules for all formats that might appear
            let module_main =
                kernels::get_or_load_module(self.context(), device_index, QUANT_GEMV_MODULE)?;
            let func_q4k = kernels::get_kernel_function(&module_main, "quant_gemv_q4_k_q8_1_mwr")?;
            let func_q6k = kernels::get_kernel_function(&module_main, "quant_gemv_q6_k_q8_1_mwr")?;
            let func_q8_0 = kernels::get_kernel_function(&module_main, "quant_gemv_q8_0_q8_1_mwr")?;

            // Lazily load per-format modules only if needed
            let has_q5k = weights.iter().any(|w| w.format() == QuantFormat::Q5K);
            let has_q3k = weights.iter().any(|w| w.format() == QuantFormat::Q3K);
            let has_q2k = weights.iter().any(|w| w.format() == QuantFormat::Q2K);

            let func_q5k = if has_q5k {
                let m =
                    kernels::get_or_load_module(self.context(), device_index, GEMV_Q5_K_MODULE)?;
                Some(kernels::get_kernel_function(
                    &m,
                    "quant_gemv_q5_k_q8_1_mwr",
                )?)
            } else {
                None
            };
            let func_q3k = if has_q3k {
                let m =
                    kernels::get_or_load_module(self.context(), device_index, GEMV_Q3_K_MODULE)?;
                Some(kernels::get_kernel_function(
                    &m,
                    "quant_gemv_q3_k_q8_1_mwr",
                )?)
            } else {
                None
            };
            let func_q2k = if has_q2k {
                let m =
                    kernels::get_or_load_module(self.context(), device_index, GEMV_Q2_K_MODULE)?;
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

                let func = match w.format() {
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
                let output =
                    Tensor::<CudaRuntime>::empty(&out_shape, DType::F32, activation.device());
                let output_ptr = output.ptr();
                let weight_ptr = w.storage().ptr();

                // MWR: one output column per block, 128 threads (4 warps)
                let cfg = LaunchConfig {
                    grid_dim: (n_u32, m_u32, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                };

                unsafe {
                    let mut builder = self.stream().launch_builder(func);
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
                .map(|w| self.quant_matmul(activation, w))
                .collect()
        }
    }

    fn quant_swiglu(
        &self,
        activation: &Tensor<CudaRuntime>,
        gate_weight: &QuantTensor<CudaRuntime>,
        up_weight: &QuantTensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        let (m, k) = validate_input_cuda(activation)?;
        let n = gate_weight.shape()[0];
        let device_index = activation.device().id();

        if up_weight.shape()[0] != n || up_weight.shape()[1] != k {
            return Err(Error::QuantError {
                reason: format!(
                    "gate_weight shape {:?} vs up_weight shape {:?}",
                    gate_weight.shape(),
                    up_weight.shape()
                ),
            });
        }
        if gate_weight.format() != up_weight.format() {
            return Err(Error::QuantError {
                reason: format!(
                    "gate format {:?} != up format {:?}",
                    gate_weight.format(),
                    up_weight.format()
                ),
            });
        }

        let act_contig = activation.contiguous();
        let a_shape = activation.shape();
        let mut out_shape = a_shape[..a_shape.len() - 1].to_vec();
        out_shape.push(n);
        let output = Tensor::<CudaRuntime>::empty(&out_shape, DType::F32, activation.device());
        let output_ptr = output.ptr();
        let m_u32 = m as u32;
        let k_u32 = k as u32;
        let n_u32 = n as u32;

        // Use fused kernel for GEMV path (decode + short prefill)
        let use_fused = m <= 64
            && matches!(
                gate_weight.format(),
                QuantFormat::Q4K
                    | QuantFormat::Q6K
                    | QuantFormat::Q8_0
                    | QuantFormat::Q5K
                    | QuantFormat::Q3K
                    | QuantFormat::Q2K
            )
            && k % 32 == 0;

        if use_fused {
            let q8_buf = quantize_activation_q8_1(self, &act_contig, m, k)?;
            let q8_ptr = q8_buf.ptr();
            let gate_ptr = gate_weight.storage().ptr();
            let up_ptr = up_weight.storage().ptr();

            let (kernel_name, module_name) = match gate_weight.format() {
                QuantFormat::Q4K => ("fused_swiglu_q4k_q8_1_mwr", QUANT_GEMV_MODULE),
                QuantFormat::Q6K => ("fused_swiglu_q6k_q8_1_mwr", QUANT_GEMV_MODULE),
                QuantFormat::Q8_0 => ("fused_swiglu_q8_0_q8_1_mwr", QUANT_GEMV_MODULE),
                QuantFormat::Q5K => ("fused_swiglu_q5k_q8_1_mwr", GEMV_Q5_K_MODULE),
                QuantFormat::Q3K => ("fused_swiglu_q3k_q8_1_mwr", GEMV_Q3_K_MODULE),
                QuantFormat::Q2K => ("fused_swiglu_q2k_q8_1_mwr", GEMV_Q2_K_MODULE),
                _ => unreachable!(),
            };

            let cfg = LaunchConfig {
                grid_dim: (n_u32, m_u32, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };

            let module = kernels::get_or_load_module(self.context(), device_index, module_name)?;
            let func = kernels::get_kernel_function(&module, kernel_name)?;

            unsafe {
                let mut builder = self.stream().launch_builder(&func);
                builder.arg(&q8_ptr);
                builder.arg(&gate_ptr);
                builder.arg(&up_ptr);
                builder.arg(&output_ptr);
                builder.arg(&m_u32);
                builder.arg(&k_u32);
                builder.arg(&n_u32);
                builder.launch(cfg).map_err(|e| Error::QuantError {
                    reason: format!("CUDA {} launch failed: {:?}", kernel_name, e),
                })?;
            }

            Ok(output)
        } else if m <= 64 {
            // Generic fused SwiGLU: gate+up matmul + silu in one kernel
            quant_swiglu_via_dequant(self, &act_contig, gate_weight, up_weight, &output, m, k, n)
                .map(|_| output)
        } else {
            // Large batch: separate matmuls + fused silu_mul
            let gate = self.quant_matmul(activation, gate_weight)?;
            let up = self.quant_matmul(activation, up_weight)?;
            use numr::ops::ActivationOps;
            self.silu_mul(&gate, &up).map_err(Error::Numr)
        }
    }
}
