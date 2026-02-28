//! CUDA implementation of QuantMatmulOps

use crate::error::{Error, Result};
use crate::quant::traits::QuantMatmulOps;
use crate::quant::{QuantFormat, QuantTensor};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::int4_gemm as int4_dispatch;
use super::kernels::{self, QUANT_ACT_MODULE, QUANT_GEMV_MODULE, QUANT_MATMUL_MODULE};

/// Validate input is F32 and extract (M, K).
fn validate_input_cuda(input: &Tensor<CudaRuntime>) -> Result<(usize, usize)> {
    if input.dtype() != DType::F32 {
        return Err(Error::QuantError {
            reason: format!("input must be F32, got {:?}", input.dtype()),
        });
    }
    let shape = input.shape();
    if shape.len() < 2 {
        return Err(Error::QuantError {
            reason: format!("input must be at least 2D, got {:?}", shape),
        });
    }
    let k = shape[shape.len() - 1];
    let m: usize = shape.iter().product::<usize>() / k;
    Ok((m, k))
}

/// Quantize F32 activation to Q8_1 format on GPU.
/// Returns a raw byte tensor of shape [m * num_blocks * 36] containing Q8_1 blocks.
fn quantize_activation_q8_1(
    client: &CudaClient,
    activation: &Tensor<CudaRuntime>,
    m: usize,
    k: usize,
) -> Result<Tensor<CudaRuntime>> {
    let device_index = activation.device().id();
    let num_blocks = k / 32;
    let q8_bytes = m * num_blocks * 36;

    // Allocate Q8_1 buffer as U8 tensor
    let q8_buf = Tensor::<CudaRuntime>::empty(&[q8_bytes], DType::U8, activation.device());

    let module = kernels::get_or_load_module(client.context(), device_index, QUANT_ACT_MODULE)?;
    let func = kernels::get_kernel_function(&module, "quantize_f32_q8_1")?;

    let act_ptr = activation.ptr();
    let q8_ptr = q8_buf.ptr();
    let m_u32 = m as u32;
    let k_u32 = k as u32;

    // Grid: (num_blocks, M, 1), Block: (32, 1, 1) — one warp per Q8_1 block
    let cfg = LaunchConfig {
        grid_dim: (num_blocks as u32, m_u32, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&act_ptr);
        builder.arg(&q8_ptr);
        builder.arg(&m_u32);
        builder.arg(&k_u32);
        builder.launch(cfg).map_err(|e| Error::QuantError {
            reason: format!("CUDA quantize_f32_q8_1 kernel launch failed: {:?}", e),
        })?;
    }

    Ok(q8_buf)
}

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
        // Validate activation dtype
        if activation.dtype() != DType::F32 {
            return Err(Error::QuantError {
                reason: format!(
                    "quant_matmul activation must be F32, got {:?}",
                    activation.dtype()
                ),
            });
        }

        // Validate weight is 2D: [N, K]
        let w_shape = weight.shape();
        if w_shape.len() != 2 {
            return Err(Error::QuantError {
                reason: format!("quant_matmul weight must be 2D [N, K], got {:?}", w_shape),
            });
        }
        let n = w_shape[0];
        let k = w_shape[1];

        // Validate activation shape: [..., K]
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

        // Compute M from activation shape
        let total_elements: usize = a_shape.iter().product();
        let m = total_elements / k;

        // Ensure activation is contiguous
        let act_contig = activation.contiguous();

        let device_index = activation.device().id();

        // Allocate output: [..., N]
        let mut out_shape = a_shape[..a_shape.len() - 1].to_vec();
        out_shape.push(n);
        let output = Tensor::<CudaRuntime>::empty(&out_shape, DType::F32, activation.device());
        let output_ptr = output.ptr();

        let m_u32 = m as u32;
        let k_u32 = k as u32;
        let n_u32 = n as u32;

        // Use GEMV for small M (decode), matmul for large M (prefill)
        if m <= 4 {
            let warps_per_block = 8u32;
            let block_size = 256u32;
            let grid_x = (n as u32).div_ceil(warps_per_block);
            let grid_y = m as u32;

            let cfg = LaunchConfig {
                grid_dim: (grid_x, grid_y, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };

            // For Q4_K: use dp4a path with Q8_1 quantized activation
            // (Q6_K dp4a is slower due to unaligned loads — 210-byte blocks)
            if weight.format() == QuantFormat::Q4K && k % 32 == 0 {
                let q8_buf = quantize_activation_q8_1(self, &act_contig, m, k)?;
                let q8_ptr = q8_buf.ptr();
                let weight_ptr = weight.storage().ptr();

                let kernel_name = "quant_gemv_q4_k_q8_1";

                let module =
                    kernels::get_or_load_module(self.context(), device_index, QUANT_GEMV_MODULE)?;
                let func = kernels::get_kernel_function(&module, kernel_name)?;

                unsafe {
                    let mut builder = self.stream().launch_builder(&func);
                    builder.arg(&q8_ptr);
                    builder.arg(&weight_ptr);
                    builder.arg(&output_ptr);
                    builder.arg(&m_u32);
                    builder.arg(&k_u32);
                    builder.arg(&n_u32);
                    builder.launch(cfg).map_err(|e| Error::QuantError {
                        reason: format!("CUDA {} launch failed: {:?}", kernel_name, e),
                    })?;
                }
            } else {
                // F32 activation path for other formats
                let act_ptr = act_contig.ptr();
                let weight_ptr = weight.storage().ptr();

                let kernel_name = match weight.format() {
                    QuantFormat::Q4_0 => "quant_gemv_q4_0_f32",
                    QuantFormat::Q8_0 => "quant_gemv_q8_0_f32",
                    QuantFormat::Q4K => "quant_gemv_q4_k_f32",
                    QuantFormat::Q6K => "quant_gemv_q6_k_f32",
                    other => {
                        return Err(Error::UnsupportedQuantFormat {
                            format: format!("{} (CUDA quant_gemv not implemented)", other),
                        });
                    }
                };

                let module =
                    kernels::get_or_load_module(self.context(), device_index, QUANT_GEMV_MODULE)?;
                let func = kernels::get_kernel_function(&module, kernel_name)?;

                unsafe {
                    let mut builder = self.stream().launch_builder(&func);
                    builder.arg(&act_ptr);
                    builder.arg(&weight_ptr);
                    builder.arg(&output_ptr);
                    builder.arg(&m_u32);
                    builder.arg(&k_u32);
                    builder.arg(&n_u32);
                    builder.launch(cfg).map_err(|e| Error::QuantError {
                        reason: format!("CUDA quant_gemv kernel launch failed: {:?}", e),
                    })?;
                }
            }
        } else {
            let act_ptr = act_contig.ptr();
            let weight_ptr = weight.storage().ptr();

            let kernel_name = match weight.format() {
                QuantFormat::Q4_0 => "quant_matmul_q4_0_f32",
                QuantFormat::Q8_0 => "quant_matmul_q8_0_f32",
                QuantFormat::Q4K => "quant_matmul_q4_k_f32",
                QuantFormat::Q6K => "quant_matmul_q6_k_f32",
                other => {
                    return Err(Error::UnsupportedQuantFormat {
                        format: format!("{} (CUDA quant_matmul not implemented)", other),
                    });
                }
            };

            let module =
                kernels::get_or_load_module(self.context(), device_index, QUANT_MATMUL_MODULE)?;
            let func = kernels::get_kernel_function(&module, kernel_name)?;

            let block_x = 16u32;
            let block_y = 16u32;
            let grid_x = (n as u32).div_ceil(block_x);
            let grid_y = (m as u32).div_ceil(block_y);

            let cfg = LaunchConfig {
                grid_dim: (grid_x, grid_y, 1),
                block_dim: (block_x, block_y, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                let mut builder = self.stream().launch_builder(&func);
                builder.arg(&act_ptr);
                builder.arg(&weight_ptr);
                builder.arg(&output_ptr);
                builder.arg(&m_u32);
                builder.arg(&k_u32);
                builder.arg(&n_u32);
                builder.launch(cfg).map_err(|e| Error::QuantError {
                    reason: format!("CUDA quant_matmul kernel launch failed: {:?}", e),
                })?;
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

        // Validate activation
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
        let total_elements: usize = a_shape.iter().product();
        let m = total_elements / k;
        let act_contig = activation.contiguous();

        // Check if all weights support dp4a (Q4_K only — Q6_K dp4a is slower due to unaligned loads)
        let all_dp4a = weights.iter().all(|w| w.format() == QuantFormat::Q4K);
        let use_dp4a = all_dp4a && m <= 4 && k % 32 == 0;

        if use_dp4a {
            // Quantize activation to Q8_1 ONCE
            let q8_buf = quantize_activation_q8_1(self, &act_contig, m, k)?;
            let q8_ptr = q8_buf.ptr();
            let device_index = activation.device().id();

            let warps_per_block = 8u32;
            let block_size = 256u32;
            let m_u32 = m as u32;
            let k_u32 = k as u32;

            let module =
                kernels::get_or_load_module(self.context(), device_index, QUANT_GEMV_MODULE)?;
            let func = kernels::get_kernel_function(&module, "quant_gemv_q4_k_q8_1")?;

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

                let mut out_shape = a_shape[..a_shape.len() - 1].to_vec();
                out_shape.push(n);
                let output =
                    Tensor::<CudaRuntime>::empty(&out_shape, DType::F32, activation.device());
                let output_ptr = output.ptr();
                let weight_ptr = w.storage().ptr();

                let grid_x = n_u32.div_ceil(warps_per_block);
                let cfg = LaunchConfig {
                    grid_dim: (grid_x, m_u32, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                };

                unsafe {
                    let mut builder = self.stream().launch_builder(&func);
                    builder.arg(&q8_ptr);
                    builder.arg(&weight_ptr);
                    builder.arg(&output_ptr);
                    builder.arg(&m_u32);
                    builder.arg(&k_u32);
                    builder.arg(&n_u32);
                    builder.launch(cfg).map_err(|e| Error::QuantError {
                        reason: format!("CUDA dp4a batch launch failed: {:?}", e),
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
}
