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
use super::kernels::{self, QUANT_MATMUL_MODULE};

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

        // Compute M from activation shape
        let total_elements: usize = a_shape.iter().product();
        let m = total_elements / k;

        // Ensure activation is contiguous
        let act_contig = activation.contiguous();

        let act_ptr = act_contig.ptr();
        let weight_ptr = weight.storage().ptr();
        let device_index = activation.device().id();

        // Allocate output: [..., N]
        let mut out_shape = a_shape[..a_shape.len() - 1].to_vec();
        out_shape.push(n);
        let output = Tensor::<CudaRuntime>::empty(&out_shape, DType::F32, activation.device());
        let output_ptr = output.ptr();

        // Load module and launch kernel
        let module =
            kernels::get_or_load_module(self.context(), device_index, QUANT_MATMUL_MODULE)?;
        let func = kernels::get_kernel_function(&module, kernel_name)?;

        // 2D grid: each thread computes one (row, col) of output
        let block_x = 16u32;
        let block_y = 16u32;
        let grid_x = (n as u32 + block_x - 1) / block_x;
        let grid_y = (m as u32 + block_y - 1) / block_y;

        let m_u32 = m as u32;
        let k_u32 = k as u32;
        let n_u32 = n as u32;

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

        Ok(output)
    }
}
