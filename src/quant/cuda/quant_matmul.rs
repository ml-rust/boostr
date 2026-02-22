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

use super::kernels::{self, QUANT_MATMUL_MODULE};

impl QuantMatmulOps<CudaRuntime> for CudaClient {
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
