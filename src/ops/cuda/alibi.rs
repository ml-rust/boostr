//! ALiBi attention bias CUDA launcher

use crate::error::{Error, Result};
use crate::ops::traits::AlibiOps;
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::kernels::{self, ALIBI_MODULE};

impl AlibiOps<CudaRuntime> for CudaClient {
    fn alibi_add_bias(
        &self,
        scores: &Tensor<CudaRuntime>,
        batch_size: usize,
        num_heads: usize,
        seq_len_q: usize,
        seq_len_k: usize,
    ) -> Result<()> {
        let dtype = scores.dtype();
        let kernel_name = match dtype {
            DType::F32 => "alibi_add_bias_fp32",
            DType::F16 => "alibi_add_bias_fp16",
            DType::BF16 => "alibi_add_bias_bf16",
            _ => {
                return Err(Error::KernelError {
                    reason: format!("ALiBi: unsupported dtype {dtype:?}"),
                });
            }
        };

        let device = scores.device();
        let device_index = device.id();
        let module = kernels::get_or_load_module(self.context(), device_index, ALIBI_MODULE)?;
        let func = kernels::get_kernel_function(&module, kernel_name)?;

        let total = (batch_size * num_heads * seq_len_q * seq_len_k) as u32;
        let block_size = 256u32;
        let grid_size = (total + block_size - 1) / block_size;

        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let s_ptr = scores.ptr();
        let b_i32 = batch_size as i32;
        let nh_i32 = num_heads as i32;
        let sq_i32 = seq_len_q as i32;
        let sk_i32 = seq_len_k as i32;

        unsafe {
            let mut builder = self.stream().launch_builder(&func);
            builder.arg(&s_ptr);
            builder.arg(&b_i32);
            builder.arg(&nh_i32);
            builder.arg(&sq_i32);
            builder.arg(&sk_i32);
            builder.launch(cfg).map_err(|e| Error::KernelError {
                reason: format!("ALiBi kernel failed: {e:?}"),
            })?;
        }

        Ok(())
    }
}
