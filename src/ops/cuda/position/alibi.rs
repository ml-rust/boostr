//! ALiBi attention bias CUDA launcher

use crate::error::{Error, Result};
use crate::ops::traits::AlibiOps;
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use crate::ops::cuda::kernels::{self, ALIBI_BF16_MODULE, ALIBI_MODULE};

/// Picks the BF16 kernel name and module, after checking the device supports BF16.
/// `label` names the caller in the error message (e.g. "ALiBi" or "ALiBi causal").
fn bf16_kernel(
    device_index: usize,
    kernel_name: &'static str,
    label: &str,
) -> Result<(&'static str, &'static str)> {
    if !numr::runtime::cuda::CudaDevice::new(device_index)
        .profile()
        .caps
        .bf16
    {
        return Err(Error::KernelError {
            reason: format!("{label}: bf16 requires an Ampere or newer device"),
        });
    }
    Ok((kernel_name, ALIBI_BF16_MODULE))
}

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
        let device = scores.device();
        let device_index = device.id();

        let (kernel_name, module_const) = match dtype {
            DType::F32 => ("alibi_add_bias_fp32", ALIBI_MODULE),
            DType::F16 => ("alibi_add_bias_fp16", ALIBI_MODULE),
            DType::BF16 => bf16_kernel(device_index, "alibi_add_bias_bf16", "ALiBi")?,
            _ => {
                return Err(Error::KernelError {
                    reason: format!("ALiBi: unsupported dtype {dtype:?}"),
                });
            }
        };

        let module = kernels::get_or_load_module(self.context(), device_index, module_const)?;
        let func = kernels::get_kernel_function(&module, kernel_name)?;

        let total = (batch_size * num_heads * seq_len_q * seq_len_k) as u32;
        let block_size = 256u32;
        let grid_size = total.div_ceil(block_size);

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

    fn alibi_add_bias_causal(
        &self,
        scores: &Tensor<CudaRuntime>,
        batch_size: usize,
        num_heads: usize,
        seq_len_q: usize,
        seq_len_k: usize,
        position: usize,
    ) -> Result<()> {
        let dtype = scores.dtype();
        let device = scores.device();
        let device_index = device.id();

        let (kernel_name, module_const) = match dtype {
            DType::F32 => ("alibi_add_bias_causal_fp32", ALIBI_MODULE),
            DType::F16 => ("alibi_add_bias_causal_fp16", ALIBI_MODULE),
            DType::BF16 => bf16_kernel(device_index, "alibi_add_bias_causal_bf16", "ALiBi causal")?,
            _ => {
                return Err(Error::KernelError {
                    reason: format!("ALiBi causal: unsupported dtype {dtype:?}"),
                });
            }
        };

        let module = kernels::get_or_load_module(self.context(), device_index, module_const)?;
        let func = kernels::get_kernel_function(&module, kernel_name)?;

        let total = (batch_size * num_heads * seq_len_q * seq_len_k) as u32;
        let block_size = 256u32;
        let grid_size = total.div_ceil(block_size);

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
        let pos_i32 = position as i32;

        unsafe {
            let mut builder = self.stream().launch_builder(&func);
            builder.arg(&s_ptr);
            builder.arg(&b_i32);
            builder.arg(&nh_i32);
            builder.arg(&sq_i32);
            builder.arg(&sk_i32);
            builder.arg(&pos_i32);
            builder.launch(cfg).map_err(|e| Error::KernelError {
                reason: format!("ALiBi causal kernel failed: {e:?}"),
            })?;
        }

        Ok(())
    }
}
