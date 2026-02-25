//! CUDA implementation of RoPEOps — fused kernel dispatch

use crate::error::{Error, Result};
use crate::ops::cuda::kernels::{self, ROPE_MODULE};
use crate::ops::traits::RoPEOps;
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};

impl RoPEOps<CudaRuntime> for CudaClient {
    fn apply_rope(
        &self,
        x: &Var<CudaRuntime>,
        cos_cache: &Var<CudaRuntime>,
        sin_cache: &Var<CudaRuntime>,
    ) -> Result<Var<CudaRuntime>> {
        let x_tensor = x.tensor();
        let cos_tensor = cos_cache.tensor();
        let sin_tensor = sin_cache.tensor();

        // Validate shapes
        let x_shape = x_tensor.shape();
        if x_shape.len() != 4 {
            return Err(Error::InvalidArgument {
                arg: "x",
                reason: format!("expected 4D [B, H, S, D], got {:?}", x_shape),
            });
        }

        let batch_size = x_shape[0];
        let num_heads = x_shape[1];
        let seq_len = x_shape[2];
        let head_dim = x_shape[3];

        if head_dim % 2 != 0 {
            return Err(Error::InvalidArgument {
                arg: "head_dim",
                reason: format!("head_dim must be even, got {}", head_dim),
            });
        }

        // Validate cache shapes
        let cos_shape = cos_tensor.shape();
        let sin_shape = sin_tensor.shape();

        if cos_shape.len() != 2 || sin_shape.len() != 2 {
            return Err(Error::InvalidArgument {
                arg: "cache",
                reason: format!(
                    "expected 2D [S, D/2], got cos: {:?}, sin: {:?}",
                    cos_shape, sin_shape
                ),
            });
        }

        if cos_shape[1] != head_dim / 2 || sin_shape[1] != head_dim / 2 {
            return Err(Error::InvalidArgument {
                arg: "cache",
                reason: format!(
                    "cache second dimension should be {}, got cos: {}, sin: {}",
                    head_dim / 2,
                    cos_shape[1],
                    sin_shape[1]
                ),
            });
        }

        // Narrow caches to current seq_len if needed
        let cos_to_use = if cos_shape[0] > seq_len {
            cos_tensor.narrow(0, 0, seq_len)?
        } else {
            cos_tensor.clone()
        };

        let sin_to_use = if sin_shape[0] > seq_len {
            sin_tensor.narrow(0, 0, seq_len)?
        } else {
            sin_tensor.clone()
        };

        // Verify dtype consistency
        let dtype = x_tensor.dtype();
        if cos_to_use.dtype() != dtype || sin_to_use.dtype() != dtype {
            return Err(Error::InvalidArgument {
                arg: "dtype",
                reason: format!(
                    "all inputs must have same dtype: x={:?}, cos={:?}, sin={:?}",
                    dtype,
                    cos_to_use.dtype(),
                    sin_to_use.dtype()
                ),
            });
        }

        // Select kernel
        let kernel_name = match dtype {
            DType::F32 => "rope_apply_f32",
            DType::F16 => "rope_apply_f16",
            DType::BF16 => "rope_apply_bf16",
            _ => {
                return Err(Error::KernelError {
                    reason: format!("RoPE: unsupported dtype {:?}", dtype),
                });
            }
        };

        // Create output tensor
        let device = x_tensor.device();
        let output = numr::tensor::Tensor::<CudaRuntime>::empty(&x_shape, dtype, device);

        // Get kernel function
        let device_index = device.id();
        let module = kernels::get_or_load_module(self.context(), device_index, ROPE_MODULE)?;
        let func = kernels::get_kernel_function(&module, kernel_name)?;

        // Configure launch
        // Total threads = batch_size * num_heads * seq_len * head_dim (one per element)
        let total_threads = (batch_size * num_heads * seq_len * head_dim) as u32;
        let block_size = 256u32;
        let grid_size = (total_threads + block_size - 1) / block_size;

        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // Extract pointers
        let x_ptr = x_tensor.ptr();
        let cos_ptr = cos_to_use.ptr();
        let sin_ptr = sin_to_use.ptr();
        let out_ptr = output.ptr();

        let b_i32 = batch_size as i32;
        let nh_i32 = num_heads as i32;
        let sl_i32 = seq_len as i32;
        let hd_i32 = head_dim as i32;

        // Launch kernel
        unsafe {
            let mut builder = self.stream().launch_builder(&func);
            builder.arg(&x_ptr);
            builder.arg(&cos_ptr);
            builder.arg(&sin_ptr);
            builder.arg(&out_ptr);
            builder.arg(&b_i32);
            builder.arg(&nh_i32);
            builder.arg(&sl_i32);
            builder.arg(&hd_i32);
            builder.launch(cfg).map_err(|e| Error::KernelError {
                reason: format!("RoPE kernel launch failed: {:?}", e),
            })?;
        }

        Ok(Var::new(output, false))
    }
}
