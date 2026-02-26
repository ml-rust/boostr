//! CUDA implementation of RoPEOps — fused kernel dispatch

use crate::error::{Error, Result};
use crate::ops::cuda::kernels::{self, ROPE_INTERLEAVED_MODULE, ROPE_MODULE, ROPE_YARN_MODULE};
use crate::ops::traits::RoPEOps;
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};

/// Validate shapes and narrow caches. Returns (batch_size, num_heads, seq_len, head_dim, dtype,
/// narrowed cos tensor, narrowed sin tensor, device).
fn validate_rope_inputs(
    x: &Var<CudaRuntime>,
    cos_cache: &Var<CudaRuntime>,
    sin_cache: &Var<CudaRuntime>,
) -> Result<(
    usize,
    usize,
    usize,
    usize,
    DType,
    numr::tensor::Tensor<CudaRuntime>,
    numr::tensor::Tensor<CudaRuntime>,
    numr::runtime::cuda::CudaDevice,
)> {
    let x_tensor = x.tensor();
    let cos_tensor = cos_cache.tensor();
    let sin_tensor = sin_cache.tensor();

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

    let device = x_tensor.device().clone();

    Ok((
        batch_size, num_heads, seq_len, head_dim, dtype, cos_to_use, sin_to_use, device,
    ))
}

fn select_kernel_name(prefix: &str, dtype: DType) -> Result<&'static str> {
    match (prefix, dtype) {
        ("rope_apply", DType::F32) => Ok("rope_apply_f32"),
        ("rope_apply", DType::F16) => Ok("rope_apply_f16"),
        ("rope_apply", DType::BF16) => Ok("rope_apply_bf16"),
        ("rope_interleaved", DType::F32) => Ok("rope_interleaved_f32"),
        ("rope_interleaved", DType::F16) => Ok("rope_interleaved_f16"),
        ("rope_interleaved", DType::BF16) => Ok("rope_interleaved_bf16"),
        ("rope_yarn", DType::F32) => Ok("rope_yarn_f32"),
        ("rope_yarn", DType::F16) => Ok("rope_yarn_f16"),
        ("rope_yarn", DType::BF16) => Ok("rope_yarn_bf16"),
        _ => Err(Error::KernelError {
            reason: format!("RoPE {}: unsupported dtype {:?}", prefix, dtype),
        }),
    }
}

impl RoPEOps<CudaRuntime> for CudaClient {
    fn apply_rope(
        &self,
        x: &Var<CudaRuntime>,
        cos_cache: &Var<CudaRuntime>,
        sin_cache: &Var<CudaRuntime>,
    ) -> Result<Var<CudaRuntime>> {
        let (batch_size, num_heads, seq_len, head_dim, dtype, cos_to_use, sin_to_use, device) =
            validate_rope_inputs(x, cos_cache, sin_cache)?;

        let kernel_name = select_kernel_name("rope_apply", dtype)?;
        let x_shape = x.tensor().shape().to_vec();
        let output = numr::tensor::Tensor::<CudaRuntime>::empty(&x_shape, dtype, &device);

        let device_index = device.id();
        let module = kernels::get_or_load_module(self.context(), device_index, ROPE_MODULE)?;
        let func = kernels::get_kernel_function(&module, kernel_name)?;

        let total_threads = (batch_size * num_heads * seq_len * head_dim) as u32;
        let block_size = 256u32;
        let grid_size = (total_threads + block_size - 1) / block_size;

        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let x_ptr = x.tensor().ptr();
        let cos_ptr = cos_to_use.ptr();
        let sin_ptr = sin_to_use.ptr();
        let out_ptr = output.ptr();

        let b_i32 = batch_size as i32;
        let nh_i32 = num_heads as i32;
        let sl_i32 = seq_len as i32;
        let hd_i32 = head_dim as i32;

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

    fn apply_rope_interleaved(
        &self,
        x: &Var<CudaRuntime>,
        cos_cache: &Var<CudaRuntime>,
        sin_cache: &Var<CudaRuntime>,
    ) -> Result<Var<CudaRuntime>> {
        let (batch_size, num_heads, seq_len, head_dim, dtype, cos_to_use, sin_to_use, device) =
            validate_rope_inputs(x, cos_cache, sin_cache)?;

        let kernel_name = select_kernel_name("rope_interleaved", dtype)?;
        let x_shape = x.tensor().shape().to_vec();
        let output = numr::tensor::Tensor::<CudaRuntime>::empty(&x_shape, dtype, &device);

        let device_index = device.id();
        let module =
            kernels::get_or_load_module(self.context(), device_index, ROPE_INTERLEAVED_MODULE)?;
        let func = kernels::get_kernel_function(&module, kernel_name)?;

        // One thread per pair (half the total elements)
        let total_pairs = (batch_size * num_heads * seq_len * head_dim / 2) as u32;
        let block_size = 256u32;
        let grid_size = (total_pairs + block_size - 1) / block_size;

        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let x_ptr = x.tensor().ptr();
        let cos_ptr = cos_to_use.ptr();
        let sin_ptr = sin_to_use.ptr();
        let out_ptr = output.ptr();

        let b_i32 = batch_size as i32;
        let nh_i32 = num_heads as i32;
        let sl_i32 = seq_len as i32;
        let hd_i32 = head_dim as i32;

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
                reason: format!("RoPE interleaved kernel launch failed: {:?}", e),
            })?;
        }

        Ok(Var::new(output, false))
    }

    fn apply_rope_yarn(
        &self,
        x: &Var<CudaRuntime>,
        cos_cache: &Var<CudaRuntime>,
        sin_cache: &Var<CudaRuntime>,
        attn_scale: f32,
    ) -> Result<Var<CudaRuntime>> {
        let (batch_size, num_heads, seq_len, head_dim, dtype, cos_to_use, sin_to_use, device) =
            validate_rope_inputs(x, cos_cache, sin_cache)?;

        let kernel_name = select_kernel_name("rope_yarn", dtype)?;
        let x_shape = x.tensor().shape().to_vec();
        let output = numr::tensor::Tensor::<CudaRuntime>::empty(&x_shape, dtype, &device);

        let device_index = device.id();
        let module = kernels::get_or_load_module(self.context(), device_index, ROPE_YARN_MODULE)?;
        let func = kernels::get_kernel_function(&module, kernel_name)?;

        let total_threads = (batch_size * num_heads * seq_len * head_dim) as u32;
        let block_size = 256u32;
        let grid_size = (total_threads + block_size - 1) / block_size;

        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let x_ptr = x.tensor().ptr();
        let cos_ptr = cos_to_use.ptr();
        let sin_ptr = sin_to_use.ptr();
        let out_ptr = output.ptr();

        let b_i32 = batch_size as i32;
        let nh_i32 = num_heads as i32;
        let sl_i32 = seq_len as i32;
        let hd_i32 = head_dim as i32;

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
            builder.arg(&attn_scale);
            builder.launch(cfg).map_err(|e| Error::KernelError {
                reason: format!("RoPE YaRN kernel launch failed: {:?}", e),
            })?;
        }

        Ok(Var::new(output, false))
    }
}
