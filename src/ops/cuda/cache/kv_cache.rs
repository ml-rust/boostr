//! CUDA KV cache operations — fused update, reshape-and-cache, and INT4 append

mod append_kv_int4;
mod batched;
mod block_ops;

use crate::error::{Error, Result};
use crate::ops::traits::{Int4GroupSize, KvCacheOps};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use crate::ops::cuda::kernels::{self, KV_CACHE_UPDATE_MODULE, RESHAPE_AND_CACHE_MODULE};

impl KvCacheOps<CudaRuntime> for CudaClient {
    fn kv_cache_update(
        &self,
        k_cache: &Tensor<CudaRuntime>,
        v_cache: &Tensor<CudaRuntime>,
        new_k: &Tensor<CudaRuntime>,
        new_v: &Tensor<CudaRuntime>,
        position: usize,
    ) -> Result<()> {
        let cache_shape = k_cache.shape();
        let new_shape = new_k.shape();

        if cache_shape.len() != 4 || new_shape.len() != 4 {
            return Err(Error::InvalidArgument {
                arg: "shape",
                reason: "expected 4D [B, H, S, D] tensors".into(),
            });
        }

        let batch = cache_shape[0];
        let num_heads = cache_shape[1];
        let max_seq_len = cache_shape[2];
        let head_dim = cache_shape[3];
        let new_len = new_shape[2];
        let outer_size = batch * num_heads;

        // Validate that batch, heads, and head_dim are compatible.
        if new_shape[0] != batch || new_shape[1] != num_heads || new_shape[3] != head_dim {
            return Err(Error::InvalidArgument {
                arg: "shape",
                reason: format!(
                    "new_k shape [{},{},{},{}] is incompatible with cache shape [{},{},{},{}]",
                    new_shape[0],
                    new_shape[1],
                    new_len,
                    new_shape[3],
                    batch,
                    num_heads,
                    max_seq_len,
                    head_dim,
                ),
            });
        }

        // One kernel serves all four tensors, so they must share a dtype.
        let dtype = k_cache.dtype();
        if v_cache.dtype() != dtype || new_k.dtype() != dtype || new_v.dtype() != dtype {
            return Err(Error::InvalidArgument {
                arg: "dtype",
                reason: format!(
                    "k_cache {:?}, v_cache {:?}, new_k {:?}, new_v {:?} must all match",
                    dtype,
                    v_cache.dtype(),
                    new_k.dtype(),
                    new_v.dtype()
                ),
            });
        }

        if position + new_len > max_seq_len {
            return Err(Error::InvalidArgument {
                arg: "position",
                reason: format!(
                    "position {} + new_len {} > max_seq_len {}",
                    position, new_len, max_seq_len
                ),
            });
        }

        let dtype = k_cache.dtype();
        let kernel_name = match dtype {
            DType::F32 => "kv_cache_update_f32",
            DType::F16 => "kv_cache_update_f16",
            DType::BF16 => "kv_cache_update_bf16",
            _ => {
                return Err(Error::InvalidArgument {
                    arg: "dtype",
                    reason: format!("unsupported dtype {:?} for kv_cache_update", dtype),
                });
            }
        };

        let total_elements = outer_size * new_len * head_dim;
        let threads = 256;
        let blocks = total_elements.div_ceil(threads);

        let device = k_cache.device();
        let device_index = device.id();
        let module =
            kernels::get_or_load_module(self.context(), device_index, KV_CACHE_UPDATE_MODULE)?;
        let func = kernels::get_kernel_function(&module, kernel_name)?;

        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let kc_ptr = k_cache.ptr();
        let vc_ptr = v_cache.ptr();
        let nk_ptr = new_k.ptr();
        let nv_ptr = new_v.ptr();
        let outer_i32 = outer_size as i32;
        let msl_i32 = max_seq_len as i32;
        let nl_i32 = new_len as i32;
        let hd_i32 = head_dim as i32;
        let pos_i32 = position as i32;
        let total_i32 = total_elements as i32;

        unsafe {
            let mut builder = self.stream().launch_builder(&func);
            builder.arg(&kc_ptr);
            builder.arg(&vc_ptr);
            builder.arg(&nk_ptr);
            builder.arg(&nv_ptr);
            builder.arg(&outer_i32);
            builder.arg(&msl_i32);
            builder.arg(&nl_i32);
            builder.arg(&hd_i32);
            builder.arg(&pos_i32);
            builder.arg(&total_i32);
            builder.launch(cfg).map_err(|e| Error::KernelError {
                reason: format!("kv_cache_update kernel launch failed: {:?}", e),
            })?;
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn kv_cache_update_batched(
        &self,
        k_caches: &[&Tensor<CudaRuntime>],
        v_caches: &[&Tensor<CudaRuntime>],
        new_ks: &[&Tensor<CudaRuntime>],
        new_vs: &[&Tensor<CudaRuntime>],
        max_seq_len: usize,
        position: usize,
    ) -> Result<()> {
        batched::kv_cache_update_batched(
            self,
            k_caches,
            v_caches,
            new_ks,
            new_vs,
            max_seq_len,
            position,
        )
    }

    fn reshape_and_cache(
        &self,
        key: &Tensor<CudaRuntime>,
        value: &Tensor<CudaRuntime>,
        key_cache: &Tensor<CudaRuntime>,
        value_cache: &Tensor<CudaRuntime>,
        slot_mapping: &Tensor<CudaRuntime>,
        block_size: usize,
    ) -> Result<()> {
        let k_shape = key.shape();
        let cache_shape = key_cache.shape();

        if k_shape.len() != 3 {
            return Err(Error::InvalidArgument {
                arg: "key",
                reason: format!(
                    "expected 3D [num_tokens, num_heads, head_dim], got {}D",
                    k_shape.len()
                ),
            });
        }
        if cache_shape.len() != 4 {
            return Err(Error::InvalidArgument {
                arg: "key_cache",
                reason: format!(
                    "expected 4D [num_blocks, block_size, num_heads, head_dim], got {}D",
                    cache_shape.len()
                ),
            });
        }

        let num_tokens = k_shape[0];
        let num_heads = k_shape[1];
        let head_dim = k_shape[2];

        let dtype = key.dtype();
        let kernel_name = match dtype {
            DType::F32 => "reshape_and_cache_f32",
            DType::F16 => "reshape_and_cache_f16",
            DType::BF16 => "reshape_and_cache_bf16",
            _ => {
                return Err(Error::InvalidArgument {
                    arg: "dtype",
                    reason: format!("unsupported dtype {:?} for reshape_and_cache", dtype),
                });
            }
        };

        // Grid: (num_tokens, num_heads), Block: ceil(head_dim / vec_size) threads
        let vec_size = match dtype {
            DType::F32 => 4usize,
            _ => 8usize,
        };
        let threads_per_block = head_dim.div_ceil(vec_size);
        let threads = threads_per_block.max(1);

        let device = key.device();
        let device_index = device.id();
        let module =
            kernels::get_or_load_module(self.context(), device_index, RESHAPE_AND_CACHE_MODULE)?;
        let func = kernels::get_kernel_function(&module, kernel_name)?;

        let cfg = LaunchConfig {
            grid_dim: (num_tokens as u32, num_heads as u32, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let k_ptr = key.ptr();
        let v_ptr = value.ptr();
        let kc_ptr = key_cache.ptr();
        let vc_ptr = value_cache.ptr();
        let sm_ptr = slot_mapping.ptr();
        let nt_i32 = num_tokens as i32;
        let nh_i32 = num_heads as i32;
        let hd_i32 = head_dim as i32;
        let bs_i32 = block_size as i32;

        unsafe {
            let mut builder = self.stream().launch_builder(&func);
            builder.arg(&k_ptr);
            builder.arg(&v_ptr);
            builder.arg(&kc_ptr);
            builder.arg(&vc_ptr);
            builder.arg(&sm_ptr);
            builder.arg(&nt_i32);
            builder.arg(&nh_i32);
            builder.arg(&hd_i32);
            builder.arg(&bs_i32);
            builder.launch(cfg).map_err(|e| Error::KernelError {
                reason: format!("reshape_and_cache kernel launch failed: {:?}", e),
            })?;
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn append_kv_int4(
        &self,
        k_cache: &Tensor<CudaRuntime>,
        v_cache: &Tensor<CudaRuntime>,
        k_scales: &Tensor<CudaRuntime>,
        k_zeros: &Tensor<CudaRuntime>,
        v_scales: &Tensor<CudaRuntime>,
        v_zeros: &Tensor<CudaRuntime>,
        new_k: &Tensor<CudaRuntime>,
        new_v: &Tensor<CudaRuntime>,
        position: usize,
        group_size: Int4GroupSize,
    ) -> Result<()> {
        append_kv_int4::append_kv_int4(
            self, k_cache, v_cache, k_scales, k_zeros, v_scales, v_zeros, new_k, new_v, position,
            group_size,
        )
    }

    fn copy_blocks(
        &self,
        key_cache: &Tensor<CudaRuntime>,
        value_cache: &Tensor<CudaRuntime>,
        block_mapping: &Tensor<CudaRuntime>,
        num_heads: usize,
        head_dim: usize,
        block_size: usize,
    ) -> Result<()> {
        block_ops::copy_blocks(
            self,
            key_cache,
            value_cache,
            block_mapping,
            num_heads,
            head_dim,
            block_size,
        )
    }

    fn swap_blocks(
        &self,
        src_cache: &Tensor<CudaRuntime>,
        dst_cache: &Tensor<CudaRuntime>,
        block_mapping: &Tensor<CudaRuntime>,
        num_heads: usize,
        head_dim: usize,
        block_size: usize,
    ) -> Result<()> {
        block_ops::swap_blocks(
            self,
            src_cache,
            dst_cache,
            block_mapping,
            num_heads,
            head_dim,
            block_size,
        )
    }
}
