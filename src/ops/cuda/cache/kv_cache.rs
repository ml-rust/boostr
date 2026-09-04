//! CUDA KV cache operations — fused update, reshape-and-cache, and INT4 append

use crate::error::{Error, Result};
use crate::ops::traits::{Int4GroupSize, KvCacheOps};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use crate::ops::cuda::kernels::{
    self, KV_CACHE_INT4_MODULE, KV_CACHE_UPDATE_MODULE, RESHAPE_AND_CACHE_MODULE,
};

/// Threads per block for `append_kv_int4_*`.
///
/// The kernel's shared reduction buffers (`k_min_s[64]`, `k_max_s[64]`, and
/// the matching V arrays) hold exactly 64 slots. Its reduction loop starts
/// at stride 32 and expects every slot to hold live data. Fewer than 64
/// threads leaves slots uninitialized. More than 64 threads overflows the
/// arrays. The launch must use exactly 64 threads.
const APPEND_KV_INT4_BLOCK: u32 = 64;

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

        let max_seq_len = cache_shape[2];
        let head_dim = cache_shape[3];
        let new_len = new_shape[2];
        let outer_size = cache_shape[0] * cache_shape[1];

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
        let new_shape = new_k.shape();
        let cache_shape = k_cache.shape();

        if new_shape.len() != 3 {
            return Err(Error::InvalidArgument {
                arg: "new_k",
                reason: format!(
                    "expected 3D [batch, num_heads, head_dim], got {}D",
                    new_shape.len()
                ),
            });
        }
        if cache_shape.len() != 4 {
            return Err(Error::InvalidArgument {
                arg: "k_cache",
                reason: format!(
                    "expected 4D [batch, num_heads, max_seq_len, head_dim/2], got {}D",
                    cache_shape.len()
                ),
            });
        }

        let batch_size = new_shape[0];
        let num_heads = new_shape[1];
        let head_dim = new_shape[2];
        let max_seq_len = cache_shape[2];

        if cache_shape[0] != batch_size || cache_shape[1] != num_heads {
            return Err(Error::InvalidArgument {
                arg: "k_cache",
                reason: format!(
                    "k_cache batch/head [{},{}] does not match new_k [{},{}]",
                    cache_shape[0], cache_shape[1], batch_size, num_heads
                ),
            });
        }
        if cache_shape[3] * 2 != head_dim {
            return Err(Error::InvalidArgument {
                arg: "k_cache",
                reason: format!(
                    "k_cache last dim {} must be head_dim/2 for head_dim {}",
                    cache_shape[3], head_dim
                ),
            });
        }
        if position >= max_seq_len {
            return Err(Error::InvalidArgument {
                arg: "position",
                reason: format!("position {} >= max_seq_len {}", position, max_seq_len),
            });
        }

        let required_scale_elems =
            batch_size * num_heads * max_seq_len * head_dim.div_ceil(group_size as usize);

        for (name, t) in [
            ("k_scales", k_scales),
            ("k_zeros", k_zeros),
            ("v_scales", v_scales),
            ("v_zeros", v_zeros),
        ] {
            if t.dtype() != DType::F16 {
                return Err(Error::InvalidArgument {
                    arg: name,
                    reason: format!("append_kv_int4 requires F16, got {:?}", t.dtype()),
                });
            }
            // The kernel indexes scales up to `required_scale_elems`. An
            // undersized tensor writes past its allocation on the device.
            if t.numel() < required_scale_elems {
                return Err(Error::InvalidArgument {
                    arg: name,
                    reason: format!(
                        "append_kv_int4 needs {} elements, got {}",
                        required_scale_elems,
                        t.numel()
                    ),
                });
            }
        }

        let dtype = new_k.dtype();
        let kernel_name = match dtype {
            DType::F32 => "append_kv_int4_fp32",
            DType::F16 => "append_kv_int4_fp16",
            DType::BF16 => "append_kv_int4_bf16",
            _ => {
                return Err(Error::InvalidArgument {
                    arg: "dtype",
                    reason: format!("unsupported dtype {:?} for append_kv_int4", dtype),
                });
            }
        };

        let gs = group_size as usize;
        let groups_per_token = head_dim.div_ceil(gs);

        let device = new_k.device();
        let device_index = device.id();
        let module =
            kernels::get_or_load_module(self.context(), device_index, KV_CACHE_INT4_MODULE)?;
        let func = kernels::get_kernel_function(&module, kernel_name)?;

        let cfg = LaunchConfig {
            grid_dim: ((batch_size * num_heads) as u32, groups_per_token as u32, 1),
            block_dim: (APPEND_KV_INT4_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };

        let nk_ptr = new_k.ptr();
        let nv_ptr = new_v.ptr();
        let kc_ptr = k_cache.ptr();
        let vc_ptr = v_cache.ptr();
        let ks_ptr = k_scales.ptr();
        let kz_ptr = k_zeros.ptr();
        let vs_ptr = v_scales.ptr();
        let vz_ptr = v_zeros.ptr();
        let batch_i32 = batch_size as i32;
        let heads_i32 = num_heads as i32;
        let pos_i32 = position as i32;
        let hd_i32 = head_dim as i32;
        let msl_i32 = max_seq_len as i32;
        let gs_i32 = gs as i32;

        unsafe {
            let mut builder = self.stream().launch_builder(&func);
            builder.arg(&nk_ptr);
            builder.arg(&nv_ptr);
            builder.arg(&kc_ptr);
            builder.arg(&vc_ptr);
            builder.arg(&ks_ptr);
            builder.arg(&kz_ptr);
            builder.arg(&vs_ptr);
            builder.arg(&vz_ptr);
            builder.arg(&batch_i32);
            builder.arg(&heads_i32);
            builder.arg(&pos_i32);
            builder.arg(&hd_i32);
            builder.arg(&msl_i32);
            builder.arg(&gs_i32);
            builder.launch(cfg).map_err(|e| Error::KernelError {
                reason: format!("append_kv_int4 kernel launch failed: {:?}", e),
            })?;
        }

        Ok(())
    }
}
