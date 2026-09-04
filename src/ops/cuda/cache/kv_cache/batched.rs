//! `kv_cache_update_batched` CUDA launcher.
//!
//! Split out of `kv_cache.rs` to keep it under the `cuda/*.rs` 400-line
//! limit. Updates every layer's K and V cache in one 2D-grid launch instead
//! of one `kv_cache_update` launch per layer.

use crate::error::{Error, Result};
use crate::ops::cuda::kernels::{self, KV_CACHE_UPDATE_MODULE};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

/// Upload a host array of device pointers to a device buffer, so the kernel
/// can index it as `T**`. This uploads only the small pointer table, never
/// tensor contents.
fn upload_ptrs(client: &CudaClient, host: &[u64]) -> Result<cudarc::driver::CudaSlice<u64>> {
    let stream = client.stream_arc();
    let mut dev = unsafe {
        stream
            .alloc::<u64>(host.len())
            .map_err(|e| Error::KernelError {
                reason: format!("kv_cache_update_batched: alloc pointer buffer: {:?}", e),
            })?
    };
    stream
        .memcpy_htod(host, &mut dev)
        .map_err(|e| Error::KernelError {
            reason: format!("kv_cache_update_batched: upload pointer buffer: {:?}", e),
        })?;
    Ok(dev)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn kv_cache_update_batched(
    client: &CudaClient,
    k_caches: &[&Tensor<CudaRuntime>],
    v_caches: &[&Tensor<CudaRuntime>],
    new_ks: &[&Tensor<CudaRuntime>],
    new_vs: &[&Tensor<CudaRuntime>],
    max_seq_len: usize,
    position: usize,
) -> Result<()> {
    let num_layers = k_caches.len();
    if num_layers == 0
        || v_caches.len() != num_layers
        || new_ks.len() != num_layers
        || new_vs.len() != num_layers
    {
        return Err(Error::InvalidArgument {
            arg: "k_caches",
            reason: format!(
                "kv_cache_update_batched requires non-empty, equal-length slices, got {} k_caches, {} v_caches, {} new_ks, {} new_vs",
                num_layers,
                v_caches.len(),
                new_ks.len(),
                new_vs.len(),
            ),
        });
    }

    let dtype = k_caches[0].dtype();
    let cache_shape = k_caches[0].shape().to_vec();
    let new_shape = new_ks[0].shape().to_vec();

    if cache_shape.len() != 4 || new_shape.len() != 4 {
        return Err(Error::InvalidArgument {
            arg: "k_caches",
            reason: "expected 4D [B, H, S, D] tensors".into(),
        });
    }
    if cache_shape[2] != max_seq_len {
        return Err(Error::InvalidArgument {
            arg: "max_seq_len",
            reason: format!(
                "cache seq dim {} does not match max_seq_len {max_seq_len}",
                cache_shape[2]
            ),
        });
    }

    let outer_size = cache_shape[0] * cache_shape[1];
    let head_dim = cache_shape[3];
    let new_len = new_shape[2];

    // Validate that batch, heads, and head_dim are compatible.
    if new_shape[0] != cache_shape[0] || new_shape[1] != cache_shape[1] || new_shape[3] != head_dim
    {
        return Err(Error::InvalidArgument {
            arg: "shape",
            reason: format!(
                "new_k shape [{},{},{},{}] is incompatible with cache shape [{},{},{},{}]",
                new_shape[0],
                new_shape[1],
                new_len,
                new_shape[3],
                cache_shape[0],
                cache_shape[1],
                max_seq_len,
                head_dim,
            ),
        });
    }

    if position + new_len > max_seq_len {
        return Err(Error::InvalidArgument {
            arg: "position",
            reason: format!("position {position} + new_len {new_len} > max_seq_len {max_seq_len}"),
        });
    }

    for layer in 0..num_layers {
        let (kc, vc, nk, nv) = (
            k_caches[layer],
            v_caches[layer],
            new_ks[layer],
            new_vs[layer],
        );
        if kc.shape() != cache_shape.as_slice() || vc.shape() != cache_shape.as_slice() {
            return Err(Error::InvalidArgument {
                arg: "k_caches",
                reason: format!(
                    "layer {layer}: cache shape {:?} does not match layer 0's {:?} — every layer must share one shape",
                    kc.shape(),
                    cache_shape
                ),
            });
        }
        if nk.shape() != new_shape.as_slice() || nv.shape() != new_shape.as_slice() {
            return Err(Error::InvalidArgument {
                arg: "new_ks",
                reason: format!(
                    "layer {layer}: new tensor shape {:?} does not match layer 0's {:?} — every layer must share one shape",
                    nk.shape(),
                    new_shape
                ),
            });
        }
        if kc.dtype() != dtype || vc.dtype() != dtype || nk.dtype() != dtype || nv.dtype() != dtype
        {
            return Err(Error::InvalidArgument {
                arg: "dtype",
                reason: format!(
                    "layer {layer}: all tensors must share one dtype, expected {:?}",
                    dtype
                ),
            });
        }
    }

    let kernel_name = match dtype {
        DType::F32 => "kv_cache_update_batched_f32",
        DType::F16 => "kv_cache_update_batched_f16",
        DType::BF16 => "kv_cache_update_batched_bf16",
        _ => {
            return Err(Error::InvalidArgument {
                arg: "dtype",
                reason: format!("unsupported dtype {:?} for kv_cache_update_batched", dtype),
            });
        }
    };

    let k_ptrs_host: Vec<u64> = k_caches.iter().map(|t| t.ptr()).collect();
    let v_ptrs_host: Vec<u64> = v_caches.iter().map(|t| t.ptr()).collect();
    let nk_ptrs_host: Vec<u64> = new_ks.iter().map(|t| t.ptr()).collect();
    let nv_ptrs_host: Vec<u64> = new_vs.iter().map(|t| t.ptr()).collect();

    let k_ptrs_dev = upload_ptrs(client, &k_ptrs_host)?;
    let v_ptrs_dev = upload_ptrs(client, &v_ptrs_host)?;
    let nk_ptrs_dev = upload_ptrs(client, &nk_ptrs_host)?;
    let nv_ptrs_dev = upload_ptrs(client, &nv_ptrs_host)?;

    let total_elements_per_layer = outer_size * new_len * head_dim;
    let threads = 256;
    let blocks_x = total_elements_per_layer.div_ceil(threads);

    let device = k_caches[0].device();
    let device_index = device.id();
    let module =
        kernels::get_or_load_module(client.context(), device_index, KV_CACHE_UPDATE_MODULE)?;
    let func = kernels::get_kernel_function(&module, kernel_name)?;

    let cfg = LaunchConfig {
        grid_dim: (blocks_x as u32, num_layers as u32, 1),
        block_dim: (threads as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let num_layers_i32 = num_layers as i32;
    let outer_i32 = outer_size as i32;
    let msl_i32 = max_seq_len as i32;
    let nl_i32 = new_len as i32;
    let hd_i32 = head_dim as i32;
    let pos_i32 = position as i32;
    let total_i32 = total_elements_per_layer as i32;

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&k_ptrs_dev);
        builder.arg(&v_ptrs_dev);
        builder.arg(&nk_ptrs_dev);
        builder.arg(&nv_ptrs_dev);
        builder.arg(&num_layers_i32);
        builder.arg(&outer_i32);
        builder.arg(&msl_i32);
        builder.arg(&nl_i32);
        builder.arg(&hd_i32);
        builder.arg(&pos_i32);
        builder.arg(&total_i32);
        builder.launch(cfg).map_err(|e| Error::KernelError {
            reason: format!("kv_cache_update_batched kernel launch failed: {:?}", e),
        })?;
    }

    Ok(())
}
