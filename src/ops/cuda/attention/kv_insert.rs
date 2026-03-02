//! KV insert kernel dispatch — writes one decode step's K/V into the full-capacity
//! cache buffer at a device-side position. Used by the CUDA graph decode loop.

use crate::error::{Error, Result};
use crate::ops::cuda::kernels::{self, KV_INSERT_MODULE};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

/// Insert one token's K and V into the full-capacity cache at `write_pos_ptr`.
///
/// # Arguments
/// - `k_new` / `v_new` : new K/V for this step, shape `[B, H_kv, 1, D]`
/// - `k_cache` / `v_cache` : full-capacity cache `[B, H_kv, capacity, D]`
/// - `write_pos_ptr` : raw device pointer to an i32 indicating where to insert
///
/// The kernel reads `*write_pos_ptr` once at the start; it must be updated before
/// each graph replay via an H2D copy.
pub fn kv_insert(
    client: &CudaClient,
    k_new: &Tensor<CudaRuntime>,
    v_new: &Tensor<CudaRuntime>,
    k_cache: &Tensor<CudaRuntime>,
    v_cache: &Tensor<CudaRuntime>,
    write_pos_ptr: u64,
) -> Result<()> {
    let shape = k_new.shape();
    if shape.len() != 4 {
        return Err(Error::InvalidArgument {
            arg: "k_new",
            reason: format!("expected 4D [B, H_kv, 1, D], got {}D", shape.len()),
        });
    }
    let b = shape[0];
    let h_kv = shape[1];
    let d = shape[3];

    let cache_shape = k_cache.shape();
    if cache_shape.len() != 4
        || cache_shape[0] != b
        || cache_shape[1] != h_kv
        || cache_shape[3] != d
    {
        return Err(Error::InvalidArgument {
            arg: "k_cache",
            reason: format!(
                "expected [B={}, H_kv={}, capacity, D={}], got {:?}",
                b, h_kv, d, cache_shape
            ),
        });
    }
    let capacity = cache_shape[2];

    let dtype = k_new.dtype();
    let kernel_name = match dtype {
        DType::F32 => "kv_insert_f32",
        DType::F16 => "kv_insert_f16",
        _ => {
            return Err(Error::InvalidArgument {
                arg: "dtype",
                reason: format!("kv_insert only supports F32/F16, got {:?}", dtype),
            });
        }
    };

    let device = k_new.device();
    let device_index = device.id();
    let module = kernels::get_or_load_module(client.context(), device_index, KV_INSERT_MODULE)?;
    let func = kernels::get_kernel_function(&module, kernel_name)?;

    let total = b * h_kv * d;
    const BLOCK: u32 = 256;
    let grid = ((total as u32 + BLOCK - 1) / BLOCK, 1, 1);
    let cfg = LaunchConfig {
        grid_dim: grid,
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    let kn_ptr = k_new.ptr();
    let vn_ptr = v_new.ptr();
    let kc_ptr = k_cache.ptr();
    let vc_ptr = v_cache.ptr();
    let b_i32 = b as i32;
    let h_i32 = h_kv as i32;
    let d_i32 = d as i32;
    let cap_i32 = capacity as i32;

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&kn_ptr);
        builder.arg(&vn_ptr);
        builder.arg(&kc_ptr);
        builder.arg(&vc_ptr);
        builder.arg(&b_i32);
        builder.arg(&h_i32);
        builder.arg(&d_i32);
        builder.arg(&cap_i32);
        builder.arg(&write_pos_ptr);
        builder.launch(cfg).map_err(|e| Error::KernelError {
            reason: format!("kv_insert kernel launch failed: {:?}", e),
        })?;
    }

    Ok(())
}
