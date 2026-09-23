//! KV insert kernel dispatch — writes one decode step's K/V into the full-capacity
//! cache buffer at a device-side per-row position. Used by the CUDA graph decode
//! loop.

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
/// - `write_pos_ptr` : raw device pointer to a `[B]` i32 buffer; batch row `b`
///   inserts at `write_pos_ptr[b]`, so lanes at different lengths share one
///   launch. At `B == 1` this is the single-element buffer it has always been.
///
/// The kernel reads `write_pos_ptr[b]` once per thread; the buffer must be
/// updated before each graph replay via a device-side write.
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
    for (name, t) in [("v_new", v_new), ("k_cache", k_cache), ("v_cache", v_cache)] {
        if t.dtype() != dtype {
            return Err(Error::InvalidArgument {
                arg: name,
                reason: format!(
                    "kv_insert: dtype {:?} != k_new dtype {:?}",
                    t.dtype(),
                    dtype
                ),
            });
        }
    }
    // The 16-bit kernel copies elements without arithmetic, so F16 and BF16
    // share it.
    let kernel_name = match dtype {
        DType::F32 => "kv_insert_f32",
        DType::F16 | DType::BF16 => "kv_insert_f16",
        _ => {
            return Err(Error::InvalidArgument {
                arg: "dtype",
                reason: format!("kv_insert only supports F32/F16/BF16, got {:?}", dtype),
            });
        }
    };

    let device = k_new.device();
    let device_index = device.id();
    let module = kernels::get_or_load_module(client.context(), device_index, KV_INSERT_MODULE)?;
    let func = kernels::get_kernel_function(&module, kernel_name)?;

    let total = b * h_kv * d;
    const BLOCK: u32 = 256;
    let grid = ((total as u32).div_ceil(BLOCK), 1, 1);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cuda_setup;

    const B: usize = 3;
    const H_KV: usize = 2;
    const D: usize = 8;
    const CAPACITY: usize = 6;
    /// Row `b` inserts at `WRITE_POS[b]`; all three differ, so a kernel that
    /// reads one shared position lands two rows in the wrong slot.
    const WRITE_POS: [i32; B] = [0, 3, 5];
    /// Value every cache slot holds before the insert.
    const BLANK: f32 = -1.0;

    /// Each lane writes its own K/V at its own slot, and no other slot moves.
    #[test]
    fn each_lane_writes_at_its_own_position() {
        let Some(cuda) = cuda_setup() else { return };
        let (client, device) = (&cuda.client, &cuda.device);

        let new_len = B * H_KV * D;
        let k_vals: Vec<f32> = (0..new_len).map(|i| i as f32 + 1.0).collect();
        let v_vals: Vec<f32> = (0..new_len).map(|i| -(i as f32) - 1.0).collect();
        let new_shape = [B, H_KV, 1, D];
        let k_new = Tensor::<CudaRuntime>::from_slice(&k_vals, &new_shape, device).expect("k_new");
        let v_new = Tensor::<CudaRuntime>::from_slice(&v_vals, &new_shape, device).expect("v_new");

        let cache_shape = [B, H_KV, CAPACITY, D];
        let blank = vec![BLANK; B * H_KV * CAPACITY * D];
        let k_cache =
            Tensor::<CudaRuntime>::from_slice(&blank, &cache_shape, device).expect("k_cache");
        let v_cache =
            Tensor::<CudaRuntime>::from_slice(&blank, &cache_shape, device).expect("v_cache");
        let write_pos =
            Tensor::<CudaRuntime>::from_slice(&WRITE_POS, &[B], device).expect("write_pos");

        kv_insert(client, &k_new, &v_new, &k_cache, &v_cache, write_pos.ptr()).expect("kv_insert");

        let k_out = k_cache.to_vec::<f32>();
        let v_out = v_cache.to_vec::<f32>();
        for (b, &lane_pos) in WRITE_POS.iter().enumerate() {
            for h in 0..H_KV {
                for pos in 0..CAPACITY {
                    for d in 0..D {
                        let dst = ((b * H_KV + h) * CAPACITY + pos) * D + d;
                        let src = (b * H_KV + h) * D + d;
                        let (want_k, want_v) = if pos as i32 == lane_pos {
                            (k_vals[src], v_vals[src])
                        } else {
                            (BLANK, BLANK)
                        };
                        assert_eq!(
                            k_out[dst], want_k,
                            "K at (b={b}, h={h}, pos={pos}, d={d}): {} != {want_k}",
                            k_out[dst]
                        );
                        assert_eq!(
                            v_out[dst], want_v,
                            "V at (b={b}, h={h}, pos={pos}, d={d}): {} != {want_v}",
                            v_out[dst]
                        );
                    }
                }
            }
        }
    }
}
