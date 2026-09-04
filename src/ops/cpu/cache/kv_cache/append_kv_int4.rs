//! `append_kv_int4` CPU reference implementation.
//!
//! Split out of `kv_cache.rs` to keep it under the `cpu/*.rs` 400-line limit.

use crate::error::{Error, Result};
use crate::ops::traits::Int4GroupSize;
use numr::dtype::DType;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

/// Read a float tensor of any dtype into an f32 buffer.
fn read_f32(t: &Tensor<CpuRuntime>, arg: &'static str) -> Result<Vec<f32>> {
    match t.dtype() {
        DType::F32 => Ok(t.to_vec::<f32>()),
        DType::F16 => Ok(t
            .to_vec::<half::f16>()
            .into_iter()
            .map(|v| v.to_f32())
            .collect()),
        DType::BF16 => Ok(t
            .to_vec::<half::bf16>()
            .into_iter()
            .map(|v| v.to_f32())
            .collect()),
        dt => Err(Error::InvalidArgument {
            arg,
            reason: format!("append_kv_int4: unsupported dtype {dt:?}"),
        }),
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn append_kv_int4(
    k_cache: &Tensor<CpuRuntime>,
    v_cache: &Tensor<CpuRuntime>,
    k_scales: &Tensor<CpuRuntime>,
    k_zeros: &Tensor<CpuRuntime>,
    v_scales: &Tensor<CpuRuntime>,
    v_zeros: &Tensor<CpuRuntime>,
    new_k: &Tensor<CpuRuntime>,
    new_v: &Tensor<CpuRuntime>,
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

    let gs = group_size as usize;
    let groups_per_token = head_dim.div_ceil(gs);
    let required_scale_elems = batch_size * num_heads * max_seq_len * groups_per_token;

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
        // The scale writes below index up to `required_scale_elems`. An
        // undersized tensor writes past its allocation.
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

    let k_data = read_f32(new_k, "new_k")?;
    let v_data = read_f32(new_v, "new_v")?;

    let packed_head_dim = head_dim / 2;

    // Raw pointers: packed cache is u8, scales/zeros are F16, both mutated in place.
    let kc_ptr = k_cache.ptr() as *mut u8;
    let vc_ptr = v_cache.ptr() as *mut u8;
    let ks_ptr = k_scales.ptr() as *mut half::f16;
    let kz_ptr = k_zeros.ptr() as *mut half::f16;
    let vs_ptr = v_scales.ptr() as *mut half::f16;
    let vz_ptr = v_zeros.ptr() as *mut half::f16;

    for b in 0..batch_size {
        for h in 0..num_heads {
            let token_base = (b * num_heads + h) * head_dim;
            let cache_row_base =
                (b * num_heads + h) * max_seq_len * packed_head_dim + position * packed_head_dim;
            let scale_row_base =
                (b * num_heads + h) * max_seq_len * groups_per_token + position * groups_per_token;

            for g in 0..groups_per_token {
                let group_start = g * gs;
                let group_end = (group_start + gs).min(head_dim);

                // Per-group asymmetric min-max, same formula as `quantize_kv_int4`:
                // scale = range / 15, zero = min. Matches the CUDA kernel's
                // 1e-8 flat-range guard exactly (append_kv_int4_impl).
                let mut k_min = f32::MAX;
                let mut k_max = f32::MIN;
                let mut v_min = f32::MAX;
                let mut v_max = f32::MIN;
                for i in group_start..group_end {
                    let kv = k_data[token_base + i];
                    let vv = v_data[token_base + i];
                    k_min = k_min.min(kv);
                    k_max = k_max.max(kv);
                    v_min = v_min.min(vv);
                    v_max = v_max.max(vv);
                }

                let k_range = k_max - k_min;
                let k_scale = if k_range > 1e-8 { k_range / 15.0 } else { 1.0 };
                let k_zero = k_min;
                let v_range = v_max - v_min;
                let v_scale = if v_range > 1e-8 { v_range / 15.0 } else { 1.0 };
                let v_zero = v_min;

                let scale_idx = scale_row_base + g;
                unsafe {
                    *ks_ptr.add(scale_idx) = half::f16::from_f32(k_scale);
                    *kz_ptr.add(scale_idx) = half::f16::from_f32(k_zero);
                    *vs_ptr.add(scale_idx) = half::f16::from_f32(v_scale);
                    *vz_ptr.add(scale_idx) = half::f16::from_f32(v_zero);
                }

                let mut i = group_start;
                while i < group_end {
                    let kv0 = k_data[token_base + i];
                    let vv0 = v_data[token_base + i];
                    let (kv1, vv1) = if i + 1 < group_end {
                        (k_data[token_base + i + 1], v_data[token_base + i + 1])
                    } else {
                        (0.0, 0.0)
                    };

                    let kq0 = ((kv0 - k_zero) / k_scale).round().clamp(0.0, 15.0) as u8;
                    let kq1 = ((kv1 - k_zero) / k_scale).round().clamp(0.0, 15.0) as u8;
                    let vq0 = ((vv0 - v_zero) / v_scale).round().clamp(0.0, 15.0) as u8;
                    let vq1 = ((vv1 - v_zero) / v_scale).round().clamp(0.0, 15.0) as u8;

                    let byte_idx = cache_row_base + i / 2;
                    unsafe {
                        *kc_ptr.add(byte_idx) = (kq0 & 0xF) | ((kq1 & 0xF) << 4);
                        *vc_ptr.add(byte_idx) = (vq0 & 0xF) | ((vq1 & 0xF) << 4);
                    }

                    i += 2;
                }
            }
        }
    }

    Ok(())
}
