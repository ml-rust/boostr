//! Position-id-aware (packed/varlen) split-half RoPE — impl_generic.

use crate::error::{Error, Result};
use numr::autograd::{Var, var_add, var_cat, var_mul, var_narrow, var_reshape, var_sub};
use numr::ops::{IndexingOps, ScalarOps, ShapeOps, TensorOps, TypeConversionOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Apply position-id-aware (packed) split-half RoPE.
///
/// # Arguments
///
/// - `x`: `[total_tokens, num_heads, head_dim]`
/// - `cos_cache`: `[max_seq_len, head_dim/2]`
/// - `sin_cache`: `[max_seq_len, head_dim/2]`
/// - `position_ids`: `[total_tokens]` integer tensor (I32 or I64)
///
/// # Numerics
///
/// For token `t` with position `p = position_ids[t]`, head `h`, pair index `d`:
/// ```text
/// out[t,h,d]       = x[t,h,d] * cos[p,d] - x[t,h,d+D/2] * sin[p,d]
/// out[t,h,d+D/2]   = x[t,h,d] * sin[p,d] + x[t,h,d+D/2] * cos[p,d]
/// ```
pub fn apply_rope_packed_impl<R, C>(
    client: &C,
    x: &Var<R>,
    cos_cache: &Var<R>,
    sin_cache: &Var<R>,
    position_ids: &Tensor<R>,
) -> Result<Var<R>>
where
    R: Runtime<DType = numr::dtype::DType>,
    C: RuntimeClient<R> + ScalarOps<R> + ShapeOps<R> + TypeConversionOps<R> + IndexingOps<R>,
    R::Client: TensorOps<R> + ShapeOps<R> + TypeConversionOps<R>,
{
    // --- Validate inputs ---
    let x_shape = x.tensor().shape().to_vec();
    if x_shape.len() != 3 {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: format!(
                "expected 3D [total_tokens, num_heads, head_dim], got {}D",
                x_shape.len()
            ),
        });
    }

    let total_tokens = x_shape[0];
    let num_heads = x_shape[1];
    let head_dim = x_shape[2];

    if !head_dim.is_multiple_of(2) {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: format!("head_dim D={} must be even for RoPE", head_dim),
        });
    }

    let half_d = head_dim / 2;

    let pid_shape = position_ids.shape();
    if pid_shape.len() != 1 || pid_shape[0] != total_tokens {
        return Err(Error::InvalidArgument {
            arg: "position_ids",
            reason: format!(
                "expected 1D [total_tokens={}], got {:?}",
                total_tokens, pid_shape
            ),
        });
    }

    let cos_shape = cos_cache.tensor().shape();
    let sin_shape = sin_cache.tensor().shape();
    if cos_shape.len() != 2 || cos_shape[1] != half_d {
        return Err(Error::InvalidArgument {
            arg: "cos_cache",
            reason: format!("expected [max_seq_len, {}], got {:?}", half_d, cos_shape),
        });
    }
    if sin_shape.len() != 2 || sin_shape[1] != half_d {
        return Err(Error::InvalidArgument {
            arg: "sin_cache",
            reason: format!("expected [max_seq_len, {}], got {:?}", half_d, sin_shape),
        });
    }

    // --- Gather cos/sin rows by position_ids ---
    // embedding_lookup(cache[max_seq_len, D/2], position_ids[total_tokens])
    //   → [total_tokens, D/2]
    // Uses embedding_lookup (not gather) because it is CUDA-graph-capture-safe:
    // shape/strides are passed as kernel args, not device-side arrays.
    let cos_gathered = client
        .embedding_lookup(cos_cache.tensor(), position_ids)
        .map_err(Error::Numr)?;
    let sin_gathered = client
        .embedding_lookup(sin_cache.tensor(), position_ids)
        .map_err(Error::Numr)?;

    // --- Cast gathered cos/sin to x dtype if needed ---
    let x_dtype = x.tensor().dtype();
    let cos_matched = if cos_gathered.dtype() != x_dtype {
        let v = numr::autograd::var_cast(&Var::new(cos_gathered, false), x_dtype, client)
            .map_err(Error::Numr)?;
        v.tensor().clone()
    } else {
        cos_gathered
    };
    let sin_matched = if sin_gathered.dtype() != x_dtype {
        let v = numr::autograd::var_cast(&Var::new(sin_gathered, false), x_dtype, client)
            .map_err(Error::Numr)?;
        v.tensor().clone()
    } else {
        sin_gathered
    };

    // --- Reshape to [total_tokens, 1, D/2] so they broadcast over num_heads ---
    let cos_reshaped = var_reshape(&Var::new(cos_matched, false), &[total_tokens, 1, half_d])
        .map_err(Error::Numr)?;
    let sin_reshaped = var_reshape(&Var::new(sin_matched, false), &[total_tokens, 1, half_d])
        .map_err(Error::Numr)?;

    // --- Split x on last dim ---
    let x1 = var_narrow(x, -1, 0, half_d).map_err(Error::Numr)?;
    let x2 = var_narrow(x, -1, half_d, half_d).map_err(Error::Numr)?;

    // out1 = x1 * cos - x2 * sin
    let x1_cos = var_mul(&x1, &cos_reshaped, client).map_err(Error::Numr)?;
    let x2_sin = var_mul(&x2, &sin_reshaped, client).map_err(Error::Numr)?;
    let out1 = var_sub(&x1_cos, &x2_sin, client).map_err(Error::Numr)?;

    // out2 = x1 * sin + x2 * cos
    let x1_sin = var_mul(&x1, &sin_reshaped, client).map_err(Error::Numr)?;
    let x2_cos = var_mul(&x2, &cos_reshaped, client).map_err(Error::Numr)?;
    let out2 = var_add(&x1_sin, &x2_cos, client).map_err(Error::Numr)?;

    // Concatenate back along last dim → [total_tokens, num_heads, head_dim]
    let _ = num_heads; // used in shape validation above; cat restores it
    var_cat(&[&out1, &out2], -1, client).map_err(Error::Numr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::impl_generic::attention::apply_rope_impl;
    use crate::test_utils::cpu_setup;
    use numr::autograd::Var;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

    // Helper: build position_ids [0, 1, 2, ..., n-1]
    fn seq_pids(n: usize, device: &numr::runtime::cpu::CpuDevice) -> Tensor<CpuRuntime> {
        let ids: Vec<i32> = (0..n as i32).collect();
        Tensor::<CpuRuntime>::from_slice(&ids, &[n], device)
    }

    #[test]
    fn test_rope_packed_identity_cos1_sin0() {
        // cos=1, sin=0 → RoPE is identity
        let (client, device) = cpu_setup();
        let total_tokens = 3;
        let num_heads = 2;
        let head_dim = 8;
        let half_d = head_dim / 2;
        let max_seq = 8;

        let x_data: Vec<f32> = (0..total_tokens * num_heads * head_dim)
            .map(|i| i as f32)
            .collect();
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(
                &x_data,
                &[total_tokens, num_heads, head_dim],
                &device,
            ),
            false,
        );
        let cos = Var::new(
            Tensor::<CpuRuntime>::from_slice(
                &vec![1.0f32; max_seq * half_d],
                &[max_seq, half_d],
                &device,
            ),
            false,
        );
        let sin = Var::new(
            Tensor::<CpuRuntime>::from_slice(
                &vec![0.0f32; max_seq * half_d],
                &[max_seq, half_d],
                &device,
            ),
            false,
        );
        let pids = seq_pids(total_tokens, &device);

        let out = apply_rope_packed_impl(&client, &x, &cos, &sin, &pids)
            .expect("apply_rope_packed_impl failed");
        let out_data = out.tensor().contiguous().unwrap().to_vec::<f32>();

        assert_eq!(out_data.len(), x_data.len());
        for (i, (&a, &b)) in out_data.iter().zip(x_data.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "identity test mismatch at {i}: got {a}, expected {b}"
            );
        }
    }

    #[test]
    fn test_rope_packed_position_reset() {
        // Three tokens: sequence 0 has tokens [0,1], sequence 1 has token [0].
        // position_ids = [0, 1, 0]  (the third token uses position 0 again).
        // Verify that out[2] equals what you'd get from applying RoPE with position 0.
        let (client, device) = cpu_setup();
        let num_heads = 1;
        let head_dim = 4;
        let half_d = head_dim / 2;
        let max_seq = 8;

        // Distinct cos/sin values per position so the test is non-trivial
        let cos_data: Vec<f32> = (0..max_seq * half_d)
            .map(|i| (i as f32 * 0.5).cos())
            .collect();
        let sin_data: Vec<f32> = (0..max_seq * half_d)
            .map(|i| (i as f32 * 0.5).sin())
            .collect();

        let cos = Var::new(
            Tensor::<CpuRuntime>::from_slice(&cos_data, &[max_seq, half_d], &device),
            false,
        );
        let sin = Var::new(
            Tensor::<CpuRuntime>::from_slice(&sin_data, &[max_seq, half_d], &device),
            false,
        );

        let x_data: Vec<f32> = vec![
            // token 0: heads × dim = 1×4
            1.0, 2.0, 3.0, 4.0, // token 1
            5.0, 6.0, 7.0, 8.0, // token 2 (same as token 0 to make comparison easy)
            1.0, 2.0, 3.0, 4.0,
        ];
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&x_data, &[3, num_heads, head_dim], &device),
            false,
        );

        // position_ids: token0=pos0, token1=pos1, token2=pos0 (reset!)
        let pids = Tensor::<CpuRuntime>::from_slice(&[0i32, 1, 0], &[3], &device);

        let out = apply_rope_packed_impl(&client, &x, &cos, &sin, &pids)
            .expect("apply_rope_packed_impl failed");
        let out_data = out.tensor().contiguous().unwrap().to_vec::<f32>();

        // Token 0 and token 2 have same x value AND same position id (0),
        // so their outputs must be equal.
        for d in 0..head_dim {
            let t0_val = out_data[d];
            let t2_val = out_data[2 * head_dim + d];
            assert!(
                (t0_val - t2_val).abs() < 1e-5,
                "packed reset: token0 and token2 should match at dim {d}: {t0_val} vs {t2_val}"
            );
        }

        // Token 1 (position 1) must differ from token 0 (position 0) in at least one element
        // (non-trivial cos/sin ensures this).
        let any_diff = (0..head_dim).any(|d| (out_data[d] - out_data[head_dim + d]).abs() > 1e-5);
        assert!(
            any_diff,
            "tokens at different positions should produce different RoPE outputs"
        );
    }

    #[test]
    fn test_rope_packed_matches_standard_single_sequence() {
        // For a single complete sequence (B=1), apply_rope_packed with
        // position_ids=[0..S-1] must equal apply_rope on x reshaped to [1, H, S, D].
        let (client, device) = cpu_setup();
        let s = 4;
        let h = 2;
        let d = 8;
        let half_d = d / 2;

        let x_data: Vec<f32> = (0..s * h * d).map(|i| (i as f32 * 0.1).sin()).collect();
        let cos_data: Vec<f32> = (0..s * half_d).map(|i| (i as f32 * 0.3).cos()).collect();
        let sin_data: Vec<f32> = (0..s * half_d).map(|i| (i as f32 * 0.3).sin()).collect();

        let cos = Var::new(
            Tensor::<CpuRuntime>::from_slice(&cos_data, &[s, half_d], &device),
            false,
        );
        let sin = Var::new(
            Tensor::<CpuRuntime>::from_slice(&sin_data, &[s, half_d], &device),
            false,
        );

        // Packed: x is [S*H, 1, D] — wait, spec says [total_tokens, num_heads, head_dim]
        // For a single sequence: total_tokens=S, num_heads=H.
        // But standard rope expects [B, H, S, D]. We need to permute.
        //
        // Standard RoPE: x_4d[b=0, h, s, d] = x_data[h*S*D + s*D + d] for row-major
        // Packed RoPE:   x_3d[t=s, h, d]    = x_data[s*H*D + h*D + d]
        //
        // They differ in layout. Use packed layout (tokens-first) and reshape standard:
        // packed input layout: [S, H, D] — x_data[s, h, d]
        // standard layout:     [1, H, S, D] — we build x_4d with same per-position values.
        //
        // Simplest: use the same raw buffer but different shapes, verify outputs agree
        // after reshaping back.
        //
        // Build x_packed as [S, H, D]:
        let x_packed = Var::new(
            Tensor::<CpuRuntime>::from_slice(&x_data, &[s, h, d], &device),
            false,
        );
        let pids = seq_pids(s, &device);

        let out_packed =
            apply_rope_packed_impl(&client, &x_packed, &cos, &sin, &pids).expect("packed failed");
        let packed_vec = out_packed.tensor().contiguous().unwrap().to_vec::<f32>();

        // Build x_standard as [1, H, S, D] by permuting packed [S, H, D] → [1, H, S, D].
        // Permutation: [S, H, D] → [H, S, D] → [1, H, S, D]
        // To keep it simple, build x_standard with the same values but transposed layout.
        // We construct the 4D tensor manually with the layout standard RoPE expects:
        // x_4d[0, h, s, d] = x_data[s * h * d ... ] — but x_data is in [S,H,D] order.
        // x_data index: s*H*D + h*D + d
        // x_4d index:   0*H*S*D + h*S*D + s*D + d
        let mut x_4d_data = vec![0.0f32; s * h * d];
        for sv in 0..s {
            for hv in 0..h {
                for dv in 0..d {
                    let src = sv * h * d + hv * d + dv;
                    let dst = hv * s * d + sv * d + dv;
                    x_4d_data[dst] = x_data[src];
                }
            }
        }
        let x_standard = Var::new(
            Tensor::<CpuRuntime>::from_slice(&x_4d_data, &[1, h, s, d], &device),
            false,
        );

        let out_standard =
            apply_rope_impl(&client, &x_standard, &cos, &sin).expect("standard failed");
        let standard_4d = out_standard.tensor().contiguous().unwrap().to_vec::<f32>();

        // Convert standard output [1, H, S, D] back to [S, H, D] order for comparison:
        let mut standard_vec = vec![0.0f32; s * h * d];
        for sv in 0..s {
            for hv in 0..h {
                for dv in 0..d {
                    let src = hv * s * d + sv * d + dv;
                    let dst = sv * h * d + hv * d + dv;
                    standard_vec[dst] = standard_4d[src];
                }
            }
        }

        assert_eq!(packed_vec.len(), standard_vec.len());
        for (i, (&a, &b)) in packed_vec.iter().zip(standard_vec.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "packed vs standard mismatch at {i}: packed={a}, standard={b}"
            );
        }
    }

    #[test]
    fn test_rope_packed_invalid_odd_dim() {
        let (client, device) = cpu_setup();
        // head_dim=3 is odd — should be rejected
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32; 3], &[1, 1, 3], &device),
            false,
        );
        // cos/sin with half_d=1 (closest valid), but head_dim is odd so validation fails before cache check
        let cos = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32; 4], &[4, 1], &device),
            false,
        );
        let sin = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.0f32; 4], &[4, 1], &device),
            false,
        );
        let pids = Tensor::<CpuRuntime>::from_slice(&[0i32], &[1], &device);

        let result = apply_rope_packed_impl(&client, &x, &cos, &sin, &pids);
        assert!(result.is_err());
    }

    #[test]
    fn test_rope_packed_invalid_wrong_ndim() {
        let (client, device) = cpu_setup();
        // 4D input should be rejected (requires 3D)
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32; 8], &[1, 1, 2, 4], &device),
            false,
        );
        let cos = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32; 8], &[4, 2], &device),
            false,
        );
        let sin = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.0f32; 8], &[4, 2], &device),
            false,
        );
        let pids = Tensor::<CpuRuntime>::from_slice(&[0i32; 2], &[2], &device);

        let result = apply_rope_packed_impl(&client, &x, &cos, &sin, &pids);
        assert!(result.is_err());
    }

    #[test]
    fn test_rope_packed_dtype_f32() {
        // Smoke test: shape check passes, output shape is correct
        let (client, device) = cpu_setup();
        let total = 2usize;
        let h = 1usize;
        let d = 4usize;

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(
                &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                &[total, h, d],
                &device,
            ),
            false,
        );
        // cos/sin: [max_seq, half_d=2] — need max_seq * 2 elements
        let cos = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32; 16], &[8, 2], &device),
            false,
        );
        let sin = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.0f32; 16], &[8, 2], &device),
            false,
        );
        let pids = Tensor::<CpuRuntime>::from_slice(&[0i32, 1], &[2], &device);

        let out = apply_rope_packed_impl(&client, &x, &cos, &sin, &pids).unwrap();
        assert_eq!(out.tensor().shape(), &[total, h, d]);
    }
}
