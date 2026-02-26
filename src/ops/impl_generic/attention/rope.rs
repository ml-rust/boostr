//! Generic RoPE implementation
//!
//! THE algorithm — same for all backends.
//! Composes numr autograd primitives: narrow, mul, sub, add, cat.

use crate::error::{Error, Result};
use numr::autograd::{Var, var_add, var_cat, var_mul, var_narrow, var_sub};
use numr::ops::{ScalarOps, ShapeOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};

/// Validate inputs common to all RoPE variants.
/// Returns `(seq_len, half_d, cos_narrowed, sin_narrowed)` with caches
/// reshaped to `[1, 1, S, D/2]` for broadcasting.
fn validate_and_prepare<R, C>(
    x: &Var<R>,
    cos_cache: &Var<R>,
    sin_cache: &Var<R>,
) -> Result<(Vec<usize>, usize, usize, Var<R>, Var<R>)>
where
    R: Runtime<DType = numr::dtype::DType>,
    C: RuntimeClient<R> + ScalarOps<R> + ShapeOps<R>,
    R::Client: TensorOps<R> + ShapeOps<R>,
{
    let shape = x.tensor().shape().to_vec();
    if shape.len() != 4 {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: format!("expected 4D [B, H, S, D], got {}D", shape.len()),
        });
    }

    let d = shape[3];
    if d % 2 != 0 {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: format!("head dim D={} must be even for RoPE", d),
        });
    }

    let half_d = d / 2;
    let seq_len = shape[2];

    // Validate cos/sin cache shapes
    let cos_shape = cos_cache.tensor().shape();
    let sin_shape = sin_cache.tensor().shape();
    if cos_shape.len() != 2 || cos_shape[1] != half_d {
        return Err(Error::InvalidArgument {
            arg: "cos_cache",
            reason: format!("expected [S, {}], got {:?}", half_d, cos_shape),
        });
    }
    if sin_shape.len() != 2 || sin_shape[1] != half_d {
        return Err(Error::InvalidArgument {
            arg: "sin_cache",
            reason: format!("expected [S, {}], got {:?}", half_d, sin_shape),
        });
    }

    // Narrow cos/sin cache from [max_S, D/2] to [S, D/2] if needed
    let cos_narrowed = if cos_shape[0] > seq_len {
        var_narrow(cos_cache, 0, 0, seq_len).map_err(Error::Numr)?
    } else {
        cos_cache.clone()
    };
    let sin_narrowed = if sin_shape[0] > seq_len {
        var_narrow(sin_cache, 0, 0, seq_len).map_err(Error::Numr)?
    } else {
        sin_cache.clone()
    };

    // Reshape to [1, 1, S, D/2] — broadcasting handles B and H
    let cos_reshaped = numr::autograd::var_reshape(&cos_narrowed, &[1, 1, seq_len, half_d])
        .map_err(Error::Numr)?;
    let sin_reshaped = numr::autograd::var_reshape(&sin_narrowed, &[1, 1, seq_len, half_d])
        .map_err(Error::Numr)?;

    Ok((shape, seq_len, half_d, cos_reshaped, sin_reshaped))
}

/// Apply Rotary Position Embedding (RoPE) — split-half variant.
///
/// For each pair of dimensions (x[..., d], x[..., d+D/2]):
///   x_rot[..., d]     = x[..., d] * cos - x[..., d+D/2] * sin
///   x_rot[..., d+D/2] = x[..., d] * sin + x[..., d+D/2] * cos
///
/// # Arguments
///
/// - `x`: `[B, H, S, D]` — input tensor
/// - `cos_cache`: `[S, D/2]` — precomputed cosines (broadcast to [1, 1, S, D/2])
/// - `sin_cache`: `[S, D/2]` — precomputed sines (broadcast to [1, 1, S, D/2])
pub fn apply_rope_impl<R, C>(
    client: &C,
    x: &Var<R>,
    cos_cache: &Var<R>,
    sin_cache: &Var<R>,
) -> Result<Var<R>>
where
    R: Runtime<DType = numr::dtype::DType>,
    C: RuntimeClient<R> + ScalarOps<R> + ShapeOps<R>,
    R::Client: TensorOps<R> + ShapeOps<R>,
{
    let (_shape, _seq_len, half_d, cos_reshaped, sin_reshaped) =
        validate_and_prepare::<R, C>(x, cos_cache, sin_cache)?;

    // Split x into two halves along last dim: x1 = x[..., :D/2], x2 = x[..., D/2:]
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

    // Concatenate back: [out1, out2] along last dim
    var_cat(&[&out1, &out2], -1, client).map_err(Error::Numr)
}

/// Apply Rotary Position Embedding — interleaved variant (GPT-NeoX/Qwen style).
///
/// Pairs adjacent elements: `(x[..., 2d], x[..., 2d+1])`.
///   x_rot[..., 2d]   = x[..., 2d] * cos[d] - x[..., 2d+1] * sin[d]
///   x_rot[..., 2d+1] = x[..., 2d] * sin[d] + x[..., 2d+1] * cos[d]
///
/// # Arguments
///
/// - `x`: `[B, H, S, D]` — input tensor (D must be even)
/// - `cos_cache`: `[S, D/2]` — precomputed cosines
/// - `sin_cache`: `[S, D/2]` — precomputed sines
pub fn apply_rope_interleaved_impl<R, C>(
    client: &C,
    x: &Var<R>,
    cos_cache: &Var<R>,
    sin_cache: &Var<R>,
) -> Result<Var<R>>
where
    R: Runtime<DType = numr::dtype::DType>,
    C: RuntimeClient<R> + ScalarOps<R> + ShapeOps<R>,
    R::Client: TensorOps<R> + ShapeOps<R>,
{
    let (shape, seq_len, half_d, cos_reshaped, sin_reshaped) =
        validate_and_prepare::<R, C>(x, cos_cache, sin_cache)?;

    let b = shape[0];
    let h = shape[1];
    let d = shape[3];

    // For interleaved: pairs are (x[2d], x[2d+1]).
    // We can reuse split-half by first deinterleaving, applying rotation, then re-interleaving.
    //
    // Reshape x: [B, H, S, D] -> [B, H, S, D/2, 2]
    // Then split along last dim to get evens and odds, compute, and recombine.
    //
    // However var_narrow on non-contiguous views causes issues with var_reshape.
    // Instead, we use the split-half approach: extract evens via stride trick.
    //
    // Safest approach: flatten to do element-wise gather.
    // Actually simplest: reshape to [B*H*S, D/2, 2], gather evens/odds, compute, scatter.
    //
    // Even simpler: use the fact that for contiguous [B,H,S,D] data laid out in memory,
    // we can reshape to [..., D/2, 2] since D is the innermost contiguous dimension.

    // Reshape to [..., D/2, 2] — this is valid because the last dim D is contiguous
    let total_bhsd = b * h * seq_len;
    let x_flat = numr::autograd::var_reshape(x, &[total_bhsd, half_d, 2]).map_err(Error::Numr)?;

    // narrow along last dim (size 2) to get evens and odds
    let x_even_3d = var_narrow(&x_flat, -1, 0, 1).map_err(Error::Numr)?; // [total, D/2, 1]
    let x_odd_3d = var_narrow(&x_flat, -1, 1, 1).map_err(Error::Numr)?;

    // The narrow produces a non-contiguous view. We need to make it contiguous
    // before reshape. Use var_add with zero to force materialization.
    // Actually, let's try a different approach: just use the contiguous() on tensor level.

    // Create a zero tensor for the add trick to force contiguous
    let zero_shape = &[total_bhsd, half_d, 1];
    let zero = Var::new(
        numr::tensor::Tensor::<R>::zeros(zero_shape, x.tensor().dtype(), x.tensor().device()),
        false,
    );
    let x_even_contig = var_add(&x_even_3d, &zero, client).map_err(Error::Numr)?;
    let x_odd_contig = var_add(&x_odd_3d, &zero, client).map_err(Error::Numr)?;

    // Now reshape to [B, H, S, D/2]
    let x_even = numr::autograd::var_reshape(&x_even_contig, &[b, h, seq_len, half_d])
        .map_err(Error::Numr)?;
    let x_odd = numr::autograd::var_reshape(&x_odd_contig, &[b, h, seq_len, half_d])
        .map_err(Error::Numr)?;

    // out_even = x_even * cos - x_odd * sin
    let even_cos = var_mul(&x_even, &cos_reshaped, client).map_err(Error::Numr)?;
    let odd_sin = var_mul(&x_odd, &sin_reshaped, client).map_err(Error::Numr)?;
    let out_even = var_sub(&even_cos, &odd_sin, client).map_err(Error::Numr)?;

    // out_odd = x_even * sin + x_odd * cos
    let even_sin = var_mul(&x_even, &sin_reshaped, client).map_err(Error::Numr)?;
    let odd_cos = var_mul(&x_odd, &cos_reshaped, client).map_err(Error::Numr)?;
    let out_odd = var_add(&even_sin, &odd_cos, client).map_err(Error::Numr)?;

    // Re-interleave: [B, H, S, D/2] -> [total, D/2, 1] -> cat -> [total, D/2, 2] -> [B, H, S, D]
    let out_even_3d =
        numr::autograd::var_reshape(&out_even, &[total_bhsd, half_d, 1]).map_err(Error::Numr)?;
    let out_odd_3d =
        numr::autograd::var_reshape(&out_odd, &[total_bhsd, half_d, 1]).map_err(Error::Numr)?;

    let interleaved = var_cat(&[&out_even_3d, &out_odd_3d], -1, client).map_err(Error::Numr)?;
    numr::autograd::var_reshape(&interleaved, &[b, h, seq_len, d]).map_err(Error::Numr)
}

/// Apply YaRN RoPE (extended context via temperature scaling).
///
/// Same rotation as standard split-half RoPE, with an additional `attn_scale`
/// multiplied into the output. The cos/sin caches should be precomputed with
/// YaRN-scaled frequencies.
///
/// # Arguments
///
/// - `x`: `[B, H, S, D]` — input tensor
/// - `cos_cache`: `[S, D/2]` — precomputed cosines (with YaRN frequency scaling)
/// - `sin_cache`: `[S, D/2]` — precomputed sines (with YaRN frequency scaling)
/// - `attn_scale`: attention scaling factor (`1.0` = no additional scaling)
pub fn apply_rope_yarn_impl<R, C>(
    client: &C,
    x: &Var<R>,
    cos_cache: &Var<R>,
    sin_cache: &Var<R>,
    attn_scale: f32,
) -> Result<Var<R>>
where
    R: Runtime<DType = numr::dtype::DType>,
    C: RuntimeClient<R> + ScalarOps<R> + ShapeOps<R>,
    R::Client: TensorOps<R> + ShapeOps<R> + ScalarOps<R>,
{
    // YaRN is standard RoPE with scaled frequencies (baked into cos/sin caches)
    // plus an attention scaling factor on the output
    let rotated = apply_rope_impl(client, x, cos_cache, sin_cache)?;

    if (attn_scale - 1.0).abs() < 1e-7 {
        return Ok(rotated);
    }

    // Scale the output: rotated * attn_scale
    numr::autograd::var_mul_scalar(&rotated, attn_scale as f64, client).map_err(Error::Numr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

    #[test]
    fn test_rope_output_shape() {
        let (client, device) = cpu_setup();
        let b = 1;
        let h = 2;
        let s = 4;
        let d = 8;

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&vec![1.0f32; b * h * s * d], &[b, h, s, d], &device),
            false,
        );
        // cos/sin cache: [S, D/2]
        let cos = Var::new(
            Tensor::<CpuRuntime>::from_slice(&vec![1.0f32; s * d / 2], &[s, d / 2], &device),
            false,
        );
        let sin = Var::new(
            Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; s * d / 2], &[s, d / 2], &device),
            false,
        );

        let out = apply_rope_impl(&client, &x, &cos, &sin).unwrap();
        assert_eq!(out.tensor().shape(), &[b, h, s, d]);
    }

    #[test]
    fn test_rope_identity_with_zero_angle() {
        let (client, device) = cpu_setup();
        // cos=1, sin=0 → RoPE is identity
        let x_data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&x_data, &[1, 1, 2, 8], &device),
            false,
        );
        let cos = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32; 8], &[2, 4], &device),
            false,
        );
        let sin = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.0f32; 8], &[2, 4], &device),
            false,
        );

        let out = apply_rope_impl(&client, &x, &cos, &sin).unwrap();
        let out_data: Vec<f32> = out.tensor().contiguous().to_vec();

        // With sin=0, cos=1: out1 = x1*1 - x2*0 = x1, out2 = x2*1 + x1*0 = x2
        // So cat(x1, x2) = x (identity)
        for (i, (&a, &b)) in out_data.iter().zip(x_data.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "mismatch at {}: got {}, expected {}",
                i,
                a,
                b
            );
        }
    }

    #[test]
    fn test_rope_90_degree_rotation() {
        let (client, device) = cpu_setup();
        // cos=0, sin=1 → 90° rotation
        // out1 = x1*0 - x2*1 = -x2
        // out2 = x2*0 + x1*1 = x1
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(
                &[1.0f32, 2.0, 3.0, 4.0], // x1=[1,2], x2=[3,4]
                &[1, 1, 1, 4],
                &device,
            ),
            false,
        );
        let cos = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0], &[1, 2], &device),
            false,
        );
        let sin = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0], &[1, 2], &device),
            false,
        );

        let out = apply_rope_impl(&client, &x, &cos, &sin).unwrap();
        let out_data: Vec<f32> = out.tensor().contiguous().to_vec();

        // out1 = -x2 = [-3, -4], out2 = x1 = [1, 2] → [-3, -4, 1, 2]
        assert!((out_data[0] - (-3.0)).abs() < 1e-5);
        assert!((out_data[1] - (-4.0)).abs() < 1e-5);
        assert!((out_data[2] - 1.0).abs() < 1e-5);
        assert!((out_data[3] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_rope_narrowing_matches_exact_cache() {
        // Verifies that a pre-allocated cache larger than seq_len produces
        // identical output to an exact-size cache (exercises the narrowing path).
        let (client, device) = cpu_setup();

        let x_data: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&x_data, &[1, 1, 2, 8], &device),
            false,
        );

        // Exact-size caches: [S=2, D/2=4]
        let cos_exact = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32; 8], &[2, 4], &device),
            false,
        );
        let sin_exact = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.0f32; 8], &[2, 4], &device),
            false,
        );

        // Pre-allocated caches: [max_S=8, D/2=4] — will be narrowed to S=2
        let cos_large = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32; 32], &[8, 4], &device),
            false,
        );
        let sin_large = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.0f32; 32], &[8, 4], &device),
            false,
        );

        let out_exact = apply_rope_impl(&client, &x, &cos_exact, &sin_exact).unwrap();
        let out_narrowed = apply_rope_impl(&client, &x, &cos_large, &sin_large).unwrap();

        let exact_data: Vec<f32> = out_exact.tensor().contiguous().to_vec();
        let narrowed_data: Vec<f32> = out_narrowed.tensor().contiguous().to_vec();

        assert_eq!(exact_data.len(), narrowed_data.len());
        for (i, (&a, &b)) in exact_data.iter().zip(narrowed_data.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "mismatch at {i}: exact={a}, narrowed={b}"
            );
        }
    }

    #[test]
    fn test_rope_invalid_odd_dim() {
        let (client, device) = cpu_setup();
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32; 3], &[1, 1, 1, 3], &device),
            false,
        );
        let cos = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1, 1], &device),
            false,
        );
        let sin = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.0f32], &[1, 1], &device),
            false,
        );

        let result = apply_rope_impl(&client, &x, &cos, &sin);
        assert!(result.is_err());
    }

    // ========================================================================
    // Interleaved RoPE tests
    // ========================================================================

    #[test]
    fn test_rope_interleaved_output_shape() {
        let (client, device) = cpu_setup();
        let (b, h, s, d) = (1, 2, 4, 8);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&vec![1.0f32; b * h * s * d], &[b, h, s, d], &device),
            false,
        );
        let cos = Var::new(
            Tensor::<CpuRuntime>::from_slice(&vec![1.0f32; s * d / 2], &[s, d / 2], &device),
            false,
        );
        let sin = Var::new(
            Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; s * d / 2], &[s, d / 2], &device),
            false,
        );

        let out = apply_rope_interleaved_impl(&client, &x, &cos, &sin).unwrap();
        assert_eq!(out.tensor().shape(), &[b, h, s, d]);
    }

    #[test]
    fn test_rope_interleaved_identity_with_zero_angle() {
        let (client, device) = cpu_setup();
        // cos=1, sin=0 → identity
        let x_data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&x_data, &[1, 1, 1, 8], &device),
            false,
        );
        let cos = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32; 4], &[1, 4], &device),
            false,
        );
        let sin = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.0f32; 4], &[1, 4], &device),
            false,
        );

        let out = apply_rope_interleaved_impl(&client, &x, &cos, &sin).unwrap();
        let out_data: Vec<f32> = out.tensor().contiguous().to_vec();

        for (i, (&a, &b)) in out_data.iter().zip(x_data.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "mismatch at {}: got {}, expected {}",
                i,
                a,
                b
            );
        }
    }

    #[test]
    fn test_rope_interleaved_90_degree_rotation() {
        let (client, device) = cpu_setup();
        // cos=0, sin=1 → 90° rotation on interleaved pairs
        // x = [1, 2, 3, 4] → pairs (1,2), (3,4)
        // out_even = x_even*0 - x_odd*1 = -x_odd → [-2, -4]
        // out_odd  = x_even*1 + x_odd*0 = x_even → [1, 3]
        // interleaved: [-2, 1, -4, 3]
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 1, 4], &device),
            false,
        );
        let cos = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0], &[1, 2], &device),
            false,
        );
        let sin = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0], &[1, 2], &device),
            false,
        );

        let out = apply_rope_interleaved_impl(&client, &x, &cos, &sin).unwrap();
        let out_data: Vec<f32> = out.tensor().contiguous().to_vec();

        assert!((out_data[0] - (-2.0)).abs() < 1e-5, "got {}", out_data[0]);
        assert!((out_data[1] - 1.0).abs() < 1e-5, "got {}", out_data[1]);
        assert!((out_data[2] - (-4.0)).abs() < 1e-5, "got {}", out_data[2]);
        assert!((out_data[3] - 3.0).abs() < 1e-5, "got {}", out_data[3]);
    }

    // ========================================================================
    // YaRN RoPE tests
    // ========================================================================

    #[test]
    fn test_rope_yarn_identity_scale_1() {
        let (client, device) = cpu_setup();
        // With attn_scale=1.0, YaRN = standard RoPE
        let x_data: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&x_data, &[1, 1, 2, 8], &device),
            false,
        );
        let cos = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32; 8], &[2, 4], &device),
            false,
        );
        let sin = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.0f32; 8], &[2, 4], &device),
            false,
        );

        let rope_out = apply_rope_impl(&client, &x, &cos, &sin).unwrap();
        let yarn_out = apply_rope_yarn_impl(&client, &x, &cos, &sin, 1.0).unwrap();

        let rope_data: Vec<f32> = rope_out.tensor().contiguous().to_vec();
        let yarn_data: Vec<f32> = yarn_out.tensor().contiguous().to_vec();

        for (i, (&a, &b)) in rope_data.iter().zip(yarn_data.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "mismatch at {}: rope={}, yarn={}",
                i,
                a,
                b
            );
        }
    }

    #[test]
    fn test_rope_yarn_scaling() {
        let (client, device) = cpu_setup();
        // With attn_scale=2.0, output should be 2x standard RoPE
        let x_data: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&x_data, &[1, 1, 1, 8], &device),
            false,
        );
        let cos = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32; 4], &[1, 4], &device),
            false,
        );
        let sin = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.0f32; 4], &[1, 4], &device),
            false,
        );

        let rope_out = apply_rope_impl(&client, &x, &cos, &sin).unwrap();
        let yarn_out = apply_rope_yarn_impl(&client, &x, &cos, &sin, 2.0).unwrap();

        let rope_data: Vec<f32> = rope_out.tensor().contiguous().to_vec();
        let yarn_data: Vec<f32> = yarn_out.tensor().contiguous().to_vec();

        for (i, (&a, &b)) in rope_data.iter().zip(yarn_data.iter()).enumerate() {
            assert!(
                (b - a * 2.0).abs() < 1e-5,
                "mismatch at {}: rope={}, yarn={} (expected {})",
                i,
                a,
                b,
                a * 2.0
            );
        }
    }
}
