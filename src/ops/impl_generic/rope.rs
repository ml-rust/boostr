//! Generic RoPE implementation
//!
//! THE algorithm — same for all backends.
//! Composes numr autograd primitives: narrow, mul, sub, add, cat.

use crate::error::{Error, Result};
use numr::autograd::{Var, var_add, var_cat, var_mul, var_narrow, var_sub};
use numr::ops::{ScalarOps, ShapeOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};

/// Apply Rotary Position Embedding (RoPE)
///
/// For each pair of dimensions (x[..., 2i], x[..., 2i+1]):
///   x_rot[..., 2i]   = x[..., 2i] * cos - x[..., 2i+1] * sin
///   x_rot[..., 2i+1] = x[..., 2i] * sin + x[..., 2i+1] * cos
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

    // Split x into two halves along last dim: x1 = x[..., :D/2], x2 = x[..., D/2:]
    let x1 = var_narrow(x, -1, 0, half_d).map_err(Error::Numr)?;
    let x2 = var_narrow(x, -1, half_d, half_d).map_err(Error::Numr)?;

    let seq_len = shape[2];

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

    // out1 = x1 * cos - x2 * sin
    let x1_cos = var_mul(&x1, &cos_reshaped, client).map_err(Error::Numr)?;
    let x2_sin = var_mul(&x2, &sin_reshaped, client).map_err(Error::Numr)?;
    let out1 = var_sub(&x1_cos, &x2_sin, client).map_err(Error::Numr)?;

    // out2 = x2 * cos + x1 * sin
    let x2_cos = var_mul(&x2, &cos_reshaped, client).map_err(Error::Numr)?;
    let x1_sin = var_mul(&x1, &sin_reshaped, client).map_err(Error::Numr)?;
    let out2 = var_add(&x2_cos, &x1_sin, client).map_err(Error::Numr)?;

    // Concatenate back: [out1, out2] along last dim
    var_cat(&[&out1, &out2], -1, client).map_err(Error::Numr)
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
}
