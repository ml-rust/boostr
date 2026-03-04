//! Interleaved RoPE implementation (GPT-NeoX/Qwen style).

use super::rope_common::validate_and_prepare;
use crate::error::{Error, Result};
use numr::autograd::{Var, var_add, var_cat, var_mul, var_narrow, var_sub};
use numr::ops::{ScalarOps, ShapeOps, TensorOps, TypeConversionOps};
use numr::runtime::{Runtime, RuntimeClient};

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
    C: RuntimeClient<R> + ScalarOps<R> + ShapeOps<R> + TypeConversionOps<R>,
    R::Client: TensorOps<R> + ShapeOps<R> + TypeConversionOps<R>,
{
    let (shape, seq_len, half_d, cos_reshaped, sin_reshaped) =
        validate_and_prepare::<R, C>(client, x, cos_cache, sin_cache)?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

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
}
