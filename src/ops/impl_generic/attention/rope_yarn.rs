//! YaRN RoPE implementation (extended context via temperature scaling).

use super::rope_standard::apply_rope_impl;
use crate::error::{Error, Result};
use numr::autograd::Var;
use numr::ops::{ScalarOps, ShapeOps, TensorOps, TypeConversionOps};
use numr::runtime::{Runtime, RuntimeClient};

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
    C: RuntimeClient<R> + ScalarOps<R> + ShapeOps<R> + TypeConversionOps<R>,
    R::Client: TensorOps<R> + ShapeOps<R> + ScalarOps<R> + TypeConversionOps<R>,
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
    use super::super::rope_standard::apply_rope_impl;
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::autograd::Var;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

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
