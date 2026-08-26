//! LongRoPE cos/sin cache construction for `feat_encoder`.
//!
//! `RoPE::precompute_freqs` (`crate::nn::rope`) supports linear/llama3/yarn
//! scaling, all of which rescale every dimension by ONE shared factor.
//! LongRoPE rescales PER DIMENSION with `short_factor` (length `dim/2`), a
//! shape of scaling that method has no path for — so the cache is built by
//! hand here. The rotation itself (`rotate_half`, upcast to f32) is
//! unchanged and reused as-is via [`RoPE::forward`] / `client.apply_rope`.

use crate::error::{Error, Result};
use crate::nn::RoPE;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Build the LongRoPE cos/sin caches for `feat_encoder`'s `num_positions`
/// (5) query/key positions.
///
/// `dim` is `head_dim` (128); `short_factor` must have `dim/2` entries.
/// `long_factor` is never consulted: VoxCPM2's local encoder always runs
/// `num_positions <= original_max_position_embeddings`, the regime
/// `short_factor` covers.
///
/// `scaling_factor = sqrt(1 + ln(max_pos/orig_max_pos) / ln(orig_max_pos))`,
/// computed generically rather than hardcoded — it happens to be exactly
/// `1.0` on this checkpoint because `max_position_embeddings ==
/// original_max_position_embeddings` (32768 == 32768).
pub fn build_long_rope_cache<R: Runtime<DType = DType>>(
    num_positions: usize,
    dim: usize,
    theta: f32,
    short_factor: &[f32],
    max_position_embeddings: usize,
    original_max_position_embeddings: usize,
    device: &R::Device,
) -> Result<RoPE<R>> {
    let half_dim = dim / 2;
    if short_factor.len() != half_dim {
        return Err(Error::InvalidArgument {
            arg: "short_factor",
            reason: format!(
                "expected {half_dim} entries (head_dim/2={half_dim}), got {}",
                short_factor.len()
            ),
        });
    }
    if original_max_position_embeddings == 0 {
        return Err(Error::InvalidArgument {
            arg: "original_max_position_embeddings",
            reason: "must be nonzero".to_string(),
        });
    }

    let scaling_factor = if max_position_embeddings == original_max_position_embeddings {
        1.0
    } else {
        let ratio = max_position_embeddings as f64 / original_max_position_embeddings as f64;
        (1.0 + ratio.ln() / (original_max_position_embeddings as f64).ln()).sqrt() as f32
    };

    let mut cos_data = vec![0.0f32; num_positions * half_dim];
    let mut sin_data = vec![0.0f32; num_positions * half_dim];
    for pos in 0..num_positions {
        for (i, &factor) in short_factor.iter().enumerate() {
            let inv_freq = 1.0 / theta.powf(2.0 * i as f32 / dim as f32);
            let angle = pos as f32 * inv_freq / factor;
            cos_data[pos * half_dim + i] = angle.cos() * scaling_factor;
            sin_data[pos * half_dim + i] = angle.sin() * scaling_factor;
        }
    }

    let cos_cache = Tensor::<R>::from_slice(&cos_data, &[num_positions, half_dim], device)?;
    let sin_cache = Tensor::<R>::from_slice(&sin_data, &[num_positions, half_dim], device)?;
    Ok(RoPE::new(cos_cache, sin_cache))
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn identity_short_factor_matches_vanilla_rope() {
        let device = <CpuRuntime as Runtime>::default_device();
        let short_factor = vec![1.0f32; 64];
        let rope = build_long_rope_cache::<CpuRuntime>(
            5,
            128,
            10000.0,
            &short_factor,
            32768,
            32768,
            &device,
        )
        .unwrap();
        let cos: Vec<f32> = rope.cos_cache().tensor().to_vec();
        // pos=0 -> angle 0 for every dim -> cos == 1.
        assert!((cos[0] - 1.0).abs() < 1e-6);
        // pos=1, dim=0 -> angle == inv_freq[0] == 1.0.
        assert!((cos[64] - 1.0f32.cos()).abs() < 1e-6);
    }

    #[test]
    fn scaling_factor_is_one_when_max_equals_original() {
        let device = <CpuRuntime as Runtime>::default_device();
        let short_factor = vec![1.0f32; 4];
        let rope =
            build_long_rope_cache::<CpuRuntime>(2, 8, 10000.0, &short_factor, 100, 100, &device)
                .unwrap();
        let cos: Vec<f32> = rope.cos_cache().tensor().to_vec();
        assert!((cos[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rejects_wrong_short_factor_len() {
        let device = <CpuRuntime as Runtime>::default_device();
        let short_factor = vec![1.0f32; 3];
        assert!(
            build_long_rope_cache::<CpuRuntime>(
                5,
                128,
                10000.0,
                &short_factor,
                32768,
                32768,
                &device
            )
            .is_err()
        );
    }
}
