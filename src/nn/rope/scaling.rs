//! YaRN and LongRoPE frequency scaling for `RoPE::precompute_freqs`.

use crate::error::{Error, Result};
use crate::model::config::RopeScalingConfig;

/// YaRN frequency scaling: ramped interpolation/extrapolation blend.
///
/// Mirrors HuggingFace `_compute_yarn_parameters`. Mutates `freqs` in place and
/// returns the `attention_factor` (mscale), which the caller folds into the
/// cos/sin caches since it scales both alike.
pub(super) fn apply_yarn_scaling(
    freqs: &mut [f32],
    head_dim: usize,
    base: f32,
    cfg: &RopeScalingConfig,
) -> Result<f32> {
    let original = cfg
        .original_max_position_embeddings
        .ok_or_else(|| Error::InvalidArgument {
            arg: "rope_scaling.original_max_position_embeddings",
            reason: "yarn RoPE scaling requires original_max_position_embeddings; \
                     set it in the checkpoint's rope_scaling config"
                .to_string(),
        })? as f64;
    let factor = cfg.factor as f64;
    let beta_fast = cfg.beta_fast.unwrap_or(32.0) as f64;
    let beta_slow = cfg.beta_slow.unwrap_or(1.0) as f64;
    let base_f64 = base as f64;
    let dim = head_dim as f64;

    // find_correction_dim, evaluated over the full dim (not dim/2).
    let correction_dim = |num_rotations: f64| {
        (dim * (original / (num_rotations * 2.0 * std::f64::consts::PI)).ln())
            / (2.0 * base_f64.ln())
    };
    let low = correction_dim(beta_fast).floor().max(0.0);
    let mut high = correction_dim(beta_slow).ceil().min(dim - 1.0);
    if low == high {
        high += 0.001;
    }

    for (i, f) in freqs.iter_mut().enumerate() {
        // linear_ramp_factor over dim/2 entries.
        let ramp = ((i as f64 - low) / (high - low)).clamp(0.0, 1.0);
        let extrapolation_factor = 1.0 - ramp;
        let inv_freq_extrapolation = 1.0 / base_f64.powf(2.0 * i as f64 / dim);
        let inv_freq_interpolation = inv_freq_extrapolation / factor;
        *f = (inv_freq_interpolation * (1.0 - extrapolation_factor)
            + inv_freq_extrapolation * extrapolation_factor) as f32;
    }

    // mscale: folds into the caches because it scales cos and sin alike.
    Ok(cfg.attention_factor.unwrap_or_else(|| {
        if factor <= 1.0 {
            1.0
        } else {
            (0.1 * factor.ln() + 1.0) as f32
        }
    }))
}

/// LongRoPE frequency scaling: per-dimension divisor list (Phi-3,
/// MiniCPM4-style), unlike YaRN/llama3's single shared factor.
///
/// `max_seq_len` plays the role HF's `rope_scaling` config calls
/// `max_position_embeddings`: the length the cache is precomputed for.
/// Picks `short_factor` when `max_seq_len <= original_max_position_embeddings`,
/// else `long_factor`, ONCE for the whole cache. Mutates `freqs` in place and
/// returns the `attention_scaling` (mscale), which the caller folds into the
/// cos/sin caches since it scales both alike.
pub(super) fn apply_longrope_scaling(
    freqs: &mut [f32],
    head_dim: usize,
    base: f32,
    max_seq_len: usize,
    cfg: &RopeScalingConfig,
) -> Result<f32> {
    let half_dim = head_dim / 2;
    let original = cfg
        .original_max_position_embeddings
        .filter(|&o| o > 0)
        .ok_or_else(|| Error::InvalidArgument {
            arg: "rope_scaling.original_max_position_embeddings",
            reason: "longrope RoPE scaling requires a nonzero \
                     original_max_position_embeddings; set it in the checkpoint's \
                     rope_scaling config"
                .to_string(),
        })?;

    let use_short = max_seq_len <= original;
    let factor_list = if use_short {
        cfg.short_factor
            .as_ref()
            .ok_or_else(|| Error::InvalidArgument {
                arg: "rope_scaling.short_factor",
                reason: "longrope RoPE scaling requires rope_scaling.short_factor when \
                     max_seq_len <= original_max_position_embeddings; set it in the \
                     checkpoint's rope_scaling config"
                    .to_string(),
            })?
    } else {
        cfg.long_factor
            .as_ref()
            .ok_or_else(|| Error::InvalidArgument {
                arg: "rope_scaling.long_factor",
                reason: "longrope RoPE scaling requires rope_scaling.long_factor when \
                     max_seq_len > original_max_position_embeddings; set it in the \
                     checkpoint's rope_scaling config"
                    .to_string(),
            })?
    };
    if factor_list.len() != half_dim {
        return Err(Error::InvalidArgument {
            arg: if use_short {
                "rope_scaling.short_factor"
            } else {
                "rope_scaling.long_factor"
            },
            reason: format!(
                "expected {half_dim} entries (head_dim/2={half_dim}), got {}",
                factor_list.len()
            ),
        });
    }

    for (i, f) in freqs.iter_mut().enumerate() {
        let inv_freq = 1.0 / base.powf(2.0 * i as f32 / head_dim as f32);
        *f = inv_freq / factor_list[i];
    }

    let attention_scaling = if max_seq_len == original {
        1.0
    } else {
        let ratio = max_seq_len as f64 / original as f64;
        (1.0 + ratio.ln() / (original as f64).ln()).sqrt() as f32
    };
    Ok(attention_scaling)
}

/// A YaRN scaling config with the defaults the tests in this module and in
/// `precompute` rely on.
#[cfg(test)]
pub(super) fn yarn_cfg() -> RopeScalingConfig {
    RopeScalingConfig {
        scaling_type: "yarn".to_string(),
        factor: 4.0,
        original_max_position_embeddings: Some(2048),
        low_freq_factor: None,
        high_freq_factor: None,
        attention_factor: None,
        beta_fast: Some(32.0),
        beta_slow: Some(1.0),
        short_factor: None,
        long_factor: None,
    }
}

#[cfg(test)]
mod tests {
    use super::super::RoPE;
    use super::*;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    /// Recover the per-dim frequencies from the caches: at pos=1 the angle equals
    /// the frequency, and `atan2` cancels any mscale applied to both cos and sin.
    fn freqs_at_pos1(rope: &RoPE<CpuRuntime>, half_dim: usize) -> Vec<f32> {
        let cos: Vec<f32> = rope.cos_cache().tensor().to_vec();
        let sin: Vec<f32> = rope.sin_cache().tensor().to_vec();
        (0..half_dim)
            .map(|i| sin[half_dim + i].atan2(cos[half_dim + i]))
            .collect()
    }

    #[test]
    fn test_rope_yarn_frequencies() {
        // head_dim=8, base=10000, factor=4, original=2048, beta_fast=32, beta_slow=1.
        //   correction range: low = floor(1.008) = 1, high = ceil(2.513) = 3
        //   ramp over dim/2=4 entries: [0, 0, 0.5, 1] -> extrapolation [1, 1, 0.5, 0]
        //   inv_freq_extrapolation = 10000^(-i/4) = [1, 0.1, 0.01, 0.001]
        let device = CpuDevice::new();
        let cfg = yarn_cfg();
        let rope =
            RoPE::<CpuRuntime>::precompute_freqs(4, 8, 10000.0, Some(&cfg), &device).unwrap();

        let expected = [1.0f32, 0.1, 0.00625, 0.00025];
        let got = freqs_at_pos1(&rope, 4);
        for (i, (&e, &g)) in expected.iter().zip(got.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-6,
                "yarn freq[{i}]: expected {e}, got {g}"
            );
        }
    }

    #[test]
    fn test_rope_yarn_attention_factor_scales_caches() {
        let device = CpuDevice::new();
        let mut cfg = yarn_cfg();
        cfg.attention_factor = Some(1.0);
        let unit =
            RoPE::<CpuRuntime>::precompute_freqs(4, 8, 10000.0, Some(&cfg), &device).unwrap();
        cfg.attention_factor = Some(0.25);
        let scaled =
            RoPE::<CpuRuntime>::precompute_freqs(4, 8, 10000.0, Some(&cfg), &device).unwrap();

        let cos_unit: Vec<f32> = unit.cos_cache().tensor().to_vec();
        let sin_unit: Vec<f32> = unit.sin_cache().tensor().to_vec();
        let cos_scaled: Vec<f32> = scaled.cos_cache().tensor().to_vec();
        let sin_scaled: Vec<f32> = scaled.sin_cache().tensor().to_vec();

        for i in 0..cos_unit.len() {
            assert!(
                (cos_scaled[i] - cos_unit[i] * 0.25).abs() < 1e-6,
                "cos[{i}]: expected {}, got {}",
                cos_unit[i] * 0.25,
                cos_scaled[i]
            );
            assert!(
                (sin_scaled[i] - sin_unit[i] * 0.25).abs() < 1e-6,
                "sin[{i}]: expected {}, got {}",
                sin_unit[i] * 0.25,
                sin_scaled[i]
            );
        }
    }

    #[test]
    fn test_rope_yarn_requires_original_max_position_embeddings() {
        let device = CpuDevice::new();
        let mut cfg = yarn_cfg();
        cfg.original_max_position_embeddings = None;
        let err = RoPE::<CpuRuntime>::precompute_freqs(4, 8, 10000.0, Some(&cfg), &device)
            .err()
            .expect("yarn without original_max_position_embeddings must error");
        assert!(
            err.to_string().contains("original_max_position_embeddings"),
            "unexpected error: {err}"
        );
    }

    fn longrope_cfg(
        short_factor: Vec<f32>,
        long_factor: Vec<f32>,
        original_max_position_embeddings: Option<usize>,
    ) -> RopeScalingConfig {
        RopeScalingConfig {
            scaling_type: "longrope".to_string(),
            factor: 1.0,
            original_max_position_embeddings,
            low_freq_factor: None,
            high_freq_factor: None,
            attention_factor: None,
            beta_fast: None,
            beta_slow: None,
            short_factor: Some(short_factor),
            long_factor: Some(long_factor),
        }
    }

    #[test]
    fn test_rope_longrope_short_factor_frequencies() {
        // head_dim=8, half_dim=4, base=10000, original=8, max_seq_len=8 (<=
        // original -> short_factor path, and max_seq_len == original ->
        // attention_scaling == 1).
        //   inv_freq = 10000^(-2i/8) = [1, 0.1, 0.01, 0.001]
        //   short_factor = [2, 4, 5, 8] -> divided = [0.5, 0.025, 0.002, 0.000125]
        let device = CpuDevice::new();
        let cfg = longrope_cfg(vec![2.0, 4.0, 5.0, 8.0], vec![1.0; 4], Some(8));
        let rope =
            RoPE::<CpuRuntime>::precompute_freqs(8, 8, 10000.0, Some(&cfg), &device).unwrap();

        let expected = [0.5f32, 0.025, 0.002, 0.000125];
        let got = freqs_at_pos1(&rope, 4);
        for (i, (&e, &g)) in expected.iter().zip(got.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-6,
                "longrope short_factor freq[{i}]: expected {e}, got {g}"
            );
        }
    }

    #[test]
    fn test_rope_longrope_long_factor_and_attention_scaling() {
        // max_seq_len=16 > original=8 selects long_factor (short_factor=999 would
        // give wildly different frequencies if wrongly selected, so this also
        // pins the selection, not just the scaling).
        //   inv_freq = 10000^(-2i/8) = [1, 0.1, 0.01, 0.001]
        //   long_factor = [2, 2, 2, 2] -> divided = [0.5, 0.05, 0.005, 0.0005]
        //   attention_scaling = sqrt(1 + ln(16/8) / ln(8)) ~= 1.1547005
        let device = CpuDevice::new();
        let cfg = longrope_cfg(vec![999.0; 4], vec![2.0; 4], Some(8));
        let rope =
            RoPE::<CpuRuntime>::precompute_freqs(16, 8, 10000.0, Some(&cfg), &device).unwrap();

        let expected = [0.5f32, 0.05, 0.005, 0.0005];
        let got = freqs_at_pos1(&rope, 4);
        for (i, (&e, &g)) in expected.iter().zip(got.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-6,
                "longrope long_factor freq[{i}]: expected {e}, got {g}"
            );
        }

        let cos: Vec<f32> = rope.cos_cache().tensor().to_vec();
        let expected_scaling = 1.154_700_5f32;
        for (i, &c) in cos.iter().take(4).enumerate() {
            assert!(
                (c - expected_scaling).abs() < 1e-5,
                "attention_scaling cos[0,{i}]: expected {expected_scaling}, got {c}"
            );
        }
    }

    #[test]
    fn test_rope_longrope_wrong_length_factor_errors() {
        let device = CpuDevice::new();
        let cfg = longrope_cfg(vec![1.0; 3], vec![1.0; 4], Some(8));
        let err = RoPE::<CpuRuntime>::precompute_freqs(8, 8, 10000.0, Some(&cfg), &device)
            .err()
            .expect("wrong-length short_factor must error");
        let msg = err.to_string();
        assert!(msg.contains("expected 4"), "unexpected error: {msg}");
        assert!(msg.contains("got 3"), "unexpected error: {msg}");
    }

    #[test]
    fn test_rope_longrope_requires_original_max_position_embeddings() {
        let device = CpuDevice::new();
        let cfg = longrope_cfg(vec![1.0; 4], vec![1.0; 4], None);
        let err = RoPE::<CpuRuntime>::precompute_freqs(8, 8, 10000.0, Some(&cfg), &device)
            .err()
            .expect("longrope without original_max_position_embeddings must error");
        assert!(
            err.to_string().contains("original_max_position_embeddings"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_rope_longrope_requires_short_factor() {
        let device = CpuDevice::new();
        let mut cfg = longrope_cfg(vec![1.0; 4], vec![1.0; 4], Some(8));
        cfg.short_factor = None;
        let err = RoPE::<CpuRuntime>::precompute_freqs(8, 8, 10000.0, Some(&cfg), &device)
            .err()
            .expect("longrope without short_factor must error");
        assert!(
            err.to_string().contains("short_factor"),
            "unexpected error: {err}"
        );
    }
}
