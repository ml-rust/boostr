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

/// LongRoPE frequencies (Phi-3, MiniCPM4): a per-dimension divisor list,
/// `short_factor` for positions below `original_max_position_embeddings` and
/// `long_factor` from there on.
pub(super) struct LongRope {
    /// Frequencies of positions `< original`.
    pub short: Vec<f32>,
    /// Frequencies of positions `>= original`. `None` when the cache never
    /// reaches `original`.
    pub long: Option<Vec<f32>>,
    /// `original_max_position_embeddings`.
    pub original: usize,
    /// mscale, folded into the cos/sin caches.
    pub attention_scaling: f32,
}

/// Build the LongRoPE frequency sets for a cache of `max_seq_len` rows.
///
/// transformers picks one factor list per forward from the sequence length.
/// A precomputed cache picks per row instead, which equals it for every
/// sequence up to `original` and for each new token past it.
///
/// `attention_factor` comes from the config. Without it, the mscale follows
/// `max_seq_len / original`.
pub(super) fn longrope_freqs(
    head_dim: usize,
    base: f32,
    max_seq_len: usize,
    cfg: &RopeScalingConfig,
) -> Result<LongRope> {
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
    let scaled = |arg: &'static str, factors: Option<&Vec<f32>>| -> Result<Vec<f32>> {
        let factors = factors.ok_or_else(|| Error::InvalidArgument {
            arg,
            reason: format!("longrope RoPE scaling requires {arg}; set it in the checkpoint's rope_scaling config"),
        })?;
        if factors.len() != half_dim {
            return Err(Error::InvalidArgument {
                arg,
                reason: format!(
                    "expected {half_dim} entries (head_dim/2={half_dim}), got {}",
                    factors.len()
                ),
            });
        }
        Ok((0..half_dim)
            .map(|i| 1.0 / base.powf(2.0 * i as f32 / head_dim as f32) / factors[i])
            .collect())
    };
    let short = scaled("rope_scaling.short_factor", cfg.short_factor.as_ref())?;
    let long = if max_seq_len > original {
        Some(scaled(
            "rope_scaling.long_factor",
            cfg.long_factor.as_ref(),
        )?)
    } else {
        None
    };
    let attention_scaling = cfg.attention_factor.unwrap_or_else(|| {
        if max_seq_len <= original {
            1.0
        } else {
            let ratio = max_seq_len as f64 / original as f64;
            (1.0 + ratio.ln() / (original as f64).ln()).sqrt() as f32
        }
    });
    Ok(LongRope {
        short,
        long,
        original,
        attention_scaling,
    })
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
        // max_seq_len=16 > original=8: rows 0..8 use short_factor, rows 8..16
        // use long_factor, as transformers does for a sequence up to 8 and
        // for each token past it. short_factor=999 makes a wrong pick obvious.
        //   inv_freq = 10000^(-2i/8) = [1, 0.1, 0.01, 0.001]
        //   long_factor = [2, 2, 2, 2] -> divided = [0.5, 0.05, 0.005, 0.0005]
        //   attention_scaling = sqrt(1 + ln(16/8) / ln(8)) ~= 1.1547005, every row
        let device = CpuDevice::new();
        let cfg = longrope_cfg(vec![999.0; 4], vec![2.0; 4], Some(8));
        let rope =
            RoPE::<CpuRuntime>::precompute_freqs(16, 8, 10000.0, Some(&cfg), &device).unwrap();
        let cos: Vec<f32> = rope.cos_cache().tensor().to_vec();
        let sin: Vec<f32> = rope.sin_cache().tensor().to_vec();
        let scaling = 1.154_700_5f32;

        let short = [1.0f32 / 999.0, 0.1 / 999.0, 0.01 / 999.0, 0.001 / 999.0];
        let long = [0.5f32, 0.05, 0.005, 0.0005];
        for (pos, freqs) in [(1usize, short), (7, short), (8, long), (15, long)] {
            for (i, f) in freqs.iter().enumerate() {
                let angle = pos as f32 * f;
                let (c, s) = (cos[pos * 4 + i], sin[pos * 4 + i]);
                assert!(
                    (c - angle.cos() * scaling).abs() < 1e-5
                        && (s - angle.sin() * scaling).abs() < 1e-5,
                    "row {pos} dim {i}: cos {c} sin {s}, want angle {angle}"
                );
            }
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
