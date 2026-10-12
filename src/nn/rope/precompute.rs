//! `RoPE::precompute_freqs`: base frequencies, scaling-type dispatch, and
//! cos/sin cache construction.

use crate::error::{Error, Result};
use crate::model::config::RopeScalingConfig;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::RoPE;
use super::scaling::{apply_yarn_scaling, longrope_freqs};

impl<R: Runtime> RoPE<R> {
    /// Compute frequency caches: cos(pos * freq), sin(pos * freq)
    /// where freq_i = 1 / (base^(2i/dim)), optionally with scaling.
    ///
    /// Supports:
    /// - No scaling (standard RoPE)
    /// - Linear scaling: `freq /= factor`
    /// - Llama3 (NTK-aware): frequency-dependent scaling with low/high freq factors
    /// - YaRN: ramped interpolation/extrapolation blend plus `attention_factor` (mscale)
    ///   folded into the cos/sin caches
    /// - LongRoPE: per-dimension divisor list (`short_factor` below/at
    ///   `original_max_position_embeddings`, else `long_factor`) plus an
    ///   `attention_scaling` (mscale) folded into the cos/sin caches
    /// - `rope_freqs`: llama.cpp's per-dimension divisor tensor, carried in
    ///   `long_factor` and applied at every length with no mscale. A GGUF
    ///   stores llama3 scaling this way, as the `rope_freqs.weight` tensor.
    ///
    /// Any other `scaling_type` is an error. `"dynamic"` is rejected: it recomputes
    /// frequencies per forward as the sequence grows, which a precomputed cache cannot do.
    ///
    /// Returns `RoPE` with cos/sin caches `[max_seq_len, dim/2]`.
    pub fn precompute_freqs(
        max_seq_len: usize,
        head_dim: usize,
        base: f32,
        scaling: Option<&RopeScalingConfig>,
        device: &<R as Runtime>::Device,
    ) -> Result<Self>
    where
        R: Runtime<DType = numr::dtype::DType>,
    {
        Self::precompute_freqs_partial(max_seq_len, head_dim, head_dim, base, scaling, device)
    }

    /// [`precompute_freqs`](Self::precompute_freqs) for a head whose leading
    /// `rotary_dim` dims rotate, as GPT-NeoX `rotary_pct` and Phi
    /// `partial_rotary_factor` ask.
    ///
    /// The frequencies are those of a `rotary_dim` head. The caches are still
    /// `[max_seq_len, head_dim/2]`: columns past `rotary_dim/2` hold `cos = 1`,
    /// `sin = 0`, so the split-half kernel leaves those dims untouched. The
    /// caller orders each head's dims so the rotated pairs come first in each
    /// half; see `model::llama` partial rotary loading.
    pub fn precompute_freqs_partial(
        max_seq_len: usize,
        head_dim: usize,
        rotary_dim: usize,
        base: f32,
        scaling: Option<&RopeScalingConfig>,
        device: &<R as Runtime>::Device,
    ) -> Result<Self>
    where
        R: Runtime<DType = numr::dtype::DType>,
    {
        if rotary_dim == 0
            || rotary_dim > head_dim
            || !rotary_dim.is_multiple_of(2)
            || !head_dim.is_multiple_of(2)
        {
            return Err(Error::InvalidArgument {
                arg: "rotary_dim",
                reason: format!(
                    "rotary_dim {rotary_dim} must be even, nonzero and at most head_dim \
                     {head_dim}"
                ),
            });
        }
        let cache_half = head_dim / 2;
        let head_dim = rotary_dim;
        let half_dim = head_dim / 2;

        // Compute base frequencies
        let mut freqs: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / base.powf(2.0 * i as f32 / head_dim as f32))
            .collect();

        // Apply scaling. LongRoPE swaps in its long frequencies from row
        // `original` on.
        let mut attention_scaling = 1.0f32;
        let mut long_rows: Option<(usize, Vec<f32>)> = None;
        if let Some(cfg) = scaling {
            match cfg.scaling_type.as_str() {
                "linear" => {
                    for f in &mut freqs {
                        *f /= cfg.factor;
                    }
                }
                "llama3" => {
                    let old_context_len =
                        cfg.original_max_position_embeddings.unwrap_or(8192) as f32;
                    let low_freq_factor = cfg.low_freq_factor.unwrap_or(1.0);
                    let high_freq_factor = cfg.high_freq_factor.unwrap_or(4.0);
                    let low_freq_wavelen = old_context_len / low_freq_factor;
                    let high_freq_wavelen = old_context_len / high_freq_factor;

                    for f in &mut freqs {
                        let wavelen = 2.0 * std::f32::consts::PI / *f;
                        if wavelen < high_freq_wavelen {
                            // High frequency: no scaling
                        } else if wavelen > low_freq_wavelen {
                            // Low frequency: full linear scaling
                            *f /= cfg.factor;
                        } else {
                            // Middle: smooth interpolation
                            let smooth = (old_context_len / wavelen - low_freq_factor)
                                / (high_freq_factor - low_freq_factor);
                            *f = (1.0 - smooth) * (*f / cfg.factor) + smooth * *f;
                        }
                    }
                }
                "yarn" => {
                    attention_scaling = apply_yarn_scaling(&mut freqs, head_dim, base, cfg)?;
                }
                "longrope" => {
                    let lr = longrope_freqs(head_dim, base, max_seq_len, cfg)?;
                    attention_scaling = lr.attention_scaling;
                    freqs = lr.short;
                    long_rows = lr.long.map(|long| (lr.original, long));
                }
                "rope_freqs" => {
                    let factors =
                        cfg.long_factor
                            .as_ref()
                            .ok_or_else(|| Error::InvalidArgument {
                                arg: "rope_scaling.long_factor",
                                reason:
                                    "rope_freqs RoPE scaling reads its per-dimension divisors from \
                                 long_factor, and none is set"
                                        .to_string(),
                            })?;
                    if factors.len() != half_dim {
                        return Err(Error::InvalidArgument {
                            arg: "rope_scaling.long_factor",
                            reason: format!(
                                "rope_freqs needs {half_dim} divisors (head_dim/2), got {}",
                                factors.len()
                            ),
                        });
                    }
                    for (f, divisor) in freqs.iter_mut().zip(factors) {
                        *f /= divisor;
                    }
                }
                "dynamic" => {
                    return Err(Error::InvalidArgument {
                        arg: "rope_scaling.type",
                        reason: "'dynamic' RoPE scaling recomputes frequencies per forward as the \
                                 sequence length grows; this precomputed cos/sin cache cannot do \
                                 that, and precomputing at max_seq_len would silently apply \
                                 max-length scaling to short sequences. Convert the checkpoint to \
                                 'linear', 'llama3', or 'yarn' scaling"
                            .to_string(),
                    });
                }
                other => {
                    return Err(Error::InvalidArgument {
                        arg: "rope_scaling.type",
                        reason: format!(
                            "unsupported RoPE scaling type '{other}'; supported: \
                             'linear', 'llama3', 'yarn', 'longrope', 'rope_freqs'"
                        ),
                    });
                }
            }
        }

        // Build caches. Columns past the rotary half stay identity.
        let mut cos_data = vec![1.0f32; max_seq_len * cache_half];
        let mut sin_data = vec![0.0f32; max_seq_len * cache_half];

        for pos in 0..max_seq_len {
            let row_freqs = match &long_rows {
                Some((original, long)) if pos >= *original => long,
                _ => &freqs,
            };
            for (i, &freq) in row_freqs.iter().enumerate() {
                let angle = pos as f32 * freq;
                cos_data[pos * cache_half + i] = angle.cos() * attention_scaling;
                sin_data[pos * cache_half + i] = angle.sin() * attention_scaling;
            }
        }

        let cos_cache = Tensor::<R>::from_slice(&cos_data, &[max_seq_len, cache_half], device)?;
        let sin_cache = Tensor::<R>::from_slice(&sin_data, &[max_seq_len, cache_half], device)?;

        Ok(Self::new(cos_cache, sin_cache))
    }
}

#[cfg(test)]
mod tests {
    use super::super::scaling::yarn_cfg;
    use super::*;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn test_rope_precompute_shape() {
        let device = CpuDevice::new();
        let rope = RoPE::<CpuRuntime>::precompute_freqs(128, 64, 10000.0, None, &device).unwrap();
        assert_eq!(rope.cos_cache().shape(), &[128, 32]);
        assert_eq!(rope.sin_cache().shape(), &[128, 32]);
    }

    #[test]
    fn test_rope_precompute_values() {
        let device = CpuDevice::new();
        let rope = RoPE::<CpuRuntime>::precompute_freqs(4, 8, 10000.0, None, &device).unwrap();

        let cos: Vec<f32> = rope.cos_cache().tensor().to_vec();
        let sin: Vec<f32> = rope.sin_cache().tensor().to_vec();

        // pos=0: all cos=1, sin=0
        for i in 0..4 {
            assert!((cos[i] - 1.0).abs() < 1e-6, "cos[0,{i}]={}", cos[i]);
            assert!(sin[i].abs() < 1e-6, "sin[0,{i}]={}", sin[i]);
        }
    }

    #[test]
    fn test_rope_linear_scaling() {
        let device = CpuDevice::new();
        let cfg = RopeScalingConfig {
            scaling_type: "linear".to_string(),
            factor: 2.0,
            original_max_position_embeddings: None,
            low_freq_factor: None,
            high_freq_factor: None,
            attention_factor: None,
            beta_fast: None,
            beta_slow: None,
            short_factor: None,
            long_factor: None,
        };

        let unscaled = RoPE::<CpuRuntime>::precompute_freqs(4, 8, 10000.0, None, &device).unwrap();
        let scaled =
            RoPE::<CpuRuntime>::precompute_freqs(4, 8, 10000.0, Some(&cfg), &device).unwrap();

        let cos_unscaled: Vec<f32> = unscaled.cos_cache().tensor().to_vec();
        let cos_scaled: Vec<f32> = scaled.cos_cache().tensor().to_vec();

        // At pos=0, both should be all 1s (cos(0)=1)
        assert!((cos_scaled[0] - 1.0).abs() < 1e-6);

        // At pos=2 scaled should match pos=1 unscaled (freq halved → angle halved)
        let half_dim = 4;
        for i in 0..half_dim {
            let expected = cos_unscaled[half_dim + i]; // pos=1 unscaled
            let actual = cos_scaled[2 * half_dim + i]; // pos=2 scaled
            assert!(
                (actual - expected).abs() < 1e-5,
                "dim {i}: expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn test_rope_llama3_scaling() {
        let device = CpuDevice::new();
        let cfg = RopeScalingConfig {
            scaling_type: "llama3".to_string(),
            factor: 8.0,
            original_max_position_embeddings: Some(8192),
            low_freq_factor: Some(1.0),
            high_freq_factor: Some(4.0),
            attention_factor: None,
            beta_fast: None,
            beta_slow: None,
            short_factor: None,
            long_factor: None,
        };

        let rope =
            RoPE::<CpuRuntime>::precompute_freqs(128, 64, 500000.0, Some(&cfg), &device).unwrap();
        assert_eq!(rope.cos_cache().shape(), &[128, 32]);
        // Verify it doesn't panic and produces valid values
        let cos: Vec<f32> = rope.cos_cache().tensor().to_vec();
        for &v in &cos {
            assert!(v.is_finite(), "non-finite cos value: {v}");
            assert!((-1.0..=1.0).contains(&v), "cos out of range: {v}");
        }
    }

    #[test]
    fn test_rope_dynamic_scaling_rejected() {
        let device = CpuDevice::new();
        let mut cfg = yarn_cfg();
        cfg.scaling_type = "dynamic".to_string();
        let err = RoPE::<CpuRuntime>::precompute_freqs(4, 8, 10000.0, Some(&cfg), &device)
            .err()
            .expect("dynamic scaling must error");
        assert!(
            err.to_string().contains("dynamic"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_rope_unknown_scaling_rejected() {
        let device = CpuDevice::new();
        let mut cfg = yarn_cfg();
        cfg.scaling_type = "nope".to_string();
        let err = RoPE::<CpuRuntime>::precompute_freqs(4, 8, 10000.0, Some(&cfg), &device)
            .err()
            .expect("unknown scaling type must error");
        let msg = err.to_string();
        assert!(
            msg.contains("nope"),
            "error must name the offending value: {msg}"
        );
        assert!(
            msg.contains("yarn"),
            "error must list supported types: {msg}"
        );
    }

    #[test]
    fn test_rope_linear_llama3_unscaled_by_attention_factor() {
        // Regression guard: mscale applies to the yarn arm only, so linear and
        // llama3 caches keep cos(0)=1 even when attention_factor is present.
        let device = CpuDevice::new();
        for scaling_type in ["linear", "llama3"] {
            let cfg = RopeScalingConfig {
                scaling_type: scaling_type.to_string(),
                factor: 4.0,
                original_max_position_embeddings: Some(2048),
                low_freq_factor: Some(1.0),
                high_freq_factor: Some(4.0),
                attention_factor: Some(0.25),
                beta_fast: Some(32.0),
                beta_slow: Some(1.0),
                short_factor: None,
                long_factor: None,
            };
            let rope =
                RoPE::<CpuRuntime>::precompute_freqs(4, 8, 10000.0, Some(&cfg), &device).unwrap();
            let cos: Vec<f32> = rope.cos_cache().tensor().to_vec();
            for (i, c) in cos.iter().take(4).enumerate() {
                assert!((c - 1.0).abs() < 1e-6, "{scaling_type} cos[0,{i}]={c}");
            }
        }
    }

    /// A GGUF's `rope_freqs.weight` divisors reproduce llama3 scaling. The
    /// divisors are what `convert_hf_to_gguf.py` writes for the same config.
    #[test]
    fn test_rope_freqs_divisors_match_llama3() {
        let device = CpuDevice::new();
        let (head_dim, base, len) = (64usize, 500000.0f32, 16usize);
        let mut llama3 = RopeScalingConfig {
            scaling_type: "llama3".to_string(),
            factor: 32.0,
            original_max_position_embeddings: Some(8192),
            low_freq_factor: Some(1.0),
            high_freq_factor: Some(4.0),
            attention_factor: None,
            beta_fast: None,
            beta_slow: None,
            short_factor: None,
            long_factor: None,
        };
        let divisors: Vec<f32> = (0..head_dim / 2)
            .map(|i| {
                let freq = 1.0 / base.powf(2.0 * i as f32 / head_dim as f32);
                let wavelen = 2.0 * std::f32::consts::PI / freq;
                if wavelen < 8192.0 / 4.0 {
                    1.0
                } else if wavelen > 8192.0 {
                    32.0
                } else {
                    let smooth = (8192.0 / wavelen - 1.0) / (4.0 - 1.0);
                    1.0 / ((1.0 - smooth) / 32.0 + smooth)
                }
            })
            .collect();
        let expected =
            RoPE::<CpuRuntime>::precompute_freqs(len, head_dim, base, Some(&llama3), &device)
                .unwrap();
        llama3.scaling_type = "rope_freqs".to_string();
        llama3.long_factor = Some(divisors);
        let got = RoPE::<CpuRuntime>::precompute_freqs(len, head_dim, base, Some(&llama3), &device)
            .unwrap();
        let want: Vec<f32> = expected.sin_cache().tensor().to_vec();
        let have: Vec<f32> = got.sin_cache().tensor().to_vec();
        for (i, (w, h)) in want.iter().zip(&have).enumerate() {
            assert!((w - h).abs() < 1e-5, "sin[{i}]: llama3 {w}, rope_freqs {h}");
        }

        llama3.long_factor = Some(vec![1.0; 3]);
        assert!(
            RoPE::<CpuRuntime>::precompute_freqs(len, head_dim, base, Some(&llama3), &device)
                .is_err()
        );
    }
}
