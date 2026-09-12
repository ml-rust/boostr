//! `RoPE::precompute_freqs`: base frequencies, scaling-type dispatch, and
//! cos/sin cache construction.

use crate::error::{Error, Result};
use crate::model::config::RopeScalingConfig;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::RoPE;
use super::scaling::{apply_longrope_scaling, apply_yarn_scaling};

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
        let half_dim = head_dim / 2;

        // Compute base frequencies
        let mut freqs: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / base.powf(2.0 * i as f32 / head_dim as f32))
            .collect();

        // Apply scaling
        let mut attention_scaling = 1.0f32;
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
                    attention_scaling =
                        apply_longrope_scaling(&mut freqs, head_dim, base, max_seq_len, cfg)?;
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
                             'linear', 'llama3', 'yarn', 'longrope'"
                        ),
                    });
                }
            }
        }

        // Build caches
        let mut cos_data = vec![0.0f32; max_seq_len * half_dim];
        let mut sin_data = vec![0.0f32; max_seq_len * half_dim];

        for pos in 0..max_seq_len {
            for (i, &freq) in freqs.iter().enumerate() {
                let angle = pos as f32 * freq;
                cos_data[pos * half_dim + i] = angle.cos() * attention_scaling;
                sin_data[pos * half_dim + i] = angle.sin() * attention_scaling;
            }
        }

        let cos_cache = Tensor::<R>::from_slice(&cos_data, &[max_seq_len, half_dim], device)?;
        let sin_cache = Tensor::<R>::from_slice(&sin_data, &[max_seq_len, half_dim], device)?;

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
}
