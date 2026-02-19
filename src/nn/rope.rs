//! RoPE (Rotary Position Embedding) module
//!
//! Wraps the RoPEOps trait as a reusable module with precomputed frequency caches.

use crate::error::Result;
use crate::model::config::RopeScalingConfig;
use crate::ops::RoPEOps;
use numr::autograd::Var;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Rotary Position Embedding module
///
/// Stores precomputed cos/sin caches for each position.
/// cos_cache, sin_cache: `[max_seq_len, head_dim/2]`
pub struct RoPE<R: Runtime> {
    cos_cache: Var<R>,
    sin_cache: Var<R>,
}

impl<R: Runtime> RoPE<R> {
    /// Create from precomputed cos/sin caches.
    ///
    /// cos_cache, sin_cache: `[max_seq_len, head_dim/2]`
    pub fn new(cos_cache: Tensor<R>, sin_cache: Tensor<R>) -> Self {
        Self {
            cos_cache: Var::new(cos_cache, false),
            sin_cache: Var::new(sin_cache, false),
        }
    }

    /// Compute frequency caches: cos(pos * freq), sin(pos * freq)
    /// where freq_i = 1 / (base^(2i/dim)), optionally with scaling.
    ///
    /// Supports:
    /// - No scaling (standard RoPE)
    /// - Linear scaling: `freq /= factor`
    /// - Llama3 (NTK-aware): frequency-dependent scaling with low/high freq factors
    ///
    /// Returns `RoPE` with cos/sin caches `[max_seq_len, dim/2]`.
    pub fn precompute_freqs(
        max_seq_len: usize,
        head_dim: usize,
        base: f32,
        scaling: Option<&RopeScalingConfig>,
        device: &<R as Runtime>::Device,
    ) -> Self
    where
        R: Runtime<DType = numr::dtype::DType>,
    {
        let half_dim = head_dim / 2;

        // Compute base frequencies
        let mut freqs: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / base.powf(2.0 * i as f32 / head_dim as f32))
            .collect();

        // Apply scaling
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
                _ => {
                    // Unknown scaling type — fall through with unscaled frequencies
                }
            }
        }

        // Build caches
        let mut cos_data = vec![0.0f32; max_seq_len * half_dim];
        let mut sin_data = vec![0.0f32; max_seq_len * half_dim];

        for pos in 0..max_seq_len {
            for (i, &freq) in freqs.iter().enumerate() {
                let angle = pos as f32 * freq;
                cos_data[pos * half_dim + i] = angle.cos();
                sin_data[pos * half_dim + i] = angle.sin();
            }
        }

        let cos_cache = Tensor::<R>::from_slice(&cos_data, &[max_seq_len, half_dim], device);
        let sin_cache = Tensor::<R>::from_slice(&sin_data, &[max_seq_len, half_dim], device);

        Self::new(cos_cache, sin_cache)
    }

    /// Apply RoPE to input tensor `x: [B, H, S, D]`
    ///
    /// Uses the cached cos/sin values for positions 0..S.
    pub fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        C: RuntimeClient<R> + RoPEOps<R>,
    {
        client.apply_rope(x, &self.cos_cache, &self.sin_cache)
    }

    pub fn cos_cache(&self) -> &Var<R> {
        &self.cos_cache
    }

    pub fn sin_cache(&self) -> &Var<R> {
        &self.sin_cache
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn test_rope_precompute_shape() {
        let device = CpuDevice::new();
        let rope = RoPE::<CpuRuntime>::precompute_freqs(128, 64, 10000.0, None, &device);
        assert_eq!(rope.cos_cache().shape(), &[128, 32]);
        assert_eq!(rope.sin_cache().shape(), &[128, 32]);
    }

    #[test]
    fn test_rope_precompute_values() {
        let device = CpuDevice::new();
        let rope = RoPE::<CpuRuntime>::precompute_freqs(4, 8, 10000.0, None, &device);

        let cos: Vec<f32> = rope.cos_cache().tensor().to_vec();
        let sin: Vec<f32> = rope.sin_cache().tensor().to_vec();

        // pos=0: all cos=1, sin=0
        for i in 0..4 {
            assert!((cos[i] - 1.0).abs() < 1e-6, "cos[0,{i}]={}", cos[i]);
            assert!(sin[i].abs() < 1e-6, "sin[0,{i}]={}", sin[i]);
        }
    }

    #[test]
    fn test_rope_forward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let rope = RoPE::<CpuRuntime>::precompute_freqs(8, 16, 10000.0, None, &device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.1f32; 2 * 4 * 16], &[1, 2, 4, 16], &device),
            false,
        );
        let out = rope.forward(&client, &x).unwrap();
        assert_eq!(out.shape(), &[1, 2, 4, 16]);
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
        };

        let unscaled = RoPE::<CpuRuntime>::precompute_freqs(4, 8, 10000.0, None, &device);
        let scaled = RoPE::<CpuRuntime>::precompute_freqs(4, 8, 10000.0, Some(&cfg), &device);

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
        };

        let rope = RoPE::<CpuRuntime>::precompute_freqs(128, 64, 500000.0, Some(&cfg), &device);
        assert_eq!(rope.cos_cache().shape(), &[128, 32]);
        // Verify it doesn't panic and produces valid values
        let cos: Vec<f32> = rope.cos_cache().tensor().to_vec();
        for &v in &cos {
            assert!(v.is_finite(), "non-finite cos value: {v}");
            assert!((-1.0..=1.0).contains(&v), "cos out of range: {v}");
        }
    }
}
