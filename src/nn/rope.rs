//! RoPE (Rotary Position Embedding) module
//!
//! Wraps the RoPEOps trait as a reusable module with precomputed frequency caches.

use crate::error::{Error, Result};
use crate::model::config::RopeScalingConfig;
use crate::ops::RoPEOps;
use numr::autograd::Var;
use numr::ops::TypeConversionOps;
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
    /// - YaRN: ramped interpolation/extrapolation blend plus `attention_factor` (mscale)
    ///   folded into the cos/sin caches
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
                             'linear', 'llama3', 'yarn'"
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

        let cos_cache = Tensor::<R>::from_slice(&cos_data, &[max_seq_len, half_dim], device);
        let sin_cache = Tensor::<R>::from_slice(&sin_data, &[max_seq_len, half_dim], device);

        Ok(Self::new(cos_cache, sin_cache))
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

    /// Cast cos/sin caches to the given dtype (e.g. BF16) so that
    /// per-token casts are avoided during inference.
    pub fn cast_caches(&mut self, dtype: numr::dtype::DType)
    where
        R: Runtime<DType = numr::dtype::DType>,
        R::Client: numr::ops::TypeConversionOps<R>,
    {
        if self.cos_cache.tensor().dtype() != dtype {
            let device = self.cos_cache.tensor().device().clone();
            let client = R::default_client(&device);
            if let Ok(cos) = client.cast(self.cos_cache.tensor(), dtype) {
                self.cos_cache = Var::new(cos, false);
            }
            if let Ok(sin) = client.cast(self.sin_cache.tensor(), dtype) {
                self.sin_cache = Var::new(sin, false);
            }
        }
    }

    pub fn cos_cache(&self) -> &Var<R> {
        &self.cos_cache
    }

    pub fn sin_cache(&self) -> &Var<R> {
        &self.sin_cache
    }
}

/// YaRN frequency scaling: ramped interpolation/extrapolation blend.
///
/// Mirrors HuggingFace `_compute_yarn_parameters`. Mutates `freqs` in place and
/// returns the `attention_factor` (mscale), which the caller folds into the
/// cos/sin caches since it scales both alike.
fn apply_yarn_scaling(
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

#[cfg(test)]
mod tests;
