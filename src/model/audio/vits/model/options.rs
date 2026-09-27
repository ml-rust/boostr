//! [`VitsSynthOptions`] (inference-time generation controls) and
//! [`VitsSynthesis`] ([`super::VitsModel::synthesize`]'s result).

use crate::model::config::VitsConfig;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Inference-time generation controls: HF's `model.noise_scale`,
/// `model.noise_scale_duration`, `model.speaking_rate` (mutable attributes
/// set from the config at load time, but overridable per call here) plus
/// the deterministic-replay seed and the optional speaker selection.
#[derive(Debug, Clone)]
pub struct VitsSynthOptions {
    /// Gaussian noise scale applied to the sampled prior latents.
    pub noise_scale: f64,
    /// Gaussian noise scale applied inside the stochastic duration
    /// predictor's flow. Unused by the deterministic duration predictor.
    pub noise_scale_duration: f64,
    /// `1.0 / speaking_rate` scales predicted durations; `1.0` is the
    /// checkpoint's natural pace.
    pub speaking_rate: f64,
    /// Picks the deterministic PRNG stream for both noise draws (duration
    /// noise and prior-latent noise each derive their own seed from this
    /// one — see [`super::VitsModel::synthesize`]'s docs).
    pub seed: u64,
    /// Which speaker embedding row to use. Only read when
    /// `config.num_speakers > 1`; ignored (as in HF) otherwise.
    pub speaker_id: Option<usize>,
}

impl VitsSynthOptions {
    /// Defaults straight from the checkpoint's config: HF sets
    /// `model.noise_scale` / `noise_scale_duration` / `speaking_rate` from
    /// these same fields at load time and never changes them unless a
    /// caller does.
    pub fn from_config(config: &VitsConfig) -> Self {
        Self {
            noise_scale: config.noise_scale,
            noise_scale_duration: config.noise_scale_duration,
            speaking_rate: config.speaking_rate,
            seed: 0,
            speaker_id: None,
        }
    }
}

/// [`super::VitsModel::synthesize`]'s result: HF's `VitsModelOutput.waveform`
/// and `.sequence_lengths` (the `spectrogram` and hidden-state fields are
/// training/debugging outputs this crate does not surface).
pub struct VitsSynthesis<R: Runtime> {
    /// `[batch, samples]`.
    pub waveform: Tensor<R>,
    /// `predicted_lengths * prod(upsample_rates)`, `I64`, `[batch]`.
    pub sequence_lengths: Tensor<R>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_config_copies_the_three_generation_fields() {
        let config = VitsConfig {
            noise_scale: 0.5,
            noise_scale_duration: 0.9,
            speaking_rate: 1.2,
            ..Default::default()
        };
        let opts = VitsSynthOptions::from_config(&config);
        assert_eq!(opts.noise_scale, 0.5);
        assert_eq!(opts.noise_scale_duration, 0.9);
        assert_eq!(opts.speaking_rate, 1.2);
        assert_eq!(opts.seed, 0);
        assert_eq!(opts.speaker_id, None);
    }
}
