//! [`KokoroSynthOptions`]: the per-call controls of
//! [`super::KokoroModelV2::synthesize`] and
//! [`super::KokoroModelV2::forward_to_spectrogram`].

/// Inference-time synthesis controls: the speaking pace and the seed of the
/// harmonic source's random draws.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KokoroSynthOptions {
    /// Divides every phoneme's duration before rounding. `1.0` is the
    /// voice's natural pace, `2.0` twice as fast.
    pub speed: f32,
    /// Picks the PRNG stream of the source's random initial phase and its
    /// additive noise. Each draw derives its own sub-seed from this one (see
    /// [`super::SineGen::forward`]). The same text, voice, speed and seed
    /// give the same samples on one backend. Backends run separate PRNGs, so
    /// one seed gives different samples on CPU and on CUDA. `None` draws
    /// from the runtime's unseeded RNG.
    pub seed: Option<u64>,
}

impl Default for KokoroSynthOptions {
    fn default() -> Self {
        Self {
            speed: 1.0,
            seed: None,
        }
    }
}

impl KokoroSynthOptions {
    /// The options at `speed` with an unseeded source.
    pub fn with_speed(speed: f32) -> Self {
        Self {
            speed,
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_natural_pace_and_unseeded() {
        let opts = KokoroSynthOptions::default();
        assert_eq!(opts.speed, 1.0);
        assert_eq!(opts.seed, None);
    }

    #[test]
    fn with_speed_keeps_the_source_unseeded() {
        let opts = KokoroSynthOptions::with_speed(0.8);
        assert_eq!(opts.speed, 0.8);
        assert_eq!(opts.seed, None);
    }
}
