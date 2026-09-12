//! Sample-rate-dependent geometry for Silero VAD.
//!
//! The 8 kHz and 16 kHz checkpoints share every structural choice — four
//! encoder convolutions, a 128-wide LSTM, a 1x1 output conv. Only the chunk
//! size, the carried context and the STFT window differ, so they are parameters
//! here rather than a second implementation.

/// Strides of the four encoder convolutions, in order. Every one uses
/// `kernel_size = 3` and `padding = 1`.
pub const ENCODER_STRIDES: [usize; 4] = [1, 2, 2, 1];

/// Kernel size shared by all four encoder convolutions.
pub const ENCODER_KERNEL: usize = 3;

/// Hidden size of the decoder LSTM, and the channel width entering the head.
pub const HIDDEN_SIZE: usize = 128;

/// Number of STFT frames the encoder sees for one chunk. Fixed by the chunk
/// geometry: `(chunk + 2 * context - n_fft) / (n_fft / 2) + 1` is 4 at both
/// supported sample rates.
pub const STFT_FRAMES: usize = 4;

/// Sample-rate-dependent geometry. Everything else about the model is
/// identical between the 8 kHz and 16 kHz checkpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VadConfig {
    /// Audio sample rate the checkpoint was trained for.
    pub sample_rate: usize,
    /// Samples of new audio per step (512 at 16 kHz).
    pub chunk_samples: usize,
    /// Samples carried over from the previous chunk (64 at 16 kHz). Also the
    /// width of the reflection pad appended after the chunk.
    pub context_samples: usize,
    /// STFT window length, i.e. the convolutional basis's kernel size.
    pub n_fft: usize,
}

impl VadConfig {
    /// The 16 kHz checkpoint (`silero_vad_16k.safetensors`).
    pub const fn silero_16k() -> Self {
        Self {
            sample_rate: 16000,
            chunk_samples: 512,
            context_samples: 64,
            n_fft: 256,
        }
    }

    /// The 8 kHz checkpoint (`silero_vad_8k.safetensors`).
    pub const fn silero_8k() -> Self {
        Self {
            sample_rate: 8000,
            chunk_samples: 256,
            context_samples: 32,
            n_fft: 128,
        }
    }

    /// STFT hop, i.e. the basis convolution's stride.
    pub const fn hop(&self) -> usize {
        self.n_fft / 2
    }

    /// Frequency bins in the magnitude spectrum, and so the first encoder
    /// convolution's input channel count (129 at 16 kHz, 65 at 8 kHz).
    pub const fn freq_bins(&self) -> usize {
        self.n_fft / 2 + 1
    }

    /// Total samples fed to the network per step: context + chunk + the
    /// trailing reflection pad.
    pub const fn window_samples(&self) -> usize {
        self.context_samples + self.chunk_samples + self.context_samples
    }

    /// `(in_channels, out_channels)` of the four encoder convolutions, in
    /// order.
    pub const fn encoder_channels(&self) -> [(usize, usize); 4] {
        [
            (self.freq_bins(), 128),
            (128, 64),
            (64, 64),
            (64, HIDDEN_SIZE),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sixteen_khz_geometry_matches_the_onnx_graph() {
        let config = VadConfig::silero_16k();
        assert_eq!(config.chunk_samples, 512);
        assert_eq!(config.context_samples, 64);
        assert_eq!(config.freq_bins(), 129);
        assert_eq!(config.hop(), 128);
        // 64 context + 512 chunk + a 64-sample reflection pad.
        assert_eq!(config.window_samples(), 640);
        // The STFT convolution must produce exactly 4 frames.
        assert_eq!(
            (config.window_samples() - config.n_fft) / config.hop() + 1,
            STFT_FRAMES
        );
        assert_eq!(
            config.encoder_channels(),
            [(129, 128), (128, 64), (64, 64), (64, 128)]
        );
    }

    #[test]
    fn eight_khz_geometry_matches_its_checkpoint() {
        let config = VadConfig::silero_8k();
        assert_eq!(config.freq_bins(), 65);
        assert_eq!(config.window_samples(), 320);
        assert_eq!(
            (config.window_samples() - config.n_fft) / config.hop() + 1,
            STFT_FRAMES
        );
        // The 8 kHz first encoder conv is [128, 65, 3] in the checkpoint.
        assert_eq!(config.encoder_channels()[0], (65, 128));
    }
}
