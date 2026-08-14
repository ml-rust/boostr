//! Kaldi-compatible log-mel filterbank frontend for NeuCodec's semantic branch.
//!
//! Bit-for-bit target: HuggingFace `SeamlessM4TFeatureExtractor`. See
//! [`pipeline`] for the extraction pipeline and what it deliberately omits,
//! [`window`] for the analysis window, and [`filterbank`] for the mel scale
//! and triangulation. Every convention has a plausible-but-wrong alternative
//! that still produces correctly *shaped* output; each is called out at its
//! definition with what breaks if it is changed.

mod constants;
mod filterbank;
mod pipeline;
mod window;

pub use constants::{
    FFT_LENGTH, FRAME_LENGTH, FRAME_SHIFT, HIGH_FREQ, LOW_FREQ, MEL_FLOOR, NORM_EPS, NUM_FFT_BINS,
    NUM_MEL_BINS, POVEY_EXPONENT, PREEMPHASIS, SAMPLE_RATE, STACKED_DIM, WAVEFORM_SCALE,
};
pub use filterbank::{hz_to_mel, mel_filterbank, mel_to_hz};
pub use pipeline::{num_frames, seamless_fbank};
pub use window::povey_window;
