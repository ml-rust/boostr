//! Constants for the Kaldi-compatible log-mel filterbank frontend.
//!
//! See the [`super`] module doc for the overall bit-for-bit target; each
//! constant here documents what breaks if its value or role is changed.

/// Sample rate the frontend assumes. Callers must resample before entry.
pub const SAMPLE_RATE: usize = 16_000;

/// Analysis window length in samples (25 ms at 16 kHz).
pub const FRAME_LENGTH: usize = 400;

/// Frame advance in samples (10 ms at 16 kHz).
pub const FRAME_SHIFT: usize = 160;

/// FFT size. Frames are zero-padded from 400 to 512; 512 is a power of two,
/// which is what `numr`'s real FFT requires.
pub const FFT_LENGTH: usize = 512;

/// Number of non-negative-frequency FFT bins (`FFT_LENGTH / 2 + 1`).
pub const NUM_FFT_BINS: usize = FFT_LENGTH / 2 + 1;

/// Mel channels produced per frame.
pub const NUM_MEL_BINS: usize = 80;

/// Feature width after stride-2 stacking of consecutive frames.
pub const STACKED_DIM: usize = NUM_MEL_BINS * 2;

/// Waveform scale applied before anything else.
///
/// Kaldi operates on 16-bit integer PCM; `transformers` reproduces that by
/// multiplying float samples by `2^15`. Dropping this shifts every log-mel
/// energy by a constant `2 * ln(2^15)` — which per-utterance normalization
/// mostly hides, except where the mel floor clamps, so the error is sparse and
/// hard to spot.
pub const WAVEFORM_SCALE: f64 = 32_768.0;

/// Pre-emphasis coefficient.
pub const PREEMPHASIS: f64 = 0.97;

/// Povey window exponent: a Hann window raised to this power.
pub const POVEY_EXPONENT: f64 = 0.85;

/// Low edge of the mel filterbank, in Hz.
///
/// 20 Hz, NOT 0 Hz. Starting at 0 shifts every triangle and rewrites all 80
/// channels.
pub const LOW_FREQ: f64 = 20.0;

/// High edge of the mel filterbank, in Hz (Nyquist).
pub const HIGH_FREQ: f64 = 8_000.0;

/// Floor applied to mel energies before the log.
///
/// This is the exact literal `transformers` uses (it is `f32::EPSILON`
/// expressed in f64). Using `1e-7` or `1e-10` instead changes the value of
/// every floored channel — silence and high-frequency bins in particular — and
/// then propagates through the per-utterance mean/variance into *all* channels.
pub const MEL_FLOOR: f64 = 1.192_092_955_078_125e-07;

/// Epsilon added to the variance during per-mel-bin normalization.
pub const NORM_EPS: f64 = 1e-7;
