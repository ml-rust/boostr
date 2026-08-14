//! Frame counting and the end-to-end extraction pipeline: windowing, FFT,
//! mel filterbank, log, and per-bin normalization.
//!
//! # Not implemented (deliberately)
//!
//! Batched extraction. The reference pads a batch with `padding_value` and
//! downsamples the attention mask via `mask[indices % 2 == 1]`. We process one
//! utterance at a time with no padding, so neither is needed — and padding
//! without the mask would corrupt the per-utterance normalization, which must
//! see real frames only.

use crate::error::{Error, Result};
use numr::algorithm::fft::{FftAlgorithms, FftNormalization};
use numr::dtype::{Complex128, DType};
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

use super::constants::{
    FFT_LENGTH, FRAME_LENGTH, FRAME_SHIFT, MEL_FLOOR, NORM_EPS, NUM_FFT_BINS, NUM_MEL_BINS,
    PREEMPHASIS, STACKED_DIM, WAVEFORM_SCALE,
};
use super::filterbank::mel_filterbank;
use super::window::povey_window;

/// Number of frames produced for `n_samples` input samples.
///
/// `center = false`: there is NO padding, and frame `t` starts at sample
/// `t * FRAME_SHIFT`. Under `center = true` (torch's STFT default) the count
/// would instead be `1 + n_samples / hop`, which both changes the frame count
/// and misaligns every frame against the reference.
#[must_use]
pub fn num_frames(n_samples: usize) -> usize {
    if n_samples < FRAME_LENGTH {
        0
    } else {
        1 + (n_samples - FRAME_LENGTH) / FRAME_SHIFT
    }
}

/// Compute Kaldi-compatible stacked log-mel features for one utterance.
///
/// `samples` is a mono 16 kHz waveform in `[-1, 1]`. The returned tensor is
/// `[num_frames / 2, 160]`, F32, on `device`.
///
/// All arithmetic runs host-side in f64 (matching the reference's numpy
/// float64) and exactly one tensor is uploaded at the end. The FFT is numr's
/// `rfft`, batched over all frames at once on numr's CPU runtime — boostr does
/// not reimplement FFT.
///
/// # Errors
///
/// Returns [`Error::InvalidArgument`] if fewer than two frames can be formed
/// (`samples.len() < 560`), since per-utterance variance is undefined for a
/// single frame and stacking needs a pair.
pub fn seamless_fbank<R: Runtime<DType = DType>>(
    samples: &[f32],
    device: &R::Device,
) -> Result<Tensor<R>> {
    let frames = num_frames(samples.len());
    if frames < 2 {
        return Err(Error::InvalidArgument {
            arg: "samples",
            reason: format!(
                "need at least {} samples for 2 frames, got {}",
                FRAME_LENGTH + FRAME_SHIFT,
                samples.len()
            ),
        });
    }

    let windowed = window_frames(samples, frames);
    let power = power_spectra(&windowed, frames)?;
    let mut features = log_mel(&power, frames);
    normalize_per_bin(&mut features, frames);

    // Stride-2 stacking is a pure reinterpretation of the row-major buffer:
    // row `i` of `[frames/2, 160]` is exactly frames `2i` and `2i+1`
    // back to back. An odd trailing frame is DROPPED, not zero-padded —
    // padding it would inject a fabricated frame into the codec's input.
    let out_rows = frames / 2;
    let out: Vec<f32> = features
        .iter()
        .take(out_rows * STACKED_DIM)
        .map(|&v| v as f32)
        .collect();

    Ok(Tensor::<R>::try_from_slice(
        &out,
        &[out_rows, STACKED_DIM],
        device,
    )?)
}

/// Frame, DC-remove, pre-emphasize and window into a `[frames, 512]` buffer.
///
/// Slots `400..512` of every frame stay zero (the FFT zero-pad).
fn window_frames(samples: &[f32], frames: usize) -> Vec<f64> {
    let window = povey_window();
    let mut buffers = vec![0.0f64; frames * FFT_LENGTH];

    for (f, buf) in buffers.chunks_mut(FFT_LENGTH).enumerate() {
        let start = f * FRAME_SHIFT;
        let (Some(src), Some(frame)) = (
            samples.get(start..start + FRAME_LENGTH),
            buf.get_mut(..FRAME_LENGTH),
        ) else {
            continue;
        };

        // 1. Raw samples, scaled to Kaldi's 16-bit-int convention. Dither is
        //    disabled (dither = 0), so there is no noise term here.
        for (dst, &s) in frame.iter_mut().zip(src.iter()) {
            *dst = f64::from(s) * WAVEFORM_SCALE;
        }

        // 2. Remove DC offset — BEFORE pre-emphasis. Kaldi's order is
        //    dither -> DC removal -> pre-emphasis -> window. Swapping the last
        //    two leaves a residual offset in every tap.
        let mean = frame.iter().sum::<f64>() / FRAME_LENGTH as f64;
        for v in frame.iter_mut() {
            *v -= mean;
        }

        // 3. Pre-emphasis, walked in DESCENDING index order so each tap reads
        //    its still-unmodified left neighbour. Ascending order would feed
        //    already-filtered values back in, turning the intended FIR into an
        //    IIR filter and blowing up the low band.
        for i in (1..FRAME_LENGTH).rev() {
            let prev = frame.get(i - 1).copied().unwrap_or(0.0);
            if let Some(cur) = frame.get_mut(i) {
                *cur -= PREEMPHASIS * prev;
            }
        }
        if let Some(first) = frame.first_mut() {
            // Boundary tap behaves as if x[-1] == x[0].
            *first *= 1.0 - PREEMPHASIS;
        }

        // 4. Analysis window.
        for (v, w) in frame.iter_mut().zip(window.iter()) {
            *v *= w;
        }
    }

    buffers
}

/// Batched real FFT over `[frames, 512]`, returning `|X[k]|^2` as
/// `[frames, 257]`.
///
/// Power spectrum (`power = 2`), not magnitude — a magnitude spectrum would
/// halve every log-mel value before normalization and is not recoverable
/// downstream.
fn power_spectra(windowed: &[f64], frames: usize) -> Result<Vec<f64>> {
    let cpu_device = CpuDevice::new();
    let client = CpuClient::new(cpu_device.clone());
    let input = Tensor::<CpuRuntime>::try_from_slice(windowed, &[frames, FFT_LENGTH], &cpu_device)?;
    let spectrum = client.rfft(&input, FftNormalization::None)?.contiguous()?;
    let bins: Vec<Complex128> = spectrum.to_vec();

    Ok(bins.iter().map(|c| c.re * c.re + c.im * c.im).collect())
}

/// Apply the mel filterbank, floor, and natural log.
///
/// Returns `[frames, 80]` row-major. The log is plain `ln` with no offset.
fn log_mel(power: &[f64], frames: usize) -> Vec<f64> {
    let filters = mel_filterbank();
    let mut out = vec![0.0f64; frames * NUM_MEL_BINS];

    for (frame_power, frame_out) in power
        .chunks(NUM_FFT_BINS)
        .take(frames)
        .zip(out.chunks_mut(NUM_MEL_BINS))
    {
        for (slot, filter) in frame_out.iter_mut().zip(filters.iter()) {
            let energy: f64 = filter
                .iter()
                .zip(frame_power.iter())
                .map(|(&w, &p)| w * p)
                .sum();
            *slot = energy.max(MEL_FLOOR).ln();
        }
    }

    out
}

/// Per-mel-bin normalization over the time axis, in place on `[frames, 80]`.
///
/// Each of the 80 channels is standardized independently across frames, using
/// the **sample** variance (ddof = 1, divide by `n - 1`). The population
/// variance (ddof = 0) is the natural default in most tensor libraries and is
/// wrong here: for short utterances the `n/(n-1)` factor is a percent-level
/// scale error on every feature.
///
/// This runs BEFORE stride-2 stacking; normalizing the stacked 160-dim rows
/// instead would pool each mel channel with its time-shifted self.
fn normalize_per_bin(features: &mut [f64], frames: usize) {
    if frames < 2 {
        return;
    }
    let n = frames as f64;

    for m in 0..NUM_MEL_BINS {
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        for row in features.chunks(NUM_MEL_BINS).take(frames) {
            let v = row.get(m).copied().unwrap_or(0.0);
            sum += v;
        }
        let mean = sum / n;
        for row in features.chunks(NUM_MEL_BINS).take(frames) {
            let d = row.get(m).copied().unwrap_or(0.0) - mean;
            sum_sq += d * d;
        }
        let var = sum_sq / (n - 1.0);
        let inv_std = 1.0 / (var + NORM_EPS).sqrt();

        for row in features.chunks_mut(NUM_MEL_BINS).take(frames) {
            if let Some(v) = row.get_mut(m) {
                *v = (*v - mean) * inv_std;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_count_formula() {
        assert_eq!(num_frames(0), 0);
        assert_eq!(num_frames(399), 0);
        assert_eq!(num_frames(400), 1);
        assert_eq!(num_frames(559), 1);
        assert_eq!(num_frames(560), 2);
        // center = false: 1 second of audio gives 98 frames, not 100.
        assert_eq!(num_frames(16_000), 98);
    }

    #[test]
    fn odd_frame_count_is_truncated_not_padded() {
        // 1040 samples => exactly 5 frames; the 5th must be dropped.
        let n = FRAME_LENGTH + 4 * FRAME_SHIFT;
        assert_eq!(num_frames(n), 5);
        let samples: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
        let device = CpuDevice::new();
        let out = seamless_fbank::<CpuRuntime>(&samples, &device).expect("fbank");
        assert_eq!(out.shape(), &[2, STACKED_DIM]);
        assert_eq!(out.dtype(), DType::F32);

        // 6 frames must give 3 rows (no truncation).
        let n6 = FRAME_LENGTH + 5 * FRAME_SHIFT;
        let samples6: Vec<f32> = (0..n6).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
        let out6 = seamless_fbank::<CpuRuntime>(&samples6, &device).expect("fbank");
        assert_eq!(out6.shape(), &[3, STACKED_DIM]);
    }

    #[test]
    fn too_short_input_is_an_error() {
        let device = CpuDevice::new();
        let samples = vec![0.0f32; FRAME_LENGTH];
        assert!(seamless_fbank::<CpuRuntime>(&samples, &device).is_err());
    }

    #[test]
    fn normalization_uses_sample_variance() {
        // One active channel with values 1, 2, 3 across 3 frames.
        // mean = 2; ddof=1 var = ((-1)^2 + 0 + 1^2) / 2 = 1.0
        // (ddof=0 would give 2/3, i.e. inv_std ~= 1.2247 — a visible mismatch.)
        let mut features = vec![0.0f64; 3 * NUM_MEL_BINS];
        for (f, row) in features.chunks_mut(NUM_MEL_BINS).enumerate() {
            row[0] = f as f64 + 1.0;
        }
        normalize_per_bin(&mut features, 3);

        let expected = 1.0 / (1.0 + NORM_EPS).sqrt();
        let col: Vec<f64> = features.chunks(NUM_MEL_BINS).map(|row| row[0]).collect();
        assert!((col[0] + expected).abs() < 1e-12, "got {}", col[0]);
        assert!(col[1].abs() < 1e-12, "got {}", col[1]);
        assert!((col[2] - expected).abs() < 1e-12, "got {}", col[2]);

        // Constant channels collapse to zero rather than dividing by zero.
        assert!(features.chunks(NUM_MEL_BINS).all(|row| row[1] == 0.0));
    }

    #[test]
    fn output_is_zero_mean_per_mel_channel() {
        let n = FRAME_LENGTH + 63 * FRAME_SHIFT;
        let samples: Vec<f32> = (0..n).map(|i| (i as f32 * 0.017).sin() * 0.3).collect();
        let device = CpuDevice::new();
        let out = seamless_fbank::<CpuRuntime>(&samples, &device).expect("fbank");
        let rows = out.shape()[0];
        let data: Vec<f32> = out.to_vec();

        // Channel m appears at columns m and m + 80 of alternating rows;
        // together they are the full time series for that mel bin.
        for m in 0..NUM_MEL_BINS {
            let sum: f64 = data
                .chunks(STACKED_DIM)
                .take(rows)
                .map(|row| f64::from(row[m]) + f64::from(row[m + NUM_MEL_BINS]))
                .sum();
            assert!(sum.abs() < 1e-3, "channel {m} mean drifted: {sum}");
        }
    }
}
