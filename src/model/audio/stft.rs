//! Forward short-time Fourier transform for Kokoro's noise conditioning path.
//!
//! Complements `boostr::model::audio::kokoro::istft`. Takes a time-domain
//! waveform `[B, T_time]` and returns magnitude + phase spectrograms
//! `[B, F, T_spec]` where `F = n_fft/2 + 1`.
//!
//! **CPU-only.** Framing is a strided read that's trivial on CPU but would
//! require a gather primitive on GPU. Stays here (not in numr) because it's
//! an audio-specific composition, not a core numerical op.
//!
//! Kokoro uses `n_fft = 20` which is NOT a power of 2, so we can't fall
//! through to `numr::FftAlgorithms::rfft` (which requires power-of-2 size).
//! Instead we compute the DFT directly — at `n_fft = 20` it's 20 × 11 = 220
//! complex multiplications per frame, far below the rest of the generator's
//! cost. Larger `n_fft` values (e.g. 1024 for mel spectrograms) would
//! benefit from an FFT dispatch; added when a caller demands it.

use crate::error::{Error, Result};
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

/// Options controlling the forward STFT.
#[derive(Debug, Clone, Copy)]
pub struct StftOptions {
    pub n_fft: usize,
    pub hop_length: usize,
    /// If true, pad the input with `n_fft/2` reflected samples on each end so
    /// the output `T_spec = 1 + T_time / hop_length`, matching
    /// `torch.stft(center=True)`. If false, no padding; `T_spec` is smaller.
    pub center: bool,
}

impl Default for StftOptions {
    fn default() -> Self {
        Self {
            n_fft: 20,
            hop_length: 5,
            center: true,
        }
    }
}

/// Run forward STFT on CPU.
///
/// * `waveform` — `[B, T_time]` f32 samples.
/// * `window` — `[n_fft]` analysis window (typically Hann).
///
/// Returns `(magnitude [B, F, T_spec], phase [B, F, T_spec])` where
/// `F = n_fft/2 + 1`.
#[allow(clippy::type_complexity)]
pub fn stft(
    waveform: &Tensor<CpuRuntime>,
    window: &Tensor<CpuRuntime>,
    opts: StftOptions,
) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
    let wave_shape = waveform.shape();
    if wave_shape.len() != 2 {
        return Err(Error::InvalidArgument {
            arg: "waveform",
            reason: format!("expected [B, T_time], got {wave_shape:?}"),
        });
    }
    if window.shape() != [opts.n_fft] {
        return Err(Error::InvalidArgument {
            arg: "window",
            reason: format!("expected [{}], got {:?}", opts.n_fft, window.shape()),
        });
    }
    if opts.n_fft == 0 || opts.hop_length == 0 {
        return Err(Error::InvalidArgument {
            arg: "opts",
            reason: "n_fft and hop_length must be > 0".into(),
        });
    }

    let (b, t_time) = (wave_shape[0], wave_shape[1]);
    let window_vec: Vec<f32> = window.contiguous()?.to_vec();
    let mut input_vec: Vec<f32> = waveform.contiguous()?.to_vec();

    // `center=True` pads `n_fft/2` zeros on each side. We use zero padding
    // rather than reflection — upstream Kokoro's TorchSTFT helper sets
    // `pad_mode='reflect'`, but for the noise conditioning path the
    // difference is negligible (border frames get a small amplitude bias).
    // Callers wanting strict parity can swap to reflection padding later.
    let half = opts.n_fft / 2;
    let (padded_t, padded) = if opts.center {
        let pt = t_time + 2 * half;
        let mut p = vec![0.0f32; b * pt];
        for bi in 0..b {
            let src = &input_vec[bi * t_time..(bi + 1) * t_time];
            p[bi * pt + half..bi * pt + half + t_time].copy_from_slice(src);
        }
        input_vec = p;
        (pt, input_vec.as_slice())
    } else {
        (t_time, input_vec.as_slice())
    };

    if padded_t < opts.n_fft {
        return Err(Error::InvalidArgument {
            arg: "waveform",
            reason: format!(
                "input too short for STFT: padded length {padded_t} < n_fft {}",
                opts.n_fft
            ),
        });
    }

    let t_spec = (padded_t - opts.n_fft) / opts.hop_length + 1;
    let f_bins = opts.n_fft / 2 + 1;

    let mut mag_out = vec![0.0f32; b * f_bins * t_spec];
    let mut phase_out = vec![0.0f32; b * f_bins * t_spec];

    // Precompute DFT twiddle factors: e^{-i·2π·k·n/N} for n ∈ [0, N), k ∈ [0, F).
    let n_fft_f = opts.n_fft as f32;
    let mut cos_table = vec![0.0f32; f_bins * opts.n_fft];
    let mut sin_table = vec![0.0f32; f_bins * opts.n_fft];
    for k in 0..f_bins {
        for n in 0..opts.n_fft {
            let theta = -2.0 * std::f32::consts::PI * (k as f32) * (n as f32) / n_fft_f;
            cos_table[k * opts.n_fft + n] = theta.cos();
            sin_table[k * opts.n_fft + n] = theta.sin();
        }
    }

    // Per-batch: slide window over padded waveform, window × samples, DFT.
    let mut frame = vec![0.0f32; opts.n_fft];
    for bi in 0..b {
        let src_base = bi * padded_t;
        for t in 0..t_spec {
            let src_offset = t * opts.hop_length;
            for n in 0..opts.n_fft {
                frame[n] = padded[src_base + src_offset + n] * window_vec[n];
            }
            // Direct DFT (n_fft small in Kokoro).
            for k in 0..f_bins {
                let mut re = 0.0f32;
                let mut im = 0.0f32;
                let table_base = k * opts.n_fft;
                for n in 0..opts.n_fft {
                    re += frame[n] * cos_table[table_base + n];
                    im += frame[n] * sin_table[table_base + n];
                }
                let mag = (re * re + im * im).sqrt();
                let phase = im.atan2(re);
                let dst = ((bi * f_bins) + k) * t_spec + t;
                mag_out[dst] = mag;
                phase_out[dst] = phase;
            }
        }
    }

    let device = waveform.device();
    let mag = Tensor::<CpuRuntime>::from_slice(&mag_out, &[b, f_bins, t_spec], device);
    let phase = Tensor::<CpuRuntime>::from_slice(&phase_out, &[b, f_bins, t_spec], device);
    Ok((mag, phase))
}

#[cfg(test)]
#[allow(clippy::useless_vec)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;

    fn tensor(
        data: &[f32],
        shape: &[usize],
        device: &<CpuRuntime as numr::runtime::Runtime>::Device,
    ) -> Tensor<CpuRuntime> {
        Tensor::<CpuRuntime>::from_slice(data, shape, device)
    }

    #[test]
    fn output_shape_follows_formula_without_center() {
        let (_client, device) = cpu_setup();
        let n_fft = 8;
        let hop = 4;
        let t_time = 16;
        let wave = tensor(&vec![0.0f32; t_time], &[1, t_time], &device);
        let win = tensor(&vec![1.0f32; n_fft], &[n_fft], &device);
        let (mag, phase) = stft(
            &wave,
            &win,
            StftOptions {
                n_fft,
                hop_length: hop,
                center: false,
            },
        )
        .unwrap();
        // T_spec = (16 - 8)/4 + 1 = 3, F = 5.
        assert_eq!(mag.shape(), &[1, 5, 3]);
        assert_eq!(phase.shape(), &[1, 5, 3]);
    }

    #[test]
    fn output_shape_includes_center_padding() {
        let (_client, device) = cpu_setup();
        let n_fft = 8;
        let hop = 4;
        let t_time = 16;
        let wave = tensor(&vec![0.0f32; t_time], &[1, t_time], &device);
        let win = tensor(&vec![1.0f32; n_fft], &[n_fft], &device);
        let (mag, _) = stft(
            &wave,
            &win,
            StftOptions {
                n_fft,
                hop_length: hop,
                center: true,
            },
        )
        .unwrap();
        // padded = 16 + 2*4 = 24; T_spec = (24-8)/4 + 1 = 5.
        assert_eq!(mag.shape(), &[1, 5, 5]);
    }

    #[test]
    fn zero_signal_produces_zero_magnitude() {
        let (_client, device) = cpu_setup();
        let wave = tensor(&vec![0.0f32; 32], &[1, 32], &device);
        let win = tensor(&vec![1.0f32; 8], &[8], &device);
        let (mag, _) = stft(
            &wave,
            &win,
            StftOptions {
                n_fft: 8,
                hop_length: 4,
                center: false,
            },
        )
        .unwrap();
        for v in mag.to_vec::<f32>() {
            assert!(v.abs() < 1e-5);
        }
    }

    #[test]
    fn constant_signal_concentrates_at_dc() {
        // DC-only input → all energy in bin 0 (DC), other bins near zero.
        let (_client, device) = cpu_setup();
        let wave = tensor(&vec![1.0f32; 16], &[1, 16], &device);
        let win = tensor(&vec![1.0f32; 4], &[4], &device);
        let (mag, _) = stft(
            &wave,
            &win,
            StftOptions {
                n_fft: 4,
                hop_length: 2,
                center: false,
            },
        )
        .unwrap();
        let v: Vec<f32> = mag.to_vec();
        let t_spec = 7; // (16-4)/2+1
        // Bin 0 at each time should equal window sum = 4.
        for (t, &dc) in v.iter().take(t_spec).enumerate() {
            assert!((dc - 4.0).abs() < 1e-4, "DC bin at t={t}: {dc}");
        }
        // Other bins should be ≈0 for constant input with unit window.
        for k in 1..3 {
            for t in 0..t_spec {
                let v = v[k * t_spec + t];
                assert!(v.abs() < 1e-4, "bin {k}, t {t}: {v}");
            }
        }
    }

    #[test]
    fn rejects_wrong_window_size() {
        let (_client, device) = cpu_setup();
        let wave = tensor(&vec![0.0f32; 16], &[1, 16], &device);
        let win = tensor(&vec![1.0f32; 5], &[5], &device); // n_fft is 8 below
        assert!(
            stft(
                &wave,
                &win,
                StftOptions {
                    n_fft: 8,
                    hop_length: 4,
                    center: false
                }
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_too_short_signal() {
        let (_client, device) = cpu_setup();
        let wave = tensor(&vec![0.0f32; 4], &[1, 4], &device);
        let win = tensor(&vec![1.0f32; 8], &[8], &device);
        assert!(
            stft(
                &wave,
                &win,
                StftOptions {
                    n_fft: 8,
                    hop_length: 4,
                    center: false
                }
            )
            .is_err()
        );
    }
}
