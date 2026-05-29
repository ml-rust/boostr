//! Inverse short-time Fourier transform for ISTFTNet vocoder output.
//!
//! Kokoro's decoder emits magnitude + phase spectrograms `[B, F, T_frames]`
//! where `F = n_fft/2 + 1`. This module turns that pair back into a time-domain
//! waveform via Hermitian irfft + windowed overlap-add with window-square
//! normalization (matching `torch.istft` / `librosa.istft` defaults).
//!
//! **CPU-only.** Overlap-add is an accumulating scatter along strided output
//! positions. numr does not yet expose `scatter_add`; rather than add it
//! speculatively, we do the accumulation on `CpuRuntime` where tensor data is
//! directly addressable. Promoting to GPU backends requires a `scatter_add`
//! primitive — a single-session addition — plus an index-tensor build. Until
//! then, callers on CUDA/WebGPU must transfer the spectrograms to CPU.

use crate::error::{Error, Result};
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

/// Options for `istft`.
#[derive(Debug, Clone, Copy)]
pub struct IStftOptions {
    /// Frame hop in samples.
    pub hop_length: usize,
    /// If true, trim the first and last `n_fft/2` samples (mirrors
    /// `torch.istft(center=True)` — forward STFT padded both ends with
    /// `n_fft/2` zeros, iSTFT undoes that crop).
    pub center: bool,
    /// Minimum window-square sum below which a sample is masked to zero
    /// (avoids divide-by-near-zero at the waveform boundaries).
    pub eps: f32,
}

impl Default for IStftOptions {
    fn default() -> Self {
        Self {
            hop_length: 256,
            center: true,
            eps: 1e-8,
        }
    }
}

/// Run iSTFT on CPU, returning waveform `[B, N_samples]` (f32).
///
/// * `mag` — `[B, F, T_frames]`
/// * `phase` — `[B, F, T_frames]`
/// * `window` — `[n_fft]` analysis / synthesis window (must match forward STFT)
pub fn istft(
    client: &CpuClient,
    mag: &Tensor<CpuRuntime>,
    phase: &Tensor<CpuRuntime>,
    window: &Tensor<CpuRuntime>,
    opts: IStftOptions,
) -> Result<Tensor<CpuRuntime>> {
    if mag.shape() != phase.shape() {
        return Err(Error::InvalidArgument {
            arg: "phase",
            reason: format!(
                "shape must match mag ({:?}), got {:?}",
                mag.shape(),
                phase.shape()
            ),
        });
    }
    let m_shape = mag.shape();
    if m_shape.len() != 3 {
        return Err(Error::InvalidArgument {
            arg: "mag",
            reason: format!("expected [B, F, T], got {m_shape:?}"),
        });
    }
    let (b, f, t_frames) = (m_shape[0], m_shape[1], m_shape[2]);

    let w_shape = window.shape();
    if w_shape.len() != 1 {
        return Err(Error::InvalidArgument {
            arg: "window",
            reason: format!("expected 1D window [n_fft], got {w_shape:?}"),
        });
    }
    let n_fft = w_shape[0];
    if f != n_fft / 2 + 1 {
        return Err(Error::InvalidArgument {
            arg: "mag",
            reason: format!("F ({f}) must equal n_fft/2+1 ({})", n_fft / 2 + 1),
        });
    }
    if opts.hop_length == 0 {
        return Err(Error::InvalidArgument {
            arg: "hop_length",
            reason: "must be > 0".into(),
        });
    }
    if t_frames == 0 {
        return Err(Error::InvalidArgument {
            arg: "mag",
            reason: "T_frames must be > 0".into(),
        });
    }

    // 1. Extract magnitude/phase to host buffers. Kokoro's `n_fft = 20` is
    // NOT power-of-2 so numr's `irfft` rejects it; we run a direct inverse
    // DFT per frame instead. At `n_fft = 20` it's 11 × 20 = 220 complex
    // multiplies per frame — negligible vs the rest of the generator.
    let _ = client; // kept for future GPU path (irfft when n_fft is PoT).
    let mag_flat: Vec<f32> = mag.contiguous().to_vec();
    let phase_flat: Vec<f32> = phase.contiguous().to_vec();
    let window_samples: Vec<f32> = window.contiguous().to_vec();

    // Precompute twiddle tables e^{i·2π·k·n/N} (positive sign = inverse DFT).
    let n_fft_f = n_fft as f32;
    let f_bins = n_fft / 2 + 1;
    let mut cos_table = vec![0.0f32; f_bins * n_fft];
    let mut sin_table = vec![0.0f32; f_bins * n_fft];
    for k in 0..f_bins {
        for n in 0..n_fft {
            let theta = 2.0 * std::f32::consts::PI * (k as f32) * (n as f32) / n_fft_f;
            cos_table[k * n_fft + n] = theta.cos();
            sin_table[k * n_fft + n] = theta.sin();
        }
    }

    // 2. Inverse DFT + windowing per frame.
    let mut windowed_flat = vec![0.0f32; b * t_frames * n_fft];
    let inv_n = 1.0f32 / n_fft_f;
    for b_idx in 0..b {
        for t_idx in 0..t_frames {
            for (n, &w) in window_samples.iter().take(n_fft).enumerate() {
                let mut acc = 0.0f32;
                // Full Hermitian sum: bin 0 and bin n_fft/2 (if n_fft even)
                // contribute once; bins in between contribute twice (conjugate).
                for k in 0..f_bins {
                    let src = (b_idx * f_bins + k) * t_frames + t_idx;
                    let mag_k = mag_flat[src];
                    let ph_k = phase_flat[src];
                    let theta =
                        ph_k + 2.0 * std::f32::consts::PI * (k as f32) * (n as f32) / n_fft_f;
                    let term = mag_k * theta.cos();
                    let mirror = k != 0 && !(n_fft % 2 == 0 && k == n_fft / 2);
                    acc += term * if mirror { 2.0 } else { 1.0 };
                }
                let dst = (b_idx * t_frames + t_idx) * n_fft + n;
                windowed_flat[dst] = acc * inv_n * w;
            }
        }
    }
    // The cos/sin tables are only kept for potential future use (caching
    // across frames inside one call); drop to keep memory predictable.
    drop(cos_table);
    drop(sin_table);

    let raw_len = (t_frames - 1) * opts.hop_length + n_fft;
    let mut waveform = vec![0.0f32; b * raw_len];
    let mut norm = vec![0.0f32; raw_len];

    // Precompute window^2 once.
    let window_sq: Vec<f32> = window_samples.iter().map(|w| w * w).collect();

    for b_idx in 0..b {
        for t_idx in 0..t_frames {
            let frame_base = (b_idx * t_frames + t_idx) * n_fft;
            let wave_base = b_idx * raw_len + t_idx * opts.hop_length;
            for n in 0..n_fft {
                waveform[wave_base + n] += windowed_flat[frame_base + n];
            }
        }
    }
    // Normalization vector is batch-independent.
    for t_idx in 0..t_frames {
        let base = t_idx * opts.hop_length;
        for n in 0..n_fft {
            norm[base + n] += window_sq[n];
        }
    }

    // Apply normalization.
    for b_idx in 0..b {
        for n in 0..raw_len {
            let nrm = norm[n];
            if nrm > opts.eps {
                waveform[b_idx * raw_len + n] /= nrm;
            } else {
                waveform[b_idx * raw_len + n] = 0.0;
            }
        }
    }

    // 6. Optionally crop the `center=True` padding.
    let (out_len, output) = if opts.center {
        let half = n_fft / 2;
        if raw_len < 2 * half {
            return Err(Error::InvalidArgument {
                arg: "mag",
                reason: "signal too short to remove center padding".into(),
            });
        }
        let out_len = raw_len - 2 * half;
        let mut cropped = vec![0.0f32; b * out_len];
        for b_idx in 0..b {
            let src = &waveform[b_idx * raw_len + half..b_idx * raw_len + half + out_len];
            cropped[b_idx * out_len..(b_idx + 1) * out_len].copy_from_slice(src);
        }
        (out_len, cropped)
    } else {
        (raw_len, waveform)
    };

    let device = mag.device();
    Ok(Tensor::<CpuRuntime>::from_slice(
        &output,
        &[b, out_len],
        device,
    ))
}

#[cfg(test)]
#[allow(clippy::useless_vec)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::Runtime;

    fn make_tensor(
        data: &[f32],
        shape: &[usize],
        device: &<CpuRuntime as Runtime>::Device,
    ) -> Tensor<CpuRuntime> {
        Tensor::<CpuRuntime>::from_slice(data, shape, device)
    }

    #[test]
    fn output_shape_matches_overlap_add_formula() {
        let (client, device) = cpu_setup();
        let n_fft = 8;
        let hop = 4;
        let t_frames = 5;
        let f = n_fft / 2 + 1;
        let mag = make_tensor(&vec![0.0f32; f * t_frames], &[1, f, t_frames], &device);
        let phase = make_tensor(&vec![0.0f32; f * t_frames], &[1, f, t_frames], &device);
        let window = make_tensor(&vec![1.0f32; n_fft], &[n_fft], &device);

        let opts = IStftOptions {
            hop_length: hop,
            center: false,
            eps: 1e-8,
        };
        let out = istft(&client, &mag, &phase, &window, opts).unwrap();
        let expected_len = (t_frames - 1) * hop + n_fft;
        assert_eq!(out.shape(), &[1, expected_len]);
    }

    #[test]
    fn zero_spectrogram_yields_zero_waveform() {
        let (client, device) = cpu_setup();
        let n_fft = 8;
        let t_frames = 4;
        let f = n_fft / 2 + 1;
        let mag = make_tensor(&vec![0.0f32; f * t_frames], &[1, f, t_frames], &device);
        let phase = make_tensor(&vec![0.0f32; f * t_frames], &[1, f, t_frames], &device);
        let window = make_tensor(&vec![0.5f32; n_fft], &[n_fft], &device);

        let out = istft(&client, &mag, &phase, &window, IStftOptions::default()).unwrap();
        for v in out.to_vec::<f32>() {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn dc_only_spectrogram_reconstructs_constant() {
        // A pure-DC spectrogram (mag[0]=C, rest 0; phase=0) back-transforms to a
        // constant signal. Through windowed overlap-add with unit window, the
        // normalized output is that constant across the valid region.
        let (client, device) = cpu_setup();
        let n_fft = 4;
        let hop = 2;
        let t_frames = 3;
        let f = n_fft / 2 + 1;

        // mag[0, f_bin=0, :] = C; all other bins zero.
        let c = 4.0f32;
        let mut mag_data = vec![0.0f32; f * t_frames];
        mag_data.iter_mut().take(t_frames).for_each(|v| *v = c);
        let mag = make_tensor(&mag_data, &[1, f, t_frames], &device);
        let phase = make_tensor(&vec![0.0f32; f * t_frames], &[1, f, t_frames], &device);
        let window = make_tensor(&vec![1.0f32; n_fft], &[n_fft], &device);

        let opts = IStftOptions {
            hop_length: hop,
            center: false,
            eps: 1e-8,
        };
        let out = istft(&client, &mag, &phase, &window, opts).unwrap();
        let samples: Vec<f32> = out.to_vec();
        // DC bin coefficient in irfft(Backward norm) contributes C / n_fft per sample.
        // Middle samples should be approximately that constant (exact with unit window
        // and proper normalization).
        let expected = c / n_fft as f32;
        // Check middle region (where all T_frames overlap).
        let mid_start = n_fft;
        let mid_end = samples.len().saturating_sub(n_fft);
        for (i, sample) in samples.iter().enumerate().take(mid_end).skip(mid_start) {
            assert!(
                (sample - expected).abs() < 1e-4,
                "sample {i}: {sample} vs expected {expected}"
            );
        }
    }

    #[test]
    fn center_trim_removes_n_fft_over_2_from_each_end() {
        let (client, device) = cpu_setup();
        let n_fft = 8;
        let hop = 4;
        let t_frames = 6;
        let f = n_fft / 2 + 1;
        let mag = make_tensor(&vec![0.0f32; f * t_frames], &[1, f, t_frames], &device);
        let phase = make_tensor(&vec![0.0f32; f * t_frames], &[1, f, t_frames], &device);
        let window = make_tensor(&vec![1.0f32; n_fft], &[n_fft], &device);

        let with_center = istft(
            &client,
            &mag,
            &phase,
            &window,
            IStftOptions {
                hop_length: hop,
                center: true,
                eps: 1e-8,
            },
        )
        .unwrap();
        let without_center = istft(
            &client,
            &mag,
            &phase,
            &window,
            IStftOptions {
                hop_length: hop,
                center: false,
                eps: 1e-8,
            },
        )
        .unwrap();
        assert_eq!(without_center.shape()[1] - with_center.shape()[1], n_fft);
    }

    #[test]
    fn rejects_mismatched_mag_phase_shapes() {
        let (client, device) = cpu_setup();
        let mag = make_tensor(&vec![0.0f32; 9], &[1, 3, 3], &device);
        let phase = make_tensor(&vec![0.0f32; 6], &[1, 3, 2], &device);
        let window = make_tensor(&vec![1.0f32; 4], &[4], &device);
        assert!(istft(&client, &mag, &phase, &window, IStftOptions::default()).is_err());
    }

    #[test]
    fn rejects_bad_window_size() {
        let (client, device) = cpu_setup();
        // n_fft=4 implies F=3, but mag here has F=5 (would need n_fft=8).
        let mag = make_tensor(&vec![0.0f32; 15], &[1, 5, 3], &device);
        let phase = make_tensor(&vec![0.0f32; 15], &[1, 5, 3], &device);
        let window = make_tensor(&vec![1.0f32; 4], &[4], &device);
        assert!(istft(&client, &mag, &phase, &window, IStftOptions::default()).is_err());
    }
}
