//! Inverse short-time Fourier transform for ISTFTNet vocoder output.
//!
//! Kokoro's decoder and NeuCodec's vocoder both emit magnitude + phase
//! spectrograms `[B, F, T_frames]` where `F = n_fft/2 + 1`. This module turns
//! that pair back into a time-domain waveform via Hermitian irfft + windowed
//! overlap-add with window-square normalization (matching `torch.istft` /
//! `librosa.istft` defaults).
//!
//! **Runs on every backend.** Overlap-add is an accumulating scatter along
//! strided output positions, expressed here as
//! [`IndexingOps::scatter_reduce`] with [`ScatterReduceOp::Sum`], which numr
//! implements on CPU, CUDA and WebGPU. An earlier version of this file was
//! `CpuRuntime`-only on the grounds that "numr does not yet expose
//! `scatter_add`" and that "numr's `irfft` rejects non-power-of-2" sizes;
//! neither is true any more — `scatter_reduce` covers the first and numr's
//! Bluestein path covers the second, including NeuCodec's `n_fft = 1920` and
//! Kokoro's 20.
//!
//! The vocoder tail is the last stage of every generation, so keeping it on
//! CPU forced a device round-trip per utterance on anything built with boostr.

use crate::error::{Error, Result};
use crate::model::traits::ModelClient;
use numr::algorithm::fft::{FftAlgorithms, FftNormalization};
use numr::dtype::DType;
use numr::ops::traits::{ComplexOps, ScatterReduceOp};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Client capabilities [`istft`] needs beyond [`ModelClient`].
///
/// `ModelClient` already carries the elementwise, compare, conditional,
/// indexing and shape ops; only the complex construction and the transform
/// itself are extra.
pub trait IStftClient<R: Runtime>: ModelClient<R> + ComplexOps<R> + FftAlgorithms<R> {}

impl<R, C> IStftClient<R> for C
where
    R: Runtime,
    C: ModelClient<R> + ComplexOps<R> + FftAlgorithms<R>,
{
}

/// How much of the overlap-added signal to trim from each end.
///
/// The overlap-add itself is identical in every case; only the crop differs,
/// and it changes both the output length and the alignment, so it must match
/// whatever forward transform produced the spectrogram.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IStftPadding {
    /// Trim `n_fft/2` from each end — mirrors `torch.istft(center=True)`,
    /// undoing a forward STFT that zero-padded both ends by `n_fft/2`.
    /// Output length is `(T-1)*hop`.
    Center,
    /// Trim `(n_fft - hop)/2` from each end — the convention used by Vocos-style
    /// ISTFT heads (`padding="same"`), including NeuCodec's acoustic decoder.
    /// Output length is `T*hop`, i.e. exactly `hop` samples per input frame.
    Same,
    /// No trim: return the full `(T-1)*hop + n_fft` overlap-added signal.
    None,
}

/// Options for `istft`.
#[derive(Debug, Clone, Copy)]
pub struct IStftOptions {
    /// Frame hop in samples.
    pub hop_length: usize,
    /// Which end-trim convention to apply.
    pub padding: IStftPadding,
    /// Minimum window-square sum below which a sample is masked to zero
    /// (avoids divide-by-near-zero at the waveform boundaries).
    pub eps: f32,
}

impl Default for IStftOptions {
    fn default() -> Self {
        Self {
            hop_length: 256,
            padding: IStftPadding::Center,
            eps: 1e-8,
        }
    }
}

/// Run iSTFT on CPU, returning waveform `[B, N_samples]` (f32).
///
/// * `mag` — `[B, F, T_frames]`
/// * `phase` — `[B, F, T_frames]`
/// * `window` — `[n_fft]` analysis / synthesis window (must match forward STFT)
pub fn istft<R, C>(
    client: &C,
    mag: &Tensor<R>,
    phase: &Tensor<R>,
    window: &Tensor<R>,
    opts: IStftOptions,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: IStftClient<R>,
{
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

    let device = mag.device();
    let dtype = mag.dtype();
    let raw_len = (t_frames - 1) * opts.hop_length + n_fft;

    // 1. Polar -> rectangular, then a real inverse transform per frame.
    //    `irfft` works along the LAST axis, so the spectrogram is permuted from
    //    [B, F, T] to [B, T, F] first — which is also the frame-major layout the
    //    overlap-add below wants, so the permute is not wasted work.
    let real = client.mul(mag, &client.cos(phase)?)?;
    let imag = client.mul(mag, &client.sin(phase)?)?;
    let spectrum = client.make_complex(&real, &imag)?;
    let spectrum = spectrum.permute(&[0, 2, 1])?.contiguous()?;

    // `Backward` is the numpy/torch default: the inverse divides by n_fft.
    // `None` would return n_fft times the intended amplitude.
    let frames = client.irfft(&spectrum, Some(n_fft), FftNormalization::Backward)?;

    // 2. Synthesis window, broadcast over batch and frame.
    let windowed = client.mul(&frames, &window.reshape(&[1, 1, n_fft])?)?;
    let windowed = windowed.contiguous()?.reshape(&[b, t_frames * n_fft])?;

    // 3. Overlap-add. Frame `t` sample `n` lands at output position
    //    `t * hop + n`; several frames hit the same position, which is exactly
    //    the accumulating scatter `ScatterReduceOp::Sum` performs.
    let mut positions = Vec::with_capacity(t_frames * n_fft);
    for t_idx in 0..t_frames {
        let base = t_idx * opts.hop_length;
        for n in 0..n_fft {
            positions.push((base + n) as i32);
        }
    }
    let index_row = Tensor::<R>::from_slice(&positions, &[1, t_frames * n_fft], device)?;
    let index = index_row
        .broadcast_to(&[b, t_frames * n_fft])?
        .contiguous()?;

    let wave_dst = Tensor::<R>::zeros(&[b, raw_len], dtype, device)?;
    // `include_self = true` keeps the zeros in `wave_dst`, which under Sum
    // contribute nothing — that is what makes the destination a clean accumulator.
    let waveform =
        client.scatter_reduce(&wave_dst, 1, &index, &windowed, ScatterReduceOp::Sum, true)?;

    // 4. Window-square normalization. Batch-independent: every row of the batch
    //    overlaps identically, so this scatters ONE row and broadcasts it.
    let window_sq = client.mul(window, window)?;
    let norm_src = window_sq
        .reshape(&[1, 1, n_fft])?
        .broadcast_to(&[1, t_frames, n_fft])?
        .contiguous()?
        .reshape(&[1, t_frames * n_fft])?;
    let norm_dst = Tensor::<R>::zeros(&[1, raw_len], dtype, device)?;
    let norm = client.scatter_reduce(
        &norm_dst,
        1,
        &index_row,
        &norm_src,
        ScatterReduceOp::Sum,
        true,
    )?;

    // Divide only where the window-square sum is meaningful. At the very edges
    // of the signal few or no frames overlap, so the sum approaches zero and the
    // quotient would blow up; those samples are defined as silence.
    let eps_t = Tensor::<R>::full_scalar(&[1, raw_len], dtype, opts.eps as f64, device)?;
    let valid = client.gt(&norm, &eps_t)?;
    let ones = Tensor::<R>::ones(&[1, raw_len], dtype, device)?;
    let safe_norm = client.where_cond(&valid, &norm, &ones)?;

    let normalized = client.div(&waveform, &safe_norm)?;
    let zeros_full = Tensor::<R>::zeros(&[b, raw_len], dtype, device)?;
    let valid_full = valid.broadcast_to(&[b, raw_len])?.contiguous()?;
    let normalized = client.where_cond(&valid_full, &normalized, &zeros_full)?;

    // 5. Crop to the requested padding convention.
    let trim = match opts.padding {
        IStftPadding::Center => n_fft / 2,
        IStftPadding::Same => n_fft.saturating_sub(opts.hop_length) / 2,
        IStftPadding::None => 0,
    };
    if trim == 0 {
        return Ok(normalized);
    }
    if raw_len < 2 * trim {
        return Err(Error::InvalidArgument {
            arg: "mag",
            reason: format!(
                "signal of {raw_len} samples is too short to trim {trim} from each end \
                 ({:?} padding)",
                opts.padding
            ),
        });
    }
    Ok(normalized
        .narrow(1, trim, raw_len - 2 * trim)?
        .contiguous()?)
}

#[cfg(test)]
#[allow(clippy::useless_vec)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::Runtime;
    use numr::runtime::cpu::CpuRuntime;

    fn make_tensor(
        data: &[f32],
        shape: &[usize],
        device: &<CpuRuntime as Runtime>::Device,
    ) -> Tensor<CpuRuntime> {
        Tensor::<CpuRuntime>::from_slice(data, shape, device).unwrap()
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
            padding: IStftPadding::None,
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
            padding: IStftPadding::None,
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
                padding: IStftPadding::Center,
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
                padding: IStftPadding::None,
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
