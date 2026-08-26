//! Forward short-time Fourier transform.
//!
//! Takes a time-domain waveform `[B, T_time]` and returns magnitude + phase
//! spectrograms `[B, F, T_spec]` where `F = n_fft/2 + 1`. Exact inverse:
//! [`crate::model::audio::kokoro::istft`].
//!
//! **Runs on every backend.** Framing is an [`IndexingOps::index_select`] with
//! a strided position table — the mirror image of the `scatter_reduce` that
//! `istft` uses for overlap-add — so the whole transform stays on device.
//!
//! An earlier version was `CpuRuntime`-only and evaluated a direct DFT, on the
//! grounds that framing "would require a gather primitive on GPU" and that
//! Kokoro's `n_fft = 20` is not a power of two so `rfft` was unavailable.
//! Both premises are gone: numr exposes `index_select` on every backend, and
//! numr's Bluestein path takes arbitrary `n_fft`. The direct DFT also cost
//! `F * n_fft` multiply-adds per frame, which is fine at Kokoro's `n_fft = 20`
//! and roughly fifty times an FFT's work at the `n_fft = 1024` that spectral
//! denoising uses.

use crate::error::{Error, Result};
use crate::model::traits::ModelClient;
use numr::algorithm::fft::{FftAlgorithms, FftNormalization};
use numr::dtype::DType;
use numr::ops::traits::ComplexOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Client capabilities [`stft`] needs beyond [`ModelClient`].
///
/// Deliberately identical to [`crate::model::audio::kokoro::IStftClient`]:
/// anything that can invert a spectrogram can produce one.
pub trait StftClient<R: Runtime>: ModelClient<R> + ComplexOps<R> + FftAlgorithms<R> {}

impl<R, C> StftClient<R> for C
where
    R: Runtime,
    C: ModelClient<R> + ComplexOps<R> + FftAlgorithms<R>,
{
}

/// Options controlling the forward STFT.
#[derive(Debug, Clone, Copy)]
pub struct StftOptions {
    pub n_fft: usize,
    pub hop_length: usize,
    /// If true, pad the input with `n_fft/2` zeros on each end so the output
    /// `T_spec = 1 + T_time / hop_length`, matching `torch.stft(center=True)`.
    /// If false, no padding; `T_spec` is smaller.
    ///
    /// Upstream Kokoro's `TorchSTFT` helper sets `pad_mode='reflect'`. Zero
    /// padding gives the border frames a small amplitude bias and is what the
    /// matching [`crate::model::audio::kokoro::IStftPadding::Center`] trim
    /// undoes.
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

/// Run forward STFT, returning `(magnitude, phase)`, each `[B, F, T_spec]`.
///
/// * `waveform` — `[B, T_time]` real samples.
/// * `window` — `[n_fft]` analysis window (typically Hann).
#[allow(clippy::type_complexity)]
pub fn stft<R, C>(
    client: &C,
    waveform: &Tensor<R>,
    window: &Tensor<R>,
    opts: StftOptions,
) -> Result<(Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    C: StftClient<R>,
{
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
    let device = waveform.device();
    let dtype = waveform.dtype();
    let half = opts.n_fft / 2;

    // `center=True` pads `n_fft/2` zeros on each side.
    let (padded_t, padded) = if opts.center {
        let pad = Tensor::<R>::zeros(&[b, half], dtype, device)?;
        let joined = client.cat(&[&pad, &waveform.contiguous()?, &pad], 1)?;
        (t_time + 2 * half, joined)
    } else {
        (t_time, waveform.contiguous()?)
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

    // Framing. Frame `t` sample `n` reads padded position `t * hop + n` — the
    // exact inverse of the position table `istft` scatters into.
    let mut positions = Vec::with_capacity(t_spec * opts.n_fft);
    for t_idx in 0..t_spec {
        let base = t_idx * opts.hop_length;
        for n in 0..opts.n_fft {
            positions.push((base + n) as i32);
        }
    }
    let index = Tensor::<R>::from_slice(&positions, &[t_spec * opts.n_fft], device)?;
    let frames = client
        .index_select(&padded, 1, &index)?
        .contiguous()?
        .reshape(&[b, t_spec, opts.n_fft])?;

    // Analysis window, broadcast over batch and frame.
    let windowed = client.mul(&frames, &window.reshape(&[1, 1, opts.n_fft])?)?;

    // `rfft` runs along the last axis, which is already `n_fft`.
    // `None` normalization: the forward transform applies no scaling, matching
    // `torch.stft` and the `Backward` inverse `istft` uses.
    let spectrum = client.rfft(&windowed.contiguous()?, FftNormalization::None)?;

    let real = client.real(&spectrum)?;
    let imag = client.imag(&spectrum)?;
    let power = client.add(&client.mul(&real, &real)?, &client.mul(&imag, &imag)?)?;
    let mag = client.sqrt(&power)?;
    let phase = client.angle(&spectrum)?;

    // [B, T_spec, F] -> [B, F, T_spec].
    debug_assert_eq!(mag.shape(), [b, t_spec, f_bins]);
    Ok((
        mag.permute(&[0, 2, 1])?.contiguous()?,
        phase.permute(&[0, 2, 1])?.contiguous()?,
    ))
}

#[cfg(test)]
mod tests;
