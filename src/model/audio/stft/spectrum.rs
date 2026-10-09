//! The complex spectrum of a forward STFT: padding, framing, the analysis
//! window and the real FFT, shared by [`super::stft`] and [`stft_complex`].

use super::forward::{CenterPad, StftClient, StftOptions};
use crate::error::{Error, Result};
use numr::algorithm::fft::FftNormalization;
use numr::dtype::DType;
use numr::ops::PadMode;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// `imag` `[B, T, F]` with the DC bin, and the Nyquist bin of an even
/// `n_fft`, set to `+0.0`: the spectrum of a real signal holds no imaginary
/// part there, and pocketfft (`torch.stft`) gives exactly zero. A rounding
/// residue, or a `-0.0`, turns a negative bin's phase from `π` to `-π`.
/// Multiplying by 0 keeps the sign of a negative residue, and adding `+0.0`
/// then clears it.
fn real_signal_imag<R, C>(client: &C, imag: &Tensor<R>, n_fft: usize) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: StftClient<R>,
{
    let f_bins = n_fft / 2 + 1;
    let mut keep = vec![1.0f32; f_bins];
    keep[0] = 0.0;
    if n_fft.is_multiple_of(2) {
        keep[f_bins - 1] = 0.0;
    }
    let mut mask = Tensor::<R>::from_slice(&keep, &[1, 1, f_bins], imag.device())?;
    if imag.dtype() != DType::F32 {
        mask = client.cast(&mask, imag.dtype())?;
    }
    let kept = client.mul(imag, &mask)?;
    Ok(client.add_scalar(&kept, 0.0)?)
}

/// The real and imaginary parts of the forward STFT, each `[B, T_spec, F]`.
/// The checks and padding are [`super::stft`]'s.
pub(super) fn spectrum<R, C>(
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

    // `center=True` pads `n_fft/2` samples on each side.
    let (padded_t, padded) = if opts.center {
        let joined = match opts.pad {
            CenterPad::Zeros => {
                let pad = Tensor::<R>::zeros(&[b, half], dtype, device)?;
                client.cat(&[&pad, &waveform.contiguous()?, &pad], 1)?
            }
            CenterPad::Reflect => {
                if t_time <= half {
                    return Err(Error::InvalidArgument {
                        arg: "waveform",
                        reason: format!(
                            "reflect padding of {half} samples needs more than {half} samples, got {t_time}"
                        ),
                    });
                }
                client.pad_mode(&waveform.contiguous()?, &[half, half], PadMode::Reflect)?
            }
        };
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
    let imag = real_signal_imag(client, &client.imag(&spectrum)?, opts.n_fft)?;
    Ok((real, imag))
}

/// Run forward STFT, returning `(real, imag)`, each `[B, F, T_spec]`: the
/// complex spectrum [`super::stft`] takes its magnitude and phase from.
///
/// * `waveform` — `[B, T_time]` real samples.
/// * `window` — `[n_fft]` analysis window.
#[allow(clippy::type_complexity)]
pub fn stft_complex<R, C>(
    client: &C,
    waveform: &Tensor<R>,
    window: &Tensor<R>,
    opts: StftOptions,
) -> Result<(Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    C: StftClient<R>,
{
    let (real, imag) = spectrum(client, waveform, window, opts)?;
    Ok((
        real.permute(&[0, 2, 1])?.contiguous()?,
        imag.permute(&[0, 2, 1])?.contiguous()?,
    ))
}

#[cfg(test)]
#[allow(clippy::useless_vec)]
mod tests {
    use super::*;
    use crate::model::audio::stft::stft;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    fn tensor(
        data: &[f32],
        shape: &[usize],
        device: &<CpuRuntime as Runtime>::Device,
    ) -> Tensor<CpuRuntime> {
        Tensor::<CpuRuntime>::from_slice(data, shape, device).unwrap()
    }

    #[test]
    fn complex_parts_carry_a_tone_in_its_bin() {
        let (client, device) = cpu_setup();
        let n_fft = 16;
        let samples: Vec<f32> = (0..64)
            .map(|i| (std::f64::consts::TAU * 2.0 * i as f64 / n_fft as f64).cos() as f32)
            .collect();
        let wave = tensor(&samples, &[1, 64], &device);
        let win = tensor(&vec![1.0f32; n_fft], &[n_fft], &device);
        let opts = StftOptions {
            n_fft,
            hop_length: n_fft,
            center: false,
            pad: CenterPad::Zeros,
        };
        let (re, im) = stft_complex(&client, &wave, &win, opts).unwrap();
        assert_eq!(re.shape(), &[1, n_fft / 2 + 1, 4]);
        let re: Vec<f32> = re.to_vec();
        let im: Vec<f32> = im.to_vec();
        // A cosine at bin 2 is real: n_fft / 2 in bin 2, nothing elsewhere.
        for t in 0..4 {
            assert!((re[2 * 4 + t] - 8.0).abs() < 1e-4);
            assert!(im[2 * 4 + t].abs() < 1e-4);
            assert!(re[4 + t].abs() < 1e-4);
        }
    }

    #[test]
    fn rejects_wrong_window_size() {
        let (client, device) = cpu_setup();
        let wave = tensor(&vec![0.0f32; 16], &[1, 16], &device);
        let win = tensor(&vec![1.0f32; 5], &[5], &device);
        assert!(
            stft_complex(
                &client,
                &wave,
                &win,
                StftOptions {
                    n_fft: 8,
                    hop_length: 4,
                    center: false,
                    pad: CenterPad::Zeros,
                }
            )
            .is_err()
        );
    }

    /// A real signal's DC and Nyquist bins hold no imaginary part, so a
    /// negative one's phase is exactly `π`, as `torch.stft` gives it, never
    /// `-π`. An alternating signal is all Nyquist and a negative constant
    /// all DC.
    #[test]
    fn dc_and_nyquist_bins_of_a_negative_signal_have_phase_pi() {
        let (client, device) = cpu_setup();
        let n_fft = 8;
        let win = tensor(&vec![1.0f32; n_fft], &[n_fft], &device);
        let opts = StftOptions {
            n_fft,
            hop_length: 4,
            center: false,
            pad: CenterPad::Zeros,
        };
        let dc = tensor(&vec![-0.5f32; 16], &[1, 16], &device);
        let (_, phase) = stft(&client, &dc, &win, opts).unwrap();
        let phase: Vec<f32> = phase.to_vec();
        let frames = phase.len() / (n_fft / 2 + 1);
        assert!(
            phase[..frames].iter().all(|&p| p == std::f32::consts::PI),
            "{phase:?}"
        );
        let alternating: Vec<f32> = (0..16)
            .map(|i| if i % 2 == 0 { -0.5 } else { 0.5 })
            .collect();
        let nyquist = tensor(&alternating, &[1, 16], &device);
        let (_, phase) = stft(&client, &nyquist, &win, opts).unwrap();
        let phase: Vec<f32> = phase.to_vec();
        let last = &phase[phase.len() - frames..];
        assert!(last.iter().all(|&p| p == std::f32::consts::PI), "{last:?}");
    }

    #[test]
    fn rejects_too_short_signal() {
        let (client, device) = cpu_setup();
        let wave = tensor(&vec![0.0f32; 4], &[1, 4], &device);
        let win = tensor(&vec![1.0f32; 8], &[8], &device);
        assert!(
            stft_complex(
                &client,
                &wave,
                &win,
                StftOptions {
                    n_fft: 8,
                    hop_length: 4,
                    center: false,
                    pad: CenterPad::Zeros,
                }
            )
            .is_err()
        );
    }
}
