//! Forward short-time Fourier transform.
//!
//! Takes a time-domain waveform `[B, T_time]` and returns magnitude + phase
//! spectrograms `[B, F, T_spec]` where `F = n_fft/2 + 1`. Exact inverse:
//! [`crate::model::audio::stft::istft`].
//!
//! **Runs on every backend.** Framing is an [`numr::ops::IndexingOps::index_select`] with
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

use super::spectrum::spectrum;
use crate::error::Result;
use crate::model::traits::ModelClient;
use numr::algorithm::fft::FftAlgorithms;
use numr::dtype::DType;
use numr::ops::traits::ComplexOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Client capabilities [`stft`] needs beyond [`ModelClient`].
///
/// Deliberately identical to [`crate::model::audio::stft::IStftClient`]:
/// anything that can invert a spectrogram can produce one.
pub trait StftClient<R: Runtime>: ModelClient<R> + ComplexOps<R> + FftAlgorithms<R> {}

impl<R, C> StftClient<R> for C
where
    R: Runtime,
    C: ModelClient<R> + ComplexOps<R> + FftAlgorithms<R>,
{
}

/// What fills the `n_fft/2` samples a centered STFT adds at each end.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CenterPad {
    /// Zeros: librosa's `pad_mode='constant'`.
    #[default]
    Zeros,
    /// The signal mirrored without its edge sample: `torch.stft`'s default
    /// `pad_mode='reflect'`, which Kokoro's `TorchSTFT` takes. The signal
    /// must be longer than `n_fft/2`.
    Reflect,
}

/// Options controlling the forward STFT.
#[derive(Debug, Clone, Copy)]
pub struct StftOptions {
    pub n_fft: usize,
    pub hop_length: usize,
    /// If true, pad the input with `n_fft/2` samples on each end, filled as
    /// [`pad`](Self::pad) says, so the output
    /// `T_spec = 1 + T_time / hop_length`, matching `torch.stft(center=True)`.
    /// If false, no padding; `T_spec` is smaller.
    pub center: bool,
    /// What a centered STFT pads with. Ignored when `center` is false.
    pub pad: CenterPad,
}

impl Default for StftOptions {
    fn default() -> Self {
        Self {
            n_fft: 20,
            hop_length: 5,
            center: true,
            pad: CenterPad::Zeros,
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
    let (real, imag) = spectrum(client, waveform, window, opts)?;
    let (b, t_spec, f_bins) = (real.shape()[0], real.shape()[1], real.shape()[2]);
    let power = client.add(&client.mul(&real, &real)?, &client.mul(&imag, &imag)?)?;
    let mag = client.sqrt(&power)?;
    let phase = client.angle(&client.make_complex(&real, &imag)?)?;

    // [B, T_spec, F] -> [B, F, T_spec].
    debug_assert_eq!(mag.shape(), [b, t_spec, f_bins]);
    Ok((
        mag.permute(&[0, 2, 1])?.contiguous()?,
        phase.permute(&[0, 2, 1])?.contiguous()?,
    ))
}

#[cfg(test)]
#[allow(clippy::useless_vec)]
mod tests {
    use super::*;
    use crate::model::audio::stft::{IStftOptions, IStftPadding, istft};
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    fn tensor(
        data: &[f32],
        shape: &[usize],
        device: &<CpuRuntime as Runtime>::Device,
    ) -> Tensor<CpuRuntime> {
        Tensor::<CpuRuntime>::from_slice(data, shape, device).unwrap()
    }

    /// Periodic Hann window, the analysis/synthesis pair `istft` normalizes for.
    fn hann(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / n as f64).cos() as f32)
            .collect()
    }

    #[test]
    fn output_shape_follows_formula_without_center() {
        let (client, device) = cpu_setup();
        let n_fft = 8;
        let wave = tensor(&vec![0.0f32; 16], &[1, 16], &device);
        let win = tensor(&vec![1.0f32; n_fft], &[n_fft], &device);
        let (mag, phase) = stft(
            &client,
            &wave,
            &win,
            StftOptions {
                n_fft,
                hop_length: 4,
                center: false,
                pad: CenterPad::Zeros,
            },
        )
        .unwrap();
        // T_spec = (16 - 8)/4 + 1 = 3, F = 5.
        assert_eq!(mag.shape(), &[1, 5, 3]);
        assert_eq!(phase.shape(), &[1, 5, 3]);
    }

    #[test]
    fn output_shape_includes_center_padding() {
        let (client, device) = cpu_setup();
        let n_fft = 8;
        let wave = tensor(&vec![0.0f32; 16], &[1, 16], &device);
        let win = tensor(&vec![1.0f32; n_fft], &[n_fft], &device);
        let (mag, _) = stft(
            &client,
            &wave,
            &win,
            StftOptions {
                n_fft,
                hop_length: 4,
                center: true,
                pad: CenterPad::Zeros,
            },
        )
        .unwrap();
        // padded = 16 + 2*4 = 24; T_spec = (24-8)/4 + 1 = 5.
        assert_eq!(mag.shape(), &[1, 5, 5]);
    }

    #[test]
    fn zero_signal_produces_zero_magnitude() {
        let (client, device) = cpu_setup();
        let wave = tensor(&vec![0.0f32; 32], &[1, 32], &device);
        let win = tensor(&vec![1.0f32; 8], &[8], &device);
        let (mag, _) = stft(
            &client,
            &wave,
            &win,
            StftOptions {
                n_fft: 8,
                hop_length: 4,
                center: false,
                pad: CenterPad::Zeros,
            },
        )
        .unwrap();
        for v in mag.to_vec::<f32>() {
            assert!(v.abs() < 1e-5);
        }
    }

    #[test]
    fn constant_signal_concentrates_at_dc() {
        let (client, device) = cpu_setup();
        let wave = tensor(&vec![1.0f32; 16], &[1, 16], &device);
        let win = tensor(&vec![1.0f32; 4], &[4], &device);
        let (mag, _) = stft(
            &client,
            &wave,
            &win,
            StftOptions {
                n_fft: 4,
                hop_length: 2,
                center: false,
                pad: CenterPad::Zeros,
            },
        )
        .unwrap();
        let v: Vec<f32> = mag.to_vec();
        let t_spec = 7; // (16-4)/2+1
        // Bin 0 at each time equals the window sum, 4.
        for (t, &dc) in v.iter().take(t_spec).enumerate() {
            assert!((dc - 4.0).abs() < 1e-4, "DC bin at t={t}: {dc}");
        }
        for k in 1..3 {
            for t in 0..t_spec {
                let v = v[k * t_spec + t];
                assert!(v.abs() < 1e-4, "bin {k}, t {t}: {v}");
            }
        }
    }

    #[test]
    fn a_pure_tone_lands_in_its_own_bin() {
        // The test the shape checks cannot do: proves the transform is a DFT and
        // not merely the right size. A sinusoid at exactly bin k has all its
        // energy there.
        let (client, device) = cpu_setup();
        let n_fft = 64;
        let bin = 8usize;
        let samples: Vec<f32> = (0..n_fft * 4)
            .map(|i| (std::f64::consts::TAU * bin as f64 * i as f64 / n_fft as f64).sin() as f32)
            .collect();
        let wave = tensor(&samples, &[1, samples.len()], &device);
        let win = tensor(&vec![1.0f32; n_fft], &[n_fft], &device);
        let (mag, _) = stft(
            &client,
            &wave,
            &win,
            StftOptions {
                n_fft,
                hop_length: n_fft,
                center: false,
                pad: CenterPad::Zeros,
            },
        )
        .unwrap();
        let f = n_fft / 2 + 1;
        let t_spec = 4;
        let v: Vec<f32> = mag.to_vec();
        let peak = (0..f)
            .max_by(|&a, &b| v[a * t_spec].partial_cmp(&v[b * t_spec]).unwrap())
            .unwrap();
        assert_eq!(peak, bin, "peak landed in bin {peak}, expected {bin}");
        // Amplitude 1.0 over n_fft samples with a rectangular window gives n_fft/2.
        assert!(
            (v[bin * t_spec] - n_fft as f32 / 2.0).abs() < 1e-2,
            "peak magnitude {}",
            v[bin * t_spec]
        );
    }

    #[test]
    fn non_power_of_two_n_fft_works() {
        // Kokoro's generator uses n_fft = 20. numr's Bluestein path takes
        // sizes that are not a power of two, and this test pins that.
        let (client, device) = cpu_setup();
        let n_fft = 20usize;
        let samples: Vec<f32> = (0..200)
            .map(|i| (std::f64::consts::TAU * 4.0 * i as f64 / n_fft as f64).sin() as f32)
            .collect();
        let wave = tensor(&samples, &[1, samples.len()], &device);
        let win = tensor(&vec![1.0f32; n_fft], &[n_fft], &device);
        let (mag, _) = stft(
            &client,
            &wave,
            &win,
            StftOptions {
                n_fft,
                hop_length: n_fft,
                center: false,
                pad: CenterPad::Zeros,
            },
        )
        .unwrap();
        let f = n_fft / 2 + 1;
        let t_spec = 10;
        assert_eq!(mag.shape(), &[1, f, t_spec]);
        let v: Vec<f32> = mag.to_vec();
        let peak = (0..f)
            .max_by(|&a, &b| v[a * t_spec].partial_cmp(&v[b * t_spec]).unwrap())
            .unwrap();
        assert_eq!(peak, 4, "peak landed in bin {peak}, expected 4");
    }

    #[test]
    fn stft_then_istft_reconstructs_the_waveform() {
        // The strongest available check: the forward transform must be the exact
        // inverse of the one already tested in `stft::inverse`, phase included. A
        // magnitude-only bug or an off-by-one in the framing shows up here and
        // nowhere else.
        let (client, device) = cpu_setup();
        let n_fft = 32usize;
        let hop = 8usize;
        let samples: Vec<f32> = (0..512)
            .map(|i| {
                let t = i as f64;
                (0.6 * (t * 0.11).sin() + 0.3 * (t * 0.37 + 1.0).sin()) as f32
            })
            .collect();
        let wave = tensor(&samples, &[1, samples.len()], &device);
        let win = tensor(&hann(n_fft), &[n_fft], &device);

        let (mag, phase) = stft(
            &client,
            &wave,
            &win,
            StftOptions {
                n_fft,
                hop_length: hop,
                center: true,
                pad: CenterPad::Zeros,
            },
        )
        .unwrap();
        let back = istft(
            &client,
            &mag,
            &phase,
            &win,
            IStftOptions {
                hop_length: hop,
                padding: IStftPadding::Center,
                eps: 1e-8,
            },
        )
        .unwrap();

        let got: Vec<f32> = back.to_vec();
        // Center padding gives (T_spec - 1) * hop samples back.
        let n = got.len().min(samples.len());
        assert!(n >= samples.len() - hop, "reconstruction is {n} samples");
        // Skip the first and last frame: fewer windows overlap there, so those
        // samples are attenuated by design rather than reconstructed.
        for i in n_fft..n - n_fft {
            assert!(
                (got[i] - samples[i]).abs() < 1e-3,
                "sample {i}: {} vs {}",
                got[i],
                samples[i]
            );
        }
    }
}
