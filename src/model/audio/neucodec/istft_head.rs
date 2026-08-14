//! `IstftHead` — NeuCodec's final `head.linear` projection plus the
//! magnitude/phase split that feeds `istft`.
//!
//! Checkpoint: `head.linear` is `Linear[1922, 1024]` (+ bias) applied to the
//! post-`post_net` residual stream `[B, T, 1024]`, producing `[B, T, 1922]`.
//! `1922 = 2 * (1920/2 + 1)` splits into log-magnitude and phase, each
//! `[B, T, 961]`:
//!
//! * `magnitude = clamp(exp(mag), max=1e2)` — the clamp is applied to the
//!   LINEAR magnitude, AFTER `exp`, matching upstream
//!   `neucodec/codec_decoder_vocos.py`:
//!
//!   ```python
//!   mag = torch.exp(mag)
//!   mag = torch.clip(mag, max=1e2)
//!   ```
//!
//!   Clamping the log-magnitude instead would cap the output at `exp(1e2)`
//!   ≈ 2.7e43 rather than at 100 — a ceiling so high it never binds, which
//!   defeats the guard entirely.
//! * `phase` is used directly as a radian angle (NOT `sin(phase)` — unlike
//!   Kokoro's `MagPhaseHead`, whose conv-based head bakes `sin()` into the
//!   phase channel). The complex spectrum is
//!   `magnitude * (cos(phase) + i*sin(phase))`, which
//!   [`crate::model::audio::kokoro::istft`] already computes internally from
//!   a raw phase angle, so no extra activation is applied here.
//!
//! Output layout is permuted to `[B, F, T]` (channels-first) to match
//! `istft`'s expected `(mag, phase)` shape.

use crate::error::{Error, Result};
use crate::model::audio::neucodec::client::NeuCodecClient;
use crate::nn::{Linear, var_contiguous};
use numr::autograd::{Var, var_clamp, var_exp, var_narrow, var_permute};
use numr::dtype::DType;
use numr::runtime::Runtime;

/// Bundled, already-built weights for the ISTFT head.
pub struct IstftHeadWeights<R: Runtime> {
    pub linear: Linear<R>,
}

/// Projects the decoder's residual stream to log-magnitude/phase and
/// activates them, ready for `istft`.
pub struct IstftHead<R: Runtime> {
    linear: Linear<R>,
    n_freq_bins: usize,
    mag_clamp_max: f32,
}

impl<R: Runtime> IstftHead<R> {
    /// `n_fft` must be a positive even number; `n_freq_bins = n_fft/2 + 1`
    /// and the linear layer's output width must be `2 * n_freq_bins`.
    pub fn new(weights: IstftHeadWeights<R>, n_fft: usize, mag_clamp_max: f32) -> Result<Self> {
        if n_fft == 0 || !n_fft.is_multiple_of(2) {
            return Err(Error::InvalidArgument {
                arg: "n_fft",
                reason: "must be a positive even number".into(),
            });
        }
        Ok(Self {
            linear: weights.linear,
            n_freq_bins: n_fft / 2 + 1,
            mag_clamp_max,
        })
    }

    pub fn n_freq_bins(&self) -> usize {
        self.n_freq_bins
    }
}

impl<R: Runtime<DType = DType>> IstftHead<R> {
    /// Forward: `x [B, T, hidden] -> (mag [B, F, T], phase [B, F, T])`.
    #[allow(clippy::type_complexity)]
    pub fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<(Var<R>, Var<R>)>
    where
        C: NeuCodecClient<R>,
        R::Client: NeuCodecClient<R>,
    {
        let out = self.linear.forward(client, x)?;
        let shape = out.shape().to_vec();
        let expected = 2 * self.n_freq_bins;
        if shape.len() != 3 || shape[2] != expected {
            return Err(Error::InvalidArgument {
                arg: "head.linear output",
                reason: format!("expected [B, T, {expected}], got {shape:?}"),
            });
        }

        let mag_log = var_narrow(&out, 2, 0, self.n_freq_bins).map_err(Error::Numr)?;
        let phase = var_narrow(&out, 2, self.n_freq_bins, self.n_freq_bins).map_err(Error::Numr)?;
        let mag_log = var_contiguous(&mag_log)?;
        let phase = var_contiguous(&phase)?;

        // exp FIRST, then clamp the linear magnitude (upstream order).
        let mag = var_exp(&mag_log, client).map_err(Error::Numr)?;
        let mag = var_clamp(&mag, f64::NEG_INFINITY, self.mag_clamp_max as f64, client)
            .map_err(Error::Numr)?;

        // [B, T, F] -> [B, F, T] for `istft`.
        let mag = var_permute(&mag, &[0, 2, 1]).map_err(Error::Numr)?;
        let mag = var_contiguous(&mag)?;
        let phase = var_permute(&phase, &[0, 2, 1]).map_err(Error::Numr)?;
        let phase = var_contiguous(&phase)?;

        Ok((mag, phase))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

    fn head(
        hidden: usize,
        n_fft: usize,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> IstftHead<CpuRuntime> {
        let f = n_fft / 2 + 1;
        let out = 2 * f;
        IstftHead::new(
            IstftHeadWeights {
                linear: Linear::new(
                    Tensor::<CpuRuntime>::from_slice(
                        &vec![0.0f32; out * hidden],
                        &[out, hidden],
                        device,
                    ),
                    Some(Tensor::<CpuRuntime>::from_slice(
                        &vec![0.0f32; out],
                        &[out],
                        device,
                    )),
                    false,
                ),
            },
            n_fft,
            1e2,
        )
        .unwrap()
    }

    #[test]
    fn forward_returns_correct_shapes() {
        let (client, device) = cpu_setup();
        let hidden = 8;
        let n_fft = 20;
        let h = head(hidden, n_fft, &device);
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(
                &vec![0.0f32; 2 * 6 * hidden],
                &[2, 6, hidden],
                &device,
            ),
            false,
        );
        let (mag, phase) = h.forward(&client, &x).unwrap();
        assert_eq!(mag.shape(), &[2, 11, 6]);
        assert_eq!(phase.shape(), &[2, 11, 6]);
    }

    #[test]
    fn zero_input_yields_mag_one_phase_zero() {
        // linear weight+bias are zero => head output is zero everywhere =>
        // mag_log=0 => clamp(exp(0), max=1e2) = clamp(1, max=1e2) = 1;
        // phase=0 (no activation).
        let (client, device) = cpu_setup();
        let hidden = 4;
        let n_fft = 8;
        let h = head(hidden, n_fft, &device);
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&vec![1.0f32; 3 * hidden], &[1, 3, hidden], &device),
            false,
        );
        let (mag, phase) = h.forward(&client, &x).unwrap();
        for v in mag.tensor().contiguous().unwrap().to_vec::<f32>() {
            assert!((v - 1.0).abs() < 1e-5, "mag should be 1, got {v}");
        }
        for v in phase.tensor().contiguous().unwrap().to_vec::<f32>() {
            assert!(v.abs() < 1e-5, "phase should be 0, got {v}");
        }
    }

    #[test]
    fn mag_clamp_bounds_large_input() {
        // Large positive linear output should saturate at mag_clamp_max itself
        // (the clamp is on the LINEAR magnitude, after exp), not at
        // exp(mag_clamp_max), and must not blow up to exp(huge) = inf.
        let (client, device) = cpu_setup();
        let hidden = 4;
        let n_fft = 8;
        let f = n_fft / 2 + 1;
        let out = 2 * f;
        let clamp_max = 1e2f32;
        let head = IstftHead::new(
            IstftHeadWeights {
                linear: Linear::new(
                    Tensor::<CpuRuntime>::from_slice(
                        &vec![0.0f32; out * hidden],
                        &[out, hidden],
                        &device,
                    ),
                    // Bias alone drives the pre-activation magnitude channels
                    // to 1e6, far past the clamp ceiling.
                    Some(Tensor::<CpuRuntime>::from_slice(
                        &vec![1.0e6f32; f]
                            .into_iter()
                            .chain(vec![0.0f32; f])
                            .collect::<Vec<_>>(),
                        &[out],
                        &device,
                    )),
                    false,
                ),
            },
            n_fft,
            clamp_max,
        )
        .unwrap();
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; 2 * hidden], &[1, 2, hidden], &device),
            false,
        );
        let (mag, _phase) = head.forward(&client, &x).unwrap();
        for v in mag.tensor().contiguous().unwrap().to_vec::<f32>() {
            assert!(v.is_finite(), "mag exploded to {v}");
            assert!(
                (v - clamp_max).abs() < 1e-3,
                "expected the linear-magnitude ceiling {clamp_max}, got {v} \
                 (a value near exp({clamp_max}) means the clamp was applied \
                 before exp instead of after)"
            );
        }
    }

    #[test]
    fn rejects_wrong_output_width() {
        let (client, device) = cpu_setup();
        // Linear layer's output width doesn't match 2*(n_fft/2+1).
        let head = IstftHead::new(
            IstftHeadWeights {
                linear: Linear::new(
                    Tensor::<CpuRuntime>::from_slice(&[0.0f32; 5 * 4], &[5, 4], &device),
                    None,
                    false,
                ),
            },
            8, // n_freq_bins=5, expected out=10, but linear only outputs 5
            1e2,
        )
        .unwrap();
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.0f32; 4], &[1, 1, 4], &device),
            false,
        );
        assert!(head.forward(&client, &x).is_err());
    }

    #[test]
    fn rejects_odd_n_fft() {
        let (_client, device) = cpu_setup();
        assert!(
            IstftHead::new(
                IstftHeadWeights {
                    linear: Linear::new(
                        Tensor::<CpuRuntime>::from_slice(&[0.0f32; 4], &[2, 2], &device),
                        None,
                        false,
                    ),
                },
                3,
                1e2,
            )
            .is_err()
        );
    }
}
