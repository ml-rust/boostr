//! Kokoro-faithful magnitude / phase head.
//!
//! Upstream `decoder.generator.conv_post` is a single weight-normed
//! `Conv1d(128, 22, k=7)` whose 22 output channels split as:
//!
//! * first `n_fft/2 + 1 = 11` channels → `exp()` → magnitude
//! * last  `n_fft/2 + 1 = 11` channels → `sin()` → phase
//!
//! Not two separate heads, not a `softplus` on magnitude. This module holds
//! the single conv and does the split + activation at forward time, returning
//! `(mag, phase)` ready for [`crate::model::audio::kokoro::istft`].

use crate::error::{Error, Result};
use crate::nn::Conv1d;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ConvOps, TensorOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

pub struct MagPhaseHead<R: Runtime> {
    conv_post: Conv1d<R>,
    /// `n_fft / 2 + 1`. Sum with itself = conv output channels.
    n_freq_bins: usize,
}

impl<R: Runtime> MagPhaseHead<R> {
    pub fn new(conv_post: Conv1d<R>, n_fft: usize) -> Result<Self> {
        if n_fft == 0 {
            return Err(Error::InvalidArgument {
                arg: "n_fft",
                reason: "must be > 0".into(),
            });
        }
        Ok(Self {
            conv_post,
            n_freq_bins: n_fft / 2 + 1,
        })
    }

    pub fn n_freq_bins(&self) -> usize {
        self.n_freq_bins
    }

    /// Forward: `x [B, C_last, T]` → `(mag [B, F, T], phase [B, F, T])` where
    /// `F = n_fft/2 + 1`.
    #[allow(clippy::type_complexity)]
    pub fn forward<C>(&self, client: &C, x: &Tensor<R>) -> Result<(Tensor<R>, Tensor<R>)>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R> + ConvOps<R> + UnaryOps<R> + BinaryOps<R> + TensorOps<R>,
    {
        let combined = self.conv_post.forward_inference(client, x)?;
        let shape = combined.shape();
        let expected_channels = 2 * self.n_freq_bins;
        if shape.len() != 3 || shape[1] != expected_channels {
            return Err(Error::InvalidArgument {
                arg: "conv_post output",
                reason: format!("expected [B, {expected_channels}, T], got {shape:?}"),
            });
        }
        let mag_log = combined
            .narrow(1, 0, self.n_freq_bins)
            .map_err(Error::Numr)?
            .contiguous()?;
        let phase_raw = combined
            .narrow(1, self.n_freq_bins, self.n_freq_bins)
            .map_err(Error::Numr)?
            .contiguous()?;
        let mag = client.exp(&mag_log).map_err(Error::Numr)?;
        let phase = client.sin(&phase_raw).map_err(Error::Numr)?;
        Ok((mag, phase))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::ops::PaddingMode;
    use numr::runtime::cpu::CpuRuntime;

    fn zeros(shape: &[usize], device: &<CpuRuntime as Runtime>::Device) -> Tensor<CpuRuntime> {
        let n: usize = shape.iter().product();
        Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; n], shape, device)
    }

    fn conv_post(
        c_in: usize,
        n_fft: usize,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> Conv1d<CpuRuntime> {
        let c_out = 2 * (n_fft / 2 + 1);
        Conv1d::new(
            zeros(&[c_out, c_in, 7], device),
            Some(zeros(&[c_out], device)),
            1,
            PaddingMode::Same,
            1,
            1,
            false,
        )
    }

    #[test]
    fn forward_returns_mag_and_phase_of_correct_shape() {
        let (client, device) = cpu_setup();
        let n_fft = 20;
        let head = MagPhaseHead::new(conv_post(128, n_fft, &device), n_fft).unwrap();
        let x = zeros(&[1, 128, 8], &device);
        let (mag, phase) = head.forward(&client, &x).unwrap();
        assert_eq!(mag.shape(), &[1, 11, 8]);
        assert_eq!(phase.shape(), &[1, 11, 8]);
    }

    #[test]
    fn zero_conv_yields_mag_ones_phase_zero() {
        // conv_post is all-zero (weights + bias). Output = 0 everywhere.
        // exp(0) = 1 (mag), sin(0) = 0 (phase).
        let (client, device) = cpu_setup();
        let n_fft = 4;
        let head = MagPhaseHead::new(conv_post(8, n_fft, &device), n_fft).unwrap();
        let x = zeros(&[1, 8, 3], &device);
        let (mag, phase) = head.forward(&client, &x).unwrap();
        for v in mag.to_vec::<f32>() {
            assert!((v - 1.0).abs() < 1e-5, "mag should be 1, got {v}");
        }
        for v in phase.to_vec::<f32>() {
            assert!(v.abs() < 1e-5, "phase should be 0, got {v}");
        }
    }

    #[test]
    fn rejects_zero_n_fft() {
        let (_client, device) = cpu_setup();
        assert!(MagPhaseHead::new(conv_post(8, 4, &device), 0).is_err());
    }
}
