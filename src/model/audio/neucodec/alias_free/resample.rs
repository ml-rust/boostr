//! `UpSample1d` / `DownSample1d` (Kaiser-filtered ×2 resamplers) and
//! `Activation1d`, which composes them around [`super::snake::SnakeBeta`].

use super::filter::kaiser_sinc_filter1d;
use crate::error::{Error, Result};
use crate::model::audio::neucodec::client::NeuCodecClient;
use crate::nn::var_contiguous;
use numr::autograd::{Var, var_mul_scalar, var_narrow};
use numr::dtype::DType;
use numr::ops::PaddingMode;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::filter::replicate_pad_1d;
use super::snake::SnakeBeta;

/// Resampling ratio used by every `Activation1d` in this encoder.
pub const RESAMPLE_RATIO: usize = 2;
/// Filter length used by both the up- and down-sampler.
pub const RESAMPLE_KERNEL_SIZE: usize = 12;

/// ×2 upsampler: replicate-pad, grouped transposed convolution with the Kaiser
/// filter, scale by the ratio, then crop.
pub struct UpSample1d<R: Runtime> {
    filter: Tensor<R>,
    ratio: usize,
    pad: usize,
    pad_left: usize,
    pad_right: usize,
}

impl<R: Runtime<DType = DType>> UpSample1d<R> {
    pub fn new(ratio: usize, kernel_size: usize, device: &R::Device) -> Result<Self> {
        if ratio == 0 || kernel_size < ratio {
            return Err(Error::InvalidArgument {
                arg: "ratio/kernel_size",
                reason: format!("ratio must be > 0 and <= kernel_size, got {ratio}/{kernel_size}"),
            });
        }
        let taps = kaiser_sinc_filter1d(0.5 / ratio as f64, 0.6 / ratio as f64, kernel_size);
        let pad = kernel_size / ratio - 1;
        Ok(Self {
            filter: Tensor::from_slice(&taps, &[1, 1, kernel_size], device)?,
            ratio,
            pad,
            pad_left: pad * ratio + (kernel_size - ratio) / 2,
            pad_right: pad * ratio + (kernel_size - ratio).div_ceil(2),
        })
    }

    /// `x [B, C, T] -> [B, C, T * ratio]`.
    pub fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        C: NeuCodecClient<R>,
        R::Client: NeuCodecClient<R>,
    {
        let channels = x.shape()[1];
        let padded = replicate_pad_1d(client, x, self.pad, self.pad)?;

        // Depthwise: one filter per channel, groups = C.
        let weight = self
            .filter
            .broadcast_to(&[channels, 1, self.filter.shape()[2]])
            .map_err(Error::Numr)?
            .contiguous()?;
        let weight = Var::new(weight, false);

        let up = numr::autograd::var_conv_transpose1d(
            &padded,
            &weight,
            None,
            self.ratio,
            PaddingMode::Valid,
            0,
            1,
            channels,
            client,
        )
        .map_err(Error::Numr)?;
        let up = var_mul_scalar(&up, self.ratio as f64, client).map_err(Error::Numr)?;

        let total = up.shape()[2];
        if total <= self.pad_left + self.pad_right {
            return Err(Error::InvalidArgument {
                arg: "x",
                reason: format!(
                    "upsampled length {total} is too short to crop {}+{}",
                    self.pad_left, self.pad_right
                ),
            });
        }
        let keep = total - self.pad_left - self.pad_right;
        let out = var_narrow(&up, 2, self.pad_left, keep).map_err(Error::Numr)?;
        var_contiguous(&out)
    }
}

/// ×2 downsampler: replicate-pad, then a strided grouped convolution with the
/// same Kaiser filter.
pub struct DownSample1d<R: Runtime> {
    filter: Tensor<R>,
    ratio: usize,
    pad_left: usize,
    pad_right: usize,
}

impl<R: Runtime<DType = DType>> DownSample1d<R> {
    pub fn new(ratio: usize, kernel_size: usize, device: &R::Device) -> Result<Self> {
        if ratio == 0 {
            return Err(Error::InvalidArgument {
                arg: "ratio",
                reason: "must be > 0".into(),
            });
        }
        let taps = kaiser_sinc_filter1d(0.5 / ratio as f64, 0.6 / ratio as f64, kernel_size);
        let even = kernel_size.is_multiple_of(2);
        Ok(Self {
            filter: Tensor::from_slice(&taps, &[1, 1, kernel_size], device)?,
            ratio,
            pad_left: kernel_size / 2 - usize::from(even),
            pad_right: kernel_size / 2,
        })
    }

    /// `x [B, C, T] -> [B, C, ceil(T / ratio)]`.
    pub fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        C: NeuCodecClient<R>,
        R::Client: NeuCodecClient<R>,
    {
        let channels = x.shape()[1];
        let padded = replicate_pad_1d(client, x, self.pad_left, self.pad_right)?;

        let weight = self
            .filter
            .broadcast_to(&[channels, 1, self.filter.shape()[2]])
            .map_err(Error::Numr)?
            .contiguous()?;
        let weight = Var::new(weight, false);

        let out = numr::autograd::var_conv1d(
            &padded,
            &weight,
            None,
            self.ratio,
            PaddingMode::Valid,
            1,
            channels,
            client,
        )
        .map_err(Error::Numr)?;
        var_contiguous(&out)
    }
}

/// Anti-aliased nonlinearity: upsample -> SnakeBeta -> downsample.
pub struct Activation1d<R: Runtime> {
    up: UpSample1d<R>,
    act: SnakeBeta<R>,
    down: DownSample1d<R>,
}

impl<R: Runtime<DType = DType>> Activation1d<R> {
    pub fn new(act: SnakeBeta<R>, device: &R::Device) -> Result<Self> {
        Ok(Self {
            up: UpSample1d::new(RESAMPLE_RATIO, RESAMPLE_KERNEL_SIZE, device)?,
            act,
            down: DownSample1d::new(RESAMPLE_RATIO, RESAMPLE_KERNEL_SIZE, device)?,
        })
    }

    pub fn activation(&self) -> &SnakeBeta<R> {
        &self.act
    }

    /// The upsample stage alone.
    ///
    /// Exposed so parity tests can localize a mismatch to the resampler rather
    /// than the activation — the two compose, so a single end-to-end number
    /// cannot say which half is wrong.
    pub fn upsample_for_test<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        C: NeuCodecClient<R>,
        R::Client: NeuCodecClient<R>,
    {
        self.up.forward(client, x)
    }

    /// `x [B, C, T] -> [B, C, T]` (length preserved).
    pub fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        C: NeuCodecClient<R>,
        R::Client: NeuCodecClient<R>,
    {
        let up = self.up.forward(client, x)?;
        let act = self.act.forward(client, &up)?;
        self.down.forward(client, &act)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    fn var(
        data: &[f32],
        shape: &[usize],
        device: &<CpuRuntime as Runtime>::Device,
    ) -> Var<CpuRuntime> {
        Var::new(
            Tensor::<CpuRuntime>::from_slice(data, shape, device).unwrap(),
            false,
        )
    }

    #[test]
    fn activation1d_preserves_length() {
        let (client, device) = cpu_setup();
        let c = 3;
        let t = 16;
        let alpha = Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; c], &[c], &device).unwrap();
        let beta = Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; c], &[c], &device).unwrap();
        let act = Activation1d::new(SnakeBeta::new(alpha, beta, false).unwrap(), &device).unwrap();
        let data: Vec<f32> = (0..(c * t)).map(|i| (i as f32 * 0.3).sin()).collect();
        let x = var(&data, &[1, c, t], &device);
        let out = act.forward(&client, &x).unwrap();
        assert_eq!(out.shape(), &[1, c, t]);
        for v in out.tensor().contiguous().unwrap().to_vec::<f32>() {
            assert!(v.is_finite());
        }
    }

    /// Up then down with the same filter should approximately reconstruct a
    /// smooth signal — a sanity check that the crops line up.
    #[test]
    fn upsample_then_downsample_round_trips_a_constant() {
        let (client, device) = cpu_setup();
        let t = 20;
        let up = UpSample1d::<CpuRuntime>::new(2, 12, &device).unwrap();
        let down = DownSample1d::<CpuRuntime>::new(2, 12, &device).unwrap();
        let x = var(&vec![2.5f32; t], &[1, 1, t], &device);
        let u = up.forward(&client, &x).unwrap();
        assert_eq!(u.shape(), &[1, 1, 2 * t]);
        let d = down.forward(&client, &u).unwrap();
        assert_eq!(d.shape(), &[1, 1, t]);
        for v in d.tensor().contiguous().unwrap().to_vec::<f32>() {
            assert!(
                (v - 2.5).abs() < 1e-3,
                "constant signal must survive resampling, got {v}"
            );
        }
    }
}
