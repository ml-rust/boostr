//! `AdaLayerNorm` — style-conditioned layer normalization.
//!
//! Used inside `predictor.text_encoder.lstms.{1, 3, 5}` (interleaved with
//! LSTM layers). Differs from [`crate::model::audio::kokoro::KokoroAdaIn1d`]
//! in two ways:
//!
//! * Uses `LayerNorm` (normalizes over the channel axis for each `(B, T)`
//!   position) instead of `InstanceNorm1d` (normalizes over time for each
//!   `(B, C)`).
//! * The inner LayerNorm has NO learnable affine — `fc(style)` produces the
//!   only scale/shift: `y = (1 + γ) · ln(x) + β`. Checkpoint keys are just
//!   `{prefix}.fc.{weight, bias}`, nothing else.

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::ops::{BinaryOps, MatmulOps, NormalizationOps, ScalarOps, TensorOps, UtilityOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

pub struct AdaLayerNorm<R: Runtime> {
    fc_weight: Tensor<R>, // [2*channels, style_dim]
    fc_bias: Tensor<R>,   // [2*channels]
    channels: usize,
    style_dim: usize,
    eps: f32,
}

impl<R: Runtime> AdaLayerNorm<R> {
    pub fn new(fc_weight: Tensor<R>, fc_bias: Tensor<R>, eps: f32) -> Result<Self> {
        let w = fc_weight.shape();
        if w.len() != 2 || !w[0].is_multiple_of(2) {
            return Err(Error::InvalidArgument {
                arg: "fc_weight",
                reason: format!("expected [2·C, style_dim], got {w:?}"),
            });
        }
        let channels = w[0] / 2;
        let style_dim = w[1];
        if fc_bias.shape() != [2 * channels] {
            return Err(Error::InvalidArgument {
                arg: "fc_bias",
                reason: format!("expected [{}], got {:?}", 2 * channels, fc_bias.shape()),
            });
        }
        Ok(Self {
            fc_weight,
            fc_bias,
            channels,
            style_dim,
            eps,
        })
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn style_dim(&self) -> usize {
        self.style_dim
    }

    /// Forward: `x [B, C, T]`, `style [B, style_dim]` → `[B, C, T]`.
    pub fn forward<C>(&self, client: &C, x: &Tensor<R>, style: &Tensor<R>) -> Result<Tensor<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R>
            + MatmulOps<R>
            + NormalizationOps<R>
            + BinaryOps<R>
            + ScalarOps<R>
            + TensorOps<R>
            + UtilityOps<R>,
    {
        let x_shape = x.shape();
        if x_shape.len() != 3 || x_shape[1] != self.channels {
            return Err(Error::InvalidArgument {
                arg: "x",
                reason: format!("expected [B, {}, T], got {x_shape:?}", self.channels),
            });
        }
        let b = x_shape[0];
        let t = x_shape[2];
        if style.shape() != [b, self.style_dim] {
            return Err(Error::InvalidArgument {
                arg: "style",
                reason: format!(
                    "expected [{b}, {}], got {:?}",
                    self.style_dim,
                    style.shape()
                ),
            });
        }

        // LayerNorm needs the channel axis last: transpose [B,C,T] → [B,T,C],
        // apply with unit weight / zero bias (no learnable affine upstream),
        // then transpose back.
        let x_btc = x.transpose(1, 2).map_err(Error::Numr)?.contiguous()?;
        let ones = client
            .fill(&[self.channels], 1.0, x.dtype())
            .map_err(Error::Numr)?;
        let zeros = client
            .fill(&[self.channels], 0.0, x.dtype())
            .map_err(Error::Numr)?;
        let ln = client
            .layer_norm(&x_btc, &ones, &zeros, self.eps)
            .map_err(Error::Numr)?;
        let ln_bct = ln.transpose(1, 2).map_err(Error::Numr)?.contiguous()?;

        // Style projection → gamma, beta per-sample, broadcast over T.
        let fc_w_t = self.fc_weight.transpose(0, 1).map_err(Error::Numr)?;
        let h = client
            .matmul_bias(style, &fc_w_t, &self.fc_bias)
            .map_err(Error::Numr)?;
        let gamma = h
            .narrow(1, 0, self.channels)
            .map_err(Error::Numr)?
            .reshape(&[b, self.channels, 1])
            .map_err(Error::Numr)?;
        let beta = h
            .narrow(1, self.channels, self.channels)
            .map_err(Error::Numr)?
            .reshape(&[b, self.channels, 1])
            .map_err(Error::Numr)?;
        let gamma_plus_one = client.add_scalar(&gamma, 1.0).map_err(Error::Numr)?;
        let scaled = client.mul(&ln_bct, &gamma_plus_one).map_err(Error::Numr)?;
        let _ = t; // silence unused if T becomes relevant later
        client.add(&scaled, &beta).map_err(Error::Numr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    fn zeros(shape: &[usize], device: &<CpuRuntime as Runtime>::Device) -> Tensor<CpuRuntime> {
        let n: usize = shape.iter().product();
        Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; n], shape, device)
    }

    #[test]
    fn zero_fc_gives_pure_layer_norm() {
        // With fc = 0, gamma = 0 and beta = 0, so output = (1 + 0) * LN(x) + 0 = LN(x).
        // LN over the channel axis makes each (b, t) position mean ≈ 0.
        let (client, device) = cpu_setup();
        let ada = AdaLayerNorm::new(zeros(&[8, 3], &device), zeros(&[8], &device), 1e-5).unwrap();
        // B=1, C=4, T=2. Channel values per-time: time 0 = [1,2,3,4] → mean 2.5; time 1 = [5,6,7,8].
        let x = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0],
            &[1, 4, 2],
            &device,
        );
        let style = zeros(&[1, 3], &device);
        let out = ada.forward(&client, &x, &style).unwrap();
        assert_eq!(out.shape(), &[1, 4, 2]);
        let flat: Vec<f32> = out.to_vec();
        // Each time-slice (4 channels) should sum to ~0.
        let sum_t0 = flat[0] + flat[2] + flat[4] + flat[6];
        let sum_t1 = flat[1] + flat[3] + flat[5] + flat[7];
        assert!(sum_t0.abs() < 1e-4, "t0 sum = {sum_t0}");
        assert!(sum_t1.abs() < 1e-4, "t1 sum = {sum_t1}");
    }

    #[test]
    fn rejects_wrong_input_shape() {
        let (client, device) = cpu_setup();
        let ada = AdaLayerNorm::new(zeros(&[8, 3], &device), zeros(&[8], &device), 1e-5).unwrap();
        let x = zeros(&[2, 4], &device);
        let style = zeros(&[2, 3], &device);
        assert!(ada.forward(&client, &x, &style).is_err());
    }

    #[test]
    fn rejects_odd_output_dim() {
        let (_client, device) = cpu_setup();
        let bad = AdaLayerNorm::new(zeros(&[7, 3], &device), zeros(&[7], &device), 1e-5);
        assert!(bad.is_err());
    }
}
