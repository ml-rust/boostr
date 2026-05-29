//! Kokoro-faithful `AdaIN1d` module + `AdaINResBlock1`.
//!
//! Distinct from the generic `boostr::nn::AdaIn1d` utility: this variant
//! carries its own checkpoint weights (Linear `fc` + affine InstanceNorm1d)
//! and applies the residual-style formula
//!
//! ```text
//!     y = (1 + γ) · norm(x) + β       where (γ, β) = split(fc(style))
//! ```
//!
//! exactly matching `StyleTTS2/modules.AdaIN1d`. Kokoro's ISTFTNet resblocks
//! use this module; other callers can pick the free-form utility in
//! `boostr::nn::adain`.
//!
//! `AdaINResBlock1` wraps three `(AdaIN1d → Snake → Conv1d) × 2` sub-units
//! with residual skip, as used throughout `decoder.generator.resblocks.*`.

use crate::error::{Error, Result};
use crate::model::audio::kokoro::snake;
use crate::nn::Conv1d;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, ConvOps, MatmulOps, NormalizationOps, ScalarOps, TensorOps, UnaryOps,
    UtilityOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Kokoro-style AdaIN1d.
///
/// Weights loaded from checkpoint keys:
/// * `{prefix}.fc.weight` / `{prefix}.fc.bias` — `Linear(style_dim, 2·C)`
/// * `{prefix}.norm.weight` / `{prefix}.norm.bias` — `InstanceNorm1d(affine=True)` `[C]`
pub struct AdaIn1d<R: Runtime> {
    fc_weight: Tensor<R>,   // [2*channels, style_dim]
    fc_bias: Tensor<R>,     // [2*channels]
    norm_weight: Tensor<R>, // [channels]
    norm_bias: Tensor<R>,   // [channels]
    channels: usize,
    style_dim: usize,
    eps: f32,
}

impl<R: Runtime> AdaIn1d<R> {
    pub fn new(
        fc_weight: Tensor<R>,
        fc_bias: Tensor<R>,
        norm_weight: Tensor<R>,
        norm_bias: Tensor<R>,
        eps: f32,
    ) -> Result<Self> {
        let fc_shape = fc_weight.shape();
        if fc_shape.len() != 2 || fc_shape[0] % 2 != 0 {
            return Err(Error::InvalidArgument {
                arg: "fc_weight",
                reason: format!("expected [2·C, style_dim], got {fc_shape:?}"),
            });
        }
        let channels = fc_shape[0] / 2;
        let style_dim = fc_shape[1];
        if fc_bias.shape() != [2 * channels] {
            return Err(Error::InvalidArgument {
                arg: "fc_bias",
                reason: format!("expected [{}], got {:?}", 2 * channels, fc_bias.shape()),
            });
        }
        if norm_weight.shape() != [channels] {
            return Err(Error::InvalidArgument {
                arg: "norm_weight",
                reason: format!("expected [{channels}], got {:?}", norm_weight.shape()),
            });
        }
        if norm_bias.shape() != [channels] {
            return Err(Error::InvalidArgument {
                arg: "norm_bias",
                reason: format!("expected [{channels}], got {:?}", norm_bias.shape()),
            });
        }
        Ok(Self {
            fc_weight,
            fc_bias,
            norm_weight,
            norm_bias,
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
            + TensorOps<R>,
    {
        let x_shape = x.shape();
        if x_shape.len() != 3 || x_shape[1] != self.channels {
            return Err(Error::InvalidArgument {
                arg: "x",
                reason: format!("expected [B, {}, T], got {x_shape:?}", self.channels),
            });
        }
        let b = x_shape[0];
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

        // Instance-norm with checkpoint affine (num_groups = channels).
        let normalized = client
            .group_norm(
                x,
                &self.norm_weight,
                &self.norm_bias,
                self.channels,
                self.eps,
            )
            .map_err(Error::Numr)?;

        // Style projection: h = fc(style) = style @ fc_weight^T + fc_bias → [B, 2·C].
        let fc_w_t = self.fc_weight.transpose(0, 1).map_err(Error::Numr)?;
        let h = client
            .matmul_bias(style, &fc_w_t, &self.fc_bias)
            .map_err(Error::Numr)?;

        // Split first half → gamma, second half → beta. Reshape to [B, C, 1].
        let gamma_flat = h.narrow(1, 0, self.channels).map_err(Error::Numr)?;
        let beta_flat = h
            .narrow(1, self.channels, self.channels)
            .map_err(Error::Numr)?;
        let gamma = gamma_flat
            .reshape(&[b, self.channels, 1])
            .map_err(Error::Numr)?;
        let beta = beta_flat
            .reshape(&[b, self.channels, 1])
            .map_err(Error::Numr)?;

        // (1 + gamma) * norm + beta.
        let gamma_plus_one = client.add_scalar(&gamma, 1.0).map_err(Error::Numr)?;
        let scaled = client
            .mul(&normalized, &gamma_plus_one)
            .map_err(Error::Numr)?;
        client.add(&scaled, &beta).map_err(Error::Numr)
    }
}

/// Three tiers of (AdaIN → Snake → Conv1d) × 2 with residual skip — one
/// `generator.resblocks.{i}` per instance.
pub struct AdaINResBlock1<R: Runtime> {
    convs1: [Conv1d<R>; 3],
    convs2: [Conv1d<R>; 3],
    adain1: [AdaIn1d<R>; 3],
    adain2: [AdaIn1d<R>; 3],
    /// `alpha1.{0,1,2}` — per-tier Snake parameter, shape `[1, C, 1]`.
    alpha1: [Tensor<R>; 3],
    /// `alpha2.{0,1,2}`.
    alpha2: [Tensor<R>; 3],
    /// ε floor for Snake's `1/(α+ε)` term. Upstream uses `1e-9`.
    snake_eps: f64,
}

impl<R: Runtime> AdaINResBlock1<R> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        convs1: [Conv1d<R>; 3],
        convs2: [Conv1d<R>; 3],
        adain1: [AdaIn1d<R>; 3],
        adain2: [AdaIn1d<R>; 3],
        alpha1: [Tensor<R>; 3],
        alpha2: [Tensor<R>; 3],
        snake_eps: f64,
    ) -> Result<Self> {
        // All three tiers must share a common channel count; enforce it so
        // shape errors surface at construction rather than deep in a forward.
        let channels = adain1[0].channels();
        for a in adain1.iter().chain(adain2.iter()) {
            if a.channels() != channels {
                return Err(Error::InvalidArgument {
                    arg: "adain",
                    reason: format!(
                        "all AdaIn1d sites must share channel count, got {} vs {channels}",
                        a.channels()
                    ),
                });
            }
        }
        for (i, a) in alpha1.iter().chain(alpha2.iter()).enumerate() {
            if a.shape() != [1, channels, 1] {
                return Err(Error::InvalidArgument {
                    arg: "alpha",
                    reason: format!(
                        "alpha tensor {i} expected [1, {channels}, 1], got {:?}",
                        a.shape()
                    ),
                });
            }
        }
        Ok(Self {
            convs1,
            convs2,
            adain1,
            adain2,
            alpha1,
            alpha2,
            snake_eps,
        })
    }

    pub fn channels(&self) -> usize {
        self.adain1[0].channels()
    }

    /// Forward: `x [B, C, T]`, `style [B, style_dim]` → `[B, C, T]`.
    pub fn forward<C>(&self, client: &C, x: &Tensor<R>, style: &Tensor<R>) -> Result<Tensor<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R>
            + ConvOps<R>
            + NormalizationOps<R>
            + ActivationOps<R>
            + TensorOps<R>
            + MatmulOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + ScalarOps<R>
            + UtilityOps<R>,
    {
        let mut acc = x.clone();
        for i in 0..3 {
            let mut h = self.adain1[i].forward(client, &acc, style)?;
            h = snake::snake(client, &h, &self.alpha1[i], self.snake_eps)?;
            h = self.convs1[i].forward_inference(client, &h)?;
            h = self.adain2[i].forward(client, &h, style)?;
            h = snake::snake(client, &h, &self.alpha2[i], self.snake_eps)?;
            h = self.convs2[i].forward_inference(client, &h)?;
            acc = client.add(&h, &acc).map_err(Error::Numr)?;
        }
        Ok(acc)
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

    fn ones(shape: &[usize], device: &<CpuRuntime as Runtime>::Device) -> Tensor<CpuRuntime> {
        let n: usize = shape.iter().product();
        Tensor::<CpuRuntime>::from_slice(&vec![1.0f32; n], shape, device)
    }

    fn build_adain(
        channels: usize,
        style_dim: usize,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> AdaIn1d<CpuRuntime> {
        AdaIn1d::new(
            zeros(&[2 * channels, style_dim], device),
            zeros(&[2 * channels], device),
            ones(&[channels], device),
            zeros(&[channels], device),
            1e-5,
        )
        .unwrap()
    }

    #[test]
    fn adain_zero_fc_is_plain_instance_norm() {
        // With fc_weight=0 and fc_bias=0: gamma=0, beta=0 → output = (1+0)*norm(x) + 0 = norm(x).
        // For a constant-per-(B,C)-slice input, norm has mean 0, so output mean ~0.
        let (client, device) = cpu_setup();
        let adain = build_adain(2, 3, &device);
        let data: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let x = Tensor::<CpuRuntime>::from_slice(&data, &[1, 2, 5], &device);
        let style = zeros(&[1, 3], &device);
        let out = adain.forward(&client, &x, &style).unwrap();
        let flat: Vec<f32> = out.to_vec();
        let mean_c0: f32 = flat[0..5].iter().sum::<f32>() / 5.0;
        let mean_c1: f32 = flat[5..10].iter().sum::<f32>() / 5.0;
        assert!(mean_c0.abs() < 1e-4);
        assert!(mean_c1.abs() < 1e-4);
    }

    #[test]
    fn adain_rejects_wrong_input_rank() {
        let (client, device) = cpu_setup();
        let adain = build_adain(2, 3, &device);
        let x = zeros(&[2, 2], &device);
        let style = zeros(&[2, 3], &device);
        assert!(adain.forward(&client, &x, &style).is_err());
    }

    #[test]
    fn adain_rejects_wrong_fc_shape() {
        let (_client, device) = cpu_setup();
        // fc_weight first dim must be even.
        let bad = AdaIn1d::new(
            zeros(&[5, 3], &device),
            zeros(&[5], &device),
            ones(&[2], &device),
            zeros(&[2], &device),
            1e-5,
        );
        assert!(bad.is_err());
    }

    fn build_resblock(
        channels: usize,
        style_dim: usize,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> AdaINResBlock1<CpuRuntime> {
        let conv = || {
            Conv1d::new(
                zeros(&[channels, channels, 3], device),
                Some(zeros(&[channels], device)),
                1,
                PaddingMode::Same,
                1,
                1,
                false,
            )
        };
        let alpha = || ones(&[1, channels, 1], device);
        AdaINResBlock1::new(
            [conv(), conv(), conv()],
            [conv(), conv(), conv()],
            [
                build_adain(channels, style_dim, device),
                build_adain(channels, style_dim, device),
                build_adain(channels, style_dim, device),
            ],
            [
                build_adain(channels, style_dim, device),
                build_adain(channels, style_dim, device),
                build_adain(channels, style_dim, device),
            ],
            [alpha(), alpha(), alpha()],
            [alpha(), alpha(), alpha()],
            1e-9,
        )
        .unwrap()
    }

    #[test]
    fn resblock_preserves_shape() {
        let (client, device) = cpu_setup();
        let block = build_resblock(4, 3, &device);
        let x = zeros(&[1, 4, 8], &device);
        let style = zeros(&[1, 3], &device);
        let y = block.forward(&client, &x, &style).unwrap();
        assert_eq!(y.shape(), &[1, 4, 8]);
    }

    #[test]
    fn resblock_zero_everything_gives_zero_output() {
        // With all-zero inputs, AdaIN output is 0 (norm of zero is zero), Snake of 0 = 0,
        // Conv1d of 0 with zero weights/bias = 0. Residual sum of x + 0 stays 0.
        let (client, device) = cpu_setup();
        let block = build_resblock(2, 2, &device);
        let x = zeros(&[1, 2, 4], &device);
        let style = zeros(&[1, 2], &device);
        let y = block.forward(&client, &x, &style).unwrap();
        for v in y.to_vec::<f32>() {
            assert!(v.abs() < 1e-6);
        }
    }

    #[test]
    fn resblock_rejects_channel_mismatch_between_adains() {
        let (_client, device) = cpu_setup();
        let conv = || {
            Conv1d::new(
                zeros(&[4, 4, 3], &device),
                Some(zeros(&[4], &device)),
                1,
                PaddingMode::Same,
                1,
                1,
                false,
            )
        };
        let alpha = || ones(&[1, 4, 1], &device);
        let res = AdaINResBlock1::new(
            [conv(), conv(), conv()],
            [conv(), conv(), conv()],
            [
                build_adain(4, 2, &device),
                build_adain(5, 2, &device), // mismatched channel count
                build_adain(4, 2, &device),
            ],
            [
                build_adain(4, 2, &device),
                build_adain(4, 2, &device),
                build_adain(4, 2, &device),
            ],
            [alpha(), alpha(), alpha()],
            [alpha(), alpha(), alpha()],
            1e-9,
        );
        assert!(res.is_err());
    }
}
