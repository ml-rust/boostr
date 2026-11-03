//! [`SrVggCompact`]: Real-ESRGAN's SRVGGNetCompact with PReLU, the network
//! of `realesr-general-x4v3`.
//!
//! The body is `conv(in → feat) + PReLU`, then `num_conv` ×
//! `[conv(feat → feat) + PReLU]`, then `conv(feat → out · upscale²)`. Pixel
//! shuffle turns the last conv into the upscaled image. The nearest-upsampled
//! input adds last, so the body learns a residual.

use crate::error::{Error, Result};
use crate::model::super_resolution::client::SrClient;
use crate::model::super_resolution::conv::SrConv;
use crate::model::super_resolution::layout::SrExecution;
use crate::nn::VarBuilder;
use numr::dtype::DType;
use numr::ops::{ConvActivation, NearestSize, ResizeConfig, ResizeFilter};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Kernel size of every SRVGGNetCompact conv.
const KERNEL: usize = 3;

/// Shape of an SRVGGNetCompact network.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SrVggCompactConfig {
    /// Input image channels.
    pub num_in_ch: usize,
    /// Output image channels.
    pub num_out_ch: usize,
    /// Feature channels of the body.
    pub num_feat: usize,
    /// Feature-to-feature convs between the first and the last conv.
    pub num_conv: usize,
    /// Integer upscale factor of [`SrVggCompact::forward_raw`].
    pub upscale: usize,
}

impl SrVggCompactConfig {
    /// The `realesr-general-x4v3` network: RGB in and out, 64 features,
    /// 32 body convs, ×4.
    pub fn realesr_general_x4v3() -> Self {
        Self {
            num_in_ch: 3,
            num_out_ch: 3,
            num_feat: 64,
            num_conv: 32,
            upscale: 4,
        }
    }

    /// Check every field is positive.
    pub fn validate(&self) -> Result<()> {
        let fields = [
            ("num_in_ch", self.num_in_ch),
            ("num_out_ch", self.num_out_ch),
            ("num_feat", self.num_feat),
            ("upscale", self.upscale),
        ];
        match fields.iter().find(|(_, v)| *v == 0) {
            Some((field, _)) => Err(Error::InvalidArgument {
                arg: "config",
                reason: format!("SRVGGNetCompact {field} is 0 in {self:?}; it must be at least 1"),
            }),
            None => Ok(()),
        }
    }

    /// Channel count of the last conv: `num_out_ch · upscale²`.
    pub fn last_conv_channels(&self) -> usize {
        self.num_out_ch * self.upscale * self.upscale
    }

    /// Every state-dict key the network reads, in `body.N` order.
    ///
    /// Conv `i` is `body.{2i}.weight` and `body.{2i}.bias`. The PReLU after
    /// conv `i` is `body.{2i+1}.weight`. The last conv has no PReLU.
    pub fn expected_keys(&self) -> Vec<String> {
        let convs = self.num_conv + 2;
        let mut keys = Vec::with_capacity(3 * convs);
        for i in 0..convs {
            keys.push(format!("body.{}.weight", 2 * i));
            keys.push(format!("body.{}.bias", 2 * i));
            if i + 1 < convs {
                keys.push(format!("body.{}.weight", 2 * i + 1));
            }
        }
        keys
    }
}

impl Default for SrVggCompactConfig {
    fn default() -> Self {
        Self::realesr_general_x4v3()
    }
}

/// SRVGGNetCompact, inference only.
///
/// Every conv runs fused with its bias and PReLU. Input and output follow
/// the [`SrExecution`] layout and dtype the network loads with.
pub struct SrVggCompact<R: Runtime> {
    first: SrConv<R>,
    body: Vec<SrConv<R>>,
    last: SrConv<R>,
    config: SrVggCompactConfig,
    exec: SrExecution,
}

impl<R: Runtime<DType = DType>> SrVggCompact<R> {
    /// Take every key of [`SrVggCompactConfig::expected_keys`] under `vb`'s
    /// prefix and check each shape.
    ///
    /// `vb` cannot list leftover keys. [`super::loader::load_compact`] adds
    /// the unexpected-key check.
    ///
    /// # Errors
    ///
    /// An invalid config, a missing key, or a shape mismatch. Each error
    /// names the key and both shapes.
    pub fn from_varbuilder<C: SrClient<R>>(
        client: &C,
        vb: &mut VarBuilder<'_, R>,
        config: &SrVggCompactConfig,
        exec: SrExecution,
    ) -> Result<Self> {
        config.validate()?;
        let mut body_vb = vb.pp("body");
        let feat = config.num_feat;
        let first = load_conv(client, &mut body_vb, 0, config.num_in_ch, feat, true, &exec)?;
        let body = (1..=config.num_conv)
            .map(|i| load_conv(client, &mut body_vb, i, feat, feat, true, &exec))
            .collect::<Result<Vec<_>>>()?;
        let last = load_conv(
            client,
            &mut body_vb,
            config.num_conv + 1,
            feat,
            config.last_conv_channels(),
            false,
            &exec,
        )?;
        Ok(Self {
            first,
            body,
            last,
            config: *config,
            exec,
        })
    }

    /// `pixel_shuffle(body(x), upscale) + upsample_nearest2d(x, upscale)`,
    /// the reference `forward_raw`.
    ///
    /// `x` is `[N, num_in_ch, H, W]` in `Nchw`, `[N, H, W, num_in_ch]` in
    /// `Nhwc`, with the execution dtype. The output is
    /// `num_out_ch × H·upscale × W·upscale` in the same layout. The body
    /// runs in two reused feature buffers.
    ///
    /// # Errors
    ///
    /// A wrong rank, channel count or dtype, or a conv or image op error.
    /// Each error names the input shape and dtype.
    pub fn forward_raw<C: SrClient<R>>(&self, client: &C, x: &Tensor<R>) -> Result<Tensor<R>> {
        let (n, _, h, w) = self.check_input(x)?;
        let layout = self.exec.layout;
        let feat_shape = self.exec.image_shape(n, self.config.num_feat, h, w);
        let buffers = [
            Tensor::<R>::empty(&feat_shape, self.exec.dtype, x.device())?,
            Tensor::<R>::empty(&feat_shape, self.exec.dtype, x.device())?,
        ];
        self.first
            .forward_into(client, &buffers[0], x, self.first.activation(), None)?;
        let mut current = 0;
        for conv in &self.body {
            let next = 1 - current;
            conv.forward_into(
                client,
                &buffers[next],
                &buffers[current],
                conv.activation(),
                None,
            )?;
            current = next;
        }
        let features = self
            .last
            .forward(client, &buffers[current], ConvActivation::None, None)?;
        let image_error = |op: &str, e: numr::error::Error| Error::ModelError {
            reason: format!(
                "SRVGGNetCompact {op} on input {:?} {:?}: {e}",
                x.shape(),
                x.dtype()
            ),
        };
        let shuffled = client
            .pixel_shuffle(&features, self.config.upscale, layout)
            .map_err(|e| image_error("pixel_shuffle", e))?;
        let factor = self.config.upscale as f64;
        let base = client
            .upsample_nearest2d(x, NearestSize::Scale(factor, factor), layout)
            .map_err(|e| image_error("upsample_nearest2d", e))?;
        client
            .add(&shuffled, &base)
            .map_err(|e| image_error("residual add", e))
    }

    /// [`Self::forward_raw`], then an antialiased bicubic resize to
    /// `size = (height, width)`, the reference `forward`.
    ///
    /// The resize runs in F32, as the reference `.float()` does. A non-F32
    /// output of `forward_raw` goes through `TypeConversionOps::cast` to F32,
    /// resizes, and casts back to the execution dtype with the same op.
    pub fn forward_resized<C: SrClient<R>>(
        &self,
        client: &C,
        x: &Tensor<R>,
        size: (usize, usize),
    ) -> Result<Tensor<R>> {
        let raw = self.forward_raw(client, x)?;
        let config = ResizeConfig::new(size, ResizeFilter::Bicubic).with_antialias(true);
        let resize_error = |e: numr::error::Error| Error::ModelError {
            reason: format!(
                "SRVGGNetCompact bicubic resize of {:?} to {size:?}: {e}",
                raw.shape()
            ),
        };
        if self.exec.dtype == DType::F32 {
            return client
                .resize(&raw, &config, self.exec.layout)
                .map_err(resize_error);
        }
        let wide = client.cast(&raw, DType::F32)?;
        let resized = client
            .resize(&wide, &config, self.exec.layout)
            .map_err(resize_error)?;
        Ok(client.cast(&resized, self.exec.dtype)?)
    }

    /// The network shape.
    pub fn config(&self) -> &SrVggCompactConfig {
        &self.config
    }

    /// The dtype, layout and accumulation the network runs with.
    pub fn execution(&self) -> &SrExecution {
        &self.exec
    }

    /// Every conv in `body.N` order: the first, the body, the last.
    pub fn convs(&self) -> impl Iterator<Item = &SrConv<R>> {
        std::iter::once(&self.first)
            .chain(self.body.iter())
            .chain(std::iter::once(&self.last))
    }

    fn check_input(&self, x: &Tensor<R>) -> Result<(usize, usize, usize, usize)> {
        let dims = self.exec.image_dims(x.shape())?;
        if dims.1 != self.config.num_in_ch {
            return Err(Error::InvalidArgument {
                arg: "input",
                reason: format!(
                    "SRVGGNetCompact takes {} input channels, got shape {:?} in {:?} (channel dim {}); pass an image in the load layout",
                    self.config.num_in_ch,
                    x.shape(),
                    self.exec.layout,
                    self.exec.channel_dim()
                ),
            });
        }
        if x.dtype() != self.exec.dtype {
            return Err(Error::InvalidArgument {
                arg: "input",
                reason: format!(
                    "SRVGGNetCompact loaded as {:?}, got a {:?} input; cast the input to {:?}",
                    self.exec.dtype,
                    x.dtype(),
                    self.exec.dtype
                ),
            });
        }
        Ok(dims)
    }
}

/// Conv `i` of the body: `body.{2i}`, with the PReLU `body.{2i+1}` when
/// `prelu` is set.
fn load_conv<R, C>(
    client: &C,
    vb: &mut VarBuilder<'_, R>,
    i: usize,
    in_channels: usize,
    out_channels: usize,
    prelu: bool,
    exec: &SrExecution,
) -> Result<SrConv<R>>
where
    R: Runtime<DType = DType>,
    C: SrClient<R>,
{
    let prelu_key = (2 * i + 1).to_string();
    SrConv::from_varbuilder(
        client,
        vb,
        &(2 * i).to_string(),
        prelu.then_some(prelu_key.as_str()),
        in_channels,
        out_channels,
        KERNEL,
        exec,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn x4v3_keys_cover_body_0_to_66() {
        let keys = SrVggCompactConfig::realesr_general_x4v3().expected_keys();
        // 34 convs × (weight, bias) + 33 PReLU slopes.
        assert_eq!(keys.len(), 34 * 2 + 33);
        assert_eq!(keys[0], "body.0.weight");
        assert_eq!(keys[2], "body.1.weight");
        assert_eq!(keys[keys.len() - 2], "body.66.weight");
        assert_eq!(keys[keys.len() - 1], "body.66.bias");
        assert!(!keys.iter().any(|k| k == "body.67.weight"));
    }

    #[test]
    fn last_conv_feeds_pixel_shuffle() {
        assert_eq!(SrVggCompactConfig::default().last_conv_channels(), 48);
    }

    #[test]
    fn zero_field_is_rejected() {
        let config = SrVggCompactConfig {
            num_feat: 0,
            ..SrVggCompactConfig::default()
        };
        let err = config.validate().err().map(|e| e.to_string());
        assert!(
            err.as_deref().is_some_and(|e| e.contains("num_feat")),
            "{err:?}"
        );
    }
}
