//! [`RrdbNet`]: BasicSR's RRDBNet at ×2, the network of `RealESRGAN_x2plus`
//! and `2x_PureVision`.
//!
//! The forward pass is:
//!
//! 1. Replicate-pad the bottom and right edges to an even height and width.
//! 2. `pixel_unshuffle(x, 2)`, then `conv_first`.
//! 3. `f + conv_body(body(f))`, where the body is `num_block` RRDBs.
//! 4. Twice: nearest ×2 upsample, then `conv_up{1,2}` with LeakyReLU.
//! 5. `conv_hr` with LeakyReLU, then `conv_last`.
//! 6. Crop to twice the unpadded height and width.

use crate::error::{Error, Result};
use crate::model::super_resolution::client::SrClient;
use crate::model::super_resolution::conv::SrConv;
use crate::model::super_resolution::layout::SrExecution;
use crate::model::super_resolution::rrdb::config::{RRDB_UNSHUFFLE_FACTOR, RrdbConfig};
use crate::model::super_resolution::rrdb::dense_block::{KERNEL, LEAKY_SLOPE};
use crate::model::super_resolution::rrdb::rrdb_block::RrdbBlock;
use crate::nn::VarBuilder;
use numr::dtype::DType;
use numr::ops::{ConvActivation, ConvLayout, ConvResidual, NearestSize, PadMode};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Factor of each of the two nearest upsamples.
const UPSAMPLE_FACTOR: f64 = 2.0;

/// RRDBNet at ×2, inference only.
///
/// Every conv runs fused with its bias and LeakyReLU. Input and output
/// follow the [`SrExecution`] layout and dtype the network loads with.
pub struct RrdbNet<R: Runtime> {
    conv_first: SrConv<R>,
    body: Vec<RrdbBlock<R>>,
    conv_body: SrConv<R>,
    conv_up1: SrConv<R>,
    conv_up2: SrConv<R>,
    conv_hr: SrConv<R>,
    conv_last: SrConv<R>,
    config: RrdbConfig,
    exec: SrExecution,
}

impl<R: Runtime<DType = DType>> RrdbNet<R> {
    /// Take every key of [`RrdbConfig::expected_keys`] under `vb`'s prefix
    /// and check each shape.
    ///
    /// `vb` cannot list leftover keys. [`crate::model::super_resolution::load_rrdb`]
    /// adds the unexpected-key check and the legacy ESRGAN key remap.
    ///
    /// # Errors
    ///
    /// An invalid or unsupported config, a missing key, or a shape mismatch.
    /// Each key error names the key and both shapes.
    pub fn from_varbuilder<C: SrClient<R>>(
        client: &C,
        vb: &mut VarBuilder<'_, R>,
        config: &RrdbConfig,
        exec: SrExecution,
    ) -> Result<Self> {
        config.validate()?;
        let feat = config.num_feat;
        let mut conv = |name: &str, in_channels: usize, out_channels: usize| {
            SrConv::from_varbuilder(
                client,
                &mut *vb,
                name,
                None,
                in_channels,
                out_channels,
                KERNEL,
                &exec,
            )
        };
        let conv_first = conv("conv_first", config.first_conv_in_channels(), feat)?;
        let conv_body = conv("conv_body", feat, feat)?;
        let conv_up1 = conv("conv_up1", feat, feat)?;
        let conv_up2 = conv("conv_up2", feat, feat)?;
        let conv_hr = conv("conv_hr", feat, feat)?;
        let conv_last = conv("conv_last", feat, config.num_out_ch)?;
        let body = (0..config.num_block)
            .map(|b| {
                RrdbBlock::from_varbuilder(client, &mut vb.pp(&format!("body.{b}")), config, &exec)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            conv_first,
            body,
            conv_body,
            conv_up1,
            conv_up2,
            conv_hr,
            conv_last,
            config: *config,
            exec,
        })
    }

    /// Upscale `x` by 2, the reference `forward`.
    ///
    /// `x` is `[N, num_in_ch, H, W]` in `Nchw`, `[N, H, W, num_in_ch]` in
    /// `Nhwc`, with the execution dtype. Any `H` and `W` of at
    /// least 1 works. The output is `num_out_ch × 2H × 2W` in the same layout and
    /// dtype. The body runs in two feature buffers that the call allocates
    /// once.
    ///
    /// # Errors
    ///
    /// A wrong rank, channel count or dtype, or a conv or image op error.
    /// Each error names the input shape and dtype.
    pub fn forward<C: SrClient<R>>(&self, client: &C, x: &Tensor<R>) -> Result<Tensor<R>> {
        let (n, _, h, w) = self.check_input(x)?;
        let layout = self.exec.layout;
        let image_error = |op: &str, e: numr::error::Error| Error::ModelError {
            reason: format!("RRDBNet {op} on input {:?} {:?}: {e}", x.shape(), x.dtype()),
        };

        let (pad_h, pad_w) = (h % RRDB_UNSHUFFLE_FACTOR, w % RRDB_UNSHUFFLE_FACTOR);
        let padded = if pad_h == 0 && pad_w == 0 {
            x.clone()
        } else {
            client
                .pad_mode(x, &edge_padding(layout, pad_h, pad_w), PadMode::Replicate)
                .map_err(|e| image_error("replicate pad", e))?
        };
        let folded = client
            .pixel_unshuffle(&padded, RRDB_UNSHUFFLE_FACTOR, layout)
            .map_err(|e| image_error("pixel_unshuffle", e))?;

        let trunk = self
            .conv_first
            .forward(client, &folded, ConvActivation::None, None)?;
        let feat_h = (h + pad_h) / RRDB_UNSHUFFLE_FACTOR;
        let feat_w = (w + pad_w) / RRDB_UNSHUFFLE_FACTOR;
        let buffer_shape = self
            .exec
            .image_shape(n, self.config.dense_channels(), feat_h, feat_w);
        let buffers = [
            Tensor::<R>::empty(&buffer_shape, self.exec.dtype, x.device())?,
            Tensor::<R>::empty(&buffer_shape, self.exec.dtype, x.device())?,
        ];
        let mut carry = trunk.clone();
        for block in &self.body {
            carry = block.forward(client, &carry, &buffers)?;
        }
        let mut features = self.conv_body.forward(
            client,
            &carry,
            ConvActivation::None,
            Some(ConvResidual {
                tensor: &trunk,
                scale: 1.0,
            }),
        )?;

        for conv in [&self.conv_up1, &self.conv_up2] {
            let enlarged = client
                .upsample_nearest2d(
                    &features,
                    NearestSize::Scale(UPSAMPLE_FACTOR, UPSAMPLE_FACTOR),
                    layout,
                )
                .map_err(|e| image_error("upsample_nearest2d", e))?;
            features = conv.forward(
                client,
                &enlarged,
                ConvActivation::LeakyRelu(LEAKY_SLOPE),
                None,
            )?;
        }
        let features = self.conv_hr.forward(
            client,
            &features,
            ConvActivation::LeakyRelu(LEAKY_SLOPE),
            None,
        )?;
        let image = self
            .conv_last
            .forward(client, &features, ConvActivation::None, None)?;
        if pad_h == 0 && pad_w == 0 {
            return Ok(image);
        }
        let (h_dim, w_dim) = spatial_dims(layout);
        Ok(image
            .narrow(h_dim, 0, 2 * h)?
            .narrow(w_dim, 0, 2 * w)?
            .contiguous()?)
    }

    /// The network shape.
    pub fn config(&self) -> &RrdbConfig {
        &self.config
    }

    /// The dtype, layout and accumulation the network runs with.
    pub fn execution(&self) -> &SrExecution {
        &self.exec
    }

    /// The RRDB blocks of the body, in `body.N` order.
    pub fn body(&self) -> &[RrdbBlock<R>] {
        &self.body
    }

    /// The convs outside the body in forward order: `conv_first`,
    /// `conv_body`, `conv_up1`, `conv_up2`, `conv_hr`, `conv_last`.
    pub fn trunk_convs(&self) -> [&SrConv<R>; 6] {
        [
            &self.conv_first,
            &self.conv_body,
            &self.conv_up1,
            &self.conv_up2,
            &self.conv_hr,
            &self.conv_last,
        ]
    }

    fn check_input(&self, x: &Tensor<R>) -> Result<(usize, usize, usize, usize)> {
        let dims = self.exec.image_dims(x.shape())?;
        if dims.1 != self.config.num_in_ch {
            return Err(Error::InvalidArgument {
                arg: "input",
                reason: format!(
                    "RRDBNet takes {} input channels, got shape {:?} in {:?} (channel dim {}); pass an image in the load layout",
                    self.config.num_in_ch,
                    x.shape(),
                    self.exec.layout,
                    self.exec.channel_dim()
                ),
            });
        }
        if dims.2 == 0 || dims.3 == 0 {
            return Err(Error::InvalidArgument {
                arg: "input",
                reason: format!(
                    "RRDBNet input {:?} has an empty height or width; pass an image of at least 1×1",
                    x.shape()
                ),
            });
        }
        if x.dtype() != self.exec.dtype {
            return Err(Error::InvalidArgument {
                arg: "input",
                reason: format!(
                    "RRDBNet loaded as {:?}, got a {:?} input; cast the input to {:?}",
                    self.exec.dtype,
                    x.dtype(),
                    self.exec.dtype
                ),
            });
        }
        Ok(dims)
    }
}

/// `ShapeOps::pad_mode` pairs that add `pad_h` rows at the bottom and
/// `pad_w` columns at the right. The pairs count from the last dimension.
fn edge_padding(layout: ConvLayout, pad_h: usize, pad_w: usize) -> Vec<usize> {
    match layout {
        ConvLayout::Nchw => vec![0, pad_w, 0, pad_h],
        ConvLayout::Nhwc => vec![0, 0, 0, pad_w, 0, pad_h],
    }
}

/// The height and width dimensions of a 4-D image tensor in `layout`.
fn spatial_dims(layout: ConvLayout) -> (isize, isize) {
    match layout {
        ConvLayout::Nchw => (2, 3),
        ConvLayout::Nhwc => (1, 2),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padding_counts_from_the_last_dim() {
        assert_eq!(edge_padding(ConvLayout::Nchw, 1, 0), vec![0, 0, 0, 1]);
        assert_eq!(edge_padding(ConvLayout::Nchw, 0, 1), vec![0, 1, 0, 0]);
        assert_eq!(edge_padding(ConvLayout::Nhwc, 1, 1), vec![0, 0, 0, 1, 0, 1]);
    }

    #[test]
    fn spatial_dims_follow_layout() {
        assert_eq!(spatial_dims(ConvLayout::Nchw), (2, 3));
        assert_eq!(spatial_dims(ConvLayout::Nhwc), (1, 2));
    }
}
