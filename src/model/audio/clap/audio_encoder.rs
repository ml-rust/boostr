//! [`ClapAudioEncoder`]: the HTSAT tower. The log-mel features are
//! normalized per mel bin, stretched in time to fill the image, folded
//! into a `spec_size` square, cut into patches, run through the Swin
//! stages, normalized and averaged over tokens. Port of transformers'
//! `ClapAudioEncoder` for an unfused checkpoint.

use super::config::{ClapAudioConfig, ClapConfig};
use super::linear::{dense, layer_norm, norm};
use super::resize::resize_bicubic;
use super::stage::SwinStage;
use super::swin_block::BlockShape;
use crate::error::{Error, Result};
use crate::nn::{LayerNorm, Linear, VarBuilder};
use numr::dtype::DType;
use numr::ops::{ActivationOps, BinaryOps, IndexingOps, ScalarOps, ShapeOps, TensorOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// BatchNorm2d's epsilon.
const BATCH_NORM_EPS: f64 = 1e-5;

/// The HTSAT audio tower.
pub struct ClapAudioEncoder<R: Runtime> {
    /// Per mel bin, `weight / sqrt(running_var + eps)`.
    bn_scale: Tensor<R>,
    /// Per mel bin, `bias - running_mean · scale`.
    bn_shift: Tensor<R>,
    /// The patch convolution as a matmul over flattened patches.
    patch: Linear<R>,
    patch_norm: Option<LayerNorm<R>>,
    stages: Vec<SwinStage<R>>,
    final_norm: LayerNorm<R>,
    config: ClapAudioConfig,
    freq_ratio: usize,
}

impl<R: Runtime<DType = DType>> ClapAudioEncoder<R> {
    /// Reads `batch_norm.*`, `patch_embed.*`, `layers.{i}.*` and `norm.*`
    /// under `vb` (transformers' `audio_model.audio_encoder`).
    pub fn from_varbuilder<C>(
        client: &C,
        vb: &mut VarBuilder<'_, R>,
        config: &ClapConfig,
    ) -> Result<Self>
    where
        C: RuntimeClient<R> + TensorOps<R> + ScalarOps<R> + BinaryOps<R> + UnaryOps<R>,
    {
        let a = &config.audio;
        let mut bn = vb.pp("batch_norm");
        let weight = bn.take_tensor("weight")?;
        let bias = bn.take_tensor("bias")?;
        let mean = bn.take_tensor("running_mean")?;
        let var = bn.take_tensor("running_var")?;
        drop(bn);
        // PyTorch's inference form: `1 / sqrt(var + eps)` times the weight.
        let inv_std = client.recip(&client.sqrt(&client.add_scalar(&var, BATCH_NORM_EPS)?)?)?;
        let bn_scale = client.mul(&inv_std, &weight)?;
        let bn_shift = client.sub(&bias, &client.mul(&mean, &bn_scale)?)?;

        let mut pe = vb.pp("patch_embed");
        let mut proj = pe.pp("proj");
        let kernel = proj.take_tensor("weight")?;
        let patch_bias = proj.take_tensor("bias")?;
        drop(proj);
        let kshape = kernel.shape().to_vec();
        let patch_len = a.patch_embed_input_channels * a.patch_size * a.patch_size;
        if kshape.len() != 4 || kshape[1] * kshape[2] * kshape[3] != patch_len {
            return Err(Error::ModelError {
                reason: format!("CLAP patch_embed.proj.weight has shape {kshape:?}"),
            });
        }
        let patch = Linear::new(
            kernel.reshape(&[kshape[0], patch_len])?,
            Some(patch_bias),
            false,
        );
        let patch_norm = if a.enable_patch_layer_norm {
            Some(layer_norm(&mut pe, "norm", 1e-5)?)
        } else {
            None
        };
        drop(pe);

        let grid = a.spec_size / a.patch_size;
        let mut stages = Vec::with_capacity(a.depths.len());
        let (mut height, mut width) = (grid, grid);
        for (i, (&depth, &heads)) in a.depths.iter().zip(&a.num_attention_heads).enumerate() {
            let shape = BlockShape {
                dim: a.patch_embeds_hidden_size << i,
                heads,
                height,
                width,
                window: a.window_size,
                shifted: false,
                mlp_ratio: a.mlp_ratio,
                qkv_bias: a.qkv_bias,
                eps: a.layer_norm_eps,
            };
            let merges = i + 1 < a.depths.len();
            let mut s = vb.pp(&format!("layers.{i}"));
            let stage =
                SwinStage::from_varbuilder(client, &mut s, shape, depth, merges, config.audio_act)?;
            drop(s);
            (height, width) = stage.output_grid();
            stages.push(stage);
        }
        let final_norm = layer_norm(vb, "norm", 1e-5)?;
        Ok(Self {
            bn_scale,
            bn_shift,
            patch,
            patch_norm,
            stages,
            final_norm,
            config: a.clone(),
            freq_ratio: config.freq_ratio(),
        })
    }

    /// `[batch, 1, frames, mels]` features to the square image
    /// `[batch, 1, spec_size, spec_size]`: transformers' batch norm and
    /// `reshape_mel2img`.
    fn image<C>(&self, client: &C, features: &Tensor<R>) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R> + BinaryOps<R> + IndexingOps<R>,
    {
        let shape = features.shape().to_vec();
        let a = &self.config;
        let (width, height) = (a.spec_size * self.freq_ratio, a.spec_size / self.freq_ratio);
        if shape.len() != 4 || shape[1] != 1 || shape[3] != a.num_mel_bins || shape[2] > width {
            return Err(Error::InvalidArgument {
                arg: "features",
                reason: format!(
                    "expected [batch, 1, at most {width}, {}], got {shape:?}",
                    a.num_mel_bins
                ),
            });
        }
        let mut x = client.add(&client.mul(features, &self.bn_scale)?, &self.bn_shift)?;
        if shape[2] < width {
            x = resize_bicubic(client, &x, 2, width)?;
        }
        if shape[3] < height {
            x = resize_bicubic(client, &x, 3, height)?;
        }
        let (batch, r, freq) = (shape[0], self.freq_ratio, x.shape()[3]);
        Ok(x.contiguous()?
            .reshape(&[batch, r, width / r, freq])?
            .permute(&[0, 1, 3, 2])?
            .contiguous()?
            .reshape(&[batch, 1, freq * r, width / r])?)
    }

    /// The pooled audio state, `[batch, hidden_size]`, of `features`
    /// (`[batch, 1, frames, mels]`, as [`super::ClapFeatureExtractor`]
    /// gives them).
    pub fn forward<C>(&self, client: &C, features: &Tensor<R>) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + ShapeOps<R>
            + IndexingOps<R>
            + ActivationOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        let image = self.image(client, features)?;
        let a = &self.config;
        let (batch, p, c_in) = (image.shape()[0], a.patch_size, a.patch_embed_input_channels);
        let side = a.spec_size / p;
        let patches = image
            .reshape(&[batch, c_in, side, p, side, p])?
            .permute(&[0, 2, 4, 1, 3, 5])?
            .contiguous()?
            .reshape(&[batch, side * side, c_in * p * p])?;
        let mut h = dense(client, &self.patch, &patches)?;
        if let Some(n) = &self.patch_norm {
            h = norm(client, n, &h)?;
        }
        for stage in &self.stages {
            h = stage.forward(client, &h)?;
        }
        let h = norm(client, &self.final_norm, &h)?;
        Ok(client.mean(&h, &[1], false)?)
    }
}
