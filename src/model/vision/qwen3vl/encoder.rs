//! The vision tower: patch tokens -> 27 blocks -> post norm -> 2x2 merger.

use super::block::Qwen3VlBlock;
use super::config::Qwen3VlVisionConfig;
use super::embed::PatchEmbed;
use super::preprocess::PreprocessedImage;
use super::rope2d::rope2d_tables;
use crate::error::{Error, Result};
use crate::model::traits::ModelClient;
use crate::nn::{LayerNorm, MaybeQuantLinear, VarBuilder};
use crate::quant::traits::DequantOps;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Rotary base of the 2D position encoding.
pub const ROPE_THETA: f32 = 10000.0;

/// Qwen3-VL vision encoder with its merger projection.
pub struct Qwen3VlVision<R: Runtime> {
    cfg: Qwen3VlVisionConfig,
    embed: PatchEmbed<R>,
    blocks: Vec<Qwen3VlBlock<R>>,
    post_ln: LayerNorm<R>,
    mm0: MaybeQuantLinear<R>,
    mm2: MaybeQuantLinear<R>,
}

impl<R: Runtime<DType = DType>> Qwen3VlVision<R> {
    /// Build from a map holding the mmproj tensors under their GGUF names
    /// (`v.patch_embd.*`, `v.position_embd.weight`, `v.blk.N.*`,
    /// `v.post_ln.*`, `mm.0.*`, `mm.2.*`).
    pub fn from_varbuilder(vb: &mut VarBuilder<R>, cfg: &Qwen3VlVisionConfig) -> Result<Self>
    where
        R::Client: DequantOps<R>,
    {
        cfg.check()?;
        let hidden = cfg.hidden_size;
        let embed = PatchEmbed::from_varbuilder(
            vb,
            cfg.patch_size,
            hidden,
            cfg.pos_grid_side(),
            cfg.spatial_merge,
        )?;
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(Qwen3VlBlock::from_varbuilder(
                vb,
                i,
                hidden,
                cfg.intermediate_size,
                cfg.num_heads,
                cfg.layer_norm_eps,
            )?);
        }
        let post_w = vb.take_tensor_dequant("v.post_ln.weight", DType::F32)?;
        let post_b = vb.take_tensor_dequant("v.post_ln.bias", DType::F32)?;
        if post_w.shape() != [hidden].as_slice() || post_b.shape() != [hidden].as_slice() {
            return Err(Error::ModelError {
                reason: format!(
                    "qwen3vl vision: v.post_ln has shape {:?}/{:?}, want [{hidden}]",
                    post_w.shape(),
                    post_b.shape()
                ),
            });
        }
        let post_ln = LayerNorm::new(post_w, post_b, cfg.layer_norm_eps, false);
        let merged = cfg.merged_width();
        let mm0 = vb.take_maybe_quant_linear("mm.0.weight", Some("mm.0.bias"))?;
        let mm2 = vb.take_maybe_quant_linear("mm.2.weight", Some("mm.2.bias"))?;
        for (name, layer, want) in [
            ("mm.0.weight", &mm0, [merged, merged]),
            ("mm.2.weight", &mm2, [cfg.projection_dim, merged]),
        ] {
            if layer.shape() != want.as_slice() {
                return Err(Error::ModelError {
                    reason: format!(
                        "qwen3vl vision: {name} has shape {:?}, want {want:?}",
                        layer.shape()
                    ),
                });
            }
        }
        Ok(Self {
            cfg: cfg.clone(),
            embed,
            blocks,
            post_ln,
            mm0,
            mm2,
        })
    }

    /// The sizes this tower was built with.
    pub fn config(&self) -> &Qwen3VlVisionConfig {
        &self.cfg
    }

    /// Encode one preprocessed image. Returns `[nx * ny, projection_dim]`.
    pub fn encode_image<C>(&self, client: &C, image: &PreprocessedImage) -> Result<Tensor<R>>
    where
        C: ModelClient<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + DequantOps<R>,
    {
        self.encode(client, &image.pixels, image.width, image.height)
    }

    /// Encode planar CHW pixels of a `width x height` image. Both sides must
    /// be multiples of `patch * merge`. Returns `[n_tokens, projection_dim]`
    /// with tokens in raster order over the `(height / 32, width / 32)`
    /// grid.
    pub fn encode<C>(
        &self,
        client: &C,
        pixels: &[f32],
        width: usize,
        height: usize,
    ) -> Result<Tensor<R>>
    where
        C: ModelClient<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + DequantOps<R>,
    {
        if pixels.len() != 3 * width * height {
            return Err(Error::InvalidArgument {
                arg: "pixels",
                reason: format!(
                    "qwen3vl vision: {} floats for a {width}x{height} image, want {}",
                    pixels.len(),
                    3 * width * height
                ),
            });
        }
        let input = Tensor::<R>::from_slice(pixels, &[1, 3, height, width], client.device())?;
        self.encode_tensor(client, &input)
    }

    /// Encode a `[1, 3, H, W]` device tensor. See [`Self::encode`].
    pub fn encode_tensor<C>(&self, client: &C, pixels: &Tensor<R>) -> Result<Tensor<R>>
    where
        C: ModelClient<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + DequantOps<R>,
    {
        let shape = pixels.shape();
        if shape.len() != 4 || shape[0] != 1 || shape[1] != 3 {
            return Err(Error::InvalidArgument {
                arg: "pixels",
                reason: format!("qwen3vl vision: pixel tensor is {shape:?}, want [1, 3, H, W]"),
            });
        }
        let (height, width) = (shape[2], shape[3]);
        let align = self.cfg.align();
        if width == 0 || height == 0 || width % align != 0 || height % align != 0 {
            return Err(Error::InvalidArgument {
                arg: "pixels",
                reason: format!(
                    "qwen3vl vision: image {width}x{height} is not a multiple of {align}"
                ),
            });
        }
        let patch = self.cfg.patch_size;
        let merge = self.cfg.spatial_merge;
        let (ph, pw) = (height / patch, width / patch);
        let n_patches = ph * pw;
        let device = client.device();

        let tables = rope2d_tables(ph, pw, merge, self.cfg.head_dim(), ROPE_THETA);
        let half = tables.half_dim;
        let cos = Var::new(
            Tensor::<R>::from_slice(&tables.cos, &[n_patches, half], device)?,
            false,
        );
        let sin = Var::new(
            Tensor::<R>::from_slice(&tables.sin, &[n_patches, half], device)?,
            false,
        );

        let mut h = self.embed.forward(client, pixels, ph, pw)?;
        for block in &self.blocks {
            h = block.forward(client, &h, &cos, &sin)?;
        }
        let h = self.post_ln.forward(client, &Var::new(h, false))?;

        let n_tokens = n_patches / (merge * merge);
        let merged = h.tensor().reshape(&[n_tokens, self.cfg.merged_width()])?;
        let m0 = self.mm0.forward(client, &Var::new(merged, false))?;
        let act = client.gelu(m0.tensor())?;
        let m2 = self.mm2.forward(client, &Var::new(act, false))?;
        Ok(m2.tensor().clone())
    }
}
