//! SigLIP vision encoder for multimodal LLMs (PaliGemma, etc.)
//!
//! Differs from CLIP:
//! - GELU activation instead of QuickGELU
//! - No class token
//! - Position embedding supports bilinear interpolation for variable image sizes

use crate::error::{Error, Result};
use crate::model::config::VisionConfig;
use crate::nn::{Conv2d, LayerNorm, Linear, VarBuilder};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, ConvOps, NormalizationOps, PaddingMode, ScalarOps, ShapeOps,
    TensorOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Single SigLIP encoder transformer layer.
struct SigLipEncoderLayer<R: Runtime> {
    ln1: LayerNorm<R>,
    ln2: LayerNorm<R>,
    q_proj: Linear<R>,
    k_proj: Linear<R>,
    v_proj: Linear<R>,
    out_proj: Linear<R>,
    fc1: Linear<R>,
    fc2: Linear<R>,
    num_heads: usize,
    head_dim: usize,
}

impl<R: Runtime> SigLipEncoderLayer<R> {
    fn from_varbuilder(
        vb: &mut VarBuilder<R>,
        hidden_size: usize,
        num_heads: usize,
        _intermediate_size: usize,
    ) -> Result<Self> {
        let head_dim = hidden_size / num_heads;

        let mut sa_vb = vb.pp("self_attn");
        let q_proj = Linear::new(
            sa_vb.take_tensor("q_proj.weight")?,
            sa_vb.take_tensor_optional("q_proj.bias")?,
            false,
        );
        let k_proj = Linear::new(
            sa_vb.take_tensor("k_proj.weight")?,
            sa_vb.take_tensor_optional("k_proj.bias")?,
            false,
        );
        let v_proj = Linear::new(
            sa_vb.take_tensor("v_proj.weight")?,
            sa_vb.take_tensor_optional("v_proj.bias")?,
            false,
        );
        let out_proj = Linear::new(
            sa_vb.take_tensor("out_proj.weight")?,
            sa_vb.take_tensor_optional("out_proj.bias")?,
            false,
        );

        let mut ln1_vb = vb.pp("layer_norm1");
        let ln1 = LayerNorm::new(
            ln1_vb.take_tensor("weight")?,
            ln1_vb.take_tensor("bias")?,
            1e-6,
            false,
        );

        let mut ln2_vb = vb.pp("layer_norm2");
        let ln2 = LayerNorm::new(
            ln2_vb.take_tensor("weight")?,
            ln2_vb.take_tensor("bias")?,
            1e-6,
            false,
        );

        let mut mlp_vb = vb.pp("mlp");
        let fc1 = Linear::new(
            mlp_vb.take_tensor("fc1.weight")?,
            mlp_vb.take_tensor_optional("fc1.bias")?,
            false,
        );
        let fc2 = Linear::new(
            mlp_vb.take_tensor("fc2.weight")?,
            mlp_vb.take_tensor_optional("fc2.bias")?,
            false,
        );

        Ok(Self {
            ln1,
            ln2,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            fc1,
            fc2,
            num_heads,
            head_dim,
        })
    }

    fn forward_inference<C>(&self, client: &C, input: &Tensor<R>) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + BinaryOps<R>
            + ScalarOps<R>
            + NormalizationOps<R>
            + ActivationOps<R>
            + ShapeOps<R>
            + UnaryOps<R>
            + ConvOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        let input_var = Var::new(input.clone(), false);

        // Pre-norm
        let normed = self.ln1.forward(client, &input_var)?;

        // Bidirectional self-attention
        let attn_out = self.self_attention(client, normed.tensor())?;

        // Residual
        let residual1 = client.add(input, &attn_out).map_err(Error::Numr)?;
        let residual1_var = Var::new(residual1.clone(), false);

        // LN2 -> MLP -> residual
        let normed2 = self.ln2.forward(client, &residual1_var)?;
        let mlp_out = self.mlp(client, normed2.tensor())?;
        client.add(&residual1, &mlp_out).map_err(Error::Numr)
    }

    fn self_attention<C>(&self, client: &C, x: &Tensor<R>) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + BinaryOps<R>
            + ScalarOps<R>
            + ActivationOps<R>
            + ShapeOps<R>
            + UnaryOps<R>,
        R::Client: TensorOps<R>,
    {
        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];

        let x_var = Var::new(x.clone(), false);

        let q = self.q_proj.forward(client, &x_var)?;
        let k = self.k_proj.forward(client, &x_var)?;
        let v = self.v_proj.forward(client, &x_var)?;

        let q = q
            .tensor()
            .reshape(&[batch, seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
        let k = k
            .tensor()
            .reshape(&[batch, seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
        let v = v
            .tensor()
            .reshape(&[batch, seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;

        let k_t = k.transpose(-2, -1)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = client
            .matmul(&q.contiguous(), &k_t.contiguous())
            .map_err(Error::Numr)?;
        let scores = client.mul_scalar(&scores, scale).map_err(Error::Numr)?;

        let attn_weights = client.softmax(&scores, -1).map_err(Error::Numr)?;

        let attn_out = client
            .matmul(&attn_weights, &v.contiguous())
            .map_err(Error::Numr)?;

        let attn_out = attn_out.transpose(1, 2)?.contiguous().reshape(&[
            batch,
            seq_len,
            self.num_heads * self.head_dim,
        ])?;

        let attn_var = Var::new(attn_out, false);
        let out = self.out_proj.forward(client, &attn_var)?;
        Ok(out.tensor().clone())
    }

    /// MLP: fc1 -> GELU -> fc2 (SigLIP uses standard GELU, not QuickGELU)
    fn mlp<C>(&self, client: &C, x: &Tensor<R>) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R> + TensorOps<R> + BinaryOps<R> + ActivationOps<R> + UnaryOps<R>,
        R::Client: TensorOps<R>,
    {
        let x_var = Var::new(x.clone(), false);
        let h = self.fc1.forward(client, &x_var)?;
        let h = client.gelu(h.tensor()).map_err(Error::Numr)?;
        let h_var = Var::new(h, false);
        let out = self.fc2.forward(client, &h_var)?;
        Ok(out.tensor().clone())
    }
}

/// SigLIP vision encoder.
///
/// Similar to CLIP but uses GELU activation and has no class token.
/// Position embeddings can be interpolated for variable resolution.
pub struct SigLipEncoder<R: Runtime> {
    patch_embed: Conv2d<R>,
    position_embedding: Var<R>,
    post_ln: LayerNorm<R>,
    layers: Vec<SigLipEncoderLayer<R>>,
    select_layer: Option<i32>,
    /// Number of patches per side at training resolution (for interpolation)
    num_patches_per_side: usize,
    hidden_size: usize,
}

impl<R: Runtime> SigLipEncoder<R> {
    /// Load SigLIP encoder from a VarBuilder.
    pub fn from_varbuilder(vb: &mut VarBuilder<R>, config: &VisionConfig) -> Result<Self> {
        let num_patches_per_side = config.image_size / config.patch_size;

        let mut embed_vb = vb.pp("embeddings");
        let patch_weight = embed_vb.take_tensor("patch_embedding.weight")?;
        let patch_bias = embed_vb.take_tensor_optional("patch_embedding.bias")?;
        let patch_embed = Conv2d::new(
            patch_weight,
            patch_bias,
            (config.patch_size, config.patch_size),
            PaddingMode::Valid,
            (1, 1),
            1,
            false,
        );

        let pos_emb = embed_vb.take_tensor("position_embedding.weight")?;
        let position_embedding = Var::new(pos_emb, false);

        let mut post_ln_vb = vb.pp("post_layernorm");
        let post_ln = LayerNorm::new(
            post_ln_vb.take_tensor("weight")?,
            post_ln_vb.take_tensor("bias")?,
            1e-6,
            false,
        );

        let mut layers = Vec::with_capacity(config.num_layers);
        let mut enc_vb = vb.pp("encoder");
        for i in 0..config.num_layers {
            let mut layer_vb = enc_vb.pp(&format!("layers.{i}"));
            let layer = SigLipEncoderLayer::from_varbuilder(
                &mut layer_vb,
                config.hidden_size,
                config.num_heads,
                config.intermediate_size,
            )?;
            layers.push(layer);
        }

        Ok(Self {
            patch_embed,
            position_embedding,
            post_ln,
            layers,
            select_layer: config.select_layer,
            num_patches_per_side,
            hidden_size: config.hidden_size,
        })
    }

    /// Interpolate position embeddings for a different image resolution.
    ///
    /// Uses bilinear interpolation: reshape to 2D grid, scale, flatten back.
    /// This is done on CPU since it only happens once at initialization.
    fn interpolate_pos_embed(
        &self,
        pos_embed: &Tensor<R>,
        target_patches_per_side: usize,
    ) -> Result<Tensor<R>>
    where
        R: Runtime<DType = DType>,
    {
        if target_patches_per_side == self.num_patches_per_side {
            return Ok(pos_embed.clone());
        }

        let total_patches = pos_embed.shape()[0];
        let hidden = pos_embed.shape()[1];
        let src_side = self.num_patches_per_side;

        if total_patches != src_side * src_side {
            return Err(Error::ModelError {
                reason: format!(
                    "position embedding has {} patches but expected {}x{}={}",
                    total_patches,
                    src_side,
                    src_side,
                    src_side * src_side
                ),
            });
        }

        // Reshape to [1, src_side, src_side, hidden], read data, do bilinear on CPU,
        // then create new tensor. This runs once so CPU fallback is acceptable.
        let data: Vec<f32> = pos_embed.to_vec();
        let tgt = target_patches_per_side;
        let mut out = vec![0.0f32; tgt * tgt * hidden];

        for ty in 0..tgt {
            for tx in 0..tgt {
                // Map target coords to source coords
                let sy = (ty as f64 + 0.5) * (src_side as f64) / (tgt as f64) - 0.5;
                let sx = (tx as f64 + 0.5) * (src_side as f64) / (tgt as f64) - 0.5;

                let sy0 = sy.floor().max(0.0) as usize;
                let sx0 = sx.floor().max(0.0) as usize;
                let sy1 = (sy0 + 1).min(src_side - 1);
                let sx1 = (sx0 + 1).min(src_side - 1);

                let fy = sy - sy.floor();
                let fx = sx - sx.floor();

                let w00 = (1.0 - fy) * (1.0 - fx);
                let w01 = (1.0 - fy) * fx;
                let w10 = fy * (1.0 - fx);
                let w11 = fy * fx;

                let dst_offset = (ty * tgt + tx) * hidden;
                let s00 = (sy0 * src_side + sx0) * hidden;
                let s01 = (sy0 * src_side + sx1) * hidden;
                let s10 = (sy1 * src_side + sx0) * hidden;
                let s11 = (sy1 * src_side + sx1) * hidden;

                for c in 0..hidden {
                    out[dst_offset + c] = (w00 * data[s00 + c] as f64
                        + w01 * data[s01 + c] as f64
                        + w10 * data[s10 + c] as f64
                        + w11 * data[s11 + c] as f64)
                        as f32;
                }
            }
        }

        let device = pos_embed.device();
        Ok(Tensor::from_slice(&out, &[tgt * tgt, hidden], device))
    }

    /// Forward pass for inference.
    ///
    /// pixel_values: [B, 3, H, W] normalized image tensor
    /// Returns: [B, num_patches, hidden_size] (no class token)
    pub fn forward_inference<C>(&self, client: &C, pixel_values: &Tensor<R>) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + BinaryOps<R>
            + ScalarOps<R>
            + NormalizationOps<R>
            + ActivationOps<R>
            + ShapeOps<R>
            + ConvOps<R>
            + UnaryOps<R>,
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        let batch = pixel_values.shape()[0];

        // Patch embedding: [B, 3, H, W] -> [B, hidden, grid_h, grid_w]
        let patches = self.patch_embed.forward_inference(client, pixel_values)?;
        let hidden = patches.shape()[1];
        let grid_h = patches.shape()[2];
        let grid_w = patches.shape()[3];
        let num_patches = grid_h * grid_w;
        let current_patches_per_side = grid_h; // assuming square

        // Reshape to [B, num_patches, hidden]
        let patches = patches
            .reshape(&[batch, hidden, num_patches])?
            .transpose(1, 2)?
            .contiguous();

        // Position embedding (interpolate if resolution differs)
        let pos =
            self.interpolate_pos_embed(self.position_embedding.tensor(), current_patches_per_side)?;
        let pos = pos
            .reshape(&[1, num_patches, self.hidden_size])?
            .broadcast_to(&[batch, num_patches, self.hidden_size])?;
        let embeddings = client
            .add(&patches, &pos.contiguous())
            .map_err(Error::Numr)?;

        // Run through transformer layers
        let total_layers = self.layers.len();
        let extract_at = self.select_layer.map(|sl| {
            if sl < 0 {
                (total_layers as i32 + sl) as usize
            } else {
                sl as usize
            }
        });

        let mut hidden_states = embeddings;
        for (i, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward_inference(client, &hidden_states)?;
            if extract_at == Some(i) {
                return Ok(hidden_states);
            }
        }

        // Post-LN
        let out_var = Var::new(hidden_states, false);
        let output = self.post_ln.forward(client, &out_var)?;
        Ok(output.tensor().clone())
    }
}
