//! CLIP ViT vision encoder for multimodal LLMs (LLaVA, etc.)

use crate::error::{Error, Result};
use crate::model::config::VisionConfig;
use crate::nn::{Conv2d, LayerNorm, Linear, VarBuilder};
use numr::autograd::Var;
use numr::ops::{
    ActivationOps, BinaryOps, ConvOps, NormalizationOps, PaddingMode, ScalarOps, ShapeOps,
    TensorOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// QuickGELU activation: x * sigmoid(1.702 * x)
fn quick_gelu<R, C>(client: &C, x: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + ActivationOps<R> + BinaryOps<R> + ScalarOps<R> + UnaryOps<R>,
{
    let scaled = client.mul_scalar(x, 1.702).map_err(Error::Numr)?;
    let gate = client.sigmoid(&scaled).map_err(Error::Numr)?;
    client.mul(x, &gate).map_err(Error::Numr)
}

/// Single CLIP encoder transformer layer.
struct ClipEncoderLayer<R: Runtime> {
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

impl<R: Runtime> ClipEncoderLayer<R> {
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
            1e-5,
            false,
        );

        let mut ln2_vb = vb.pp("layer_norm2");
        let ln2 = LayerNorm::new(
            ln2_vb.take_tensor("weight")?,
            ln2_vb.take_tensor("bias")?,
            1e-5,
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

    /// Forward pass for one encoder layer.
    ///
    /// input: [B, seq_len, hidden]
    /// output: [B, seq_len, hidden]
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

        // Multi-head self-attention (bidirectional - no causal mask)
        let attn_out = self.self_attention(client, normed.tensor())?;

        // Residual
        let attn_var = Var::new(attn_out, false);
        let residual1 = client.add(input, attn_var.tensor()).map_err(Error::Numr)?;
        let residual1_var = Var::new(residual1.clone(), false);

        // LN2 -> MLP -> residual
        let normed2 = self.ln2.forward(client, &residual1_var)?;
        let mlp_out = self.mlp(client, normed2.tensor())?;
        client.add(&residual1, &mlp_out).map_err(Error::Numr)
    }

    /// Bidirectional multi-head self-attention.
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

        // Project Q, K, V
        let q = self.q_proj.forward(client, &x_var)?;
        let k = self.k_proj.forward(client, &x_var)?;
        let v = self.v_proj.forward(client, &x_var)?;

        // Reshape to [B, seq, num_heads, head_dim] then transpose to [B, num_heads, seq, head_dim]
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

        // Attention scores: Q @ K^T / sqrt(head_dim)
        let k_t = k.transpose(-2, -1)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = client
            .matmul(&q.contiguous(), &k_t.contiguous())
            .map_err(Error::Numr)?;
        let scores = client.mul_scalar(&scores, scale).map_err(Error::Numr)?;

        // Softmax (no causal mask - bidirectional attention)
        let attn_weights = client.softmax(&scores, -1).map_err(Error::Numr)?;

        // Weighted sum: attn @ V -> [B, num_heads, seq, head_dim]
        let attn_out = client
            .matmul(&attn_weights, &v.contiguous())
            .map_err(Error::Numr)?;

        // Transpose back and reshape: [B, seq, hidden]
        let attn_out = attn_out.transpose(1, 2)?.contiguous().reshape(&[
            batch,
            seq_len,
            self.num_heads * self.head_dim,
        ])?;

        // Output projection
        let attn_var = Var::new(attn_out, false);
        let out = self.out_proj.forward(client, &attn_var)?;
        Ok(out.tensor().clone())
    }

    /// MLP: fc1 -> QuickGELU -> fc2
    fn mlp<C>(&self, client: &C, x: &Tensor<R>) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + BinaryOps<R>
            + ScalarOps<R>
            + ActivationOps<R>
            + UnaryOps<R>,
        R::Client: TensorOps<R>,
    {
        let x_var = Var::new(x.clone(), false);
        let h = self.fc1.forward(client, &x_var)?;
        let h = quick_gelu(client, h.tensor())?;
        let h_var = Var::new(h, false);
        let out = self.fc2.forward(client, &h_var)?;
        Ok(out.tensor().clone())
    }
}

/// CLIP ViT vision encoder.
///
/// Processes images via patch embedding + positional encoding + transformer layers.
/// Used in LLaVA and similar multimodal architectures.
pub struct ClipEncoder<R: Runtime> {
    patch_embed: Conv2d<R>,
    class_embedding: Var<R>,
    position_embedding: Var<R>,
    pre_ln: LayerNorm<R>,
    post_ln: LayerNorm<R>,
    layers: Vec<ClipEncoderLayer<R>>,
    select_layer: Option<i32>,
}

impl<R: Runtime> ClipEncoder<R> {
    /// Load CLIP encoder from a VarBuilder.
    pub fn from_varbuilder(vb: &mut VarBuilder<R>, config: &VisionConfig) -> Result<Self> {
        let _num_patches = (config.image_size / config.patch_size).pow(2);

        // Patch embedding: Conv2d with kernel_size=patch_size, stride=patch_size
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

        let class_emb = embed_vb.take_tensor("class_embedding")?;
        let class_embedding = Var::new(class_emb, false);

        let pos_emb = embed_vb.take_tensor("position_embedding.weight")?;
        let position_embedding = Var::new(pos_emb, false);

        let mut pre_ln_vb = vb.pp("pre_layrnorm");
        let pre_ln = if pre_ln_vb.contains("weight") {
            LayerNorm::new(
                pre_ln_vb.take_tensor("weight")?,
                pre_ln_vb.take_tensor("bias")?,
                1e-5,
                false,
            )
        } else {
            // Some models use "pre_layernorm" instead
            let mut alt_vb = vb.pp("pre_layernorm");
            LayerNorm::new(
                alt_vb.take_tensor("weight")?,
                alt_vb.take_tensor("bias")?,
                1e-5,
                false,
            )
        };

        let mut post_ln_vb = vb.pp("post_layernorm");
        let post_ln = LayerNorm::new(
            post_ln_vb.take_tensor("weight")?,
            post_ln_vb.take_tensor("bias")?,
            1e-5,
            false,
        );

        let mut layers = Vec::with_capacity(config.num_layers);
        let mut enc_vb = vb.pp("encoder");
        for i in 0..config.num_layers {
            let mut layer_vb = enc_vb.pp(&format!("layers.{i}"));
            let layer = ClipEncoderLayer::from_varbuilder(
                &mut layer_vb,
                config.hidden_size,
                config.num_heads,
                config.intermediate_size,
            )?;
            layers.push(layer);
        }

        Ok(Self {
            patch_embed,
            class_embedding,
            position_embedding,
            pre_ln,
            post_ln,
            layers,
            select_layer: config.select_layer,
        })
    }

    /// Forward pass for inference.
    ///
    /// pixel_values: [B, 3, H, W] normalized image tensor
    /// Returns: [B, num_patches+1, hidden_size]
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
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        let batch = pixel_values.shape()[0];

        // Patch embedding: [B, 3, H, W] -> [B, hidden, grid_h, grid_w]
        let patches = self.patch_embed.forward_inference(client, pixel_values)?;
        let hidden = patches.shape()[1];
        let grid_h = patches.shape()[2];
        let grid_w = patches.shape()[3];
        let num_patches = grid_h * grid_w;

        // Reshape to [B, hidden, num_patches] then transpose to [B, num_patches, hidden]
        let patches = patches
            .reshape(&[batch, hidden, num_patches])?
            .transpose(1, 2)?
            .contiguous();

        // Prepend class token: expand class_embedding [hidden] -> [B, 1, hidden]
        let cls = self.class_embedding.tensor();
        let cls = cls
            .reshape(&[1, 1, hidden])?
            .broadcast_to(&[batch, 1, hidden])?
            .contiguous();

        // Concatenate [cls, patches] -> [B, num_patches+1, hidden]
        let embeddings = client.cat(&[&cls, &patches], 1).map_err(Error::Numr)?;

        // Add position embedding
        let pos = self.position_embedding.tensor();
        let embeddings = client.add(&embeddings, pos).map_err(Error::Numr)?;

        // Pre-LN
        let emb_var = Var::new(embeddings, false);
        let mut hidden_states = self.pre_ln.forward(client, &emb_var)?.tensor().clone();

        // Determine which layer to extract from
        let total_layers = self.layers.len();
        let extract_at = self.select_layer.map(|sl| {
            if sl < 0 {
                (total_layers as i32 + sl) as usize
            } else {
                sl as usize
            }
        });

        // Run through transformer layers
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
