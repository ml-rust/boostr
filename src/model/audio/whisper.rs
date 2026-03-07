//! Whisper audio encoder: Conv1d stem + transformer encoder layers.
//!
//! Implements the encoder portion of OpenAI's Whisper model, which transforms
//! log-mel spectrograms into a sequence of hidden representations.

use crate::error::{Error, Result};
use crate::model::config::AudioConfig;
use crate::nn::VarBuilder;
use crate::nn::conv1d::Conv1d;
use crate::nn::layernorm::LayerNorm;
use crate::nn::linear::Linear;
use numr::autograd::Var;
use numr::ops::{
    ActivationOps, BinaryOps, ConvOps, MatmulOps, NormalizationOps, ReduceOps, ScalarOps, ShapeOps,
    TensorOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// A single Whisper encoder transformer layer.
///
/// Standard transformer encoder with bidirectional (non-causal) self-attention.
pub struct WhisperEncoderLayer<R: Runtime> {
    ln1: LayerNorm<R>,
    q_proj: Linear<R>,
    k_proj: Linear<R>,
    v_proj: Linear<R>,
    out_proj: Linear<R>,
    ln2: LayerNorm<R>,
    fc1: Linear<R>,
    fc2: Linear<R>,
    num_heads: usize,
    head_dim: usize,
}

impl<R: Runtime> WhisperEncoderLayer<R> {
    /// Load a single encoder layer from a VarBuilder scoped to this layer's prefix.
    pub fn from_varbuilder(
        vb: &mut VarBuilder<'_, R>,
        hidden_size: usize,
        num_heads: usize,
    ) -> Result<Self> {
        let head_dim = hidden_size / num_heads;

        let mut self_attn = vb.pp("self_attn");

        let q_w = self_attn.take_tensor("q_proj.weight")?;
        let q_b = self_attn.take_tensor_optional("q_proj.bias")?;
        let q_proj = Linear::new(q_w, q_b, false);

        let k_w = self_attn.take_tensor("k_proj.weight")?;
        let k_b = self_attn.take_tensor_optional("k_proj.bias")?;
        let k_proj = Linear::new(k_w, k_b, false);

        let v_w = self_attn.take_tensor("v_proj.weight")?;
        let v_b = self_attn.take_tensor_optional("v_proj.bias")?;
        let v_proj = Linear::new(v_w, v_b, false);

        let out_w = self_attn.take_tensor("out_proj.weight")?;
        let out_b = self_attn.take_tensor_optional("out_proj.bias")?;
        let out_proj = Linear::new(out_w, out_b, false);

        drop(self_attn);

        let mut ln1_vb = vb.pp("self_attn_layer_norm");
        let ln1_w = ln1_vb.take_tensor("weight")?;
        let ln1_b = ln1_vb.take_tensor("bias")?;
        let ln1 = LayerNorm::new(ln1_w, ln1_b, 1e-5, false);
        drop(ln1_vb);

        let mut fc_vb = vb.pp("fc1");
        let fc1_w = fc_vb.take_tensor("weight")?;
        let fc1_b = fc_vb.take_tensor_optional("bias")?;
        let fc1 = Linear::new(fc1_w, fc1_b, false);
        drop(fc_vb);

        let mut fc2_vb = vb.pp("fc2");
        let fc2_w = fc2_vb.take_tensor("weight")?;
        let fc2_b = fc2_vb.take_tensor_optional("bias")?;
        let fc2 = Linear::new(fc2_w, fc2_b, false);
        drop(fc2_vb);

        let mut ln2_vb = vb.pp("final_layer_norm");
        let ln2_w = ln2_vb.take_tensor("weight")?;
        let ln2_b = ln2_vb.take_tensor("bias")?;
        let ln2 = LayerNorm::new(ln2_w, ln2_b, 1e-5, false);

        Ok(Self {
            ln1,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            ln2,
            fc1,
            fc2,
            num_heads,
            head_dim,
        })
    }

    /// Forward pass (inference, no autograd).
    ///
    /// Input/output: `[B, seq_len, hidden]`
    /// Uses bidirectional attention (no causal mask).
    pub fn forward_inference<C>(&self, client: &C, x: &Tensor<R>) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + MatmulOps<R>
            + BinaryOps<R>
            + ActivationOps<R>
            + NormalizationOps<R>
            + ReduceOps<R>
            + ShapeOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];
        let hidden = shape[2];

        // Pre-norm
        let x_var = Var::new(x.clone(), false);
        let normed = self.ln1.forward(client, &x_var)?;
        let normed_t = normed.tensor().clone();

        // Q, K, V projections
        let normed_var = Var::new(normed_t.clone(), false);
        let q = self.q_proj.forward(client, &normed_var)?;
        let k = self.k_proj.forward(client, &normed_var)?;
        let v = self.v_proj.forward(client, &normed_var)?;

        // Reshape to [B, num_heads, seq_len, head_dim]
        let q = q
            .tensor()
            .reshape(&[batch, seq_len, self.num_heads, self.head_dim])
            .map_err(Error::Numr)?
            .transpose(1, 2)
            .map_err(Error::Numr)?
            .contiguous();
        let k = k
            .tensor()
            .reshape(&[batch, seq_len, self.num_heads, self.head_dim])
            .map_err(Error::Numr)?
            .transpose(1, 2)
            .map_err(Error::Numr)?
            .contiguous();
        let v = v
            .tensor()
            .reshape(&[batch, seq_len, self.num_heads, self.head_dim])
            .map_err(Error::Numr)?
            .transpose(1, 2)
            .map_err(Error::Numr)?
            .contiguous();

        // Attention: softmax(Q @ K^T / sqrt(d)) @ V
        let k_t = k.transpose(2, 3).map_err(Error::Numr)?.contiguous();
        let scale = (self.head_dim as f32).sqrt();
        let scores = client.matmul(&q, &k_t).map_err(Error::Numr)?;
        let scores = client
            .div_scalar(&scores, scale as f64)
            .map_err(Error::Numr)?;
        // Bidirectional: no causal mask
        let attn_weights = client.softmax(&scores, -1).map_err(Error::Numr)?;
        let attn_out = client.matmul(&attn_weights, &v).map_err(Error::Numr)?;

        // Reshape back to [B, seq_len, hidden]
        let attn_out = attn_out
            .transpose(1, 2)
            .map_err(Error::Numr)?
            .contiguous()
            .reshape(&[batch, seq_len, hidden])
            .map_err(Error::Numr)?;

        // Project out
        let attn_out_var = Var::new(attn_out, false);
        let projected = self.out_proj.forward(client, &attn_out_var)?;

        // Residual
        let after_attn = client.add(x, projected.tensor()).map_err(Error::Numr)?;

        // FFN with pre-norm
        let after_attn_var = Var::new(after_attn.clone(), false);
        let normed2 = self.ln2.forward(client, &after_attn_var)?;
        let normed2_var = Var::new(normed2.tensor().clone(), false);

        let fc1_out = self.fc1.forward(client, &normed2_var)?;
        let activated = client.gelu(fc1_out.tensor()).map_err(Error::Numr)?;
        let activated_var = Var::new(activated, false);
        let fc2_out = self.fc2.forward(client, &activated_var)?;

        // Residual
        client
            .add(&after_attn, fc2_out.tensor())
            .map_err(Error::Numr)
    }
}

/// Whisper audio encoder.
///
/// Architecture:
/// 1. Two Conv1d layers (stem) to downsample mel spectrogram
/// 2. Learned positional embedding
/// 3. Stack of transformer encoder layers (bidirectional attention)
/// 4. Final layer norm
pub struct WhisperEncoder<R: Runtime> {
    conv1: Conv1d<R>,
    conv2: Conv1d<R>,
    position_embedding: Var<R>,
    layers: Vec<WhisperEncoderLayer<R>>,
    ln_post: LayerNorm<R>,
}

impl<R: Runtime> WhisperEncoder<R> {
    /// Load from a VarBuilder and AudioConfig.
    pub fn from_varbuilder(vb: &mut VarBuilder<'_, R>, config: &AudioConfig) -> Result<Self> {
        let hidden = config.hidden_size;
        // Conv stem
        let mut conv1_vb = vb.pp("conv1");
        let conv1_w = conv1_vb.take_tensor("weight")?;
        let conv1_b = conv1_vb.take_tensor_optional("bias")?;
        drop(conv1_vb);
        let conv1 = Conv1d::new(
            conv1_w,
            conv1_b,
            1,                                    // stride=1
            numr::ops::PaddingMode::conv1d(1, 1), // padding=1
            1,                                    // dilation=1
            1,                                    // groups=1
            false,
        );

        let mut conv2_vb = vb.pp("conv2");
        let conv2_w = conv2_vb.take_tensor("weight")?;
        let conv2_b = conv2_vb.take_tensor_optional("bias")?;
        drop(conv2_vb);
        let conv2 = Conv1d::new(
            conv2_w,
            conv2_b,
            2,                                    // stride=2 (downsample)
            numr::ops::PaddingMode::conv1d(1, 1), // padding=1
            1,                                    // dilation=1
            1,                                    // groups=1
            false,
        );

        // Learned positional embedding: [1, max_audio_len, hidden]
        let pos_emb = vb.take_tensor("embed_positions.weight")?;
        let position_embedding = Var::new(pos_emb, false);

        // Transformer layers
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let mut layer_vb = vb.pp(&format!("layers.{i}"));
            layers.push(WhisperEncoderLayer::from_varbuilder(
                &mut layer_vb,
                hidden,
                config.num_heads,
            )?);
        }

        // Final layer norm
        let mut ln_vb = vb.pp("layer_norm");
        let ln_w = ln_vb.take_tensor("weight")?;
        let ln_b = ln_vb.take_tensor("bias")?;
        let ln_post = LayerNorm::new(ln_w, ln_b, 1e-5, false);

        Ok(Self {
            conv1,
            conv2,
            position_embedding,
            layers,
            ln_post,
        })
    }

    /// Forward pass (inference only).
    ///
    /// Input: `mel` with shape `[B, num_mel_bins, audio_len]`
    /// Output: `[B, seq_len, hidden]` where `seq_len ≈ audio_len / 2`
    pub fn forward_inference<C>(&self, client: &C, mel: &Tensor<R>) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + MatmulOps<R>
            + BinaryOps<R>
            + ActivationOps<R>
            + NormalizationOps<R>
            + ConvOps<R>
            + ReduceOps<R>
            + ShapeOps<R>,
        R::Client:
            TensorOps<R> + ScalarOps<R> + ConvOps<R> + ReduceOps<R> + BinaryOps<R> + ScalarOps<R>,
    {
        // Conv stem: [B, num_mel_bins, audio_len] -> [B, hidden, seq_len]
        let x = self.conv1.forward_inference(client, mel)?;
        let x = client.gelu(&x).map_err(Error::Numr)?;
        let x = self.conv2.forward_inference(client, &x)?;
        let x = client.gelu(&x).map_err(Error::Numr)?;

        // Transpose to [B, seq_len, hidden]
        let x = x.transpose(1, 2).map_err(Error::Numr)?.contiguous();

        let seq_len = x.shape()[1];

        // Add positional embedding (truncate if audio is shorter than max)
        let pos = self.position_embedding.tensor();
        let pos_seq_len = pos.shape()[0]; // [max_len, hidden]
        let pos_slice = if seq_len <= pos_seq_len {
            pos.narrow(0, 0, seq_len).map_err(Error::Numr)?
        } else {
            // Pad with zeros if audio exceeds max length (unusual but handle gracefully)
            pos.clone()
        };

        // Broadcast add: [B, seq_len, hidden] + [seq_len, hidden]
        let x = client.add(&x, &pos_slice).map_err(Error::Numr)?;

        // Transformer layers
        let mut hidden = x;
        for layer in &self.layers {
            hidden = layer.forward_inference(client, &hidden)?;
        }

        // Final layer norm
        let hidden_var = Var::new(hidden, false);
        let output = self.ln_post.forward(client, &hidden_var)?;

        Ok(output.tensor().clone())
    }
}
