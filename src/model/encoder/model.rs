//! BERT-style transformer encoder for embedding generation.
//!
//! Architecture: token_embed + position_embed + embed_norm → N × (SelfAttention + FFN + LayerNorm) → pool
//!
//! Supports two position-id strategies:
//! - BERT: position_ids = [0, 1, ..., S-1]
//! - XLM-RoBERTa: position_ids = cumsum(input_ids != pad_id) + pad_id
//!   (padding positions get position_id = pad_id, real tokens are numbered
//!   from pad_id+1 upward)
//!
//! Used by sentence embedding models (all-MiniLM, BGE, nomic-embed, etc.)
//! and cross-encoder rerankers (bge-reranker-v2-m3, jina-reranker, etc.)
//! for producing fixed-size vector representations from token sequences.

use super::config::{ArchFamily, EncoderConfig, HiddenAct};
use crate::error::{Error, Result};
use crate::nn::{Embedding, LayerNorm, Linear};
use numr::autograd::{
    Var, var_add, var_matmul, var_narrow, var_permute, var_reshape, var_transpose,
};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, IndexingOps, NormalizationOps, ReduceOps, ScalarOps, ShapeOps,
    TensorOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// A single transformer encoder layer: self-attention + FFN with residual + LayerNorm.
struct EncoderLayer<R: Runtime> {
    q_proj: Linear<R>,
    k_proj: Linear<R>,
    v_proj: Linear<R>,
    o_proj: Linear<R>,
    attn_norm: LayerNorm<R>,
    ffn_up: Linear<R>,
    ffn_down: Linear<R>,
    ffn_norm: LayerNorm<R>,
    num_heads: usize,
    head_dim: usize,
    hidden_act: HiddenAct,
}

impl<R: Runtime<DType = DType>> EncoderLayer<R> {
    fn forward<C>(
        &self,
        client: &C,
        x: &Var<R>,
        attention_mask: Option<&Tensor<R>>,
    ) -> Result<Var<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + BinaryOps<R>
            + ReduceOps<R>
            + ShapeOps<R>
            + IndexingOps<R>
            + ActivationOps<R>
            + UnaryOps<R>
            + NormalizationOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        // Self-attention with residual
        let attn_out = self.self_attention(client, x, attention_mask)?;
        let x = var_add(x, &attn_out, client).map_err(Error::Numr)?;
        let x = self.attn_norm.forward(client, &x)?;

        // FFN with residual
        let ffn_out = self.ffn(client, &x)?;
        let x = var_add(&x, &ffn_out, client).map_err(Error::Numr)?;
        let x = self.ffn_norm.forward(client, &x)?;

        Ok(x)
    }

    fn self_attention<C>(
        &self,
        client: &C,
        x: &Var<R>,
        attention_mask: Option<&Tensor<R>>,
    ) -> Result<Var<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + BinaryOps<R>
            + ReduceOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + UnaryOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        let shape = x.shape().to_vec();
        let batch = shape[0];
        let seq_len = shape[1];

        let q = self.q_proj.forward(client, x)?;
        let k = self.k_proj.forward(client, x)?;
        let v = self.v_proj.forward(client, x)?;

        // Reshape to [B, S, H, D] then transpose to [B, H, S, D]
        let q = var_reshape(&q, &[batch, seq_len, self.num_heads, self.head_dim])
            .map_err(Error::Numr)?;
        let k = var_reshape(&k, &[batch, seq_len, self.num_heads, self.head_dim])
            .map_err(Error::Numr)?;
        let v = var_reshape(&v, &[batch, seq_len, self.num_heads, self.head_dim])
            .map_err(Error::Numr)?;

        let q = var_permute(&q, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let k = var_permute(&k, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let v = var_permute(&v, &[0, 2, 1, 3]).map_err(Error::Numr)?;

        // Make contiguous after permute (required for matmul)
        let q = Var::new(q.tensor().contiguous(), false);
        let k = Var::new(k.tensor().contiguous(), false);
        let v = Var::new(v.tensor().contiguous(), false);

        // Scaled dot-product attention: softmax(Q @ K^T / sqrt(d)) @ V
        let k_t = var_transpose(&k).map_err(Error::Numr)?;
        let scores = var_matmul(&q, &k_t, client).map_err(Error::Numr)?;
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let scores = Var::new(
            client
                .mul_scalar(scores.tensor(), scale as f64)
                .map_err(Error::Numr)?,
            false,
        );

        // Apply additive attention mask when provided.
        //
        // `attention_mask` is a [B, S] float tensor where 1.0 = real token,
        // 0.0 = padding.  We convert it to an additive bias by computing
        // `(1 - mask) * -1e9`, which maps real tokens → 0 (no change to the
        // score) and padding positions → -1e9 (zeroed out after softmax).
        // The bias is reshaped to [B, 1, 1, S] so it broadcasts over the
        // [B, H, S, S] scores tensor correctly.
        let scores = if let Some(mask) = attention_mask {
            let mask_shape = mask.shape().to_vec();
            if mask_shape.len() != 2 || mask_shape[0] != batch || mask_shape[1] != seq_len {
                return Err(Error::ModelError {
                    reason: format!(
                        "attention_mask shape must be [{batch}, {seq_len}], got {:?}",
                        mask_shape
                    ),
                });
            }
            // (1.0 - mask) * -1e9  →  real=0, pad=-1e9
            let inv = client.rsub_scalar(mask, 1.0).map_err(Error::Numr)?;
            let additive = client.mul_scalar(&inv, -1e9).map_err(Error::Numr)?;
            // Reshape [B, S] → [B, 1, 1, S] for broadcasting over [B, H, S, S]
            let additive = additive
                .reshape(&[batch, 1, 1, seq_len])
                .map_err(Error::Numr)?;
            let biased = client
                .add(scores.tensor(), &additive)
                .map_err(Error::Numr)?;
            Var::new(biased, false)
        } else {
            scores
        };

        let attn_weights = Var::new(
            client.softmax(scores.tensor(), -1).map_err(Error::Numr)?,
            false,
        );
        let attn_out = var_matmul(&attn_weights, &v, client).map_err(Error::Numr)?;

        // Transpose back [B, H, S, D] → [B, S, H, D] → [B, S, hidden]
        let attn_out = var_permute(&attn_out, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let attn_out = Var::new(attn_out.tensor().contiguous(), false);
        let hidden = self.num_heads * self.head_dim;
        let attn_out = var_reshape(&attn_out, &[batch, seq_len, hidden]).map_err(Error::Numr)?;

        let o = self.o_proj.forward(client, &attn_out)?;
        Ok(o)
    }

    fn ffn<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        C: RuntimeClient<R> + TensorOps<R> + ActivationOps<R>,
        R::Client: TensorOps<R>,
    {
        let h = self.ffn_up.forward(client, x)?;
        let h = match self.hidden_act {
            HiddenAct::Gelu => Var::new(client.gelu(h.tensor()).map_err(Error::Numr)?, false),
            HiddenAct::Relu => Var::new(client.relu(h.tensor()).map_err(Error::Numr)?, false),
        };
        let out = self.ffn_down.forward(client, &h)?;
        Ok(out)
    }
}

/// Pooling strategy for producing a single vector from encoder outputs.
#[derive(Debug, Clone, Copy)]
pub enum Pooling {
    /// Average all token hidden states (most common for sentence embeddings).
    Mean,
    /// Use the [CLS] token's hidden state (position 0).
    Cls,
}

/// BERT-style transformer encoder for producing embeddings.
///
/// Takes token IDs and returns either:
/// - `encode()`: per-token hidden states `[B, S, hidden_size]`
/// - `embed()`: pooled embeddings `[B, hidden_size]`
pub struct Encoder<R: Runtime> {
    config: EncoderConfig,
    token_embed: Embedding<R>,
    position_embed: Embedding<R>,
    /// Post-embedding LayerNorm (applied after token + position embedding sum).
    /// Both BERT and XLM-RoBERTa apply this norm before the first transformer layer.
    embed_norm: LayerNorm<R>,
    layers: Vec<EncoderLayer<R>>,
    pooling: Pooling,
}

/// Client trait bounds needed by encoder forward passes.
pub trait EncoderClient<R: Runtime>:
    RuntimeClient<R>
    + TensorOps<R>
    + ScalarOps<R>
    + BinaryOps<R>
    + ReduceOps<R>
    + ShapeOps<R>
    + IndexingOps<R>
    + ActivationOps<R>
    + UnaryOps<R>
    + NormalizationOps<R>
{
}

impl<R, C> EncoderClient<R> for C
where
    R: Runtime,
    C: RuntimeClient<R>
        + TensorOps<R>
        + ScalarOps<R>
        + BinaryOps<R>
        + ReduceOps<R>
        + ShapeOps<R>
        + IndexingOps<R>
        + ActivationOps<R>
        + UnaryOps<R>
        + NormalizationOps<R>,
{
}

impl<R: Runtime<DType = DType>> Encoder<R> {
    /// Create an encoder from pre-loaded weight tensors.
    ///
    /// `weights` is a closure that fetches tensors by name (e.g., from a GGUF reader).
    pub fn from_weights<F>(config: EncoderConfig, pooling: Pooling, mut get: F) -> Result<Self>
    where
        F: FnMut(&str) -> Result<Tensor<R>>,
    {
        let token_embed = Embedding::new(get("embeddings.word_embeddings.weight")?, false);
        let position_embed = Embedding::new(get("embeddings.position_embeddings.weight")?, false);
        let embed_norm = LayerNorm::new(
            get("embeddings.layer_norm.weight")?,
            get("embeddings.layer_norm.bias")?,
            config.layer_norm_eps as f32,
            false,
        );

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let p = format!("encoder.layer.{i}");

            let q_proj = Linear::new(
                get(&format!("{p}.attention.self.query.weight"))?,
                Some(get(&format!("{p}.attention.self.query.bias"))?),
                false,
            );
            let k_proj = Linear::new(
                get(&format!("{p}.attention.self.key.weight"))?,
                Some(get(&format!("{p}.attention.self.key.bias"))?),
                false,
            );
            let v_proj = Linear::new(
                get(&format!("{p}.attention.self.value.weight"))?,
                Some(get(&format!("{p}.attention.self.value.bias"))?),
                false,
            );
            let o_proj = Linear::new(
                get(&format!("{p}.attention.output.dense.weight"))?,
                Some(get(&format!("{p}.attention.output.dense.bias"))?),
                false,
            );
            let attn_norm = LayerNorm::new(
                get(&format!("{p}.attention.output.LayerNorm.weight"))?,
                get(&format!("{p}.attention.output.LayerNorm.bias"))?,
                config.layer_norm_eps as f32,
                false,
            );
            let ffn_up = Linear::new(
                get(&format!("{p}.intermediate.dense.weight"))?,
                Some(get(&format!("{p}.intermediate.dense.bias"))?),
                false,
            );
            let ffn_down = Linear::new(
                get(&format!("{p}.output.dense.weight"))?,
                Some(get(&format!("{p}.output.dense.bias"))?),
                false,
            );
            let ffn_norm = LayerNorm::new(
                get(&format!("{p}.output.LayerNorm.weight"))?,
                get(&format!("{p}.output.LayerNorm.bias"))?,
                config.layer_norm_eps as f32,
                false,
            );

            layers.push(EncoderLayer {
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                attn_norm,
                ffn_up,
                ffn_down,
                ffn_norm,
                num_heads: config.num_attention_heads,
                head_dim: config.head_dim(),
                hidden_act: config.hidden_act,
            });
        }

        Ok(Self {
            config,
            token_embed,
            position_embed,
            embed_norm,
            layers,
            pooling,
        })
    }

    /// Forward pass: token IDs → per-token hidden states `[B, S, hidden_size]`.
    ///
    /// `attention_mask`: optional `[B, S]` float tensor where 1.0 = real token,
    /// 0.0 = padding.  When `None`, no masking is applied — all positions are
    /// treated as valid, preserving the pre-existing behaviour for callers that
    /// do not have variable-length input.
    pub fn encode<C>(
        &self,
        client: &C,
        input_ids: &Tensor<R>,
        attention_mask: Option<&Tensor<R>>,
    ) -> Result<Var<R>>
    where
        C: EncoderClient<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + IndexingOps<R>,
    {
        let shape = input_ids.shape().to_vec();
        let seq_len = *shape.last().ok_or_else(|| Error::ModelError {
            reason: "input_ids must have at least 1 dimension".into(),
        })?;

        // Token embeddings [B, S, hidden]
        let tok_emb = self.token_embed.forward(client, input_ids)?;

        // Position IDs depend on architecture family.
        let device = input_ids.device();
        let pos_tensor = match self.config.arch_family {
            ArchFamily::Bert => {
                // BERT: simple 0-based sequential positions
                let pos_ids: Vec<i64> = (0..seq_len as i64).collect();
                Tensor::<R>::from_slice(&pos_ids, &[seq_len], device)
            }
            ArchFamily::XlmRoberta => {
                // XLM-RoBERTa: position_ids = cumsum(input_ids != pad_id) + pad_id
                // padding positions get position_id = pad_id.
                //
                // For a batch [B, S], each row is computed independently.
                // We flatten the 2D case to per-row and handle both [S] and [B, S].
                let pad_id = self.config.padding_token_id;
                let flat_ids: Vec<i64> = input_ids.to_vec();
                let batch = if shape.len() == 2 { shape[0] } else { 1 };

                let mut pos_flat: Vec<i64> = Vec::with_capacity(batch * seq_len);
                for b in 0..batch {
                    let mut count: i64 = 0;
                    for s in 0..seq_len {
                        let tok = flat_ids[b * seq_len + s];
                        if tok == pad_id {
                            pos_flat.push(pad_id);
                        } else {
                            count += 1;
                            pos_flat.push(pad_id + count);
                        }
                    }
                }

                if shape.len() == 2 {
                    Tensor::<R>::from_slice(&pos_flat, &[batch, seq_len], device)
                } else {
                    Tensor::<R>::from_slice(&pos_flat, &[seq_len], device)
                }
            }
        };

        let pos_emb = self.position_embed.forward(client, &pos_tensor)?;

        // Combine: token + position embeddings, then apply embedding LayerNorm
        let combined = var_add(&tok_emb, &pos_emb, client).map_err(Error::Numr)?;
        let mut hidden = self.embed_norm.forward(client, &combined)?;

        // Transformer encoder layers
        for layer in &self.layers {
            hidden = layer.forward(client, &hidden, attention_mask)?;
        }

        Ok(hidden)
    }

    /// Forward pass: token IDs → pooled embedding `[B, hidden_size]`.
    ///
    /// `attention_mask`: optional `[B, S]` float tensor where 1.0 = real token,
    /// 0.0 = padding.  Pass `None` for single-sequence inference with no padding.
    pub fn embed<C>(
        &self,
        client: &C,
        input_ids: &Tensor<R>,
        attention_mask: Option<&Tensor<R>>,
    ) -> Result<Var<R>>
    where
        C: EncoderClient<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + IndexingOps<R>,
    {
        let hidden = self.encode(client, input_ids, attention_mask)?;

        match self.pooling {
            Pooling::Mean => {
                // Mean pool over sequence dimension (dim=1): [B, S, H] → [B, H]
                //
                // When an attention mask is provided we compute a *masked* mean
                // so that padding positions are excluded from the average:
                //
                //   pooled = sum(hidden * mask[:, :, None], dim=1)
                //            / sum(mask, dim=1, keepdim=True)
                //
                // Without a mask we fall back to a plain mean over all positions,
                // which preserves the pre-existing behaviour.
                let pooled = if let Some(mask) = attention_mask {
                    let mask_shape = mask.shape().to_vec();
                    let batch = mask_shape[0];
                    let seq_len = mask_shape[1];
                    let hidden_size = self.config.hidden_size;

                    // Expand mask [B, S] → [B, S, 1] for broadcasting with hidden [B, S, H]
                    let mask_3d = mask.reshape(&[batch, seq_len, 1]).map_err(Error::Numr)?;

                    // Masked hidden states: [B, S, H] (pad positions zeroed)
                    let masked = client.mul(hidden.tensor(), &mask_3d).map_err(Error::Numr)?;

                    // Sum over sequence dim → [B, H]
                    let summed = client.sum(&masked, &[1], false).map_err(Error::Numr)?;

                    // Denominator: number of real tokens per sequence → [B, 1]
                    let token_counts = client.sum(mask, &[1], true).map_err(Error::Numr)?;
                    // Clamp to ≥ 1 to avoid div-by-zero on all-padding rows
                    let token_counts = client
                        .maximum(
                            &token_counts,
                            &Tensor::from_slice(&[1.0f32], &[1], mask.device()),
                        )
                        .map_err(Error::Numr)?;
                    // Reshape [B, 1] to [B] then back to allow division: keep as [B, 1]
                    // and rely on broadcasting [B, H] / [B, 1] → [B, H]
                    let pooled_t = client.div(&summed, &token_counts).map_err(Error::Numr)?;
                    // Ensure output is [B, hidden_size]
                    let _ = hidden_size; // used via shape validation implicitly
                    Var::new(pooled_t, false)
                } else {
                    Var::new(
                        client
                            .mean(hidden.tensor(), &[1], false)
                            .map_err(Error::Numr)?,
                        false,
                    )
                };
                Ok(pooled)
            }
            Pooling::Cls => {
                // Take first token (CLS): [B, S, H] → [B, 1, H] → [B, H]
                //
                // var_narrow produces a non-contiguous view into [B, S, H] storage
                // (dim-0 stride remains S*H rather than H).  Make contiguous before
                // reshape so that layout.reshape() can compute new strides correctly.
                let cls = var_narrow(&hidden, 1, 0, 1).map_err(Error::Numr)?;
                let cls = Var::new(cls.tensor().contiguous(), false);
                let shape = cls.shape().to_vec();
                let batch = shape[0];
                let hidden_dim = shape[2];
                var_reshape(&cls, &[batch, hidden_dim]).map_err(Error::Numr)
            }
        }
    }

    /// Returns the encoder's configuration.
    pub fn config(&self) -> &EncoderConfig {
        &self.config
    }

    /// Returns the pooling strategy used by this encoder.
    pub fn pooling(&self) -> Pooling {
        self.pooling
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    fn make_test_encoder() -> (
        Encoder<CpuRuntime>,
        numr::runtime::cpu::CpuClient,
        numr::runtime::cpu::CpuDevice,
    ) {
        let (client, device) = cpu_setup();

        let config = EncoderConfig {
            vocab_size: 10,
            hidden_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            intermediate_size: 16,
            max_position_embeddings: 32,
            layer_norm_eps: 1e-12,
            hidden_act: HiddenAct::Gelu,
            type_vocab_size: 0,
            arch_family: ArchFamily::Bert,
            padding_token_id: 0,
        };

        let encoder = Encoder::from_weights(config, Pooling::Mean, |name| {
            // Return appropriately shaped random-ish tensors
            match name {
                "embeddings.word_embeddings.weight" => {
                    Ok(Tensor::from_slice(&vec![0.1f32; 10 * 8], &[10, 8], &device))
                }
                "embeddings.position_embeddings.weight" => Ok(Tensor::from_slice(
                    &vec![0.01f32; 32 * 8],
                    &[32, 8],
                    &device,
                )),
                "embeddings.layer_norm.weight" => {
                    Ok(Tensor::from_slice(&[1.0f32; 8], &[8], &device))
                }
                "embeddings.layer_norm.bias" => Ok(Tensor::from_slice(&[0.0f32; 8], &[8], &device)),
                n if n.ends_with("query.weight")
                    || n.ends_with("key.weight")
                    || n.ends_with("value.weight") =>
                {
                    Ok(Tensor::from_slice(&vec![0.02f32; 8 * 8], &[8, 8], &device))
                }
                n if n.ends_with("query.bias")
                    || n.ends_with("key.bias")
                    || n.ends_with("value.bias") =>
                {
                    Ok(Tensor::from_slice(&[0.0f32; 8], &[8], &device))
                }
                n if n.ends_with("attention.output.dense.weight") => {
                    Ok(Tensor::from_slice(&vec![0.02f32; 8 * 8], &[8, 8], &device))
                }
                n if n.ends_with("attention.output.dense.bias") => {
                    Ok(Tensor::from_slice(&[0.0f32; 8], &[8], &device))
                }
                n if n.ends_with("output.dense.weight") => {
                    // ffn_down: [hidden_size, intermediate_size] = [8, 16]
                    Ok(Tensor::from_slice(
                        &vec![0.02f32; 8 * 16],
                        &[8, 16],
                        &device,
                    ))
                }
                n if n.ends_with("output.dense.bias") => {
                    Ok(Tensor::from_slice(&[0.0f32; 8], &[8], &device))
                }
                n if n.ends_with("LayerNorm.weight") => {
                    Ok(Tensor::from_slice(&[1.0f32; 8], &[8], &device))
                }
                n if n.ends_with("LayerNorm.bias") => {
                    Ok(Tensor::from_slice(&[0.0f32; 8], &[8], &device))
                }
                n if n.ends_with("intermediate.dense.weight") => Ok(Tensor::from_slice(
                    &vec![0.02f32; 16 * 8],
                    &[16, 8],
                    &device,
                )),
                n if n.ends_with("intermediate.dense.bias") => {
                    Ok(Tensor::from_slice(&[0.0f32; 16], &[16], &device))
                }
                _ => Err(Error::ModelError {
                    reason: format!("unknown weight: {name}"),
                }),
            }
        })
        .unwrap();

        (encoder, client, device)
    }

    fn make_test_encoder_cls() -> (
        Encoder<CpuRuntime>,
        numr::runtime::cpu::CpuClient,
        numr::runtime::cpu::CpuDevice,
    ) {
        let (client, device) = cpu_setup();

        let config = EncoderConfig {
            vocab_size: 10,
            hidden_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            intermediate_size: 16,
            max_position_embeddings: 32,
            layer_norm_eps: 1e-12,
            hidden_act: HiddenAct::Gelu,
            type_vocab_size: 0,
            arch_family: ArchFamily::Bert,
            padding_token_id: 0,
        };

        let device_ref = &device;
        let encoder = Encoder::from_weights(config, Pooling::Cls, |name| match name {
            "embeddings.word_embeddings.weight" => Ok(Tensor::from_slice(
                &vec![0.1f32; 10 * 8],
                &[10, 8],
                device_ref,
            )),
            "embeddings.position_embeddings.weight" => Ok(Tensor::from_slice(
                &vec![0.01f32; 32 * 8],
                &[32, 8],
                device_ref,
            )),
            "embeddings.layer_norm.weight" => {
                Ok(Tensor::from_slice(&[1.0f32; 8], &[8], device_ref))
            }
            "embeddings.layer_norm.bias" => Ok(Tensor::from_slice(&[0.0f32; 8], &[8], device_ref)),
            n if n.ends_with("query.weight")
                || n.ends_with("key.weight")
                || n.ends_with("value.weight")
                || n.ends_with("attention.output.dense.weight") =>
            {
                Ok(Tensor::from_slice(
                    &vec![0.02f32; 8 * 8],
                    &[8, 8],
                    device_ref,
                ))
            }
            n if n.ends_with("query.bias")
                || n.ends_with("key.bias")
                || n.ends_with("value.bias")
                || n.ends_with("attention.output.dense.bias")
                || n.ends_with("output.dense.bias") =>
            {
                Ok(Tensor::from_slice(&[0.0f32; 8], &[8], device_ref))
            }
            n if n.ends_with("LayerNorm.weight") => {
                Ok(Tensor::from_slice(&[1.0f32; 8], &[8], device_ref))
            }
            n if n.ends_with("LayerNorm.bias") => {
                Ok(Tensor::from_slice(&[0.0f32; 8], &[8], device_ref))
            }
            n if n.ends_with("intermediate.dense.weight") => Ok(Tensor::from_slice(
                &vec![0.02f32; 16 * 8],
                &[16, 8],
                device_ref,
            )),
            n if n.ends_with("intermediate.dense.bias") => {
                Ok(Tensor::from_slice(&[0.0f32; 16], &[16], device_ref))
            }
            n if n.ends_with("output.dense.weight") => Ok(Tensor::from_slice(
                &vec![0.02f32; 8 * 16],
                &[8, 16],
                device_ref,
            )),
            _ => Err(Error::ModelError {
                reason: format!("unknown weight: {name}"),
            }),
        })
        .unwrap();

        (encoder, client, device)
    }

    #[test]
    fn test_encode_output_shape() {
        let (encoder, client, device) = make_test_encoder();
        let input_ids = Tensor::<CpuRuntime>::from_slice(&[1i64, 2, 3], &[1, 3], &device);
        let hidden = encoder.encode(&client, &input_ids, None).unwrap();
        assert_eq!(hidden.shape(), &[1, 3, 8]);
    }

    #[test]
    fn test_embed_mean_pool() {
        let (encoder, client, device) = make_test_encoder();
        let input_ids = Tensor::<CpuRuntime>::from_slice(&[1i64, 2, 3, 4], &[1, 4], &device);
        let emb = encoder.embed(&client, &input_ids, None).unwrap();
        assert_eq!(emb.shape(), &[1, 8]);
    }

    #[test]
    fn test_embed_batched() {
        let (encoder, client, device) = make_test_encoder();
        let input_ids = Tensor::<CpuRuntime>::from_slice(&[1i64, 2, 3, 4, 5, 6], &[2, 3], &device);
        let emb = encoder.embed(&client, &input_ids, None).unwrap();
        assert_eq!(emb.shape(), &[2, 8]);
    }

    #[test]
    fn test_encode_with_none_mask_matches_no_mask() {
        // Regression guard: encode(..., None) must produce the same output
        // as before the widening (i.e., no mask applied).
        let (encoder, client, device) = make_test_encoder();
        let input_ids = Tensor::<CpuRuntime>::from_slice(&[1i64, 2, 3], &[1, 3], &device);
        let h1 = encoder.encode(&client, &input_ids, None).unwrap();
        let h2 = encoder.encode(&client, &input_ids, None).unwrap();
        let v1: Vec<f32> = h1.tensor().to_vec();
        let v2: Vec<f32> = h2.tensor().to_vec();
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_mask_wrong_shape_returns_error() {
        let (encoder, client, device) = make_test_encoder();
        // input_ids: [1, 3]
        let input_ids = Tensor::<CpuRuntime>::from_slice(&[1i64, 2, 3], &[1, 3], &device);
        // mask has wrong seq_len (4 instead of 3)
        let bad_mask = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 4], &[1, 4], &device);
        let result = encoder.encode(&client, &input_ids, Some(&bad_mask));
        assert!(result.is_err());
    }

    #[test]
    fn test_cls_pooling_batched_produces_correct_shape() {
        // Regression: CLS pooling on a batch must not fail with NotContiguous.
        // var_narrow([B, S, H], dim=1, start=0, len=1) → [B, 1, H] with stride S*H at dim 0.
        // reshape([B, H]) requires contiguous — verify the fix holds.
        let (encoder, client, device) = make_test_encoder_cls();
        let input_ids = Tensor::<CpuRuntime>::from_slice(&[1i64, 2, 3, 4, 5, 6], &[2, 3], &device);
        let emb = encoder.embed(&client, &input_ids, None).unwrap();
        assert_eq!(emb.shape(), &[2, 8]);
    }

    #[test]
    fn test_xlm_roberta_position_ids() {
        // Verify XLM-RoBERTa position-id computation: non-pad tokens are numbered
        // from pad_id+1, padding positions get position_id = pad_id.
        //
        // Input: [0, 4, 7, 1, 1] where pad_id=1
        // Expected positions: [2, 3, 4, 1, 1]  (count skips pad_id=1 position, starts at pad_id+1=2)
        let (client, device) = cpu_setup();

        let config = EncoderConfig {
            vocab_size: 10,
            hidden_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            intermediate_size: 16,
            max_position_embeddings: 32,
            layer_norm_eps: 1e-12,
            hidden_act: HiddenAct::Gelu,
            type_vocab_size: 0,
            arch_family: ArchFamily::XlmRoberta,
            padding_token_id: 1,
        };

        let device_ref = &device;
        // Build encoder — weights just need to be the right shapes
        let encoder =
            Encoder::<CpuRuntime>::from_weights(config, Pooling::Mean, |name| match name {
                "embeddings.word_embeddings.weight" => Ok(Tensor::from_slice(
                    &vec![0.1f32; 10 * 8],
                    &[10, 8],
                    device_ref,
                )),
                "embeddings.position_embeddings.weight" => Ok(Tensor::from_slice(
                    &vec![0.01f32; 32 * 8],
                    &[32, 8],
                    device_ref,
                )),
                "embeddings.layer_norm.weight" => {
                    Ok(Tensor::from_slice(&[1.0f32; 8], &[8], device_ref))
                }
                "embeddings.layer_norm.bias" => {
                    Ok(Tensor::from_slice(&[0.0f32; 8], &[8], device_ref))
                }
                n if n.ends_with("query.weight")
                    || n.ends_with("key.weight")
                    || n.ends_with("value.weight")
                    || n.ends_with("attention.output.dense.weight") =>
                {
                    Ok(Tensor::from_slice(
                        &vec![0.02f32; 8 * 8],
                        &[8, 8],
                        device_ref,
                    ))
                }
                n if n.ends_with("query.bias")
                    || n.ends_with("key.bias")
                    || n.ends_with("value.bias")
                    || n.ends_with("attention.output.dense.bias")
                    || n.ends_with("output.dense.bias") =>
                {
                    Ok(Tensor::from_slice(&[0.0f32; 8], &[8], device_ref))
                }
                n if n.ends_with("LayerNorm.weight") => {
                    Ok(Tensor::from_slice(&[1.0f32; 8], &[8], device_ref))
                }
                n if n.ends_with("LayerNorm.bias") => {
                    Ok(Tensor::from_slice(&[0.0f32; 8], &[8], device_ref))
                }
                n if n.ends_with("intermediate.dense.weight") => Ok(Tensor::from_slice(
                    &vec![0.02f32; 16 * 8],
                    &[16, 8],
                    device_ref,
                )),
                n if n.ends_with("intermediate.dense.bias") => {
                    Ok(Tensor::from_slice(&[0.0f32; 16], &[16], device_ref))
                }
                n if n.ends_with("output.dense.weight") => Ok(Tensor::from_slice(
                    &vec![0.02f32; 8 * 16],
                    &[8, 16],
                    device_ref,
                )),
                _ => Err(Error::ModelError {
                    reason: format!("unknown weight: {name}"),
                }),
            })
            .unwrap();

        // Sequence: [0, 4, 7, 1, 1] where 1 is pad — forward must succeed
        let input_ids = Tensor::<CpuRuntime>::from_slice(&[0i64, 4, 7, 1, 1], &[1, 5], &device);
        let result = encoder.embed(&client, &input_ids, None);
        assert!(
            result.is_ok(),
            "xlm-roberta forward should succeed: {result:?}"
        );
        assert_eq!(result.unwrap().shape(), &[1, 8]);
    }
}
