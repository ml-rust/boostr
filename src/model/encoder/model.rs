//! BERT-style transformer encoder for embedding generation.
//!
//! Architecture: token_embed + position_embed → N × (SelfAttention + FFN + LayerNorm) → mean pool
//!
//! Used by sentence embedding models (all-MiniLM, BGE, nomic-embed, etc.)
//! for producing fixed-size vector representations from token sequences.

use super::config::{EncoderConfig, HiddenAct};
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
    fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
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
        let attn_out = self.self_attention(client, x)?;
        let x = var_add(x, &attn_out, client).map_err(Error::Numr)?;
        let x = self.attn_norm.forward(client, &x)?;

        // FFN with residual
        let ffn_out = self.ffn(client, &x)?;
        let x = var_add(&x, &ffn_out, client).map_err(Error::Numr)?;
        let x = self.ffn_norm.forward(client, &x)?;

        Ok(x)
    }

    fn self_attention<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
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
            layers,
            pooling,
        })
    }

    /// Forward pass: token IDs → per-token hidden states `[B, S, hidden_size]`.
    pub fn encode<C>(&self, client: &C, input_ids: &Tensor<R>) -> Result<Var<R>>
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

        // Position IDs: [0, 1, ..., S-1]
        let device = input_ids.device();
        let pos_ids: Vec<i64> = (0..seq_len as i64).collect();
        let pos_tensor = Tensor::<R>::from_slice(&pos_ids, &[seq_len], device);
        let pos_emb = self.position_embed.forward(client, &pos_tensor)?;

        // Combine: token + position embeddings
        let mut hidden = var_add(&tok_emb, &pos_emb, client).map_err(Error::Numr)?;

        // Transformer encoder layers
        for layer in &self.layers {
            hidden = layer.forward(client, &hidden)?;
        }

        Ok(hidden)
    }

    /// Forward pass: token IDs → pooled embedding `[B, hidden_size]`.
    pub fn embed<C>(&self, client: &C, input_ids: &Tensor<R>) -> Result<Var<R>>
    where
        C: EncoderClient<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + IndexingOps<R>,
    {
        let hidden = self.encode(client, input_ids)?;

        match self.pooling {
            Pooling::Mean => {
                // Mean pool over sequence dimension (dim=1): [B, S, H] → [B, H]
                let pooled = Var::new(
                    client
                        .mean(hidden.tensor(), &[1], false)
                        .map_err(Error::Numr)?,
                    false,
                );
                Ok(pooled)
            }
            Pooling::Cls => {
                // Take first token (CLS): [B, S, H] → [B, 1, H] → [B, H]
                let cls = var_narrow(&hidden, 1, 0, 1).map_err(Error::Numr)?;
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

    #[test]
    fn test_encode_output_shape() {
        let (encoder, client, device) = make_test_encoder();
        let input_ids = Tensor::<CpuRuntime>::from_slice(&[1i64, 2, 3], &[1, 3], &device);
        let hidden = encoder.encode(&client, &input_ids).unwrap();
        assert_eq!(hidden.shape(), &[1, 3, 8]);
    }

    #[test]
    fn test_embed_mean_pool() {
        let (encoder, client, device) = make_test_encoder();
        let input_ids = Tensor::<CpuRuntime>::from_slice(&[1i64, 2, 3, 4], &[1, 4], &device);
        let emb = encoder.embed(&client, &input_ids).unwrap();
        assert_eq!(emb.shape(), &[1, 8]);
    }

    #[test]
    fn test_embed_batched() {
        let (encoder, client, device) = make_test_encoder();
        let input_ids = Tensor::<CpuRuntime>::from_slice(&[1i64, 2, 3, 4, 5, 6], &[2, 3], &device);
        let emb = encoder.embed(&client, &input_ids).unwrap();
        assert_eq!(emb.shape(), &[2, 8]);
    }
}
