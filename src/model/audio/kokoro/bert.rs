//! ALBERT backbone for Kokoro's phoneme encoder.
//!
//! Kokoro wraps HuggingFace's `AlbertModel` (Lan et al. 2019) with a single
//! `Linear(768 → 512)` projection called `bert_encoder`. Distinctive ALBERT
//! features preserved here:
//!
//! * **Factorized embeddings** — a narrow `embedding_size=128` table projected
//!   up to `hidden_size=768` via a single `embedding_hidden_mapping_in` linear
//!   at the encoder boundary.
//! * **Cross-layer weight sharing** — all `num_hidden_layers=12` transformer
//!   layers are a single `albert_layer_groups[0].albert_layers[0]` block
//!   applied 12 times in a loop. State-dict contains exactly one layer's
//!   worth of parameters under that path.
//!
//! Inference-only (no masking yet — Kokoro runs one utterance at a time with
//! no padding). When we need to serve batched heterogeneous lengths, add
//! `attention_mask` support via an additive `-inf` pre-softmax mask.

use crate::error::{Error, Result};
use crate::nn::Embedding;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, IndexingOps, MatmulOps, NormalizationOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, TypeConversionOps, UnaryOps, UtilityOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Geometry for one ALBERT instance. Defaults match Kokoro-82M / ALBERT-base.
#[derive(Debug, Clone, Copy)]
pub struct AlbertConfig {
    pub hidden_size: usize,
    pub embedding_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
    pub type_vocab_size: usize,
    pub layer_norm_eps: f32,
}

impl AlbertConfig {
    /// Per-head width = `hidden_size / num_attention_heads`.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads.max(1)
    }
}

/// ALBERT input embeddings: word + position + token-type → LayerNorm.
pub struct AlbertEmbeddings<R: Runtime> {
    word_embeddings: Embedding<R>,
    position_embeddings: Embedding<R>,
    token_type_embeddings: Embedding<R>,
    ln_weight: Tensor<R>,
    ln_bias: Tensor<R>,
    eps: f32,
    max_positions: usize,
}

impl<R: Runtime> AlbertEmbeddings<R> {
    pub fn new(
        word_embeddings: Embedding<R>,
        position_embeddings: Embedding<R>,
        token_type_embeddings: Embedding<R>,
        ln_weight: Tensor<R>,
        ln_bias: Tensor<R>,
        eps: f32,
        max_positions: usize,
    ) -> Self {
        Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            ln_weight,
            ln_bias,
            eps,
            max_positions,
        }
    }

    pub fn forward<C>(&self, client: &C, token_ids: &Tensor<R>) -> Result<Tensor<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R>
            + IndexingOps<R>
            + NormalizationOps<R>
            + BinaryOps<R>
            + UtilityOps<R>
            + TensorOps<R>,
        R::Client: IndexingOps<R>,
    {
        let shape = token_ids.shape();
        if shape.len() != 2 {
            return Err(Error::InvalidArgument {
                arg: "token_ids",
                reason: format!("expected [B, T], got {shape:?}"),
            });
        }
        let (b, t) = (shape[0], shape[1]);
        if t > self.max_positions {
            return Err(Error::InvalidArgument {
                arg: "token_ids",
                reason: format!(
                    "sequence length {t} exceeds ALBERT's max_position_embeddings {}",
                    self.max_positions
                ),
            });
        }

        let word = self.word_embeddings.forward(client, token_ids)?;
        // Positions: 0, 1, …, T-1 broadcast across batch.
        let positions_1d = client
            .arange(0.0, t as f64, 1.0, DType::I64)
            .map_err(Error::Numr)?;
        let positions = positions_1d
            .reshape(&[1, t])
            .map_err(Error::Numr)?
            .broadcast_to(&[b, t])
            .map_err(Error::Numr)?
            .contiguous()?;
        let pos_emb = self.position_embeddings.forward(client, &positions)?;
        // Token types: all zeros.
        let type_ids = client.fill(&[b, t], 0.0, DType::I64).map_err(Error::Numr)?;
        let type_emb = self.token_type_embeddings.forward(client, &type_ids)?;

        // Sum the three embeddings, then LayerNorm over the last axis.
        let sum1 = client
            .add(word.tensor(), pos_emb.tensor())
            .map_err(Error::Numr)?;
        let summed = client.add(&sum1, type_emb.tensor()).map_err(Error::Numr)?;
        client
            .layer_norm(&summed, &self.ln_weight, &self.ln_bias, self.eps)
            .map_err(Error::Numr)
    }
}

/// One ALBERT transformer layer (reused 12 times by `AlbertModel`).
pub struct AlbertLayer<R: Runtime> {
    // Attention
    pub q_weight: Tensor<R>,
    pub q_bias: Tensor<R>,
    pub k_weight: Tensor<R>,
    pub k_bias: Tensor<R>,
    pub v_weight: Tensor<R>,
    pub v_bias: Tensor<R>,
    pub attn_dense_weight: Tensor<R>,
    pub attn_dense_bias: Tensor<R>,
    pub attn_ln_weight: Tensor<R>,
    pub attn_ln_bias: Tensor<R>,
    // FFN
    pub ffn_weight: Tensor<R>,
    pub ffn_bias: Tensor<R>,
    pub ffn_output_weight: Tensor<R>,
    pub ffn_output_bias: Tensor<R>,
    pub full_ln_weight: Tensor<R>,
    pub full_ln_bias: Tensor<R>,
}

impl<R: Runtime> AlbertLayer<R> {
    /// Forward: `x [B, T, H]` → `[B, T, H]`.
    pub fn forward<C>(&self, client: &C, x: &Tensor<R>, config: &AlbertConfig) -> Result<Tensor<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R>
            + MatmulOps<R>
            + NormalizationOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + ScalarOps<R>
            + TensorOps<R>
            + ReduceOps<R>
            + UnaryOps<R>
            + ShapeOps<R>
            + TypeConversionOps<R>
            + UtilityOps<R>,
    {
        let shape = x.shape();
        let (b, t, h) = (shape[0], shape[1], shape[2]);
        let n_heads = config.num_attention_heads;
        let d_head = config.head_dim();

        // Flatten batch-time for matmul.
        let flat = x.reshape(&[b * t, h]).map_err(Error::Numr)?;
        let q = linear(client, &flat, &self.q_weight, &self.q_bias)?;
        let k = linear(client, &flat, &self.k_weight, &self.k_bias)?;
        let v = linear(client, &flat, &self.v_weight, &self.v_bias)?;

        // Reshape to multi-head: [B*T, H] → [B, T, n_heads, d_head] → [B, n_heads, T, d_head].
        let q = q
            .reshape(&[b, t, n_heads, d_head])
            .map_err(Error::Numr)?
            .transpose(1, 2)
            .map_err(Error::Numr)?
            .contiguous()?;
        let k = k
            .reshape(&[b, t, n_heads, d_head])
            .map_err(Error::Numr)?
            .transpose(1, 2)
            .map_err(Error::Numr)?
            .contiguous()?;
        let v = v
            .reshape(&[b, t, n_heads, d_head])
            .map_err(Error::Numr)?
            .transpose(1, 2)
            .map_err(Error::Numr)?
            .contiguous()?;

        // Attention scores: q @ k.T / sqrt(d_head). k.T swaps the last two dims.
        let k_t = k.transpose(2, 3).map_err(Error::Numr)?.contiguous()?;
        let scores = client.matmul(&q, &k_t).map_err(Error::Numr)?;
        let scale = 1.0 / (d_head as f64).sqrt();
        let scaled = client.mul_scalar(&scores, scale).map_err(Error::Numr)?;
        let attn = client.softmax(&scaled, -1).map_err(Error::Numr)?;
        // attn: [B, H, T, T]; v: [B, H, T, d_head] → ctx: [B, H, T, d_head].
        let ctx = client.matmul(&attn, &v).map_err(Error::Numr)?;

        // [B, H, T, d_head] → [B, T, H, d_head] → [B, T, hidden].
        let ctx_merged = ctx
            .transpose(1, 2)
            .map_err(Error::Numr)?
            .contiguous()?
            .reshape(&[b, t, h])
            .map_err(Error::Numr)?;

        // Attention output projection + residual + LayerNorm.
        let ctx_flat = ctx_merged.reshape(&[b * t, h]).map_err(Error::Numr)?;
        let attn_out = linear(
            client,
            &ctx_flat,
            &self.attn_dense_weight,
            &self.attn_dense_bias,
        )?;
        let attn_out_shaped = attn_out.reshape(&[b, t, h]).map_err(Error::Numr)?;
        let residual1 = client.add(&attn_out_shaped, x).map_err(Error::Numr)?;
        let post_attn = client
            .layer_norm(
                &residual1,
                &self.attn_ln_weight,
                &self.attn_ln_bias,
                config.layer_norm_eps,
            )
            .map_err(Error::Numr)?;

        // FFN: Linear → GELU → Linear → residual → LayerNorm.
        let post_attn_flat = post_attn.reshape(&[b * t, h]).map_err(Error::Numr)?;
        let ffn1 = linear(client, &post_attn_flat, &self.ffn_weight, &self.ffn_bias)?;
        let ffn1_gelu = client.gelu(&ffn1).map_err(Error::Numr)?;
        let ffn2 = linear(
            client,
            &ffn1_gelu,
            &self.ffn_output_weight,
            &self.ffn_output_bias,
        )?;
        let ffn2_shaped = ffn2.reshape(&[b, t, h]).map_err(Error::Numr)?;
        let residual2 = client.add(&ffn2_shaped, &post_attn).map_err(Error::Numr)?;
        client
            .layer_norm(
                &residual2,
                &self.full_ln_weight,
                &self.full_ln_bias,
                config.layer_norm_eps,
            )
            .map_err(Error::Numr)
    }
}

/// Linear helper: `input @ weight.T + bias` for `input [N, in], weight [out, in]`.
fn linear<R, C>(
    client: &C,
    input: &Tensor<R>,
    weight: &Tensor<R>,
    bias: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + MatmulOps<R> + TensorOps<R>,
{
    let w_t = weight.transpose(0, 1).map_err(Error::Numr)?;
    client.matmul_bias(input, &w_t, bias).map_err(Error::Numr)
}

/// Full ALBERT model: embeddings → projection → 12× shared layer → returns
/// `[B, T, hidden_size]`. Pooler output intentionally not exposed — Kokoro
/// doesn't use it downstream.
pub struct AlbertModel<R: Runtime> {
    pub embeddings: AlbertEmbeddings<R>,
    /// Linear(embedding_size, hidden_size). Stored as raw tensors since we
    /// don't need autograd here.
    pub embedding_projection_weight: Tensor<R>,
    pub embedding_projection_bias: Tensor<R>,
    pub shared_layer: AlbertLayer<R>,
    pub config: AlbertConfig,
}

impl<R: Runtime> AlbertModel<R> {
    /// Forward: `token_ids [B, T]` → `[B, T, hidden_size]`.
    pub fn forward<C>(&self, client: &C, token_ids: &Tensor<R>) -> Result<Tensor<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R>
            + IndexingOps<R>
            + MatmulOps<R>
            + NormalizationOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + ScalarOps<R>
            + TensorOps<R>
            + ReduceOps<R>
            + UnaryOps<R>
            + ShapeOps<R>
            + TypeConversionOps<R>
            + UtilityOps<R>,
        R::Client: IndexingOps<R>,
    {
        let emb = self.embeddings.forward(client, token_ids)?; // [B, T, embedding_size]
        let shape = emb.shape();
        let (b, t) = (shape[0], shape[1]);
        let emb_flat = emb
            .reshape(&[b * t, self.config.embedding_size])
            .map_err(Error::Numr)?;
        let projected = linear(
            client,
            &emb_flat,
            &self.embedding_projection_weight,
            &self.embedding_projection_bias,
        )?;
        let mut x = projected
            .reshape(&[b, t, self.config.hidden_size])
            .map_err(Error::Numr)?;

        for _ in 0..self.config.num_hidden_layers {
            x = self.shared_layer.forward(client, &x, &self.config)?;
        }
        Ok(x)
    }
}

/// Kokoro's full text backbone: AlbertModel + `bert_encoder` Linear to the
/// main decoder hidden size.
pub struct BertEncoder<R: Runtime> {
    pub albert: AlbertModel<R>,
    /// `Linear(albert.hidden_size, kokoro_hidden_dim)`.
    pub projection_weight: Tensor<R>,
    pub projection_bias: Tensor<R>,
    pub out_dim: usize,
}

impl<R: Runtime> BertEncoder<R> {
    pub fn forward<C>(&self, client: &C, token_ids: &Tensor<R>) -> Result<Tensor<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R>
            + IndexingOps<R>
            + MatmulOps<R>
            + NormalizationOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + ScalarOps<R>
            + TensorOps<R>
            + ReduceOps<R>
            + UnaryOps<R>
            + ShapeOps<R>
            + TypeConversionOps<R>
            + UtilityOps<R>,
        R::Client: IndexingOps<R>,
    {
        let albert_out = self.albert.forward(client, token_ids)?; // [B, T, H]
        let shape = albert_out.shape();
        let (b, t, h) = (shape[0], shape[1], shape[2]);
        let flat = albert_out.reshape(&[b * t, h]).map_err(Error::Numr)?;
        let projected = linear(
            client,
            &flat,
            &self.projection_weight,
            &self.projection_bias,
        )?;
        projected
            .reshape(&[b, t, self.out_dim])
            .map_err(Error::Numr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    fn zeros(shape: &[usize], device: &<CpuRuntime as Runtime>::Device) -> Tensor<CpuRuntime> {
        let n: usize = shape.iter().product();
        Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; n], shape, device)
    }
    fn ones(shape: &[usize], device: &<CpuRuntime as Runtime>::Device) -> Tensor<CpuRuntime> {
        let n: usize = shape.iter().product();
        Tensor::<CpuRuntime>::from_slice(&vec![1.0f32; n], shape, device)
    }

    fn tiny_config() -> AlbertConfig {
        AlbertConfig {
            hidden_size: 4,
            embedding_size: 2,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            intermediate_size: 8,
            max_position_embeddings: 16,
            vocab_size: 10,
            type_vocab_size: 2,
            layer_norm_eps: 1e-5,
        }
    }

    fn build_layer(
        cfg: &AlbertConfig,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> AlbertLayer<CpuRuntime> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        AlbertLayer {
            q_weight: zeros(&[h, h], device),
            q_bias: zeros(&[h], device),
            k_weight: zeros(&[h, h], device),
            k_bias: zeros(&[h], device),
            v_weight: zeros(&[h, h], device),
            v_bias: zeros(&[h], device),
            attn_dense_weight: zeros(&[h, h], device),
            attn_dense_bias: zeros(&[h], device),
            attn_ln_weight: ones(&[h], device),
            attn_ln_bias: zeros(&[h], device),
            ffn_weight: zeros(&[i, h], device),
            ffn_bias: zeros(&[i], device),
            ffn_output_weight: zeros(&[h, i], device),
            ffn_output_bias: zeros(&[h], device),
            full_ln_weight: ones(&[h], device),
            full_ln_bias: zeros(&[h], device),
        }
    }

    fn build_embeddings(
        cfg: &AlbertConfig,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> AlbertEmbeddings<CpuRuntime> {
        AlbertEmbeddings::new(
            Embedding::new(zeros(&[cfg.vocab_size, cfg.embedding_size], device), false),
            Embedding::new(
                zeros(&[cfg.max_position_embeddings, cfg.embedding_size], device),
                false,
            ),
            Embedding::new(
                zeros(&[cfg.type_vocab_size, cfg.embedding_size], device),
                false,
            ),
            ones(&[cfg.embedding_size], device),
            zeros(&[cfg.embedding_size], device),
            cfg.layer_norm_eps,
            cfg.max_position_embeddings,
        )
    }

    #[test]
    fn albert_layer_preserves_shape() {
        let (client, device) = cpu_setup();
        let cfg = tiny_config();
        let layer = build_layer(&cfg, &device);
        let x = zeros(&[1, 5, cfg.hidden_size], &device);
        let y = layer.forward(&client, &x, &cfg).unwrap();
        assert_eq!(y.shape(), &[1, 5, cfg.hidden_size]);
    }

    #[test]
    fn albert_model_output_shape_is_b_t_hidden() {
        let (client, device) = cpu_setup();
        let cfg = tiny_config();
        let model = AlbertModel {
            embeddings: build_embeddings(&cfg, &device),
            embedding_projection_weight: zeros(&[cfg.hidden_size, cfg.embedding_size], &device),
            embedding_projection_bias: zeros(&[cfg.hidden_size], &device),
            shared_layer: build_layer(&cfg, &device),
            config: cfg,
        };
        let ids = Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 2, 3], &[1, 4], &device);
        let out = model.forward(&client, &ids).unwrap();
        assert_eq!(out.shape(), &[1, 4, cfg.hidden_size]);
    }

    #[test]
    fn bert_encoder_projects_to_out_dim() {
        let (client, device) = cpu_setup();
        let cfg = tiny_config();
        let albert = AlbertModel {
            embeddings: build_embeddings(&cfg, &device),
            embedding_projection_weight: zeros(&[cfg.hidden_size, cfg.embedding_size], &device),
            embedding_projection_bias: zeros(&[cfg.hidden_size], &device),
            shared_layer: build_layer(&cfg, &device),
            config: cfg,
        };
        let out_dim = 8;
        let encoder = BertEncoder {
            albert,
            projection_weight: zeros(&[out_dim, cfg.hidden_size], &device),
            projection_bias: zeros(&[out_dim], &device),
            out_dim,
        };
        let ids = Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 2], &[1, 3], &device);
        let out = encoder.forward(&client, &ids).unwrap();
        assert_eq!(out.shape(), &[1, 3, out_dim]);
    }

    #[test]
    fn embeddings_reject_oversized_sequence() {
        let (client, device) = cpu_setup();
        let cfg = tiny_config();
        let emb = build_embeddings(&cfg, &device);
        // max_positions is 16 in tiny_config; give 20.
        let ids_data = [0i64; 20];
        let ids = Tensor::<CpuRuntime>::from_slice(&ids_data, &[1, 20], &device);
        assert!(emb.forward(&client, &ids).is_err());
    }
}
