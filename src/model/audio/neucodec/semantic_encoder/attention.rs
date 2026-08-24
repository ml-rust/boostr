//! `Wav2Vec2BertSelfAttention` with `position_embeddings_type = "relative_key"`.
//!
//! ```text
//! q,k,v = linear_{q,k,v}(x)            -> [B, T, 16, 64] -> [B, 16, T, 64]
//! scores = (q @ k^T) / sqrt(64)
//! distance = key_pos - query_pos                  # [T, T]
//! distance = clamp(distance, -left_max, +right_max)
//! pos = distance_embedding[distance + left_max]   # [T, T, 64]
//! rel = einsum("bhld,lrd->bhlr", q, pos)          # q ONLY — k is not involved
//! scores = scores + rel / sqrt(64)                # scaled SEPARATELY, then added
//! out = softmax(scores, dim=-1) @ v -> [B, T, 1024] -> linear_out
//! ```
//!
//! ## `relative_key` is NOT the Transformer-XL `relative` variant
//!
//! The tempting wrong move is to treat this as the familiar Transformer-XL /
//! `relative` scheme, where the positional term contributes TWO products — one
//! contracted with `q` and one with `k` — plus learned global content/position
//! biases. `relative_key` has exactly ONE term, contracted with `q` alone, and
//! no bias vectors at all. The checkpoint agrees: `self_attn` carries only
//! `linear_{q,k,v,out}` and a single `distance_embedding.weight`; there is no
//! `linear_pos`, no `pos_bias_u`, no `pos_bias_v`. Implementing the XL variant
//! against these weights would need tensors that do not exist — and the
//! "obvious" repair (reusing `linear_k` for the key-side positional term)
//! produces a silently different, wrong attention.
//!
//! ## The two `sqrt(64)` divisions are separate, not one
//!
//! `rel` is divided by `sqrt(head_dim)` on its own and only then added to the
//! already-scaled content scores. Folding them into a single division of the
//! sum is equivalent here, but scaling `rel` by anything else — or forgetting
//! to scale it because "the scores are already scaled" — is not.
//!
//! ## `distance = key - query`, and the clamp window is ASYMMETRIC
//!
//! The sign convention is `key_pos - query_pos`, so a POSITIVE distance means
//! the key lies in the query's future. The natural-looking `query - key` flips
//! the table left-to-right and, because the window is asymmetric, is not even
//! a relabelling — it is a different model.
//!
//! The window is `[-left_max, +right_max]` = `[-64, +8]`: this is a streaming
//! encoder that sees 64 frames of history but only 8 of lookahead. Assuming
//! the usual symmetric `[-max, +max]` window would demand a 129-row table; the
//! checkpoint's `distance_embedding.weight` is `[73, 64]`, i.e. `64 + 8 + 1`,
//! which pins the asymmetry.

use crate::error::{Error, Result};
use crate::model::audio::neucodec::client::NeuCodecClient;
use crate::model::audio::neucodec::semantic_encoder::config::SemanticEncoderConfig;
use crate::nn::{Embedding, Linear, var_contiguous};
use numr::autograd::{
    Var, var_add, var_div_scalar, var_matmul, var_permute, var_reshape, var_softmax,
};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Already-built weights for [`SemanticSelfAttention`].
pub struct SemanticSelfAttentionWeights<R: Runtime> {
    pub linear_q: Linear<R>,
    pub linear_k: Linear<R>,
    pub linear_v: Linear<R>,
    pub linear_out: Linear<R>,
    /// `distance_embedding.weight`, `[left_max + right_max + 1, head_dim]`.
    pub distance_embedding: Embedding<R>,
}

/// Multi-head self-attention with `relative_key` positional scores.
pub struct SemanticSelfAttention<R: Runtime> {
    linear_q: Linear<R>,
    linear_k: Linear<R>,
    linear_v: Linear<R>,
    linear_out: Linear<R>,
    distance_embedding: Embedding<R>,
    config: SemanticEncoderConfig,
}

/// Row-major `[seq_len * seq_len]` lookup indices into the distance table.
///
/// `index[l * seq_len + r] = clamp(r - l, -left_max, right_max) + left_max`,
/// i.e. `distance = key_pos - query_pos`, clamped to the asymmetric window and
/// shifted so the most negative distance maps to row 0.
pub fn relative_distance_indices(seq_len: usize, left_max: usize, right_max: usize) -> Vec<i64> {
    let (lo, hi) = (-(left_max as i64), right_max as i64);
    let mut indices = Vec::with_capacity(seq_len * seq_len);
    for query in 0..seq_len {
        for key in 0..seq_len {
            let distance = key as i64 - query as i64;
            indices.push(distance.clamp(lo, hi) + left_max as i64);
        }
    }
    indices
}

/// Build the `[seq_len * seq_len]` relative-distance index tensor on `device`.
///
/// This is the host-side table from [`relative_distance_indices`], uploaded
/// once. The same table is valid for every layer that shares `seq_len`,
/// `left_max`, and `right_max` — callers with multiple layers (e.g.
/// [`crate::model::audio::neucodec::semantic_encoder::encoder::SemanticEncoder`])
/// should build it once per forward pass and reuse it via
/// [`SemanticSelfAttention::forward_with_indices`] instead of calling this
/// once per layer.
pub fn relative_distance_index_tensor<R: Runtime<DType = DType>>(
    seq_len: usize,
    left: usize,
    right: usize,
    device: &R::Device,
) -> Result<Tensor<R>> {
    let indices = relative_distance_indices(seq_len, left, right);
    Tensor::<R>::from_slice(&indices, &[seq_len * seq_len], device).map_err(Error::Numr)
}

impl<R: Runtime> SemanticSelfAttention<R> {
    /// Assemble from already-loaded weights.
    pub fn new(
        weights: SemanticSelfAttentionWeights<R>,
        config: SemanticEncoderConfig,
    ) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            linear_q: weights.linear_q,
            linear_k: weights.linear_k,
            linear_v: weights.linear_v,
            linear_out: weights.linear_out,
            distance_embedding: weights.distance_embedding,
            config,
        })
    }
}

impl<R: Runtime<DType = DType>> SemanticSelfAttention<R> {
    /// Forward: `x [B, T, hidden] -> [B, T, hidden]`.
    ///
    /// No attention mask: this port serves a single utterance, so every key is
    /// valid for every query.
    ///
    /// Builds the relative-distance index table for this call's `seq_len`.
    /// Callers driving multiple layers over the same `seq_len` (e.g.
    /// [`crate::model::audio::neucodec::semantic_encoder::encoder::SemanticEncoder`])
    /// should build the table once via [`relative_distance_index_tensor`] and
    /// call [`Self::forward_with_indices`] instead, to avoid rebuilding and
    /// re-uploading a bit-identical table per layer.
    pub fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        C: NeuCodecClient<R>,
        R::Client: NeuCodecClient<R>,
    {
        let shape = x.shape().to_vec();
        if shape.len() != 3 {
            return Err(Error::InvalidArgument {
                arg: "x",
                reason: format!(
                    "expected [B, T, {}], got {shape:?}",
                    self.config.hidden_size
                ),
            });
        }
        let seq_len = shape[1];
        let index_tensor = relative_distance_index_tensor::<R>(
            seq_len,
            self.config.left_max_position_embeddings,
            self.config.right_max_position_embeddings,
            x.tensor().device(),
        )?;
        self.forward_with_indices(client, x, &index_tensor)
    }

    /// Forward with a caller-supplied relative-distance index tensor.
    ///
    /// `index_tensor` must be the `[seq_len * seq_len]` table produced by
    /// [`relative_distance_index_tensor`] for this call's `seq_len` and this
    /// attention's configured `left_max`/`right_max` window — callers that
    /// share the same geometry across layers (e.g. every layer of
    /// [`crate::model::audio::neucodec::semantic_encoder::encoder::SemanticEncoder`])
    /// build it once per forward pass and pass the same tensor to every layer.
    pub fn forward_with_indices<C>(
        &self,
        client: &C,
        x: &Var<R>,
        index_tensor: &Tensor<R>,
    ) -> Result<Var<R>>
    where
        C: NeuCodecClient<R>,
        R::Client: NeuCodecClient<R>,
    {
        let shape = x.shape().to_vec();
        if shape.len() != 3 || shape[2] != self.config.hidden_size {
            return Err(Error::InvalidArgument {
                arg: "x",
                reason: format!(
                    "expected [B, T, {}], got {shape:?}",
                    self.config.hidden_size
                ),
            });
        }
        let (batch, seq_len, hidden) = (shape[0], shape[1], shape[2]);
        let expected_indices = seq_len * seq_len;
        if index_tensor.shape() != [expected_indices].as_slice() {
            return Err(Error::InvalidArgument {
                arg: "index_tensor",
                reason: format!(
                    "expected [{expected_indices}] elements for seq_len {seq_len}, got {:?}",
                    index_tensor.shape()
                ),
            });
        }
        let scale = (self.config.head_dim as f64).sqrt();

        let q = self.heads_first(client, &self.linear_q, x, batch, seq_len)?;
        let k = self.heads_first(client, &self.linear_k, x, batch, seq_len)?;
        let v = self.heads_first(client, &self.linear_v, x, batch, seq_len)?;

        // Content scores: [B, H, T(q), T(k)].
        let k_t = var_permute(&k, &[0, 1, 3, 2]).map_err(Error::Numr)?;
        let k_t = var_contiguous(&k_t)?;
        let scores = var_matmul(&q, &k_t, client).map_err(Error::Numr)?;
        let scores = var_div_scalar(&scores, scale, client).map_err(Error::Numr)?;

        let rel = self.relative_scores(client, &q, batch, seq_len, index_tensor)?;
        let rel = var_div_scalar(&rel, scale, client).map_err(Error::Numr)?;
        let scores = var_add(&scores, &rel, client).map_err(Error::Numr)?;

        // Softmax over the KEY axis.
        let probs = var_softmax(&scores, -1, client).map_err(Error::Numr)?;
        let out = var_matmul(&probs, &v, client).map_err(Error::Numr)?;

        let out = var_permute(&out, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let out = var_contiguous(&out)?;
        let out = var_reshape(&out, &[batch, seq_len, hidden]).map_err(Error::Numr)?;

        self.linear_out.forward(client, &out)
    }

    /// Project and reshape to `[B, H, T, head_dim]`, contiguous.
    fn heads_first<C>(
        &self,
        client: &C,
        projection: &Linear<R>,
        x: &Var<R>,
        batch: usize,
        seq_len: usize,
    ) -> Result<Var<R>>
    where
        C: NeuCodecClient<R>,
        R::Client: NeuCodecClient<R>,
    {
        let projected = projection.forward(client, x)?;
        let split = var_reshape(
            &projected,
            &[batch, seq_len, self.config.num_heads, self.config.head_dim],
        )
        .map_err(Error::Numr)?;
        let permuted = var_permute(&split, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        var_contiguous(&permuted)
    }

    /// `einsum("bhld,lrd->bhlr", q, pos)` — the UNSCALED relative-key term.
    ///
    /// Expressed as a batched matmul over the leading query axis: `q` is
    /// permuted to `[T(l), B*H, D]` and the position table to `[T(l), D, T(r)]`,
    /// so one 3-D matmul contracts `d` for every query position at once. The
    /// result `[T(l), B*H, T(r)]` is permuted back to `[B, H, T(l), T(r)]`.
    ///
    /// `index_tensor` is the `[seq_len * seq_len]` relative-distance index
    /// table, supplied by the caller (see [`Self::forward_with_indices`])
    /// rather than built here, so it can be shared across layers.
    fn relative_scores<C>(
        &self,
        client: &C,
        q: &Var<R>,
        batch: usize,
        seq_len: usize,
        index_tensor: &Tensor<R>,
    ) -> Result<Var<R>>
    where
        C: NeuCodecClient<R>,
        R::Client: NeuCodecClient<R>,
    {
        let heads = self.config.num_heads;
        let head_dim = self.config.head_dim;

        // pos: [T(l), T(r), D] -> [T(l), D, T(r)].
        let pos = self.distance_embedding.forward(client, index_tensor)?;
        let pos = var_reshape(&pos, &[seq_len, seq_len, head_dim]).map_err(Error::Numr)?;
        let pos = var_permute(&pos, &[0, 2, 1]).map_err(Error::Numr)?;
        let pos = var_contiguous(&pos)?;

        // q: [B, H, T(l), D] -> [T(l), B*H, D].
        let q_l = var_permute(q, &[2, 0, 1, 3]).map_err(Error::Numr)?;
        let q_l = var_contiguous(&q_l)?;
        let q_l = var_reshape(&q_l, &[seq_len, batch * heads, head_dim]).map_err(Error::Numr)?;

        let rel = var_matmul(&q_l, &pos, client).map_err(Error::Numr)?;
        let rel = var_reshape(&rel, &[seq_len, batch, heads, seq_len]).map_err(Error::Numr)?;
        let rel = var_permute(&rel, &[1, 2, 0, 3]).map_err(Error::Numr)?;
        var_contiguous(&rel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    fn small_config() -> SemanticEncoderConfig {
        SemanticEncoderConfig {
            hidden_size: 8,
            num_heads: 2,
            head_dim: 4,
            intermediate_size: 16,
            num_layers: 1,
            left_max_position_embeddings: 3,
            right_max_position_embeddings: 1,
            ..Default::default()
        }
    }

    fn linear(out_f: usize, in_f: usize, device: &CpuDevice) -> Linear<CpuRuntime> {
        Linear::new(
            Tensor::<CpuRuntime>::from_slice(&vec![0.02f32; out_f * in_f], &[out_f, in_f], device)
                .unwrap(),
            Some(Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; out_f], &[out_f], device).unwrap()),
            false,
        )
    }

    fn attention(
        cfg: SemanticEncoderConfig,
        device: &CpuDevice,
    ) -> SemanticSelfAttention<CpuRuntime> {
        let rows = cfg.distance_embedding_len();
        let table: Vec<f32> = (0..(rows * cfg.head_dim))
            .map(|i| (i as f32 * 0.017).sin() * 0.1)
            .collect();
        SemanticSelfAttention::new(
            SemanticSelfAttentionWeights {
                linear_q: linear(cfg.hidden_size, cfg.hidden_size, device),
                linear_k: linear(cfg.hidden_size, cfg.hidden_size, device),
                linear_v: linear(cfg.hidden_size, cfg.hidden_size, device),
                linear_out: linear(cfg.hidden_size, cfg.hidden_size, device),
                distance_embedding: Embedding::new(
                    Tensor::<CpuRuntime>::from_slice(&table, &[rows, cfg.head_dim], device)
                        .unwrap(),
                    false,
                ),
            },
            cfg,
        )
        .expect("build attention")
    }

    #[test]
    fn distance_is_key_minus_query() {
        // seq_len 3, symmetric-enough window so nothing clamps.
        let idx = relative_distance_indices(3, 3, 3);
        // Row 0 (query 0): distances 0, +1, +2 -> shifted by left_max = 3.
        assert_eq!(&idx[0..3], &[3, 4, 5]);
        // Row 2 (query 2): distances -2, -1, 0.
        assert_eq!(&idx[6..9], &[1, 2, 3]);
    }

    #[test]
    fn clamp_window_is_asymmetric() {
        // left_max = 2, right_max = 1: distances below -2 and above +1 saturate.
        let seq = 5;
        let idx = relative_distance_indices(seq, 2, 1);
        // Query 4, key 0: distance -4 -> clamped to -2 -> row 0.
        assert_eq!(idx[4 * seq], 0);
        // Query 0, key 4: distance +4 -> clamped to +1 -> row 1 + 2 = 3.
        assert_eq!(idx[4], 3);
        // Every index must be inside the table.
        assert!(idx.iter().all(|&i| (0..4).contains(&i)));
    }

    #[test]
    fn forward_preserves_shape() {
        let (client, device) = cpu_setup();
        let cfg = small_config();
        let attn = attention(cfg, &device);

        let (b, t) = (2, 6);
        let data: Vec<f32> = (0..(b * t * cfg.hidden_size))
            .map(|i| (i as f32 * 0.07).cos())
            .collect();
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&data, &[b, t, cfg.hidden_size], &device).unwrap(),
            false,
        );
        let y = attn.forward(&client, &x).expect("forward");
        assert_eq!(y.shape(), &[b, t, cfg.hidden_size]);
        for v in y.tensor().contiguous().expect("contiguous").to_vec::<f32>() {
            assert!(v.is_finite(), "attention output is not finite: {v}");
        }
    }

    #[test]
    fn forward_and_forward_with_indices_agree() {
        let (client, device) = cpu_setup();
        let cfg = small_config();
        let attn = attention(cfg, &device);

        let (b, t) = (2, 6);
        let data: Vec<f32> = (0..(b * t * cfg.hidden_size))
            .map(|i| (i as f32 * 0.07).cos())
            .collect();
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&data, &[b, t, cfg.hidden_size], &device).unwrap(),
            false,
        );

        let y_forward = attn.forward(&client, &x).expect("forward");

        let index_tensor = relative_distance_index_tensor::<CpuRuntime>(
            t,
            cfg.left_max_position_embeddings,
            cfg.right_max_position_embeddings,
            &device,
        )
        .expect("index tensor");
        let y_with_indices = attn
            .forward_with_indices(&client, &x, &index_tensor)
            .expect("forward_with_indices");

        let a = y_forward
            .tensor()
            .contiguous()
            .expect("contiguous")
            .to_vec::<f32>();
        let b_vals = y_with_indices
            .tensor()
            .contiguous()
            .expect("contiguous")
            .to_vec::<f32>();
        assert_eq!(
            a, b_vals,
            "forward and forward_with_indices must agree bit-for-bit"
        );
    }

    #[test]
    fn forward_with_indices_rejects_wrong_sized_table() {
        let (client, device) = cpu_setup();
        let cfg = small_config();
        let attn = attention(cfg, &device);

        let (b, t) = (1, 6);
        let data = vec![0.0f32; b * t * cfg.hidden_size];
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&data, &[b, t, cfg.hidden_size], &device).unwrap(),
            false,
        );

        // Table built for the wrong seq_len (t - 1) has t*t - ... elements, not t*t.
        let wrong_index_tensor = relative_distance_index_tensor::<CpuRuntime>(
            t - 1,
            cfg.left_max_position_embeddings,
            cfg.right_max_position_embeddings,
            &device,
        )
        .expect("index tensor");

        assert!(
            attn.forward_with_indices(&client, &x, &wrong_index_tensor)
                .is_err()
        );
    }

    #[test]
    fn rejects_wrong_hidden_width() {
        let (client, device) = cpu_setup();
        let cfg = small_config();
        let attn = attention(cfg, &device);
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.0f32; 4 * 3], &[1, 4, 3], &device).unwrap(),
            false,
        );
        assert!(attn.forward(&client, &x).is_err());
    }
}
