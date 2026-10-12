//! `qwen35` gated attention for lanes that share one cached prefix.
//!
//! Every lane continues the same prefix with its own suffix. The prefix
//! keys and values sit once in a batch-1 [`KvCache`]; a lane's cache holds
//! only its own suffix. Each suffix query attends to every prefix key and
//! to its own lane's suffix keys up to itself, under one softmax, so a lane
//! computes what it would compute with the prefix copied into it.
//!
//! The batch folds into the query rows for the prefix half: the scores are
//! one `[1, H_kv, rep * B * S, D] x [1, H_kv, D, P]` matmul and the values
//! one `[1, H_kv, rep * B * S, P] x [1, H_kv, P, D]` matmul, where `rep` is
//! the GQA repeat. Query head `h` reads KV head `h / rep`, as `repeat_kv`
//! maps them, and the prefix keys are never repeated or copied per lane.

use super::forward::append_shared;
use super::layer::Qwen35AttentionBlock;
use super::projections::Heads;
use crate::error::{Error, Result};
use crate::inference::KvCache;
use crate::model::attention_mask::causal_window_mask;
use crate::model::traits::ModelClient;
use crate::nn::RoPE;
use crate::quant::traits::DequantOps;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, FwhtOps, IndexingOps, ReduceOps,
    ScalarOps, ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> Qwen35AttentionBlock<R> {
    /// Inference forward of `batch` lanes that each continue `prefix`.
    ///
    /// - `x`: the `attn_norm`-ed hidden state `[batch, seq, hidden_size]`
    ///   of every lane's suffix tokens
    /// - `positions`: `[4, seq]` IMROPE streams; the suffix starts at the
    ///   prefix's rope end
    /// - `prefix`: the batch-1 cache of the shared prefix, read only
    /// - `kv_cache`: `[batch, ...]` lanes holding each lane's own suffix so
    ///   far; the new k/v rows are appended at the shared cursor
    ///
    /// An empty `prefix` runs [`forward`](Self::forward) unchanged. The
    /// result is the `attn_output` projection `[batch, seq, hidden_size]`,
    /// without the residual, as a detached `Var`.
    ///
    /// # Errors
    ///
    /// [`Error::ModelError`] when `x` is not `[batch, seq, hidden_size]`,
    /// the rope table width is not `rope_dim / 2`, or `prefix` holds more
    /// than one lane. [`Error::InferenceError`] when the lanes of
    /// `kv_cache` hold different numbers of keys. Cache and op errors
    /// propagate.
    pub fn forward_shared_prefix<C>(
        &self,
        client: &C,
        x: &Var<R>,
        rope: &RoPE<R>,
        positions: &Tensor<R>,
        prefix: &KvCache<R>,
        kv_cache: &mut KvCache<R>,
    ) -> Result<Var<R>>
    where
        C: ModelClient<R> + FwhtOps<R>,
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
        if prefix.is_empty() {
            return self.forward(client, x, rope, positions, kv_cache);
        }
        let (prefix_k, prefix_v) = prefix.get_kv()?;
        if prefix_k.shape()[0] != 1 {
            return Err(Error::ModelError {
                reason: format!(
                    "qwen35_attention: a shared prefix is one sequence, got {} lanes",
                    prefix_k.shape()[0]
                ),
            });
        }
        let cfg = &self.cfg;
        let (h, h_kv, hd) = (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
        let rep = cfg.head_repeat();
        let p = prefix.seq_len();

        let Heads {
            q,
            gate,
            k,
            v,
            batch,
            seq,
        } = self.rotated_heads(client, x, rope, positions)?;
        let (suffix_k, suffix_v) = append_shared(client, kv_cache, &k, &v, rep)?;
        let sk = kv_cache.seq_len();
        let q = q.tensor();
        let scale = (hd as f64).sqrt().recip();
        let rows = rep * batch * seq;

        // Prefix scores: [B, H, S, D] -> [H, B, S, D] -> [1, H_kv, rep*B*S, D],
        // times the prefix keys [1, H_kv, D, P], back to [B, H, S, P].
        let q_rows = q
            .permute(&[1, 0, 2, 3])?
            .contiguous()?
            .reshape(&[1, h_kv, rows, hd])?;
        let prefix_kt = prefix_k.contiguous()?.transpose(-2, -1)?.contiguous()?;
        let prefix_scores = client
            .matmul(&q_rows, &prefix_kt)?
            .reshape(&[h, batch, seq, p])?
            .permute(&[1, 0, 2, 3])?
            .contiguous()?;
        let prefix_scores = client.mul_scalar(&prefix_scores, scale)?;

        // Suffix scores: each lane's own keys, causal from the lane's
        // cursor before this append.
        let suffix_kt = suffix_k.tensor().transpose(-2, -1)?.contiguous()?;
        let suffix_scores = client.matmul(q, &suffix_kt)?;
        let suffix_scores = client.mul_scalar(&suffix_scores, scale)?;
        let mask = causal_window_mask(client, seq, sk, 0, q.dtype(), q.device())?;
        let suffix_scores = client.add(&suffix_scores, &mask)?;

        // One softmax over [prefix | suffix] keys, then split.
        let scores = client.cat(&[&prefix_scores, &suffix_scores], 3)?;
        let weights = client.softmax(&scores, -1)?;
        let prefix_w = weights
            .narrow(3, 0, p)?
            .permute(&[1, 0, 2, 3])?
            .contiguous()?
            .reshape(&[1, h_kv, rows, p])?;
        let prefix_out = client
            .matmul(&prefix_w, &prefix_v.contiguous()?)?
            .reshape(&[h, batch, seq, hd])?
            .permute(&[1, 0, 2, 3])?
            .contiguous()?;
        let suffix_w = weights.narrow(3, p, sk)?.contiguous()?;
        let suffix_out = client.matmul(&suffix_w, suffix_v.tensor())?;
        let attn = client.add(&prefix_out, &suffix_out)?;

        self.gated_output(client, &Var::new(attn, false), &gate, batch, seq)
    }
}
