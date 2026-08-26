//! Causal GQA attention for VoxCPM2's MiniCPM4 decoder.
//!
//! Unlike the `feat_encoder` sibling in this module — the one bidirectional
//! transformer in the VoxCPM2 stack, which hand-writes its own unmasked
//! orchestration — this block is a plain causal decoder, so it runs the
//! shared [`attention_core_masked`] sequence that every other causal block in
//! the crate uses (`LlamaAttention` included). That helper owns
//! reshape/permute, contiguity, RoPE, the GQA head repeat, and the causal
//! mask; nothing here re-derives any of it.
//!
//! Causality is not a flag on that path: `attention_core_masked` always
//! builds a causal mask. This is the full-sequence forward, so without it
//! every position would attend to FUTURE positions while every shape stayed
//! valid.
//!
//! `head_dim` (128) is read from config, never derived from
//! `hidden_size / num_heads` — see
//! [`MiniCpm4Config::head_dim`](crate::model::audio::voxcpm::minicpm4::MiniCpm4Config::head_dim).

use crate::error::{Error, Result};
use crate::inference::KvCache;
use crate::model::attention_core::{
    AttentionCoreSpec, AttentionKernel, attention_core_masked, prefill_attention_mask,
};
use crate::model::traits::ModelClient;
use crate::nn::var_ops::{repeat_kv, var_contiguous};
use crate::nn::{Linear, RoPE};
use crate::ops::impl_generic::attention::multi_head_attention_impl;
use numr::autograd::{Var, var_narrow, var_permute, var_reshape};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::Runtime;

/// `q_proj`: 2048 -> 2048 (16 heads x 128), `k_proj`/`v_proj`: 2048 -> 256
/// (2 heads x 128, GQA group size 8), `o_proj`: 2048 -> 2048. All bias-free.
pub struct MiniCpm4Attention<R: Runtime> {
    pub(crate) q_proj: Linear<R>,
    pub(crate) k_proj: Linear<R>,
    pub(crate) v_proj: Linear<R>,
    pub(crate) o_proj: Linear<R>,
    pub(crate) num_heads: usize,
    pub(crate) num_kv_heads: usize,
    pub(crate) head_dim: usize,
}

impl<R: Runtime<DType = DType>> MiniCpm4Attention<R> {
    /// Borrowed view of this block's attention parameters, for
    /// [`attention_core_masked`].
    ///
    /// No Q/K per-head norm and no ALiBi on this checkpoint, and
    /// `sliding_window: 0` — MiniCPM4's VoxCPM2 configuration attends the
    /// full prefix, so windowing is disabled rather than left unset.
    fn core_spec(&self) -> AttentionCoreSpec<'_, R> {
        AttentionCoreSpec {
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            q_norm: None,
            k_norm: None,
            use_alibi: false,
            sliding_window: 0,
            // Materialized-mask kernel: the same entry point `LlamaAttention`
            // uses, and the one that does not add an `R::Client:
            // FlashAttentionOps` bound here.
            kernel: AttentionKernel::Masked,
        }
    }

    /// Causal GQA attention over `x: [batch, seq, hidden]`, returning
    /// `[batch, seq, hidden]`.
    ///
    /// Softmax scale is `1/sqrt(head_dim)`, derived from `q`'s actual last
    /// dimension inside `multi_head_attention_impl`.
    pub fn forward<C>(&self, client: &C, x: &Var<R>, rope: &RoPE<R>) -> Result<Var<R>>
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
            + ConditionalOps<R>,
    {
        let q = self.q_proj.forward(client, x)?;
        let k = self.k_proj.forward(client, x)?;
        let v = self.v_proj.forward(client, x)?;

        let attn_out = attention_core_masked(
            client,
            &q,
            &k,
            &v,
            rope.cos_cache(),
            rope.sin_cache(),
            &self.core_spec(),
        )?;

        self.o_proj.forward(client, &attn_out)
    }

    /// KV-cached causal GQA attention over `x: [batch, seq, hidden]` covering
    /// absolute positions `position..position + seq`, returning
    /// `[batch, seq, hidden]`.
    ///
    /// Serves BOTH cached shapes: prefill passes the whole prefix at
    /// `position == 0`, a decode step passes `seq == 1` at the next position.
    ///
    /// This does not call [`attention_core_masked`]: that entry point takes RAW
    /// K/V projections and rotates them itself, so handing it the cache would
    /// re-apply RoPE to keys that were already rotated when they were written.
    /// The step order is otherwise the same one that function documents —
    /// reshape/permute, contiguous Q/K, (no Q/K norm here), RoPE, GQA repeat,
    /// additive causal mask, attend — and the mask still comes from the shared
    /// [`prefill_attention_mask`] builder, so causality is not re-derived here.
    ///
    /// # Mask equivalence with the reference
    ///
    /// The reference preallocates the cache to `max_length` and masks with
    /// `arange(max_length) <= position_id`, i.e. a row spanning the FULL
    /// preallocated width that admits keys `0..=position_id` and rejects the
    /// zeroed tail. [`KvCache`] instead exposes only the `seq_len` slots
    /// actually written, so `sk == position + seq` and
    /// [`prefill_attention_mask`] (key offset `sk - sq`) admits exactly keys
    /// `0..=position + i` for query row `i`. The two masks select the SAME
    /// keys; the reference additionally multiplies zeroed, never-written slots
    /// by `-inf`, which contribute nothing to the softmax either way. The
    /// equivalence holds ONLY because the caller writes positions in order
    /// starting at 0 — [`MiniCpm4Model::decode_step`] enforces that by
    /// requiring `position == kv_cache.seq_len()`.
    ///
    /// [`MiniCpm4Model::decode_step`]:
    ///     crate::model::audio::voxcpm::minicpm4::MiniCpm4Model::decode_step
    pub fn forward_cached<C>(
        &self,
        client: &C,
        x: &Var<R>,
        rope: &RoPE<R>,
        kv_cache: &mut KvCache<R>,
        position: usize,
    ) -> Result<Var<R>>
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
            + ConditionalOps<R>,
    {
        let shape = x.shape().to_vec();
        if shape.len() != 3 {
            return Err(Error::InvalidArgument {
                arg: "x",
                reason: format!("expected 3D [batch, seq, hidden], got {}D", shape.len()),
            });
        }
        let (batch, seq) = (shape[0], shape[1]);

        let q = self.q_proj.forward(client, x)?;
        let k = self.k_proj.forward(client, x)?;
        let v = self.v_proj.forward(client, x)?;

        // [B, S, H*D] -> [B, S, H, D] -> [B, H, S, D]
        let q =
            var_reshape(&q, &[batch, seq, self.num_heads, self.head_dim]).map_err(Error::Numr)?;
        let k = var_reshape(&k, &[batch, seq, self.num_kv_heads, self.head_dim])
            .map_err(Error::Numr)?;
        let v = var_reshape(&v, &[batch, seq, self.num_kv_heads, self.head_dim])
            .map_err(Error::Numr)?;
        let q = var_permute(&q, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let k = var_permute(&k, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let v = var_permute(&v, &[0, 2, 1, 3]).map_err(Error::Numr)?;

        // The fused RoPE kernel assumes contiguous layout.
        let q = var_contiguous(&q)?;
        let k = var_contiguous(&k)?;

        // Same precomputed cos/sin tables the full-sequence path uses, sliced
        // at the absolute positions this call covers. Building a second table
        // here is how the two paths would drift.
        let cos = var_narrow(rope.cos_cache(), 0, position, seq).map_err(Error::Numr)?;
        let sin = var_narrow(rope.sin_cache(), 0, position, seq).map_err(Error::Numr)?;
        let q = client.apply_rope(&q, &cos, &sin)?;
        let k = client.apply_rope(&k, &cos, &sin)?;

        let v = var_contiguous(&v)?;

        // Post-RoPE K/V land in the cache, exactly as the reference writes
        // rotated keys at index `position_id`.
        kv_cache.update(k.tensor(), v.tensor())?;
        let (k_full, v_full) = kv_cache.get_kv()?;
        let k_full = Var::new(k_full.contiguous()?, false);
        let v_full = Var::new(v_full.contiguous()?, false);

        let (k_full, v_full) = if self.num_kv_heads < self.num_heads {
            let repeat = self.num_heads / self.num_kv_heads;
            (
                repeat_kv(&k_full, repeat).map_err(Error::Numr)?,
                repeat_kv(&v_full, repeat).map_err(Error::Numr)?,
            )
        } else {
            (k_full, v_full)
        };

        let sk = k_full.shape()[2];
        let mask = prefill_attention_mask(
            client,
            batch,
            seq,
            sk,
            &self.core_spec(),
            q.tensor().device(),
        )?;
        let attn_out =
            multi_head_attention_impl(client, &q, &k_full, &v_full, Some(&mask), self.num_heads)?;

        // [B, H, S, D] -> [B, S, H, D] -> [B, S, H*D]
        let attn_out = var_permute(&attn_out, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let attn_out = var_contiguous(&attn_out)?;
        let attn_out = var_reshape(&attn_out, &[batch, seq, self.num_heads * self.head_dim])
            .map_err(Error::Numr)?;

        self.o_proj.forward(client, &attn_out)
    }
}
