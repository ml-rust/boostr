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

use crate::error::Result;
use crate::model::attention_core::{AttentionCoreSpec, AttentionKernel, attention_core_masked};
use crate::model::traits::ModelClient;
use crate::nn::{Linear, RoPE};
use numr::autograd::Var;
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
}
