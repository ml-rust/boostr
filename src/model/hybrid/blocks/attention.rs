//! The hybrid model's attention block: pre-norm, RoPE (or ALiBi), KV-cached
//! multi-head attention, and the gated MLP.

use crate::error::{Error, Result};
use crate::inference::KvCache;
use crate::model::traits::ModelClient;
use crate::nn::var_ops::{repeat_kv, var_contiguous};
use crate::nn::{Linear, RmsNorm, RoPE};
use crate::ops::impl_generic::attention::multi_head_attention_impl;
use crate::ops::impl_generic::attention::rope::apply_rope_impl;
use numr::autograd::{Var, var_add, var_mul, var_narrow, var_reshape, var_silu};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::Runtime;

/// Attention block: pre-norm → multi-head attention → residual + pre-norm → MLP → residual
pub(in crate::model::hybrid) struct AttentionBlock<R: Runtime> {
    pub(in crate::model::hybrid) input_layernorm: RmsNorm<R>,
    pub(in crate::model::hybrid) q_proj: Linear<R>,
    pub(in crate::model::hybrid) k_proj: Linear<R>,
    pub(in crate::model::hybrid) v_proj: Linear<R>,
    pub(in crate::model::hybrid) o_proj: Linear<R>,
    pub(in crate::model::hybrid) post_attention_layernorm: RmsNorm<R>,
    pub(in crate::model::hybrid) gate_proj: Linear<R>,
    pub(in crate::model::hybrid) up_proj: Linear<R>,
    pub(in crate::model::hybrid) down_proj: Linear<R>,
    pub(in crate::model::hybrid) num_heads: usize,
    pub(in crate::model::hybrid) num_kv_heads: usize,
    pub(in crate::model::hybrid) head_dim: usize,
    /// Use ALiBi instead of RoPE (Falcon v1, BLOOM, MPT).
    pub(in crate::model::hybrid) use_alibi: bool,
    /// Sliding-window attention span. `0` disables windowing (unlimited context).
    ///
    /// The window is INCLUSIVE of the current token: query at absolute position
    /// `p` may attend keys `j` with `p - sliding_window < j <= p`.
    ///
    /// IGNORED when `use_alibi` is set. ALiBi's bias kernel writes the causal
    /// structure together with the distance bias; the two mechanisms do not
    /// compose here, so ALiBi models always attend the full context.
    pub(in crate::model::hybrid) sliding_window: usize,
}

impl<R: Runtime<DType = DType>> AttentionBlock<R> {
    pub(in crate::model::hybrid) fn forward_with_kv_cache<C>(
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
        // Pre-norm attention + residual
        let normed = self.input_layernorm.forward(client, x)?;
        let attn_out = self.attention_forward(client, &normed, rope, kv_cache, position)?;
        let h = var_add(x, &attn_out, client).map_err(Error::Numr)?;

        // Pre-norm MLP + residual
        let normed = self.post_attention_layernorm.forward(client, &h)?;
        let mlp_out = self.mlp_forward(client, &normed)?;
        var_add(&h, &mlp_out, client).map_err(Error::Numr)
    }

    fn attention_forward<C>(
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
        let batch = shape[0];
        let seq_len = shape[1];

        let q = self.q_proj.forward(client, x)?;
        let k = self.k_proj.forward(client, x)?;
        let v = self.v_proj.forward(client, x)?;

        let q = var_reshape(&q, &[batch, seq_len, self.num_heads, self.head_dim])
            .map_err(Error::Numr)?;
        let k = var_reshape(&k, &[batch, seq_len, self.num_kv_heads, self.head_dim])
            .map_err(Error::Numr)?;
        let v = var_reshape(&v, &[batch, seq_len, self.num_kv_heads, self.head_dim])
            .map_err(Error::Numr)?;

        let q = numr::autograd::var_permute(&q, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let q = var_contiguous(&q)?;
        let k = numr::autograd::var_permute(&k, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let k = var_contiguous(&k)?;
        let v = numr::autograd::var_permute(&v, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let v = var_contiguous(&v)?;

        // Apply RoPE with position offset, or skip it for ALiBi models: ALiBi
        // carries positional information in the attention bias instead.
        let (q, k) = if self.use_alibi {
            (q, k)
        } else {
            let cos_offset =
                var_narrow(rope.cos_cache(), 0, position, seq_len).map_err(Error::Numr)?;
            let sin_offset =
                var_narrow(rope.sin_cache(), 0, position, seq_len).map_err(Error::Numr)?;
            let q = apply_rope_impl(client, &q, &cos_offset, &sin_offset)?;
            let k = apply_rope_impl(client, &k, &cos_offset, &sin_offset)?;
            (q, k)
        };

        // Update KV cache with new K/V tensors [B, H_kv, S, D]
        kv_cache.update(k.tensor(), v.tensor())?;

        // Get full cached K/V for attention
        let (cached_k, cached_v) = kv_cache.get_kv()?;
        let cached_k = Var::new(cached_k.contiguous()?, false);
        let cached_v = Var::new(cached_v.contiguous()?, false);

        // GQA: repeat K/V heads to match Q heads if needed
        let (cached_k, cached_v) = if self.num_kv_heads < self.num_heads {
            let repeat = self.num_heads / self.num_kv_heads;
            let k_rep = repeat_kv(&cached_k, repeat).map_err(Error::Numr)?;
            let v_rep = repeat_kv(&cached_v, repeat).map_err(Error::Numr)?;
            (k_rep, v_rep)
        } else {
            (cached_k, cached_v)
        };

        let sq = q.shape()[2];
        let sk = cached_k.shape()[2];
        let mask = self.attention_mask(
            client,
            batch,
            sq,
            sk,
            position,
            q.tensor().dtype(),
            q.tensor().device(),
        )?;

        // Multi-head attention (Q attends to full cached K/V)
        let attn_out = multi_head_attention_impl(
            client,
            &q,
            &cached_k,
            &cached_v,
            mask.as_ref(),
            self.num_heads,
        )?;

        // [B, H, S, D] -> [B, S, H, D] -> [B, S, H*D]
        let attn_out =
            numr::autograd::var_permute(&attn_out, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let attn_out = var_contiguous(&attn_out)?;
        let attn_out = var_reshape(&attn_out, &[batch, seq_len, self.num_heads * self.head_dim])
            .map_err(Error::Numr)?;

        // Output projection
        self.o_proj.forward(client, &attn_out)
    }

    fn mlp_forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
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
        let gate = self.gate_proj.forward(client, x)?;
        let up = self.up_proj.forward(client, x)?;
        let gate_silu = var_silu(&gate, client).map_err(Error::Numr)?;
        let hidden = var_mul(&gate_silu, &up, client).map_err(Error::Numr)?;
        self.down_proj.forward(client, &hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

    fn ramp(n: usize) -> Vec<f32> {
        (0..n).map(|i| (i as f32 % 7.0) * 0.25 - 0.75).collect()
    }

    /// A block with deterministic non-zero weights, so a change in the rotary
    /// frequencies actually moves the output.
    fn rope_probe_block(use_alibi: bool) -> AttentionBlock<CpuRuntime> {
        let (_, device) = cpu_setup();
        let w = || Tensor::<CpuRuntime>::from_slice(&ramp(64), &[8, 8], &device).unwrap();
        let n = || Tensor::<CpuRuntime>::from_slice(&[1.0f32; 8], &[8], &device).unwrap();
        AttentionBlock {
            input_layernorm: RmsNorm::new(n(), 1e-5, false),
            q_proj: Linear::new(w(), None, false),
            k_proj: Linear::new(w(), None, false),
            v_proj: Linear::new(w(), None, false),
            o_proj: Linear::new(w(), None, false),
            post_attention_layernorm: RmsNorm::new(n(), 1e-5, false),
            gate_proj: Linear::new(w(), None, false),
            up_proj: Linear::new(w(), None, false),
            down_proj: Linear::new(w(), None, false),
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 4,
            use_alibi,
            sliding_window: 0,
        }
    }

    /// Run one prefill through a fresh KV cache with a RoPE cache built at `base`.
    fn forward_with_rope_base(use_alibi: bool, base: f32) -> Vec<f32> {
        let (client, device) = cpu_setup();
        let block = rope_probe_block(use_alibi);
        let rope = RoPE::<CpuRuntime>::precompute_freqs(8, 4, base, None, &device).unwrap();
        let mut cache =
            crate::inference::KvCache::<CpuRuntime>::new(1, 2, 8, 8, 4, DType::F32, &device)
                .unwrap();
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&ramp(24), &[1, 3, 8], &device).unwrap(),
            false,
        );
        let out = block
            .forward_with_kv_cache(&client, &x, &rope, &mut cache, 0)
            .unwrap();
        out.tensor().to_vec::<f32>()
    }

    #[test]
    fn alibi_blocks_skip_rope() {
        // The rotary frequencies depend on `base`, so an ALiBi block that still
        // applied RoPE would produce different outputs for different bases.
        assert_eq!(
            forward_with_rope_base(true, 10_000.0),
            forward_with_rope_base(true, 100.0)
        );
    }

    #[test]
    fn non_alibi_blocks_still_apply_rope() {
        // Guards the test above against passing for the wrong reason.
        assert_ne!(
            forward_with_rope_base(false, 10_000.0),
            forward_with_rope_base(false, 100.0)
        );
    }
}
