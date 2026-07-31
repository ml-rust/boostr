//! Packed (varlen) self-attention for `EncoderLayer` (NomicBert/Gemma packed path).
//!
//! Operates on `[total_tokens, hidden]` packed input using
//! `VarLenAttentionOps::varlen_attention_fwd` and packed RoPE.
//!
//! The varlen kernel supports full and causal attention but not a bounded
//! attention span, so a windowed block cannot be evaluated here. Callers must
//! first clear
//! [`ensure_varlen_span_is_unconstrained`](super::attention_mask::ensure_varlen_span_is_unconstrained),
//! which errors rather than letting a windowed model return unwindowed results.

use super::encoder_layer::{EncoderLayer, VarlenCtx};
use crate::error::{Error, Result};
use crate::ops::{RoPEPackedOps, VarLenAttentionOps};
use crate::quant::traits::QuantMatmulOps;
use numr::autograd::{Var, var_reshape};
use numr::dtype::DType;
use numr::ops::{BinaryOps, NormalizationOps, ScalarOps, ShapeOps, TensorOps, TypeConversionOps};
use numr::runtime::{Runtime, RuntimeClient};

impl<R: Runtime<DType = DType>> EncoderLayer<R> {
    /// Packed (varlen) self-attention for NomicBert.
    ///
    /// Input `x` is `[total_tokens, hidden]` (2-D, packed sequences).
    /// Uses `VarLenAttentionOps::varlen_attention_fwd` which applies its own
    /// `1/sqrt(head_dim)` scaling — we do NOT scale Q here.
    pub(super) fn self_attention_varlen<C>(
        &self,
        client: &C,
        x: &Var<R>,
        ctx: &VarlenCtx<'_, R>,
    ) -> Result<Var<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + BinaryOps<R>
            + ShapeOps<R>
            + QuantMatmulOps<R>
            + TypeConversionOps<R>
            + NormalizationOps<R>
            + RoPEPackedOps<R>
            + VarLenAttentionOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        let x_shape = x.shape().to_vec();
        let total_tokens = x_shape[0];
        // hidden == x_shape[1], derived from total_tokens * hidden = numel

        // 1. Linear projections: [total_tokens, hidden] → [total_tokens, hidden]
        let q = self.q_proj.forward(client, x)?;
        let k = self.k_proj.forward(client, x)?;
        let v = self.v_proj.forward(client, x)?;

        // 1b. Whole-hidden QK-norm, before the reshape into heads. No packed
        //     architecture uses it today (jina-bert-v2 takes the padded path,
        //     since ALiBi has no varlen kernel), but the hook is called here so
        //     the two attention paths stay in step.
        let (q, k) = self.qk_norm_hidden(client, q, k)?;

        // 2. Reshape to [total_tokens, num_heads, head_dim] (no permute).
        let q =
            var_reshape(&q, &[total_tokens, self.num_heads, self.head_dim]).map_err(Error::Numr)?;
        let k = var_reshape(&k, &[total_tokens, self.num_kv_heads, self.head_dim])
            .map_err(Error::Numr)?;
        let v = var_reshape(&v, &[total_tokens, self.num_kv_heads, self.head_dim])
            .map_err(Error::Numr)?;

        // 3. QK-norm (Gemma): RmsNorm over head_dim applied to Q and K after
        //    reshape to [total_tokens, heads, head_dim], BEFORE RoPE.
        //    For NomicBert/BERT these are None and are a no-op.
        //    RmsNorm normalises the last axis, so it works on both
        //    [B, H, S, D] (padded) and [total, H, D] (packed) layouts.
        let (q, k) = self.qk_norm_per_head(client, q, k)?;

        // 4. Packed RoPE on Q and K.  V is left unchanged.
        //    RoPE is applied when present (NomicBert/Gemma); BERT/XLM-R have
        //    rope = None and use learned absolute position embeddings added at
        //    the embedding stage, so Q and K are used as-is here.
        let (q, k) = if let Some(rope) = self.rope.as_ref() {
            let q = client.apply_rope_packed(
                &q,
                rope.cos_cache(),
                rope.sin_cache(),
                ctx.position_ids,
            )?;
            let k = client.apply_rope_packed(
                &k,
                rope.cos_cache(),
                rope.sin_cache(),
                ctx.position_ids,
            )?;
            (q, k)
        } else {
            (q, k)
        };

        // 5. Contiguous before passing to varlen kernel.
        let q_c = Var::new(q.tensor().contiguous()?, false);
        let k_c = Var::new(k.tensor().contiguous()?, false);
        let v_c = Var::new(v.tensor().contiguous()?, false);

        // 6. Varlen attention forward.  The kernel applies 1/sqrt(head_dim) itself.
        //    K/V are already shaped [total_tokens, num_kv_heads, head_dim] from
        //    step 2 above.  For MHA num_kv_heads == num_heads (behaviour unchanged).
        let (out_tensor, _lse) = client.varlen_attention_fwd(
            q_c.tensor(),
            k_c.tensor(),
            v_c.tensor(),
            ctx.cu_seqlens,
            ctx.cu_seqlens,
            ctx.batch,
            self.num_heads,
            self.num_kv_heads,
            ctx.max_seqlen,
            ctx.max_seqlen,
            self.head_dim,
            false, // non-causal (bidirectional encoder)
        )?;

        // 7. Wrap output and reshape to [total_tokens, hidden].
        let hidden = self.num_heads * self.head_dim;
        let out_var = Var::new(out_tensor.contiguous()?, false);
        let out_var = var_reshape(&out_var, &[total_tokens, hidden]).map_err(Error::Numr)?;

        // 8. Output projection.
        self.o_proj.forward(client, &out_var)
    }
}
