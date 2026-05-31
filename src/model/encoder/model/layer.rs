//! Single transformer encoder layer: self-attention + FFN with residual + norm.
//!
//! Supports three architectural styles via field variants:
//! - BERT/XLM-RoBERTa: post-add LayerNorm (standard residual), no RoPE, no gate.
//! - NomicBert: post-add LayerNorm, RoPE, SwiGLU gate.
//! - Gemma-embedding: sandwich RMSNorm (pre-norm + post-norm before residual add),
//!   QK-norm, RoPE, GQA (KV head repeat), GeGLU gate, no biases.

use crate::error::{Error, Result};
use crate::model::encoder::config::{FfnVariant, HiddenAct};
use crate::nn::{LayerNorm, MaybeQuantLinear, RmsNorm, RoPE};
use crate::ops::{RoPEOps, RoPEPackedOps, VarLenAttentionOps};
use crate::quant::traits::QuantMatmulOps;
use numr::autograd::{Var, var_add, var_mul, var_reshape, var_silu};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, IndexingOps, NormalizationOps, ReduceOps, ScalarOps, ShapeOps,
    TensorOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use std::sync::Arc;

/// Packed (varlen) attention context for NomicBert forward passes.
///
/// Carries the pre-built host-derived metadata (cu_seqlens, position_ids, seg_ids)
/// that are constructed once per batch from host-side token lists and then threaded
/// through the layer loop.  No tensor DATA is transferred GPU↔CPU here — these
/// tensors are built on the host from host-side `Vec<i32>`/`Vec<i64>` and uploaded
/// once.
pub(in crate::model::encoder) struct VarlenCtx<'a, R: Runtime> {
    /// Cumulative sequence lengths `[batch+1]` (I32).  `cu_seqlens[0] = 0`,
    /// `cu_seqlens[i] = sum of token counts for sequences 0..i-1`.
    pub cu_seqlens: &'a Tensor<R>,
    /// Per-token absolute position ids `[total_tokens]` (I64).  Values reset to
    /// 0 at the start of each packed sequence.
    pub position_ids: &'a Tensor<R>,
    /// Number of sequences in this packed batch.
    pub batch: usize,
    /// Length of the longest individual sequence.
    pub max_seqlen: usize,
}

/// Norm layer that can be either a LayerNorm (BERT/NomicBert) or RmsNorm (Gemma).
pub(in crate::model::encoder) enum NormLayer<R: Runtime> {
    LayerNorm(LayerNorm<R>),
    RmsNorm(RmsNorm<R>),
}

impl<R: Runtime<DType = DType>> NormLayer<R> {
    pub fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        C: RuntimeClient<R> + NormalizationOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        match self {
            Self::LayerNorm(ln) => ln.forward(client, x),
            Self::RmsNorm(rn) => rn.forward(client, x),
        }
    }
}

/// A single transformer encoder layer: self-attention + FFN with residual + norm.
pub(in crate::model::encoder) struct EncoderLayer<R: Runtime> {
    pub(in crate::model::encoder) q_proj: MaybeQuantLinear<R>,
    pub(in crate::model::encoder) k_proj: MaybeQuantLinear<R>,
    pub(in crate::model::encoder) v_proj: MaybeQuantLinear<R>,
    pub(in crate::model::encoder) o_proj: MaybeQuantLinear<R>,
    /// Pre-attention norm (post-add LayerNorm for BERT/NomicBert; pre-attn RMSNorm for Gemma).
    pub(in crate::model::encoder) attn_norm: NormLayer<R>,
    pub(in crate::model::encoder) ffn_up: MaybeQuantLinear<R>,
    /// Gate projection: SwiGLU (NomicBert) or GeGLU (Gemma). `None` for BERT/XLM-R.
    pub(in crate::model::encoder) ffn_gate: Option<MaybeQuantLinear<R>>,
    pub(in crate::model::encoder) ffn_down: MaybeQuantLinear<R>,
    /// Pre-FFN norm (post-add LayerNorm for BERT/NomicBert; pre-FFN RMSNorm for Gemma).
    pub(in crate::model::encoder) ffn_norm: NormLayer<R>,
    pub(in crate::model::encoder) num_heads: usize,
    /// Number of KV heads. Equal to `num_heads` for MHA (BERT/NomicBert).
    pub(in crate::model::encoder) num_kv_heads: usize,
    pub(in crate::model::encoder) head_dim: usize,
    pub(in crate::model::encoder) hidden_act: HiddenAct,
    /// FFN variant: Standard (BERT), GatedSilu (NomicBert), or GatedGelu (Gemma).
    pub(in crate::model::encoder) ffn_variant: FfnVariant,
    /// RoPE cache (NomicBert and Gemma; `None` for BERT/XLM-R).
    pub(in crate::model::encoder) rope: Option<Arc<RoPE<R>>>,
    /// QK-norm on Q after reshape, before RoPE (Gemma only; `None` for BERT/NomicBert).
    pub(in crate::model::encoder) q_norm: Option<RmsNorm<R>>,
    /// QK-norm on K after reshape, before RoPE (Gemma only; `None` for BERT/NomicBert).
    pub(in crate::model::encoder) k_norm: Option<RmsNorm<R>>,
    /// Post-attention sandwich norm applied to attn output before residual add
    /// (Gemma only; `None` for BERT/NomicBert).
    pub(in crate::model::encoder) post_attn_norm: Option<RmsNorm<R>>,
    /// Post-FFN sandwich norm applied to FFN output before residual add
    /// (Gemma only; `None` for BERT/NomicBert).
    pub(in crate::model::encoder) post_ffn_norm: Option<RmsNorm<R>>,
}

impl<R: Runtime<DType = DType>> EncoderLayer<R> {
    /// Forward pass.
    ///
    /// Exactly one of `attention_mask` (padded BERT/Gemma path) or `varlen_ctx`
    /// (packed NomicBert path) should be non-`None`.  When both are `None`, the
    /// padded path runs without masking (single-sequence BERT inference).
    pub(super) fn forward<C>(
        &self,
        client: &C,
        x: &Var<R>,
        attention_mask: Option<&Tensor<R>>,
        varlen_ctx: Option<&VarlenCtx<'_, R>>,
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
            + NormalizationOps<R>
            + QuantMatmulOps<R>
            + TypeConversionOps<R>
            + RoPEOps<R>
            + RoPEPackedOps<R>
            + VarLenAttentionOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        // Two independent axes:
        //   1. Norm scheme: sandwich (Gemma) vs. post-norm (BERT/NomicBert).
        //   2. Attention impl: varlen (packed) vs. padded.
        //
        // All four combinations are handled correctly below.

        if self.post_attn_norm.is_some() {
            // ── Sandwich norm (Gemma) ───────────────────────────────────────
            //   attn_in  = attn_norm(x)
            //   attn_raw = ATTN(attn_in)          ← varlen OR padded
            //   x        = x + post_attn_norm(attn_raw)
            //
            //   ffn_in   = ffn_norm(x)
            //   ffn_raw  = ffn(ffn_in)
            //   x        = x + post_ffn_norm(ffn_raw)
            let attn_in = self.attn_norm.forward(client, x)?;
            let attn_raw = if let Some(ctx) = varlen_ctx {
                self.self_attention_varlen(client, &attn_in, ctx)?
            } else {
                self.self_attention_padded(client, &attn_in, attention_mask)?
            };
            let attn_out = self
                .post_attn_norm
                .as_ref()
                .ok_or_else(|| Error::ModelError {
                    reason: "post_attn_norm is None despite sandwich path branch".into(),
                })?
                .forward(client, &attn_raw)?;
            let x = var_add(x, &attn_out, client).map_err(Error::Numr)?;

            let ffn_in = self.ffn_norm.forward(client, &x)?;
            let ffn_raw = self.ffn(client, &ffn_in)?;
            let ffn_out = self
                .post_ffn_norm
                .as_ref()
                .ok_or_else(|| Error::ModelError {
                    reason: "post_ffn_norm is None despite sandwich path branch".into(),
                })?
                .forward(client, &ffn_raw)?;
            let x = var_add(&x, &ffn_out, client).map_err(Error::Numr)?;
            Ok(x)
        } else if let Some(ctx) = varlen_ctx {
            // ── Post-norm + varlen (NomicBert packed path) ──────────────────
            //   x = attn_norm(x + varlen_attn(x))
            //   x = ffn_norm(x + ffn(x))
            let attn_out = self.self_attention_varlen(client, x, ctx)?;
            let x = var_add(x, &attn_out, client).map_err(Error::Numr)?;
            let x = self.attn_norm.forward(client, &x)?;

            let ffn_out = self.ffn(client, &x)?;
            let x = var_add(&x, &ffn_out, client).map_err(Error::Numr)?;
            let x = self.ffn_norm.forward(client, &x)?;
            Ok(x)
        } else {
            // ── Post-norm + padded (BERT/XLM-R standard path) ───────────────
            //   x = attn_norm(x + attn(x))
            //   x = ffn_norm(x + ffn(x))
            let attn_out = self.self_attention_padded(client, x, attention_mask)?;
            let x = var_add(x, &attn_out, client).map_err(Error::Numr)?;
            let x = self.attn_norm.forward(client, &x)?;

            let ffn_out = self.ffn(client, &x)?;
            let x = var_add(&x, &ffn_out, client).map_err(Error::Numr)?;
            let x = self.ffn_norm.forward(client, &x)?;

            Ok(x)
        }
    }

    /// Padded (BERT/Gemma/XLM-R) self-attention: input is `[B, S, hidden]`.
    ///
    /// Uses the classic `[B, H, S, S]` score matrix.  Not called for NomicBert —
    /// use `self_attention_varlen` instead.
    fn self_attention_padded<C>(
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
            + UnaryOps<R>
            + QuantMatmulOps<R>
            + TypeConversionOps<R>
            + RoPEOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        use numr::autograd::{var_matmul, var_permute, var_transpose};

        let shape = x.shape().to_vec();
        let batch = shape[0];
        let seq_len = shape[1];

        let q = self.q_proj.forward(client, x)?;
        let k = self.k_proj.forward(client, x)?;
        let v = self.v_proj.forward(client, x)?;

        // Q: [B, S, num_heads * head_dim] → [B, num_heads, S, head_dim]
        let q = var_reshape(&q, &[batch, seq_len, self.num_heads, self.head_dim])
            .map_err(Error::Numr)?;
        let q = var_permute(&q, &[0, 2, 1, 3]).map_err(Error::Numr)?;

        // K, V: [B, S, num_kv_heads * head_dim] → [B, num_kv_heads, S, head_dim]
        let k = var_reshape(&k, &[batch, seq_len, self.num_kv_heads, self.head_dim])
            .map_err(Error::Numr)?;
        let k = var_permute(&k, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let v = var_reshape(&v, &[batch, seq_len, self.num_kv_heads, self.head_dim])
            .map_err(Error::Numr)?;
        let v = var_permute(&v, &[0, 2, 1, 3]).map_err(Error::Numr)?;

        // QK-norm (Gemma): apply RmsNorm over head_dim to Q and K after reshape,
        // before RoPE. For BERT/NomicBert these are None and this is a no-op.
        let q = if let Some(qn) = &self.q_norm {
            qn.forward(client, &q)?
        } else {
            q
        };
        let k = if let Some(kn) = &self.k_norm {
            kn.forward(client, &k)?
        } else {
            k
        };

        // RoPE (NomicBert and Gemma): applied after QK-norm.
        // BERT/XLM-R: rope is None, these are no-ops.
        let q = if let Some(rope) = &self.rope {
            rope.forward(client, &q)?
        } else {
            q
        };
        let k = if let Some(rope) = &self.rope {
            rope.forward(client, &k)?
        } else {
            k
        };

        // GQA: repeat K and V head dimension when num_kv_heads < num_heads.
        // When equal (MHA), repeats == 1 and this block is skipped.
        let repeats = self.num_heads / self.num_kv_heads;
        let k = if repeats > 1 {
            let k_t = k.tensor().contiguous()?;
            let k_rep = client
                .repeat_interleave(&k_t, repeats, Some(1))
                .map_err(Error::Numr)?;
            Var::new(k_rep, false)
        } else {
            k
        };
        let v = if repeats > 1 {
            let v_t = v.tensor().contiguous()?;
            let v_rep = client
                .repeat_interleave(&v_t, repeats, Some(1))
                .map_err(Error::Numr)?;
            Var::new(v_rep, false)
        } else {
            v
        };

        // Fold the attention scale into Q (removes a [B,H,S,D] mul_scalar pass later).
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let q_scaled = Var::new(
            client.mul_scalar(q.tensor(), scale).map_err(Error::Numr)?,
            false,
        );
        let q_scaled = Var::new(q_scaled.tensor().contiguous()?, false);
        let k = Var::new(k.tensor().contiguous()?, false);
        let v = Var::new(v.tensor().contiguous()?, false);

        let k_t = var_transpose(&k).map_err(Error::Numr)?;
        let scores = var_matmul(&q_scaled, &k_t, client).map_err(Error::Numr)?;

        let attn_weights = if let Some(mask) = attention_mask {
            let mask_shape = mask.shape().to_vec();
            if mask_shape.len() != 2 || mask_shape[0] != batch || mask_shape[1] != seq_len {
                return Err(Error::ModelError {
                    reason: format!(
                        "attention_mask shape must be [{batch}, {seq_len}], got {:?}",
                        mask_shape
                    ),
                });
            }
            // Build the additive mask. Using -30000.0 for F16 (F16 max is ±65504;
            // -1e9 would overflow to -inf). -30000.0 fits in F16 and drives masked
            // positions to zero under softmax. F32 path uses -1e9 for existing behaviour.
            let inv = client.rsub_scalar(mask, 1.0).map_err(Error::Numr)?;
            let scores_dtype = scores.tensor().dtype();
            let additive_val = if scores_dtype == DType::F16 {
                -30000.0f64
            } else {
                -1e9f64
            };
            let additive_f32 = client.mul_scalar(&inv, additive_val).map_err(Error::Numr)?;
            let additive = if scores_dtype != DType::F32 {
                client
                    .cast(&additive_f32, scores_dtype)
                    .map_err(Error::Numr)?
            } else {
                additive_f32
            };
            let additive = additive
                .reshape(&[batch, 1, 1, seq_len])
                .map_err(Error::Numr)?;
            Var::new(
                client
                    .softmax_with_bias(scores.tensor(), &additive, -1)
                    .map_err(Error::Numr)?,
                false,
            )
        } else {
            Var::new(
                client.softmax(scores.tensor(), -1).map_err(Error::Numr)?,
                false,
            )
        };
        let attn_out = var_matmul(&attn_weights, &v, client).map_err(Error::Numr)?;

        let attn_out = var_permute(&attn_out, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let attn_out = Var::new(attn_out.tensor().contiguous()?, false);
        let hidden = self.num_heads * self.head_dim;
        let attn_out = var_reshape(&attn_out, &[batch, seq_len, hidden]).map_err(Error::Numr)?;

        let o = self.o_proj.forward(client, &attn_out)?;
        Ok(o)
    }

    fn ffn<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + ActivationOps<R>
            + ScalarOps<R>
            + QuantMatmulOps<R>
            + BinaryOps<R>
            + TypeConversionOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        match self.ffn_variant {
            FfnVariant::Standard => {
                let h = self.ffn_up.forward(client, x)?;
                let h = match self.hidden_act {
                    HiddenAct::Gelu => {
                        Var::new(client.gelu(h.tensor()).map_err(Error::Numr)?, false)
                    }
                    HiddenAct::Relu => {
                        Var::new(client.relu(h.tensor()).map_err(Error::Numr)?, false)
                    }
                };
                self.ffn_down.forward(client, &h)
            }
            FfnVariant::GatedSilu => {
                // SwiGLU: ffn_down(silu(ffn_gate(x)) * ffn_up(x))
                let gate = self
                    .ffn_gate
                    .as_ref()
                    .ok_or_else(|| Error::ModelError {
                        reason: "ffn_gate is None but ffn_variant is GatedSilu".into(),
                    })?
                    .forward(client, x)?;
                let up = self.ffn_up.forward(client, x)?;
                let gate_silu = var_silu(&gate, client).map_err(Error::Numr)?;
                let hidden = var_mul(&gate_silu, &up, client).map_err(Error::Numr)?;
                self.ffn_down.forward(client, &hidden)
            }
            FfnVariant::GatedGelu => {
                // GeGLU: ffn_down(gelu(ffn_gate(x)) * ffn_up(x))
                let gate = self
                    .ffn_gate
                    .as_ref()
                    .ok_or_else(|| Error::ModelError {
                        reason: "ffn_gate is None but ffn_variant is GatedGelu".into(),
                    })?
                    .forward(client, x)?;
                let up = self.ffn_up.forward(client, x)?;
                let gate_gelu = Var::new(client.gelu(gate.tensor()).map_err(Error::Numr)?, false);
                let hidden = var_mul(&gate_gelu, &up, client).map_err(Error::Numr)?;
                self.ffn_down.forward(client, &hidden)
            }
        }
    }
}
