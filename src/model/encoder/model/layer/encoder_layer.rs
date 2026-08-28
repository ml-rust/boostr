//! [`EncoderLayer`] — one transformer block, and its residual/norm schemes.

use super::NormLayer;
use crate::error::{Error, Result};
use crate::model::encoder::config::{
    FfnVariant, HiddenAct, LayerAttention, NormScheme, QkNormScope,
};
use crate::nn::{MaybeQuantLinear, RmsNorm, RoPE};
use crate::ops::{RoPEOps, RoPEPackedOps, VarLenAttentionOps};
use crate::quant::traits::{DequantOps, QuantMatmulOps};
use numr::autograd::{Var, var_add};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, IndexingOps, NormalizationOps, ReduceOps, ScalarOps, ShapeOps,
    TensorOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use std::sync::Arc;

/// Packed (varlen) attention context for a forward pass over a ragged batch.
///
/// Carries pre-built host-derived metadata (cu_seqlens, position_ids) that is
/// constructed once per batch from host-side token lists and threaded through
/// the layer loop. No tensor DATA is transferred GPU↔CPU here.
pub(in crate::model::encoder) struct VarlenCtx<'a, R: Runtime> {
    /// Cumulative sequence lengths `[batch+1]` (I32).
    pub cu_seqlens: &'a Tensor<R>,
    /// Per-token absolute position ids `[total_tokens]` (I64), reset per sequence.
    pub position_ids: &'a Tensor<R>,
    /// Number of sequences in this packed batch.
    pub batch: usize,
    /// Length of the longest individual sequence.
    pub max_seqlen: usize,
}

/// A single transformer encoder layer: self-attention + FFN.
///
/// Supports four architectural styles via field variants:
/// - BERT/XLM-RoBERTa: post-add LayerNorm, no RoPE, no gate.
/// - NomicBert: post-add LayerNorm, RoPE, SwiGLU gate.
/// - Gemma-embedding: sandwich RMSNorm, QK-norm, RoPE, GQA, GeGLU gate.
/// - Qwen3: pre-norm RMSNorm with plain residuals, QK-norm, RoPE, GQA, SwiGLU.
pub(in crate::model::encoder) struct EncoderLayer<R: Runtime> {
    pub(in crate::model::encoder) q_proj: MaybeQuantLinear<R>,
    pub(in crate::model::encoder) k_proj: MaybeQuantLinear<R>,
    pub(in crate::model::encoder) v_proj: MaybeQuantLinear<R>,
    pub(in crate::model::encoder) o_proj: MaybeQuantLinear<R>,
    /// Attention norm. Post-add for BERT/NomicBert; pre-attention otherwise.
    pub(in crate::model::encoder) attn_norm: NormLayer<R>,
    pub(in crate::model::encoder) ffn_up: MaybeQuantLinear<R>,
    /// Gate projection: SwiGLU or GeGLU. `None` for BERT/XLM-R.
    pub(in crate::model::encoder) ffn_gate: Option<MaybeQuantLinear<R>>,
    pub(in crate::model::encoder) ffn_down: MaybeQuantLinear<R>,
    /// FFN norm. Post-add for BERT/NomicBert; pre-FFN otherwise.
    pub(in crate::model::encoder) ffn_norm: NormLayer<R>,
    pub(in crate::model::encoder) num_heads: usize,
    /// Number of KV heads. Equal to `num_heads` for MHA.
    pub(in crate::model::encoder) num_kv_heads: usize,
    pub(in crate::model::encoder) head_dim: usize,
    pub(in crate::model::encoder) hidden_act: HiddenAct,
    pub(in crate::model::encoder) ffn_variant: FfnVariant,
    /// Where normalisation sits relative to the residual add.
    pub(in crate::model::encoder) norm_scheme: NormScheme,
    /// How this block attends: RoPE base, window, causality.
    ///
    /// The RoPE cache below was built from `attn.rope_freq_base`, and the span
    /// mask handed to `forward` was built from `attn.window`/`attn.causal`.
    /// Both come from `EncoderConfig::layer_attention`, so they cannot disagree.
    pub(in crate::model::encoder) attn: LayerAttention,
    /// RoPE cache for this block's base. `None` for BERT/XLM-R.
    pub(in crate::model::encoder) rope: Option<Arc<RoPE<R>>>,
    /// QK-norm on Q, applied before RoPE. `None` for BERT/NomicBert.
    ///
    /// Both the norm type and the axis vary by architecture — RmsNorm over
    /// `head_dim` for Gemma/Qwen3, LayerNorm over the whole hidden vector for
    /// jina-bert-v2 — so `qk_norm_scope` says which axis this applies over.
    pub(in crate::model::encoder) q_norm: Option<NormLayer<R>>,
    /// QK-norm on K, applied before RoPE. `None` for BERT/NomicBert.
    pub(in crate::model::encoder) k_norm: Option<NormLayer<R>>,
    /// Which axis `q_norm`/`k_norm` normalise over. Ignored when both are
    /// `None`.
    pub(in crate::model::encoder) qk_norm_scope: QkNormScope,
    /// Second post-attention norm (jina-bert-v2's `attn_norm_2`).
    ///
    /// Applied only under [`NormScheme::PostNorm`], and it re-adds the layer
    /// input a *second* time before normalising:
    /// `x = attn_norm_2(attn_norm(x + ATTN(x)) + x)`. That extra residual is
    /// part of the architecture, not a duplicate of the first add.
    pub(in crate::model::encoder) attn_norm_2: Option<NormLayer<R>>,
    /// Post-attention sandwich norm (Gemma only).
    pub(in crate::model::encoder) post_attn_norm: Option<RmsNorm<R>>,
    /// Post-FFN sandwich norm (Gemma only).
    pub(in crate::model::encoder) post_ffn_norm: Option<RmsNorm<R>>,
}

impl<R: Runtime<DType = DType>> EncoderLayer<R> {
    /// Forward pass.
    ///
    /// `attention_mask` is the `[B, S]` padding mask (padded path only);
    /// `span_mask` is the additive `[1, 1, S, S]` window/causal mask, already
    /// selected for this layer's spec; `varlen_ctx` selects the packed path.
    pub(in crate::model::encoder) fn forward<C>(
        &self,
        client: &C,
        x: &Var<R>,
        attention_mask: Option<&Tensor<R>>,
        span_mask: Option<&Tensor<R>>,
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
        R::Client: TensorOps<R> + ScalarOps<R> + DequantOps<R>,
    {
        // Two independent axes: the norm scheme, and whether attention runs
        // packed or padded. All combinations are handled.
        let attend = |input: &Var<R>| -> Result<Var<R>> {
            match varlen_ctx {
                Some(ctx) => self.self_attention_varlen(client, input, ctx),
                None => self.self_attention_padded(client, input, attention_mask, span_mask),
            }
        };

        match self.norm_scheme {
            // x = x + post_attn_norm(ATTN(attn_norm(x)))
            // x = x + post_ffn_norm(FFN(ffn_norm(x)))
            NormScheme::Sandwich => {
                let attn_raw = attend(&self.attn_norm.forward(client, x)?)?;
                let attn_out = self
                    .post_attn_norm
                    .as_ref()
                    .ok_or_else(|| Error::ModelError {
                        reason: "post_attn_norm is required by the sandwich norm scheme".into(),
                    })?
                    .forward(client, &attn_raw)?;
                let x = var_add(x, &attn_out, client).map_err(Error::Numr)?;

                let ffn_raw = self.ffn(client, &self.ffn_norm.forward(client, &x)?)?;
                let ffn_out = self
                    .post_ffn_norm
                    .as_ref()
                    .ok_or_else(|| Error::ModelError {
                        reason: "post_ffn_norm is required by the sandwich norm scheme".into(),
                    })?
                    .forward(client, &ffn_raw)?;
                var_add(&x, &ffn_out, client).map_err(Error::Numr)
            }

            // x = x + ATTN(attn_norm(x))
            // x = x + FFN(ffn_norm(x))
            //
            // Not expressible as a sandwich layer with identity post-norms:
            // RmsNorm with an all-ones weight still divides by the RMS.
            NormScheme::PreNorm => {
                let attn_out = attend(&self.attn_norm.forward(client, x)?)?;
                let x = var_add(x, &attn_out, client).map_err(Error::Numr)?;

                let ffn_out = self.ffn(client, &self.ffn_norm.forward(client, &x)?)?;
                var_add(&x, &ffn_out, client).map_err(Error::Numr)
            }

            // x = attn_norm(x + ATTN(x))
            // x = attn_norm_2(x + input)      — jina-bert-v2 only
            // x = ffn_norm(x + FFN(x))
            NormScheme::PostNorm => {
                let attn_out = attend(x)?;
                let residual = var_add(x, &attn_out, client).map_err(Error::Numr)?;
                let normed = self.attn_norm.forward(client, &residual)?;

                // jina-bert-v2 re-adds the *layer input* (not the value just
                // normalised) and normalises again. Mirrors llama.cpp's
                // `llm_build_bert`, which does `cur = ggml_add(cur, inpL)`
                // a second time before `attn_norm_2`.
                let x = match &self.attn_norm_2 {
                    Some(norm) => {
                        let re_added = var_add(&normed, x, client).map_err(Error::Numr)?;
                        norm.forward(client, &re_added)?
                    }
                    None => normed,
                };

                let ffn_out = self.ffn(client, &x)?;
                let x = var_add(&x, &ffn_out, client).map_err(Error::Numr)?;
                self.ffn_norm.forward(client, &x)
            }
        }
    }
}
