//! `lm_head` construction for `Llama::from_varbuilder` — tied-weight reuse,
//! plus checkpoint/config vocab-row reconciliation (see `crate::model::vocab_growth`).

use crate::error::{Error, Result};
use crate::model::config::ModelConfig;
use crate::model::vocab_growth::fit_vocab_rows;
use crate::nn::{Embedding, Linear, MaybeQuantLinear};
use numr::dtype::DType;
use numr::ops::{ReduceOps, ShapeOps};
use numr::runtime::Runtime;

/// Build `lm_head`, tying it to `embed_tokens` when configured, otherwise
/// loading it from `vb` and reconciling its row count with `config.vocab_size`.
///
/// `embed_name` is the checkpoint key `embed_tokens` was read from. Both
/// branches below can mint a FRESH autograd id — the tied head wraps a clone
/// of the embedding, and a grown head wraps a reshaped tensor — and an
/// importance collection keys on that id, so each rebuilt layer re-binds its
/// checkpoint key through `imatrix::register_name`. Without it the head's
/// statistics would arrive nameless and `imatrix::finish` would refuse the
/// whole run rather than guess.
pub(super) fn build_lm_head<R: Runtime<DType = DType>>(
    vb: &mut crate::nn::VarBuilder<R>,
    config: &ModelConfig,
    embed_tokens: &Embedding<R>,
    embed_name: &str,
) -> Result<MaybeQuantLinear<R>>
where
    R::Client:
        crate::quant::DequantOps<R> + numr::ops::TypeConversionOps<R> + ReduceOps<R> + ShapeOps<R>,
{
    if config.tie_word_embeddings {
        // `embed_tokens` was already grown to `config.vocab_size` before this
        // call — cloning it here is what keeps the tied embedding and
        // lm_head structurally identical.
        let embed_w = embed_tokens.weight().tensor().clone();
        let head = Linear::new(embed_w, None, false);
        // A tied head multiplies by the EMBEDDING tensor, so that is the
        // checkpoint key a quantizer will look this importance vector up
        // under — the statistics themselves are the head's own, taken from
        // the hidden states that feed the output projection.
        crate::quant::imatrix::register_name(head.weight().id(), embed_name);
        return Ok(MaybeQuantLinear::Standard(head));
    }

    let head_name = vb.full_name("lm_head.weight");
    let lm_head = vb.take_maybe_quant_linear("lm_head.weight", None)?;
    // A head padded above vocab_size is a normal layout and loads as-is;
    // only a head with too FEW rows needs growth. Mean-init isn't
    // representable in a block-quantized layout, so that case must fail
    // loudly rather than silently skip the growth it was asked for.
    let quant_row_mismatch = |rows: usize| -> Error {
        Error::ModelError {
            reason: format!(
                "lm_head.weight: quantized head has {rows} rows but \
                 config.vocab_size is {}; mean-init into a block-quantized \
                 lm_head is not representable, dequantize the head first",
                config.vocab_size
            ),
        }
    };
    Ok(match lm_head {
        MaybeQuantLinear::Standard(linear) => {
            let rows = weight_rows(linear.weight().tensor().shape());
            if rows >= config.vocab_size {
                // Already big enough — leave the loaded `Linear` alone
                // rather than rebuilding it around an identical tensor.
                MaybeQuantLinear::Standard(linear)
            } else {
                let requires_grad = linear.weight().requires_grad();
                let bias = linear.bias().map(|b| b.tensor().clone());
                let weight = fit_vocab_rows(
                    linear.weight().tensor().clone(),
                    config,
                    vb.device(),
                    "lm_head.weight",
                )?;
                let grown = Linear::new(weight, bias, requires_grad);
                // `Linear::new` minted a fresh id; re-bind the same key the
                // loader used, or this head's statistics arrive nameless.
                crate::quant::imatrix::register_name(grown.weight().id(), &head_name);
                MaybeQuantLinear::Standard(grown)
            }
        }
        MaybeQuantLinear::Quantized(qlinear) => {
            let rows = weight_rows(qlinear.weight().shape());
            if rows < config.vocab_size {
                return Err(quant_row_mismatch(rows));
            }
            MaybeQuantLinear::Quantized(qlinear)
        }
        MaybeQuantLinear::DecomposedQuant(dqlinear) => {
            let rows = weight_rows(dqlinear.weight().shape());
            if rows < config.vocab_size {
                return Err(quant_row_mismatch(rows));
            }
            MaybeQuantLinear::DecomposedQuant(dqlinear)
        }
    })
}

/// Row count (dim 0) of a weight shape, or 0 for a rank-0 shape.
fn weight_rows(shape: &[usize]) -> usize {
    shape.first().copied().unwrap_or(0)
}

impl<R: Runtime<DType = DType>> super::Llama<R> {
    /// Token embeddings, times `block.embedding_multiplier` when set (Gemma).
    pub(crate) fn embed<C>(
        &self,
        client: &C,
        input_ids: &numr::tensor::Tensor<R>,
    ) -> Result<numr::autograd::Var<R>>
    where
        C: crate::model::traits::ModelClient<R>,
        R::Client: numr::ops::IndexingOps<R> + numr::ops::ScalarOps<R> + numr::ops::TensorOps<R>,
    {
        let hidden = self.embed_tokens.forward(client, input_ids)?;
        match self.config.block.embedding_multiplier {
            Some(m) => {
                numr::autograd::var_mul_scalar(&hidden, f64::from(m), client).map_err(Error::Numr)
            }
            None => Ok(hidden),
        }
    }

    /// `lm_head(hidden)`, times `block.logit_scale` (Cohere), then
    /// `cap * tanh(logits / cap)` for `block.final_logit_softcap` (Gemma 2).
    pub(crate) fn head<C>(
        &self,
        client: &C,
        hidden: &numr::autograd::Var<R>,
    ) -> Result<numr::autograd::Var<R>>
    where
        C: crate::model::traits::ModelClient<R>,
        R::Client: numr::ops::TensorOps<R>
            + numr::ops::ScalarOps<R>
            + numr::ops::ReduceOps<R>
            + numr::ops::IndexingOps<R>
            + numr::ops::ShapeOps<R>
            + numr::ops::ActivationOps<R>
            + numr::ops::BinaryOps<R>
            + numr::ops::UnaryOps<R>
            + numr::ops::CompareOps<R>
            + numr::ops::ConditionalOps<R>
            + crate::quant::traits::DequantOps<R>,
    {
        use numr::autograd::{var_mul_scalar, var_tanh};
        let mut logits = self.lm_head.forward(client, hidden)?;
        if let Some(scale) = self.config.block.logit_scale {
            logits = var_mul_scalar(&logits, f64::from(scale), client).map_err(Error::Numr)?;
        }
        if let Some(cap) = self.config.block.final_logit_softcap {
            let cap = f64::from(cap);
            let t = var_mul_scalar(&logits, cap.recip(), client).map_err(Error::Numr)?;
            let t = var_tanh(&t, client).map_err(Error::Numr)?;
            logits = var_mul_scalar(&t, cap, client).map_err(Error::Numr)?;
        }
        Ok(logits)
    }

    /// An error unless every block is the plain Llama block, for the CUDA
    /// graph paths, which implement only that block.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn ensure_plain_llama(&self, path: &str) -> Result<()> {
        if self.config.block.is_llama() && self.layers.iter().all(|l| l.is_plain_llama()) {
            return Ok(());
        }
        Err(Error::ModelError {
            reason: format!(
                "the {path} path runs the Llama block only; model type '{}' needs its own \
                 norms, MLP or attention options. Run it without graph mode",
                self.config.model_type
            ),
        })
    }
}
