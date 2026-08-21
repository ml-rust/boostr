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
pub(super) fn build_lm_head<R: Runtime<DType = DType>>(
    vb: &mut crate::nn::VarBuilder<R>,
    config: &ModelConfig,
    embed_tokens: &Embedding<R>,
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
        return Ok(MaybeQuantLinear::Standard(Linear::new(
            embed_w, None, false,
        )));
    }

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
                MaybeQuantLinear::Standard(Linear::new(weight, bias, requires_grad))
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
