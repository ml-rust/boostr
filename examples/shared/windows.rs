//! The corpus, the window selection, and the teacher-forced cross-entropy the
//! windows are scored with.
//!
//! Everything that decides WHICH tokens are scored lives here, because that is
//! the half of the measurement two runs must agree on exactly. There is no RNG
//! in this file and no sampling of any kind: the corpus is tokenized once in
//! file order, and window `i` starts at `i * stride`. Two runs of one command
//! line over one text file select the same windows, token for token.
//!
//! Shared by the `token_ce` and `imatrix` examples, in `examples/shared/`, so
//! there is exactly ONE definition of which tokens a run sees. Each binary
//! compiles this module separately and uses a subset of it — `imatrix`
//! selects windows but scores no loss — so an item unused by one of them is
//! not dead code.

#![allow(dead_code)]

use std::error::Error;

use boostr::model::llama::Llama;
use boostr::model::traits::{Model, ModelClient};
use boostr::nn::cross_entropy_loss;
use boostr::quant::traits::DequantOps;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use splintr::AnyTokenizer;

/// Tokenize the whole corpus, once, in file order.
///
/// `encode_raw`, never `encode`: `encode` would insert the tokenizer's BOS id,
/// and a BOS in the middle of a stride-selected window is not a token the
/// model was trained to predict there. No special token is added anywhere, so
/// what is scored is the corpus itself.
///
/// Ids widen to `i64` because that is what the embedding lookup and the
/// cross-entropy gather normalize their index tensors to; converting once here
/// keeps the per-window path free of it.
pub fn tokenize_corpus(tokenizer: &AnyTokenizer, text: &str) -> Vec<i64> {
    tokenizer
        .encode_raw(text)
        .into_iter()
        .map(|id| id as i64)
        .collect()
}

/// Select `windows` windows of `seq_len + 1` tokens each, starting at
/// `i * stride`.
///
/// `seq_len + 1`, not `seq_len`: the extra token is the last target. A window
/// feeds its first `seq_len` ids to the model and scores the model's
/// prediction of ids `1..=seq_len`, so every window contributes exactly
/// `seq_len` scored tokens and no logit slicing is needed.
///
/// A corpus too short for the request is an ERROR, never a short run. Silently
/// scoring fewer windows would make two numbers that look comparable average
/// over different token counts.
pub fn select_windows(
    tokens: &[i64],
    seq_len: usize,
    windows: usize,
    stride: usize,
) -> Result<Vec<&[i64]>, String> {
    if windows == 0 {
        return Err("--windows must be at least 1".to_string());
    }
    let span = seq_len + 1;
    let needed = (windows - 1) * stride + span;
    if tokens.len() < needed {
        return Err(format!(
            "corpus holds {} token(s); --windows {windows} at --seq-len {seq_len} and \
             --stride {stride} needs {needed}. Pass a longer --text, fewer --windows, \
             a shorter --seq-len, or a smaller --stride",
            tokens.len()
        ));
    }
    Ok((0..windows)
        .map(|i| {
            let start = i * stride;
            &tokens[start..start + span]
        })
        .collect())
}

/// Mean token cross-entropy in NATS over `windows`, teacher forced, and the
/// token count it averaged over.
///
/// One window at a time, batch size 1, in the order [`select_windows`]
/// produced them. The accumulation is `f64` over that fixed order, so the
/// summation order is pinned along with the window order.
///
/// The logits are cast to F32 before the loss whenever the model's own dtype
/// is narrower. A BF16 log-softmax over a 100k-wide vocabulary throws away
/// mantissa bits that are the same order as the differences this measurement
/// is looking for.
pub fn score_windows<R, C>(
    model: &Llama<R>,
    client: &C,
    device: &R::Device,
    windows: &[&[i64]],
) -> Result<(f64, usize), Box<dyn Error>>
where
    R: Runtime<DType = DType>,
    C: ModelClient<R> + DequantOps<R> + TypeConversionOps<R>,
    R::Client: ModelClient<R> + DequantOps<R> + TypeConversionOps<R>,
{
    let mut total = 0.0f64;
    let mut scored = 0usize;

    for (index, window) in windows.iter().enumerate() {
        let seq_len = window.len() - 1;
        let inputs = Tensor::<R>::from_slice(&window[..seq_len], &[1, seq_len], device)?;
        let targets = Tensor::<R>::from_slice(&window[1..], &[1, seq_len], device)?;

        // Teacher forced by construction: the whole window is presented at
        // once and the causal mask inside the attention block is what stops
        // position `t` from seeing its own target. No KV cache, no decode
        // loop, no sampling — the model never consumes its own output here.
        let logits = model.forward(client, &Var::new(inputs, false))?;
        let logits = if logits.tensor().dtype() == DType::F32 {
            logits.tensor().clone()
        } else {
            client.cast(logits.tensor(), DType::F32)?
        };

        let loss = cross_entropy_loss(client, &Var::new(logits, false), &targets)?;
        let mean = loss.tensor().try_to_vec::<f32>()?[0] as f64;
        total += mean * seq_len as f64;
        scored += seq_len;

        eprintln!(
            "window {}/{}: mean cross-entropy {mean:.6} nats over {seq_len} token(s)",
            index + 1,
            windows.len()
        );
    }

    Ok((total / scored as f64, scored))
}
