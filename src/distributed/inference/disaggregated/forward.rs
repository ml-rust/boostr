//! Forward closures the caller builds over its model and runtime.
//!
//! The workers own transport, sampling and stop checks. The closures own the
//! model, the KV cache and its serialized format. Both closures run on a
//! blocking thread, never on an async worker thread.
//!
//! # Logits contract
//!
//! Every closure returns logits as a contiguous `[1, seq_len, vocab]` tensor
//! of any float dtype. The sampler reads the last position only.

use anyhow::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// What prefill returns for one prompt.
///
/// - `logits`: `[1, prompt_len, vocab]`. The first token samples from the last row.
/// - `kv_cache`: the cache after the prompt, serialized for [`DecodeOpenFn`].
pub struct PrefillOutput<R: Runtime> {
    pub logits: Tensor<R>,
    pub kv_cache: Vec<u8>,
}

/// Prefill over a prompt of token ids.
///
/// An `Err` ends the request with a [`super::Stage::Prefill`] error event.
pub type PrefillFn<R> = Box<dyn Fn(&[u32]) -> Result<PrefillOutput<R>> + Send + Sync>;

/// One decode step of one request: `(token, position) -> logits`.
///
/// - `token`: the last sampled token, fed as the next input.
/// - `position`: the sequence position of `token`, which is the prompt
///   length plus the tokens sampled before it.
///
/// The closure owns the request's deserialized cache and appends `token` to
/// it. It returns `[1, 1, vocab]` logits. An `Err` ends the request with a
/// [`super::Stage::Decode`] error event. Dropping the closure frees the cache.
pub type DecodeStepFn<R> = Box<dyn FnMut(u32, u32) -> Result<Tensor<R>> + Send>;

/// Opens one request on a decode worker from the serialized cache prefill sent.
///
/// The decode worker calls it once per request and drops the returned step
/// closure when the request finishes, fails or is cancelled. An `Err` ends the
/// request with a [`super::Stage::Decode`] error event.
pub type DecodeOpenFn<R> = Box<dyn Fn(&[u8]) -> Result<DecodeStepFn<R>> + Send + Sync>;
