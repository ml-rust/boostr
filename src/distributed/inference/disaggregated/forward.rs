//! Forward closures the caller builds over its model and runtime.
//!
//! The workers own transport, sampling and stop checks. The closures own the
//! model, the KV cache and its serialized format. Both closures run on a
//! blocking thread, never on an async worker thread.
//!
//! # Logits contract
//!
//! Every closure returns logits as a contiguous `[1, n, vocab]` tensor of
//! any float dtype, whose last row scores the last token fed. The sampler
//! reads that row only, so a chunked prefill returns its last chunk's rows.

use std::sync::Arc;

use anyhow::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Whether the router cancelled the request a [`PrefillFn`] runs.
///
/// A prefill that runs its prompt in chunks reads it before each chunk and
/// returns an error once it is set. The worker then drops the request with
/// no event, whatever the closure returned. A forward that runs is never
/// interrupted, so a prefill in one forward reads it at most once, before
/// that forward.
#[derive(Clone, Default)]
pub struct PrefillCancel {
    probe: Option<Arc<dyn Fn() -> bool + Send + Sync>>,
}

impl PrefillCancel {
    /// A probe that never reads as cancelled.
    pub fn never() -> Self {
        Self { probe: None }
    }

    /// A probe that reads as cancelled once `probe` returns `true`.
    pub fn new(probe: Arc<dyn Fn() -> bool + Send + Sync>) -> Self {
        Self { probe: Some(probe) }
    }

    /// Whether a cancel for the request arrived.
    pub fn is_cancelled(&self) -> bool {
        self.probe.as_ref().is_some_and(|probe| probe())
    }
}

/// What prefill returns for one prompt.
///
/// - `logits`: `[1, n, vocab]`, `n >= 1`, whose last row scores the last
///   prompt token. The first token samples from that row. A chunked prefill
///   returns its last chunk's rows.
/// - `kv_cache`: the cache after the prompt, serialized for [`DecodeOpenFn`].
pub struct PrefillOutput<R: Runtime> {
    pub logits: Tensor<R>,
    pub kv_cache: Vec<u8>,
}

/// Prefill over a prompt of token ids, reading the request's
/// [`PrefillCancel`] between chunks.
///
/// An `Err` ends the request with a [`super::Stage::Prefill`] error event,
/// unless the request was cancelled: a cancelled request ends with no event.
pub type PrefillFn<R> =
    Box<dyn Fn(&[u32], &PrefillCancel) -> Result<PrefillOutput<R>> + Send + Sync>;

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
