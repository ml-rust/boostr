//! Disaggregated router: admission, worker choice and per-request token streams.
//!
//! - Stream: the first token arrives when prefill samples it, then one event per decode step.
//! - Stop: an EOS id, a stop id or `max_new_tokens` ends the stream. The last token carries the `FinishReason`.
//! - Errors: a failed forward, sample, transfer or peer ends the stream with an error item.
//! - Cancel: dropping a `TokenStream` early tells both workers to stop and free the request.
//! - Shutdown: `DisaggRouter::stop_workers` ends every worker's run loop.
//!
//! - `routing`: `DisaggRouter`
//! - `dispatch`: in-flight requests, event reordering, `WorkerError`
//! - `shutdown`: `DisaggRouter::stop_workers`
//! - `stream`: `TokenStream`, `Generation`

pub mod dispatch;
pub mod routing;
pub mod shutdown;
pub mod stream;

pub use dispatch::WorkerError;
pub use routing::DisaggRouter;
pub use stream::{Generation, TokenStream};
