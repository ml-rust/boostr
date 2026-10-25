//! Disaggregated prefill/decode inference over nexar.
//!
//! The router admits a request. A prefill worker runs the prompt and samples
//! the first token. A decode worker resumes from the KV handoff and samples
//! the rest. Each token streams back as one event. `router` documents
//! streaming, errors and cancel. `protocol` documents the wire format.
//! `handoff` documents how decode confirms each KV handoff.

mod cancel;
pub mod config;
pub mod decode_worker;
pub mod forward;
mod handoff;
pub mod prefill_worker;
pub mod protocol;
pub mod router;
pub mod sampler;
#[cfg(test)]
mod test_harness;
mod wire;

pub use config::{DisaggConfig, DisaggRole};
pub use decode_worker::DecodeWorker;
pub use forward::{DecodeOpenFn, DecodeStepFn, PrefillFn, PrefillOutput};
pub use prefill_worker::PrefillWorker;
pub use protocol::*;
pub use router::{DisaggRouter, Generation, TokenStream, WorkerError};
pub use sampler::RequestSampler;
