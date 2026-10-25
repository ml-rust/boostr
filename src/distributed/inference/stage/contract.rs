//! The interface a swarm worker serves its stage through.

use anyhow::Result;

use super::super::transport::{Activation, GenRequestHeader, StagePayload};

/// What one forward of a stage produces.
#[derive(Debug, Clone, PartialEq)]
pub enum StageOutput {
    /// The input of the next stage.
    Activation(Activation),
    /// The last stage's logits for the last position: one row of `vocab_size`.
    Logits(Vec<f32>),
}

/// One pipeline stage. [`SwarmWorker`](super::super::worker::SwarmWorker)
/// serves it, and the leader runs its own stage through it.
///
/// A stage keeps per-request state keyed by
/// [`GenRequestHeader::request_id`]. Calls on one stage never overlap.
pub trait Stage: Send {
    /// Runs one prefill or decode frame of a request.
    ///
    /// The worker never passes a release here: it calls [`Stage::release`].
    ///
    /// # Errors
    ///
    /// Any error fails this request only. The worker then calls
    /// [`Stage::release`] for it and keeps serving.
    fn forward(&mut self, header: &GenRequestHeader, payload: StagePayload) -> Result<StageOutput>;

    /// Frees the state of `request_id`. An unknown id is a no-op.
    fn release(&mut self, request_id: u64);

    /// Frees the state of every request. The worker calls it when its loop ends.
    fn release_all(&mut self);
}
