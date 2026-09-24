//! The decode driver a lane session steps every token.
//!
//! Both arms hold the batched KV cache, the batched GDN state and the
//! `LaneTable` that describes every lane's key range and IMROPE positions.
//! `Graph` keeps the table inside its captured graph, which is the single
//! copy the replay reads; `Eager` keeps its own beside the caches.
//!
//! One launch produces `[batch, 1, vocab]` logits, row `b` for lane `b`.
//! Neither arm picks a token: the caller samples each row on its own policy
//! and writes the results back through [`LaneDriver::set_next_tokens`].

use numr::ops::HostCopyOps;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use crate::error::{Error, Result};
use crate::inference::decode_graph::Qwen35DecodeGraph;
use crate::inference::{LaneTable, LayeredGdnState, LayeredKvCache};
use crate::model::qwen35::Qwen35Model;

/// Lane table, caches and input buffer the eager batched decode reads.
pub struct EagerLanes {
    /// Per-lane key range and IMROPE positions.
    pub lanes: LaneTable<CudaRuntime>,
    /// Batched KV cache for the attention layers, one lane per sequence.
    pub kv: LayeredKvCache<CudaRuntime>,
    /// Batched GDN state for the GDN layers, one row per lane.
    pub gdn: LayeredGdnState<CudaRuntime>,
    /// `[batch, 1]` I64 token each lane decodes from this step, allocated
    /// once and written in place by one stream-ordered host-to-device copy.
    pub token_buf: Tensor<CudaRuntime>,
}

/// Captured decode graph and the caches its kernels address.
///
/// `decode` is declared first so the graph drops before the caches it
/// writes. Its `lanes` field is the table every replay reads.
pub struct GraphLanes {
    /// The captured graph, its stable buffers and its lane table.
    pub decode: Qwen35DecodeGraph,
    /// Batched KV cache for the attention layers, one lane per sequence.
    pub kv: LayeredKvCache<CudaRuntime>,
    /// Batched GDN state for the GDN layers, one row per lane.
    pub gdn: LayeredGdnState<CudaRuntime>,
}

/// Which decode path a lane session drives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaneDecodeMode {
    /// `forward_qwen35_batched_decode`, one launch per kernel.
    Eager,
    /// One captured CUDA graph replayed per token.
    Graph,
}

/// The decode path a lane session steps, holding everything one step reads.
pub enum LaneDriver {
    /// Eager batched decode over the session's own lane table.
    Eager(Box<EagerLanes>),
    /// Captured multi-lane decode graph replayed per token.
    Graph(Box<GraphLanes>),
}

impl LaneDriver {
    /// Which path this driver takes.
    pub fn mode(&self) -> LaneDecodeMode {
        match self {
            LaneDriver::Eager(_) => LaneDecodeMode::Eager,
            LaneDriver::Graph(_) => LaneDecodeMode::Graph,
        }
    }

    /// Lane count one step serves.
    pub fn batch(&self) -> usize {
        self.lanes().batch()
    }

    /// The lane table this driver steps.
    pub fn lanes(&self) -> &LaneTable<CudaRuntime> {
        match self {
            LaneDriver::Eager(e) => &e.lanes,
            LaneDriver::Graph(g) => &g.decode.lanes,
        }
    }

    /// The lane table and both batched caches, together, for an admit.
    pub fn parts(
        &mut self,
    ) -> (
        &mut LaneTable<CudaRuntime>,
        &mut LayeredKvCache<CudaRuntime>,
        &mut LayeredGdnState<CudaRuntime>,
    ) {
        match self {
            LaneDriver::Eager(e) => (&mut e.lanes, &mut e.kv, &mut e.gdn),
            LaneDriver::Graph(g) => (&mut g.decode.lanes, &mut g.kv, &mut g.gdn),
        }
    }

    /// Set the input token every lane decodes from next, in lane order.
    ///
    /// Both arms write their `[batch, 1]` token buffer in place with one
    /// stream-ordered host-to-device copy, so the write allocates nothing and
    /// the buffer's address holds for the life of the driver.
    ///
    /// # Errors
    ///
    /// [`Error::InferenceError`] when `tokens` is not exactly `batch` long, or
    /// the write fails.
    pub fn set_next_tokens(&self, client: &CudaClient, tokens: &[i64]) -> Result<()> {
        let batch = self.batch();
        if tokens.len() != batch {
            return Err(Error::InferenceError {
                reason: format!(
                    "lane driver holds {batch} lanes but {} next tokens were given; pass one \
                     token per lane, in lane order",
                    tokens.len()
                ),
            });
        }
        match self {
            LaneDriver::Eager(e) => client
                .write_host_slice(&e.token_buf, tokens)
                .map_err(|err| Error::InferenceError {
                    reason: format!(
                        "writing {batch} next tokens into the eager [{batch}, 1] token \
                             buffer failed: {err}; check that the buffer is still allocated on \
                             this device"
                    ),
                }),
            LaneDriver::Graph(g) => g.decode.set_next_tokens(client, tokens),
        }
    }

    /// One decode step for every lane: `[batch, 1, vocab]` logits.
    ///
    /// The lane table already describes this step, so the caller writes it
    /// only through admit, release and this call. Both arms count the token
    /// every occupied lane just decoded before returning, so the table is
    /// ready for the next step.
    ///
    /// The returned tensor is the graph's stable logits buffer in `Graph`
    /// mode and the forward's own output in `Eager` mode. Row `b` is lane
    /// `b`.
    ///
    /// # Errors
    ///
    /// [`Error::InferenceError`] when the launch fails, or when an occupied
    /// lane's next rope position is outside the model's rope table.
    pub fn launch(
        &mut self,
        client: &CudaClient,
        model: &Qwen35Model<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        match self {
            LaneDriver::Graph(g) => {
                g.decode.pre_replay_and_launch(client)?;
                Ok(g.decode.logits().clone())
            }
            LaneDriver::Eager(e) => {
                let EagerLanes {
                    lanes,
                    kv,
                    gdn,
                    token_buf,
                } = e.as_mut();
                let logits = {
                    let step = lanes.step();
                    model.forward_qwen35_batched_decode(
                        client,
                        token_buf,
                        lanes.positions(),
                        kv,
                        gdn,
                        &step,
                    )?
                };
                lanes.advance(client)?;
                Ok(logits)
            }
        }
    }

    /// Clear lane `lane`'s GDN state and mark the lane free.
    ///
    /// # Errors
    ///
    /// [`Error::InferenceError`] when `lane` is outside the batch, or the
    /// clear fails.
    pub fn release(&mut self, client: &CudaClient, lane: usize) -> Result<()> {
        let (lanes, _kv, gdn) = self.parts();
        lanes.release(client, gdn, lane)
    }

    /// Reset both caches and the lane table to an empty session.
    ///
    /// # Errors
    ///
    /// [`Error::InferenceError`] when a reset or a lane release fails.
    pub fn reset(&mut self, client: &CudaClient) -> Result<()> {
        for lane in 0..self.batch() {
            self.release(client, lane)?;
        }
        let (_lanes, kv, gdn) = self.parts();
        kv.reset();
        gdn.reset(client)?;
        Ok(())
    }
}
