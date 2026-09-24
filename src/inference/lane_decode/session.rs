//! A fixed set of decode lanes and the driver that steps them.
//!
//! The session owns, allocated once: the batched KV cache and GDN state at
//! `batch` lanes, the `LaneTable` that describes every lane's key range and
//! IMROPE positions, the decode driver, and the host mirror of the token each
//! lane decodes from next.
//!
//! [`LaneDecodeSession::install`] takes a prefilled one-lane cache pair into a
//! free lane. [`LaneDecodeSession::launch`] runs one decode step for every
//! lane and returns the `[batch, 1, vocab]` logits.
//! [`LaneDecodeSession::release`] gives a lane back for the next sequence.
//!
//! The session picks no token and holds no sampling policy: the caller reads
//! each lane's row with [`LaneDecodeSession::logits_row`], samples it however
//! that lane's request asks, and writes the result back through
//! [`LaneDecodeSession::set_next_token`] plus
//! [`LaneDecodeSession::commit_next_tokens`]. A lane holds its sequence from
//! cache position 0 with its own end, so a lane produces the tokens it would
//! produce alone, whatever the other lanes hold.

use numr::runtime::Runtime;
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
use numr::tensor::Tensor;

use crate::error::{Error, Result};
use crate::inference::lanes::LaneAdmission;
use crate::inference::{LayeredGdnState, LayeredKvCache};
use crate::model::qwen35::Qwen35Model;

use super::build::{LaneDriverSpec, build_lane_driver};
use super::driver::{LaneDecodeMode, LaneDriver};

/// A fixed set of decode lanes, stepped together.
pub struct LaneDecodeSession {
    /// The caches, the lane table and the decode path.
    driver: LaneDriver,
    /// Host mirror of the input token each lane decodes from next.
    next_tokens: Vec<i64>,
    /// Tokens one lane holds at most, which both caches were built for.
    capacity: usize,
    /// Sequence id the next install hands its lane.
    next_seq: u64,
    /// Graph captures the build performed.
    graph_captures: usize,
}

impl LaneDecodeSession {
    /// Build the driver `spec` describes and open a session over it, every
    /// lane free.
    ///
    /// # Errors
    ///
    /// [`Error::InferenceError`] when an allocation fails, or when the graph
    /// arm's warmup or capture fails.
    pub fn open(spec: LaneDriverSpec<'_>) -> Result<Self> {
        let lanes = spec.lanes;
        let capacity = spec.capacity;
        let build = build_lane_driver(spec)?;
        Ok(Self {
            driver: build.driver,
            next_tokens: vec![0; lanes],
            capacity,
            next_seq: 0,
            graph_captures: build.graph_captures,
        })
    }

    /// The default client of the device the caches sit on.
    pub fn client(device: &CudaDevice) -> CudaClient {
        CudaRuntime::default_client(device)
    }

    /// Lane count one step serves.
    pub fn batch(&self) -> usize {
        self.driver.batch()
    }

    /// Tokens one lane holds at most.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Which decode path this session steps.
    pub fn mode(&self) -> LaneDecodeMode {
        self.driver.mode()
    }

    /// How many lanes are free for a new sequence.
    pub fn free_lanes(&self) -> usize {
        self.driver.lanes().free_lanes()
    }

    /// Graph captures the build performed: 0 on the eager arm, 1 or 2 on the
    /// graph arm.
    pub fn graph_captures(&self) -> usize {
        self.graph_captures
    }

    /// Install a prefilled sequence into a free lane and return that lane.
    ///
    /// `kv` and `gdn` are one sequence's own prefill, which lands at cache
    /// position 0 of its lane. `rope_pos` is the position the sequence's first
    /// decoded token sits at. The session hands the lane the next sequence id.
    ///
    /// # Errors
    ///
    /// [`Error::InferenceError`] when every lane is taken, when `rope_pos` is
    /// outside the model's rope table, or when the install fails.
    pub fn install(
        &mut self,
        client: &CudaClient,
        kv: &LayeredKvCache<CudaRuntime>,
        gdn: &LayeredGdnState<CudaRuntime>,
        rope_pos: usize,
    ) -> Result<usize> {
        let seq = self.next_seq;
        let (table, batched_kv, batched_gdn) = self.driver.parts();
        let lane = table.admit(
            client,
            batched_kv,
            batched_gdn,
            LaneAdmission {
                seq,
                kv,
                gdn,
                rope_pos,
            },
        )?;
        self.next_seq += 1;
        Ok(lane)
    }

    /// Record the token lane `lane` decodes from next.
    ///
    /// The write reaches the device on the next
    /// [`LaneDecodeSession::commit_next_tokens`].
    ///
    /// # Errors
    ///
    /// [`Error::InferenceError`] when `lane` is outside the batch.
    pub fn set_next_token(&mut self, lane: usize, token_id: u32) -> Result<()> {
        let batch = self.next_tokens.len();
        let slot = self
            .next_tokens
            .get_mut(lane)
            .ok_or_else(|| Error::InferenceError {
                reason: format!(
                    "lane {lane} is outside the {batch} lanes this session holds; set a next \
                     token only on a lane an install handed out"
                ),
            })?;
        *slot = i64::from(token_id);
        Ok(())
    }

    /// Write every lane's next input token to the device in one copy.
    ///
    /// # Errors
    ///
    /// [`Error::InferenceError`] when the write fails.
    pub fn commit_next_tokens(&self, client: &CudaClient) -> Result<()> {
        self.driver.set_next_tokens(client, &self.next_tokens)
    }

    /// One decode step for every lane: `[batch, 1, vocab]` logits, row `b`
    /// for lane `b`.
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
        self.driver.launch(client, model)
    }

    /// Lane `lane`'s own contiguous logits row out of a step's `logits`.
    ///
    /// # Errors
    ///
    /// [`Error::InferenceError`] when the narrow or the copy fails.
    pub fn logits_row(
        &self,
        logits: &Tensor<CudaRuntime>,
        lane: usize,
    ) -> Result<Tensor<CudaRuntime>> {
        logits
            .narrow(0, lane, 1)
            .and_then(|row| row.contiguous())
            .map_err(|e| Error::InferenceError {
                reason: format!(
                    "reading lane {lane}'s logits row out of {:?} failed: {e}; read a row only \
                     for a lane inside the batch",
                    logits.shape()
                ),
            })
    }

    /// Give lane `lane` back: clear its GDN state, mark it free and drop the
    /// token it was going to decode from.
    ///
    /// # Errors
    ///
    /// [`Error::InferenceError`] when `lane` is outside the batch, or the
    /// clear fails.
    pub fn release(&mut self, client: &CudaClient, lane: usize) -> Result<()> {
        self.driver.release(client, lane)?;
        self.set_next_token(lane, 0)
    }
}
