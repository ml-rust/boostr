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
//! that lane's request asks — using that lane's own [`SamplingScratch`] from
//! [`LaneDecodeSession::scratch`] to sample without allocating — and writes
//! the result back through [`LaneDecodeSession::set_next_token`] plus
//! [`LaneDecodeSession::commit_next_tokens`]. A lane holds its sequence from
//! cache position 0 with its own end, so a lane produces the tokens it would
//! produce alone, whatever the other lanes hold.
//!
//! [`logits_row`](LaneDecodeSession::logits_row) and each lane's
//! [`scratch`](LaneDecodeSession::scratch) are allocated once, per lane, at
//! [`LaneDecodeSession::open`], and every call fully overwrites them before
//! returning — so a step reads or samples a lane's row without allocating,
//! and reusing these buffers across steps or lanes never carries a value
//! from one call into another.

use numr::ops::BinaryOps;
use numr::runtime::Runtime;
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
use numr::tensor::Tensor;

use crate::error::{Error, Result};
use crate::inference::lanes::LaneAdmission;
use crate::inference::{LayeredGdnState, LayeredKvCache};
use crate::model::qwen35::Qwen35Model;
use crate::ops::SamplingScratch;

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
    /// One `[1, 1, vocab]` contiguous row buffer per lane, matching the
    /// model's own output dtype. [`Self::logits_row`] fills it in place
    /// from that lane's strided slice of a step's logits.
    row_buffers: Vec<Tensor<CudaRuntime>>,
    /// One sampling scratch per lane, sized for the model's vocab.
    /// [`Self::scratch`] hands it to the caller's own sampling call.
    scratch: Vec<SamplingScratch<CudaRuntime>>,
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
        let vocab_size = spec.model.config().vocab_size;
        let dtype = spec.model.dtype();
        let device = spec.device;
        let mut row_buffers = Vec::with_capacity(lanes);
        let mut scratch = Vec::with_capacity(lanes);
        for _ in 0..lanes {
            row_buffers.push(Tensor::<CudaRuntime>::zeros(
                &[1, 1, vocab_size],
                dtype,
                device,
            )?);
            scratch.push(SamplingScratch::new(vocab_size, device)?);
        }
        let build = build_lane_driver(spec)?;
        Ok(Self {
            driver: build.driver,
            next_tokens: vec![0; lanes],
            capacity,
            next_seq: 0,
            graph_captures: build.graph_captures,
            row_buffers,
            scratch,
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

    /// Lane `lane`'s own contiguous logits row out of a step's `logits`,
    /// written into that lane's own row buffer.
    ///
    /// Copies lane `lane`'s strided `[1, 1, vocab]` slice of `logits` into
    /// the row buffer [`Self::open`] allocated for it, so this call
    /// allocates nothing. The write fully overwrites the buffer, so lane
    /// `lane`'s row from an earlier step never survives into this one, and
    /// another lane's buffer is untouched.
    ///
    /// # Errors
    ///
    /// [`Error::InferenceError`] when `lane` is outside the batch, or the
    /// narrow or the copy fails.
    pub fn logits_row(
        &self,
        client: &CudaClient,
        logits: &Tensor<CudaRuntime>,
        lane: usize,
    ) -> Result<&Tensor<CudaRuntime>> {
        let row_buf = self
            .row_buffers
            .get(lane)
            .ok_or_else(|| Error::InferenceError {
                reason: format!(
                    "lane {lane} is outside the {} lanes this session holds; read a row only \
                     for a lane inside the batch",
                    self.row_buffers.len()
                ),
            })?;
        fill_row_buffer(client, logits, row_buf, lane)?;
        Ok(&self.row_buffers[lane])
    }

    /// Lane `lane`'s own sampling scratch, sized for the model's vocab.
    ///
    /// # Errors
    ///
    /// [`Error::InferenceError`] when `lane` is outside the batch.
    pub fn scratch(&self, lane: usize) -> Result<&SamplingScratch<CudaRuntime>> {
        self.scratch.get(lane).ok_or_else(|| Error::InferenceError {
            reason: format!(
                "lane {lane} is outside the {} lanes this session holds; read a scratch only \
                 for a lane inside the batch",
                self.scratch.len()
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

/// Copies lane `lane`'s strided `[1, 1, vocab]` slice of `logits` into
/// `row_buf`, which must already be a `[1, 1, vocab]` contiguous buffer of
/// `logits`'s dtype.
///
/// The narrow-then-copy this performs is all of what
/// [`LaneDecodeSession::logits_row`] does; it is a free function so it can
/// be exercised directly, against hand-built logits and row buffers,
/// without a full session.
fn fill_row_buffer(
    client: &CudaClient,
    logits: &Tensor<CudaRuntime>,
    row_buf: &Tensor<CudaRuntime>,
    lane: usize,
) -> Result<()> {
    let view = logits
        .narrow(0, lane, 1)
        .map_err(|e| Error::InferenceError {
            reason: format!(
                "narrowing lane {lane}'s logits row out of {:?} failed: {e}; read a row only for a \
             lane inside the batch",
                logits.shape()
            ),
        })?;
    client
        .copy_into(row_buf, &view)
        .map_err(|e| Error::InferenceError {
            reason: format!("copying lane {lane}'s logits row into its row buffer failed: {e}"),
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cuda_setup;
    use numr::dtype::DType;

    /// A `[2, 1, 4]` batch of two lanes' logits, each row a distinct,
    /// recognizable sequence of values.
    fn two_lane_logits(device: &CudaDevice) -> Tensor<CudaRuntime> {
        let data: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0];
        Tensor::<CudaRuntime>::from_slice(&data, &[2, 1, 4], device).expect("logits")
    }

    fn row_buf(device: &CudaDevice) -> Tensor<CudaRuntime> {
        Tensor::<CudaRuntime>::zeros(&[1, 1, 4], DType::F32, device).expect("row buf")
    }

    /// Each lane's filled row is contiguous and equal to that lane's own
    /// slice of the batched logits, for more than one lane.
    #[test]
    fn each_lane_reads_its_own_contiguous_slice() {
        let Some(cuda) = cuda_setup() else { return };
        let (client, device) = (&cuda.client, &cuda.device);
        let logits = two_lane_logits(device);
        let buf_a = row_buf(device);
        let buf_b = row_buf(device);

        fill_row_buffer(client, &logits, &buf_a, 0).expect("fill lane 0");
        fill_row_buffer(client, &logits, &buf_b, 1).expect("fill lane 1");

        assert!(buf_a.is_contiguous());
        assert!(buf_b.is_contiguous());
        assert_eq!(buf_a.to_vec::<f32>(), vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(buf_b.to_vec::<f32>(), vec![10.0, 11.0, 12.0, 13.0]);
    }

    /// A later fill for lane B's buffer leaves an earlier fill for lane A's
    /// buffer untouched — the two buffers never alias.
    #[test]
    fn a_later_fill_does_not_touch_an_earlier_lanes_buffer() {
        let Some(cuda) = cuda_setup() else { return };
        let (client, device) = (&cuda.client, &cuda.device);
        let logits = two_lane_logits(device);
        let buf_a = row_buf(device);
        let buf_b = row_buf(device);

        fill_row_buffer(client, &logits, &buf_a, 0).expect("fill lane 0");
        let before = buf_a.to_vec::<f32>();

        fill_row_buffer(client, &logits, &buf_b, 1).expect("fill lane 1");

        assert_eq!(buf_a.to_vec::<f32>(), before);
        assert_eq!(buf_a.to_vec::<f32>(), vec![0.0, 1.0, 2.0, 3.0]);
    }

    /// Refilling the same buffer with a different lane's row fully
    /// overwrites it — no earlier step's row survives into the next one.
    #[test]
    fn refilling_a_buffer_overwrites_its_previous_contents() {
        let Some(cuda) = cuda_setup() else { return };
        let (client, device) = (&cuda.client, &cuda.device);
        let logits = two_lane_logits(device);
        let buf = row_buf(device);

        fill_row_buffer(client, &logits, &buf, 1).expect("fill lane 1");
        assert_eq!(buf.to_vec::<f32>(), vec![10.0, 11.0, 12.0, 13.0]);

        fill_row_buffer(client, &logits, &buf, 0).expect("fill lane 0");
        assert_eq!(buf.to_vec::<f32>(), vec![0.0, 1.0, 2.0, 3.0]);
    }
}
