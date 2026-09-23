//! `Qwen35DecodeGraph`: captured decode graph and per-replay mutable state
//! for the `qwen35` hybrid (full-capacity KV cache plus in-place GDN state).
//!
//! Sibling of [`DecodeGraph`](super::DecodeGraph). The Llama graph copies a
//! RoPE cos/sin row per replay; `qwen35` reads its IMROPE rows on the device
//! from the lane table's positions tensor, so the per-replay work is the
//! per-lane scalar writes and the host's token write.
//!
//! One capture serves a fixed lane count. Lane `b` holds its own sequence
//! from cache position 0 with its own end, so lanes of different live
//! lengths replay the same graph and a free lane reads no key.
//!
//! The capture ends at the logits copy. The graph picks no token: the caller
//! reads `logits_buf` after the launch, applies its own sampling rule per
//! lane, and writes the chosen tokens back with
//! [`Qwen35DecodeGraph::set_next_tokens`]. Two lanes in one capture can
//! therefore run different sampling policies in the same replay.

use numr::dtype::DType;
use numr::ops::HostCopyOps;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::device_scalars::DeviceScalars;
use crate::error::{Error, Result};
use crate::inference::lanes::LaneTable;

/// Captured CUDA decode graph for `qwen35` plus the buffers a replay reads.
///
/// Every tensor here is allocated before capture, so its device address is
/// stable across replays. The KV cache and GDN state the graph writes are
/// owned by the caller and must outlive this struct's `graph`.
pub struct Qwen35DecodeGraph {
    /// The captured CUDA graph, replayed once per token for every lane.
    pub graph: numr::runtime::CapturedGraph<CudaRuntime>,

    /// Device-side `seq_len_k` and `write_pos`, one entry per lane.
    pub device_scalars: DeviceScalars,

    /// Static capacity of the KV cache and GDN state the graph reads.
    ///
    /// Passed to [`DeviceScalars::update_lanes`] as the key-axis clamp on
    /// every replay: a property of the buffer, so it never varies with a
    /// lane's live length.
    pub capacity: usize,

    /// Lane table holding each lane's length, rope position, `[batch]`
    /// `kv_start`/`kv_end` and the `[4, batch, 1]` IMROPE positions the
    /// graph gathers its cos/sin rows from. It is the single source for
    /// every per-lane value a replay reads.
    pub lanes: LaneTable<CudaRuntime>,

    /// Stable input tensor for the embedding lookup, `[batch, 1]` i64.
    ///
    /// Written by [`Qwen35DecodeGraph::set_next_tokens`] and read by the next
    /// launch.
    pub token_buf: Tensor<CudaRuntime>,

    /// Stable `[batch, 1, vocab]` copy of the step's logits, written by the
    /// captured copy. The caller waits on the stream, reads it, and samples
    /// each lane's next token from its own row.
    pub logits_buf: Tensor<CudaRuntime>,
}

impl Qwen35DecodeGraph {
    /// Bind a captured graph to the buffers its replays read.
    ///
    /// Every argument must describe the same lane count: `token_buf` is
    /// `[batch, 1]` i64, `logits_buf` is `[batch, 1, vocab]`, and the scalars
    /// and the lane table both hold `batch` lanes. `capacity` is the static
    /// capacity of the KV cache and GDN state this graph was captured
    /// against; the caller knows it from building those caches.
    ///
    /// # Errors
    ///
    /// [`Error::InferenceError`] when a shape, a dtype or a lane count
    /// disagrees with the others.
    pub fn new(
        graph: numr::runtime::CapturedGraph<CudaRuntime>,
        device_scalars: DeviceScalars,
        capacity: usize,
        lanes: LaneTable<CudaRuntime>,
        token_buf: Tensor<CudaRuntime>,
        logits_buf: Tensor<CudaRuntime>,
    ) -> Result<Self> {
        let batch = lanes.batch();
        if token_buf.shape() != [batch, 1].as_slice() || token_buf.dtype() != DType::I64 {
            return Err(Error::InferenceError {
                reason: format!(
                    "qwen35 decode graph holds {batch} lanes but token_buf is {:?} {:?}; \
                     allocate it as [{batch}, 1] I64",
                    token_buf.shape(),
                    token_buf.dtype()
                ),
            });
        }
        let logits_shape = logits_buf.shape();
        let logits_ok = logits_shape.len() == 3
            && logits_shape[0] == batch
            && logits_shape[1] == 1
            && logits_shape[2] > 0;
        if !logits_ok {
            return Err(Error::InferenceError {
                reason: format!(
                    "qwen35 decode graph holds {batch} lanes but logits_buf is {logits_shape:?}; \
                     allocate it as [{batch}, 1, vocab] in the model's logits dtype"
                ),
            });
        }
        if device_scalars.batch() != batch {
            return Err(Error::InferenceError {
                reason: format!(
                    "qwen35 decode graph holds {batch} lanes but the decode scalars hold {}; \
                     allocate both with the same lane count",
                    device_scalars.batch()
                ),
            });
        }
        Ok(Self {
            graph,
            device_scalars,
            capacity,
            lanes,
            token_buf,
            logits_buf,
        })
    }

    /// Lane count one replay serves.
    pub fn batch(&self) -> usize {
        self.lanes.batch()
    }

    /// Stable `[batch, 1, vocab]` logits of the last launch.
    ///
    /// Row `b` is lane `b`'s logits. Wait on the stream (event or sync)
    /// before reading it.
    pub fn logits(&self) -> &Tensor<CudaRuntime> {
        &self.logits_buf
    }

    /// Write one token per lane into `token_buf` with a single stream-ordered
    /// host-to-device copy.
    ///
    /// `tokens[b]` is the token lane `b` feeds to the next launch. Call once
    /// before the decode loop to seed every lane's first token, and once per
    /// step with the tokens the caller sampled from
    /// [`Qwen35DecodeGraph::logits`]. A free lane takes any id; the graph
    /// reads no key for it and its row is ignored.
    ///
    /// The copy is ordered on the compute stream behind the launch that
    /// produced the logits, and it stages `tokens` before returning, so the
    /// caller may drop the slice immediately. Nothing here allocates.
    ///
    /// # Errors
    ///
    /// [`Error::InferenceError`] when `tokens` is not exactly `batch` long,
    /// or the copy fails.
    pub fn set_next_tokens(&self, client: &CudaClient, tokens: &[i64]) -> Result<()> {
        let batch = self.batch();
        if tokens.len() != batch {
            return Err(Error::InferenceError {
                reason: format!(
                    "qwen35 decode graph holds {batch} lanes but {} next tokens were given; \
                     pass one token per lane, in lane order",
                    tokens.len()
                ),
            });
        }
        client
            .write_host_slice(&self.token_buf, tokens)
            .map_err(|e| Error::InferenceError {
                reason: format!(
                    "qwen35 decode graph could not write {batch} next tokens into token_buf: \
                     {e}; check that token_buf is still allocated on this device"
                ),
            })
    }

    /// Prepare every lane's inputs and replay the graph.
    ///
    /// Per token, all stream-ordered on the compute stream:
    ///
    /// 1. Write `device_scalars` from the lane table's write positions.
    /// 2. Launch the graph.
    /// 3. Count the token every occupied lane just decoded, which rewrites
    ///    `kv_end` and the IMROPE positions in place for the next replay.
    ///
    /// The lane table already describes this step when the call starts:
    /// admitting a sequence and advancing a lane each write the device
    /// tensors. Step 3 is ordered behind the launch on the same stream, so
    /// it never moves the range the in-flight replay reads. Nothing here
    /// allocates, and nothing reads a device value on the host.
    ///
    /// `token_buf` must already hold this step's input token per lane, from
    /// [`Qwen35DecodeGraph::set_next_tokens`]. After this call `logits_buf`
    /// holds each lane's logits row; the caller waits on the stream (event or
    /// sync), samples per lane, and writes the results back for the next
    /// replay.
    ///
    /// # Errors
    ///
    /// [`Error::InferenceError`] when a scalar write fails, the launch fails,
    /// or an occupied lane's next rope position is outside the model's rope
    /// table.
    pub fn pre_replay_and_launch(&mut self, client: &CudaClient) -> Result<()> {
        self.device_scalars
            .update_lanes(client, self.lanes.write_positions(), self.capacity)?;
        self.graph.launch()?;
        self.lanes.advance(client)
    }
}
