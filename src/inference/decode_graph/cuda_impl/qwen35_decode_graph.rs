//! `Qwen35DecodeGraph`: captured decode graph and per-replay mutable state
//! for the `qwen35` hybrid (full-capacity KV cache plus in-place GDN state).
//!
//! Sibling of [`DecodeGraph`](super::DecodeGraph). The Llama graph copies a
//! RoPE cos/sin row per replay; `qwen35` reads its IMROPE rows on the device
//! from the lane table's positions tensor, so the per-replay work is the
//! token copy and the per-lane scalar writes.
//!
//! One capture serves a fixed lane count. Lane `b` holds its own sequence
//! from cache position 0 with its own end, so lanes of different live
//! lengths replay the same graph and a free lane reads no key.

use numr::dtype::DType;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::device_scalars::DeviceScalars;
use super::seed_token::seed_i64_at;
use super::stable_copy::copy_into_stable;
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
    pub token_buf: Tensor<CudaRuntime>,

    /// Output token buffer `[batch]` i64, written by the graph's captured
    /// batched argmax copy. Read by the caller after `graph.launch()`
    /// completes.
    pub next_token_buf: Tensor<CudaRuntime>,
}

impl Qwen35DecodeGraph {
    /// Bind a captured graph to the buffers its replays read.
    ///
    /// Every argument must describe the same lane count: `token_buf` is
    /// `[batch, 1]` i64, `next_token_buf` is `[batch]` i64, and the scalars
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
        next_token_buf: Tensor<CudaRuntime>,
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
        if next_token_buf.shape() != [batch].as_slice() || next_token_buf.dtype() != DType::I64 {
            return Err(Error::InferenceError {
                reason: format!(
                    "qwen35 decode graph holds {batch} lanes but next_token_buf is {:?} {:?}; \
                     allocate it as [{batch}] I64",
                    next_token_buf.shape(),
                    next_token_buf.dtype()
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
            next_token_buf,
        })
    }

    /// Lane count one replay serves.
    pub fn batch(&self) -> usize {
        self.lanes.batch()
    }

    /// Write `token` into lane `lane` of `next_token_buf` with two
    /// stream-ordered `cuMemsetD32Async` calls.
    ///
    /// Call once per occupied lane before the decode loop to seed that
    /// lane's first input token.
    ///
    /// # Errors
    ///
    /// [`Error::InferenceError`] when `lane` is outside the batch, or a
    /// memset fails.
    pub fn seed_next_token(&self, client: &CudaClient, lane: usize, token: i64) -> Result<()> {
        seed_i64_at(client, &self.next_token_buf, lane, token)
    }

    /// Prepare every lane's inputs and replay the graph.
    ///
    /// Per token, all stream-ordered on the compute stream:
    ///
    /// 1. D2D async: `next_token_buf` -> `token_buf`, every lane at once.
    /// 2. Write `device_scalars` from the lane table's write positions.
    /// 3. Launch the graph.
    /// 4. Count the token every occupied lane just decoded, which rewrites
    ///    `kv_end` and the IMROPE positions in place for the next replay.
    ///
    /// The lane table already describes this step when the call starts:
    /// admitting a sequence and advancing a lane each write the device
    /// tensors. Step 4 is ordered behind the launch on the same stream, so
    /// it never moves the range the in-flight replay reads. Nothing here
    /// allocates, and nothing reads a device value on the host.
    ///
    /// After this call `next_token_buf` holds each lane's argmax. The caller
    /// waits on the stream (event or sync) before reading it.
    ///
    /// # Errors
    ///
    /// [`Error::InferenceError`] when the token copy fails, a scalar write
    /// fails, the launch fails, or an occupied lane's next rope position is
    /// outside the model's rope table.
    pub fn pre_replay_and_launch(&mut self, client: &CudaClient) -> Result<()> {
        copy_into_stable(client, &self.next_token_buf, &self.token_buf).map_err(|e| {
            Error::InferenceError {
                reason: format!("Qwen35DecodeGraph token_buf copy failed: {e}"),
            }
        })?;
        self.device_scalars
            .update_lanes(client, self.lanes.write_positions(), self.capacity)?;
        self.graph.launch()?;
        self.lanes.advance(client)
    }
}
