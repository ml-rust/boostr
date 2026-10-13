//! `DecodeGraph`: captured decode graph and per-replay mutable state for a
//! flat (non-paged) KV cache.
//!
//! ## Constraints
//!
//! CUDA graphs freeze kernel arguments at capture time. We work around
//! position-dependent values by reading them from **device memory** inside kernels.
//! Before each graph replay, the CPU updates a small set of device scalars via
//! async H2D copies on the same stream — no CPU–GPU sync required.
//!
//! ## Stable-address tensors
//!
//! All tensors that are read or written from outside the graph MUST be allocated
//! BEFORE `Runtime::capture_graph_into()` is called (before `cuStreamBeginCapture`).
//! Any tensor allocated INSIDE the capture region has a graph-managed address
//! that is only valid within the graph's execution — accessing it from the CPU
//! after the graph has run causes `CUDA_ERROR_ILLEGAL_ADDRESS`.
//!
//! The output of each decode step is therefore written into a pre-allocated
//! `next_token_buf` via a `cuMemcpyAsync` node captured inside the graph.
//! CUDA automatically patches the (graph-internal) source address when replaying;
//! the stable destination address never changes.
//!
//! ## Stream ordering
//!
//! ALL pre-launch copies MUST use stream-ordered async variants so they are
//! serialized on the compute stream before `cuGraphLaunch` executes:
//!
//! - D2D copies: `cuMemcpyDtoDAsync_v2(dst, src, bytes, stream)`
//! - H2D copies: `cuMemcpyHtoDAsync_v2(dst, src, bytes, stream)`
//!
//! `cuMemcpy` (synchronous context-ordered) is NOT serialized with respect to
//! any stream. Using it before `cuGraphLaunch` (which is stream-ordered) creates
//! a race: the graph's kernels can start before the copies finish, causing
//! `CUDA_ERROR_ILLEGAL_ADDRESS` when kernels dereference stale/invalid pointers.
//!
//! | Tensor             | Allocated   | Updated how                          |
//! |--------------------|-------------|--------------------------------------|
//! | `token_buf`        | pre-capture | D2D async (DtoDAsync) from prev step |
//! | `device_scalars`   | pre-capture | H2D async (HtoDAsync) from CPU       |
//! | `cos_slice`        | pre-capture | D2D async (DtoDAsync) from rope cache|
//! | `sin_slice`        | pre-capture | D2D async (DtoDAsync) from rope cache|
//! | `next_token_buf`   | pre-capture | written by graph (argmax→memcpy node)|

use cudarc::driver::sys;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::device_scalars::{DeviceScalars, copy_rope_slice_async};
use super::seed_token::seed_i64;
use crate::error::{Error, Result};

/// Captured CUDA decode graph + all per-replay mutable state.
///
/// All tensors here are **pre-allocated before graph capture** so their device
/// addresses are stable across replays.
pub struct DecodeGraph {
    /// The captured CUDA graph — replayed once per token.
    ///
    /// Stored as a [`numr::runtime::CapturedGraph`] so the input/output tensors whose device
    /// addresses are encoded in the graph are kept alive for as long as the
    /// graph can be replayed.
    pub graph: numr::runtime::CapturedGraph<CudaRuntime>,

    /// Device-side scalars (seq_len_k, write_pos).
    pub device_scalars: DeviceScalars,

    /// Stable input tensor for the embedding lookup `[1, 1]` i64.
    ///
    /// Filled via stream-ordered D2D async copy from `next_token_buf` before each replay.
    pub token_buf: Tensor<CudaRuntime>,

    /// RoPE cos slice `[1, head_dim]` f32 — updated with the current position.
    pub cos_slice: Tensor<CudaRuntime>,

    /// RoPE sin slice `[1, head_dim]` f32.
    pub sin_slice: Tensor<CudaRuntime>,

    /// Full RoPE cos table `[max_pos, head_dim]` — source for D2D slicing.
    pub rope_cos_cache: Tensor<CudaRuntime>,

    /// Full RoPE sin table `[max_pos, head_dim]`.
    pub rope_sin_cache: Tensor<CudaRuntime>,

    /// Output token buffer `[1]` i64 — written by the graph via a captured
    /// `cuMemcpyAsync` node (argmax result copied to this stable address).
    ///
    /// Read by the caller after each `graph.launch()` completes.
    pub next_token_buf: Tensor<CudaRuntime>,

    /// Half of the attention head dimension (used for RoPE offset computation).
    pub head_dim: usize,

    /// CPU-side token count — updated in lockstep with DeviceScalars.
    pub seq_len: usize,
}

impl DecodeGraph {
    /// Write `token` (CPU i64) into `next_token_buf` via two stream-ordered device-side writes.
    ///
    /// Call this once before the decode loop starts, to seed the first input token.
    /// Subsequent steps read `next_token_buf` from the previous graph launch.
    ///
    /// Uses two `cuMemsetD32Async` calls (low/high 32-bit words of the i64) — no host
    /// pointer, no stack-lifetime hazard.  Little-endian: low word at ptr+0, high at ptr+4.
    pub fn seed_next_token(&self, client: &CudaClient, token: i64) -> Result<()> {
        seed_i64(client, &self.next_token_buf, token)
    }

    /// Prepare per-step inputs and replay the graph.
    ///
    /// All pre-launch copies use stream-ordered async variants so they are
    /// guaranteed to complete before `cuGraphLaunch` starts executing on the
    /// same compute stream.
    ///
    /// Call order per token:
    /// 1. D2D async: `next_token_buf` → `token_buf` (8 bytes, stream-ordered).
    /// 2. H2D async: update `device_scalars` to `seq_len` (2 × 4 bytes, stream-ordered).
    /// 3. D2D async: copy RoPE slice for `seq_len` into `cos_slice`/`sin_slice` (stream-ordered).
    /// 4. Launch graph (stream-ordered; stream serialization guarantees steps 1–3 are done).
    ///
    /// After this call, `next_token_buf` holds the argmax result of this step.
    /// The caller must wait for the GPU stream before reading it (e.g. stream sync
    /// or a pipelined D2H with a CUDA event).
    pub fn pre_replay_and_launch(&mut self, client: &CudaClient) -> Result<()> {
        // 1. D2D async: next_token_buf → token_buf (8 bytes, i64, stream-ordered)
        //    Feeds the previous step's output as this step's input token.
        //    Driver enqueues made outside numr's launch path hold the capture
        //    permit themselves, scoped to the enqueue alone: a permit held
        //    across a call that takes its own would take the read side twice
        //    on one thread, which a waiting capture turns into a deadlock.
        {
            let _permit = client.stream().enqueue_permit();
            let stream = client.stream().raw().cu_stream();
            unsafe {
                let result = sys::cuMemcpyDtoDAsync_v2(
                    self.token_buf.ptr(),
                    self.next_token_buf.ptr(),
                    std::mem::size_of::<i64>(),
                    stream,
                );
                if result != sys::CUresult::CUDA_SUCCESS {
                    return Err(Error::InferenceError {
                        reason: format!("cuMemcpyDtoDAsync_v2 for token_buf failed: {:?}", result),
                    });
                }
            }
        }

        // 2. H2D async: update device scalars (stream-ordered)
        self.device_scalars.update(client, &[self.seq_len])?;

        // 3. D2D async: update RoPE slices for this position (stream-ordered)
        {
            let _permit = client.stream().enqueue_permit();
            let stream_handle = client.stream().raw().cu_stream();
            copy_rope_slice_async(
                &self.rope_cos_cache,
                self.seq_len * self.head_dim,
                &self.cos_slice,
                self.head_dim,
                stream_handle,
            )?;
            copy_rope_slice_async(
                &self.rope_sin_cache,
                self.seq_len * self.head_dim,
                &self.sin_slice,
                self.head_dim,
                stream_handle,
            )?;
        }

        // 4. Launch graph (stream serialization guarantees steps 1–3 are done)
        self.graph.launch()?;

        // Advance CPU-side tracking
        self.seq_len += 1;

        Ok(())
    }
}
