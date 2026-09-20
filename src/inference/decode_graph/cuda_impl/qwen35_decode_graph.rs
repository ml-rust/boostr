//! `Qwen35DecodeGraph`: captured decode graph and per-replay mutable state
//! for the `qwen35` hybrid (full-capacity KV cache plus in-place GDN state).
//!
//! Sibling of [`DecodeGraph`](super::DecodeGraph). The Llama graph copies a
//! RoPE cos/sin row per replay; `qwen35` reads its IMROPE rows on the device
//! from [`MropeScalars`], so the per-replay work is the token copy and the
//! two scalar writes.

use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::device_scalars::DeviceScalars;
use super::mrope_scalars::MropeScalars;
use super::seed_token::seed_i64;
use super::stable_copy::copy_into_stable;
use crate::error::{Error, Result};

/// Captured CUDA decode graph for `qwen35` plus the buffers a replay reads.
///
/// Every tensor here is allocated before capture, so its device address is
/// stable across replays. The KV cache and GDN state the graph writes are
/// owned by the caller and must outlive this struct's `graph`.
pub struct Qwen35DecodeGraph {
    /// The captured CUDA graph, replayed once per token.
    pub graph: numr::runtime::CapturedGraph<CudaRuntime>,

    /// Device-side `seq_len_k` and `write_pos`.
    pub device_scalars: DeviceScalars,

    /// Device-side IMROPE positions for the token this replay decodes.
    pub mrope: MropeScalars,

    /// Stable input tensor for the embedding lookup `[1, 1]` i64.
    pub token_buf: Tensor<CudaRuntime>,

    /// Output token buffer `[1]` i64, written by the graph's captured argmax
    /// copy. Read by the caller after `graph.launch()` completes.
    pub next_token_buf: Tensor<CudaRuntime>,

    /// CPU-side token count (the KV slot of the next token), advanced after
    /// every launch.
    pub seq_len: usize,

    /// IMROPE position of the next token, advanced after every launch.
    /// Equal to `seq_len` for a text-only context; behind it once the
    /// prefill held an image.
    pub rope_pos: usize,
}

impl Qwen35DecodeGraph {
    /// Write `token` into `next_token_buf` with two stream-ordered
    /// `cuMemsetD32Async` calls (low word at `ptr`, high word at `ptr + 4`).
    ///
    /// Call once before the decode loop to seed the first input token.
    pub fn seed_next_token(&self, client: &CudaClient, token: i64) -> Result<()> {
        seed_i64(client, &self.next_token_buf, token)
    }

    /// Prepare per-step inputs and replay the graph.
    ///
    /// Per token, all stream-ordered on the compute stream:
    /// 1. D2D async: `next_token_buf` -> `token_buf`.
    /// 2. Write `device_scalars` for `seq_len`.
    /// 3. Write `mrope` positions for `rope_pos`.
    /// 4. Launch the graph, then advance both counters.
    ///
    /// After this call `next_token_buf` holds this step's argmax. The caller
    /// waits on the stream (event or sync) before reading it.
    pub fn pre_replay_and_launch(&mut self, client: &CudaClient) -> Result<()> {
        copy_into_stable(client, &self.next_token_buf, &self.token_buf).map_err(|e| {
            Error::InferenceError {
                reason: format!("Qwen35DecodeGraph token_buf copy failed: {e}"),
            }
        })?;
        self.device_scalars.update(client, self.seq_len)?;
        self.mrope.update(client, self.rope_pos)?;
        self.graph.launch()?;
        self.seq_len += 1;
        self.rope_pos += 1;
        Ok(())
    }
}
