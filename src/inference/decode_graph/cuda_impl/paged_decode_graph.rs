//! `PagedDecodeGraph`: captured decode graph and per-replay mutable state for
//! a paged KV cache.

use cudarc::driver::sys;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::device_scalars::{DeviceScalars, copy_rope_slice_async};
use crate::error::{Error, Result};

/// Captured CUDA decode graph for **paged KV cache** mode.
///
/// Like `DecodeGraph` but additionally manages `slot_mapping` (the paged cache
/// write position).  The block_table and paged cache tensors have stable addresses
/// from pre-allocation, so they are frozen at capture time.  Only `slot_mapping[0]`
/// changes per step (set to `seq_len` — the slot for the new token).
pub struct PagedDecodeGraph {
    /// The captured CUDA graph.
    pub graph: numr::runtime::CapturedGraph<CudaRuntime>,

    /// Device-side scalars (seq_len_k for attention loop bound).
    pub device_scalars: DeviceScalars,

    /// Stable input token `[1, 1]` i64.
    pub token_buf: Tensor<CudaRuntime>,

    /// RoPE cos slice `[1, half_dim]` f32.
    pub cos_slice: Tensor<CudaRuntime>,

    /// RoPE sin slice `[1, half_dim]` f32.
    pub sin_slice: Tensor<CudaRuntime>,

    /// Full RoPE cos table `[max_pos, half_dim]`.
    pub rope_cos_cache: Tensor<CudaRuntime>,

    /// Full RoPE sin table `[max_pos, half_dim]`.
    pub rope_sin_cache: Tensor<CudaRuntime>,

    /// Output token buffer `[1]` i64 — written by graph argmax node.
    pub next_token_buf: Tensor<CudaRuntime>,

    /// Slot mapping `[1]` i32 — updated per step to `seq_len` (paged cache write position).
    pub slot_mapping: Tensor<CudaRuntime>,

    /// Half of head dimension (for RoPE offset).
    pub head_dim: usize,

    /// CPU-side token count.
    pub seq_len: usize,
}

impl PagedDecodeGraph {
    /// Seed the first input token (same as DecodeGraph).
    pub fn seed_next_token(&self, client: &CudaClient, token: i64) -> Result<()> {
        let lo = (token as u64 & 0xFFFF_FFFF) as u32;
        let hi = ((token as u64) >> 32) as u32;
        let stream = client.stream().cu_stream();
        unsafe {
            let result = sys::cuMemsetD32Async(self.next_token_buf.ptr(), lo, 1, stream);
            if result != sys::CUresult::CUDA_SUCCESS {
                return Err(Error::InferenceError {
                    reason: format!("PagedDecodeGraph seed lo failed: {:?}", result),
                });
            }
            let result = sys::cuMemsetD32Async(self.next_token_buf.ptr() + 4, hi, 1, stream);
            if result != sys::CUresult::CUDA_SUCCESS {
                return Err(Error::InferenceError {
                    reason: format!("PagedDecodeGraph seed hi failed: {:?}", result),
                });
            }
        }
        Ok(())
    }

    /// Prepare per-step inputs and replay the paged graph.
    ///
    /// Additional step vs DecodeGraph: updates `slot_mapping[0] = seq_len` so the
    /// paged cache insert kernel writes K/V to the correct slot.
    pub fn pre_replay_and_launch(&mut self, client: &CudaClient) -> Result<()> {
        let stream = client.stream().cu_stream();

        // 1. D2D async: next_token_buf → token_buf
        unsafe {
            let result = sys::cuMemcpyDtoDAsync_v2(
                self.token_buf.ptr(),
                self.next_token_buf.ptr(),
                std::mem::size_of::<i64>(),
                stream,
            );
            if result != sys::CUresult::CUDA_SUCCESS {
                return Err(Error::InferenceError {
                    reason: format!("PagedDecodeGraph token_buf copy failed: {:?}", result),
                });
            }
        }

        // 2. Update device scalars (seq_len_k, write_pos)
        self.device_scalars.update(client, self.seq_len)?;

        // 3. Update slot_mapping[0] = seq_len (the paged cache slot for this token)
        unsafe {
            let result =
                sys::cuMemsetD32Async(self.slot_mapping.ptr(), self.seq_len as u32, 1, stream);
            if result != sys::CUresult::CUDA_SUCCESS {
                return Err(Error::InferenceError {
                    reason: format!("PagedDecodeGraph slot_mapping update failed: {:?}", result),
                });
            }
        }

        // 4. D2D async: RoPE slices for this position
        let stream_handle = client.stream().cu_stream();
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

        // 5. Launch graph
        self.graph.launch()?;

        self.seq_len += 1;

        Ok(())
    }
}
