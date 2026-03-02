//! CUDA graph decode loop infrastructure.
//!
//! # Overview
//!
//! The decode loop for autoregressive generation makes ~930 individual kernel
//! launches per token. CUDA graphs replace them with a single `cuGraphLaunch`
//! (~5µs overhead instead of ~13ms).
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
//! BEFORE `Runtime::capture_graph()` is called (before `cuStreamBeginCapture`).
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

#[cfg(feature = "cuda")]
pub use cuda_impl::*;

#[cfg(feature = "cuda")]
mod cuda_impl {
    use cudarc::driver::sys;
    use numr::runtime::Graph;
    use numr::runtime::cuda::{CudaClient, CudaRuntime};
    use numr::tensor::Tensor;

    use crate::error::{Error, Result};

    /// Device-resident scalars updated before each graph replay.
    ///
    /// Both fields are backed by device memory with a stable address.
    /// Values are written via async H2D copy on the compute stream — no sync.
    pub struct DeviceScalars {
        /// `seq_len_k_ptr`: device pointer to i32 — attention loop bound.
        ///
        /// Passed to `decode_attention_*` kernels; they dereference it at runtime,
        /// so the loop bound can differ between graph replays.
        pub seq_len_k: Tensor<CudaRuntime>,

        /// `write_pos_ptr`: device pointer to i32 — KV insert position.
        ///
        /// Passed to the `kv_insert` kernel; determines where the new token's
        /// K/V vectors are written in the full-capacity cache.
        pub write_pos: Tensor<CudaRuntime>,
    }

    impl DeviceScalars {
        /// Allocate device scalars initialised to `initial_seq_len`.
        pub fn new(initial_seq_len: usize, device: &numr::runtime::cuda::CudaDevice) -> Self {
            let val = initial_seq_len as i32;
            let seq_len_k = Tensor::<CudaRuntime>::from_slice(&[val], &[1], device);
            let write_pos = Tensor::<CudaRuntime>::from_slice(&[val], &[1], device);
            Self {
                seq_len_k,
                write_pos,
            }
        }

        /// Raw device pointer to the i32 seq_len_k value. Pass to decode_attention.
        pub fn seq_len_k_ptr(&self) -> u64 {
            self.seq_len_k.ptr()
        }

        /// Raw device pointer to the i32 write_pos value. Pass to kv_insert.
        pub fn write_pos_ptr(&self) -> u64 {
            self.write_pos.ptr()
        }

        /// Update `cos_slice` and `sin_slice` stable tensors with the rope values for `position`.
        ///
        /// Performs a stream-ordered D2D async copy of `half_dim` f32 elements from
        /// `rope_cos_cache[position * half_dim ..]` into `cos_slice` (and likewise for sin).
        /// Used to prepare RoPE values before each graph replay.
        pub fn update_rope_slices(
            &self,
            client: &CudaClient,
            rope_cos_cache: &Tensor<CudaRuntime>,
            rope_sin_cache: &Tensor<CudaRuntime>,
            cos_slice: &numr::autograd::Var<CudaRuntime>,
            sin_slice: &numr::autograd::Var<CudaRuntime>,
            position: usize,
            half_dim: usize,
        ) -> Result<()> {
            let stream = client.stream().cu_stream();
            copy_rope_slice_async(
                rope_cos_cache,
                position * half_dim,
                cos_slice.tensor(),
                half_dim,
                stream,
            )?;
            copy_rope_slice_async(
                rope_sin_cache,
                position * half_dim,
                sin_slice.tensor(),
                half_dim,
                stream,
            )?;
            Ok(())
        }

        /// Stream-ordered device-side write — update scalars for the current decode step.
        ///
        /// `seq_len` is the number of tokens currently in the KV cache (before this step's insert).
        ///
        /// - `write_pos = seq_len`   — where to insert this step's K/V
        /// - `seq_len_k = seq_len + 1` — how many K/V entries to attend over AFTER insert
        ///                               (positions 0..seq_len inclusive; matches non-graph behavior
        ///                               where `update()` increments seq_len before `get_kv()`)
        ///
        /// Uses `cuMemsetD32Async` — the value is embedded in the API call itself
        /// (no host memory pointer involved), so there is NO stack-lifetime hazard.
        /// `cuMemcpyHtoDAsync_v2` from a stack variable is unsafe because the GPU
        /// reads asynchronously after the function returns, by which time the stack
        /// frame may be reused, causing garbage values and out-of-bounds accesses.
        pub fn update(&self, client: &CudaClient, seq_len: usize) -> Result<()> {
            let write_pos_val = seq_len as u32;
            let seq_len_k_val = (seq_len + 1) as u32;
            let stream = client.stream().cu_stream();
            unsafe {
                // cuMemsetD32Async(ptr, value, count, stream)
                // Sets `count` 4-byte words at ptr to `value`, stream-ordered.
                // Value is passed by copy into the driver — no host pointer lifetime hazard.
                let result = sys::cuMemsetD32Async(self.seq_len_k.ptr(), seq_len_k_val, 1, stream);
                if result != sys::CUresult::CUDA_SUCCESS {
                    return Err(Error::InferenceError {
                        reason: format!("cuMemsetD32Async for seq_len_k failed: {:?}", result),
                    });
                }
                let result = sys::cuMemsetD32Async(self.write_pos.ptr(), write_pos_val, 1, stream);
                if result != sys::CUresult::CUDA_SUCCESS {
                    return Err(Error::InferenceError {
                        reason: format!("cuMemsetD32Async for write_pos failed: {:?}", result),
                    });
                }
            }
            Ok(())
        }
    }

    /// Stream-ordered D2D async copy of `head_dim` f32 elements from `src` at element
    /// offset `src_elem_off` into `dst` starting at element 0.
    ///
    /// Uses `cuMemcpyDtoDAsync_v2` so the copy is serialized on `stream` before any
    /// subsequent stream operation (including `cuGraphLaunch`).
    fn copy_rope_slice_async(
        src: &Tensor<CudaRuntime>,
        src_elem_off: usize,
        dst: &Tensor<CudaRuntime>,
        head_dim: usize,
        stream: sys::CUstream,
    ) -> Result<()> {
        let bytes = head_dim * std::mem::size_of::<f32>();
        let src_ptr = src.ptr() + (src_elem_off * std::mem::size_of::<f32>()) as u64;
        let dst_ptr = dst.ptr();
        unsafe {
            let result = sys::cuMemcpyDtoDAsync_v2(dst_ptr, src_ptr, bytes, stream);
            if result != sys::CUresult::CUDA_SUCCESS {
                return Err(Error::InferenceError {
                    reason: format!("cuMemcpyDtoDAsync_v2 for RoPE slice failed: {:?}", result),
                });
            }
        }
        Ok(())
    }

    /// Argmax on `logits` (graph-internal tensor) and write result into `out` (stable).
    ///
    /// This function MUST be called inside a `Runtime::capture_graph()` closure.
    /// The source (`logits`) has a graph-managed address — CUDA internally patches it
    /// on each replay.  The destination (`out`) is pre-allocated before capture and
    /// has a stable address that the caller can read after each graph launch.
    ///
    /// `logits` shape: `[1, 1, vocab_size]`
    /// `out` shape: `[1]` i64
    pub fn argmax_to_buf(
        client: &CudaClient,
        logits: &Tensor<CudaRuntime>,
        out: &Tensor<CudaRuntime>,
    ) -> numr::error::Result<()> {
        use numr::ops::traits::IndexingOps;

        // Argmax along last dim: [1, 1, vocab] → [1, 1] i64
        // Allocated inside the graph (graph-managed address — unstable from CPU)
        let last_dim = logits.shape().len() - 1;
        let token_ids = client.argmax(logits, last_dim, false)?;

        // cuMemcpyAsync: from graph-internal token_ids to pre-allocated `out`.
        // CUDA records a MemCpy node; on each graph replay it patches the source
        // address to the actual execution-time allocation of `token_ids`.
        let bytes = std::mem::size_of::<i64>();
        unsafe {
            let result = sys::cuMemcpyAsync(
                out.ptr(),
                token_ids.ptr(),
                bytes,
                client.stream().cu_stream(),
            );
            if result != sys::CUresult::CUDA_SUCCESS {
                return Err(numr::error::Error::Backend(format!(
                    "argmax_to_buf cuMemcpyAsync failed: {:?}",
                    result
                )));
            }
        }
        Ok(())
    }

    /// Captured CUDA decode graph + all per-replay mutable state.
    ///
    /// All tensors here are **pre-allocated before graph capture** so their device
    /// addresses are stable across replays.
    pub struct DecodeGraph {
        /// The captured CUDA graph — replayed once per token.
        pub graph: numr::runtime::cuda::CudaGraph,

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
            let lo = (token as u64 & 0xFFFF_FFFF) as u32; // low 32 bits
            let hi = ((token as u64) >> 32) as u32; // high 32 bits
            let stream = client.stream().cu_stream();
            unsafe {
                let result = sys::cuMemsetD32Async(self.next_token_buf.ptr(), lo, 1, stream);
                if result != sys::CUresult::CUDA_SUCCESS {
                    return Err(Error::InferenceError {
                        reason: format!("seed_next_token cuMemsetD32Async lo failed: {:?}", result),
                    });
                }
                let result = sys::cuMemsetD32Async(self.next_token_buf.ptr() + 4, hi, 1, stream);
                if result != sys::CUresult::CUDA_SUCCESS {
                    return Err(Error::InferenceError {
                        reason: format!("seed_next_token cuMemsetD32Async hi failed: {:?}", result),
                    });
                }
            }
            Ok(())
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
            let stream = client.stream().cu_stream();

            // 1. D2D async: next_token_buf → token_buf (8 bytes, i64, stream-ordered)
            //    Feeds the previous step's output as this step's input token.
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

            // 2. H2D async: update device scalars (stream-ordered)
            self.device_scalars.update(client, self.seq_len)?;

            // 3. D2D async: update RoPE slices for this position (stream-ordered)
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

            // 4. Launch graph (stream serialization guarantees steps 1–3 are done)
            self.graph.launch()?;

            // Advance CPU-side tracking
            self.seq_len += 1;

            Ok(())
        }
    }
}
