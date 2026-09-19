//! Device-resident scalars updated before each graph replay, and the shared
//! stream-ordered RoPE-slice copy both decode graphs use.

use cudarc::driver::sys;
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
    pub fn new(initial_seq_len: usize, device: &numr::runtime::cuda::CudaDevice) -> Result<Self> {
        let val = initial_seq_len as i32;
        let seq_len_k = Tensor::<CudaRuntime>::from_slice(&[val], &[1], device)?;
        let write_pos = Tensor::<CudaRuntime>::from_slice(&[val], &[1], device)?;
        Ok(Self {
            seq_len_k,
            write_pos,
        })
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
    #[allow(clippy::too_many_arguments)]
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
    ///   (positions 0..seq_len inclusive; matches non-graph behavior
    ///   where `update()` increments seq_len before `get_kv()`)
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

/// Stream-ordered D2D async copy of `head_dim` elements from `src` at element
/// offset `src_elem_off` into `dst` starting at element 0.
///
/// `src` and `dst` share one dtype: a RoPE table is cast to the model dtype
/// at load, so the slice buffer must be allocated at the table's dtype.
///
/// Uses `cuMemcpyDtoDAsync_v2` so the copy is serialized on `stream` before any
/// subsequent stream operation (including `cuGraphLaunch`).
pub(super) fn copy_rope_slice_async(
    src: &Tensor<CudaRuntime>,
    src_elem_off: usize,
    dst: &Tensor<CudaRuntime>,
    head_dim: usize,
    stream: sys::CUstream,
) -> Result<()> {
    if src.dtype() != dst.dtype() {
        return Err(Error::InferenceError {
            reason: format!(
                "RoPE slice dtype {:?} != table dtype {:?}",
                dst.dtype(),
                src.dtype()
            ),
        });
    }
    let elem = src.dtype().size_in_bytes();
    let bytes = head_dim * elem;
    let src_ptr = src.ptr() + (src_elem_off * elem) as u64;
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
