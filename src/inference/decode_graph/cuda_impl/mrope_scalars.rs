//! Device-resident IMROPE positions for a graph-mode decode step.
//!
//! [`DeviceScalars`](super::DeviceScalars) carries the two i32 values every
//! graph decode path reads (`seq_len_k`, `write_pos`). The IMROPE positions
//! are a `qwen35`-only input, so they live in this sibling: the Llama graph
//! paths and their callers keep the allocation and per-replay writes they
//! have.

use cudarc::driver::sys;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use crate::error::{Error, Result};

/// `[4, 1]` i32 IMROPE positions with a stable device address.
///
/// The row order is the `t, h, w, e` stream layout `MRopeOps` reads. A text
/// token carries `t = h = w = position`, `e = 0`. `apply_mrope_interleaved`
/// gathers its cos/sin rows from this tensor on the device, so the captured
/// graph reads the values a replay wrote.
pub struct MropeScalars {
    positions: Tensor<CudaRuntime>,
}

impl MropeScalars {
    /// Allocate the positions buffer holding `initial_position`.
    pub fn new(initial_position: usize, device: &numr::runtime::cuda::CudaDevice) -> Result<Self> {
        let p = initial_position as i32;
        let positions = Tensor::<CudaRuntime>::from_slice(&[p, p, p, 0], &[4, 1], device)?;
        Ok(Self { positions })
    }

    /// The `[4, 1]` i32 positions tensor. Pass to the graph-mode forward.
    pub fn positions(&self) -> &Tensor<CudaRuntime> {
        &self.positions
    }

    /// Raw device pointer to the first i32 (`t`).
    pub fn positions_ptr(&self) -> u64 {
        self.positions.ptr()
    }

    /// Stream-ordered write of `t = h = w = rope_pos`, `e = 0` for the
    /// token this replay decodes.
    ///
    /// `rope_pos` is the IMROPE position, not the KV slot: the slot comes
    /// from [`DeviceScalars::update`](super::DeviceScalars::update). In a
    /// text-only context both equal the token count; after an image the
    /// rope position lags the slot by `n_tokens - max(nx, ny)` per image.
    /// Uses `cuMemsetD32Async`: the value travels inside the driver call,
    /// so no host pointer outlives this function.
    pub fn update(&self, client: &CudaClient, rope_pos: usize) -> Result<()> {
        let stream = client.stream().cu_stream();
        let base = self.positions.ptr();
        let word = std::mem::size_of::<i32>() as u64;
        unsafe {
            let result = sys::cuMemsetD32Async(base, rope_pos as u32, 3, stream);
            if result != sys::CUresult::CUDA_SUCCESS {
                return Err(Error::InferenceError {
                    reason: format!("cuMemsetD32Async for mrope t/h/w failed: {result:?}"),
                });
            }
            let result = sys::cuMemsetD32Async(base + 3 * word, 0, 1, stream);
            if result != sys::CUresult::CUDA_SUCCESS {
                return Err(Error::InferenceError {
                    reason: format!("cuMemsetD32Async for mrope e failed: {result:?}"),
                });
            }
        }
        Ok(())
    }
}
