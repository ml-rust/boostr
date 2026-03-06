//! CUDA-accelerated prefix-cache batch lookup.
//!
//! Exposes `gpu_prefix_cache_lookup` — a thin wrapper around the
//! `prefix_cache_lookup` PTX kernel that performs a batch probe of the
//! GPU-resident hash table built by `GpuRadixTree`.
//!
//! The caller is responsible for uploading the table arrays to the device
//! (via `numr`'s storage API) before calling this function.  The typical
//! flow is:
//!
//! ```text
//! 1. GpuRadixTree::insert / evict_lru on CPU (fast, amortised O(1)).
//! 2. Upload keys[] and values[] to device tensors (incremental diff upload
//!    is a future optimisation; full upload is correct and simple).
//! 3. gpu_prefix_cache_lookup(client, query_hashes_dev, keys_dev, values_dev)
//!    → block_ids_dev  (all on device, no CPU round-trip).
//! 4. Scheduler reads block_ids from the result tensor.
//! ```

use crate::error::{Error, Result};
use crate::inference::memory::BlockId;
use crate::ops::cuda::kernels::{self, PREFIX_CACHE_LOOKUP_MODULE};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

/// Block size for the CUDA launch.
const BLOCK_SIZE: u32 = 256;

/// Perform a batched prefix-cache probe on the GPU.
///
/// # Arguments
///
/// * `client` — the CUDA device client.
/// * `query_hashes` — device tensor of shape `[num_queries]`, dtype `U64`
///   (or equivalently stored as pairs of `U32`; must be bit-compatible with
///   `uint64_t` on device).
/// * `table_keys` — device tensor of shape `[capacity]`, dtype `U64`.
/// * `table_values` — device tensor of shape `[capacity]`, dtype `I32`.
///   Slot value `-1` means "empty".
///
/// # Returns
///
/// A device tensor of shape `[num_queries]`, dtype `I32`.  Each element is
/// the matched `BlockId` (non-negative) or `-1` for a miss.
///
/// Callers can convert the result to `Vec<Option<BlockId>>` by copying to
/// host and mapping `-1` → `None`.
pub fn gpu_prefix_cache_lookup(
    client: &CudaClient,
    query_hashes: &Tensor<CudaRuntime>,
    table_keys: &Tensor<CudaRuntime>,
    table_values: &Tensor<CudaRuntime>,
) -> Result<Tensor<CudaRuntime>> {
    let qshape = query_hashes.shape();
    if qshape.len() != 1 {
        return Err(Error::InvalidArgument {
            arg: "query_hashes",
            reason: format!("expected 1D tensor, got {}D", qshape.len()),
        });
    }
    let num_queries = qshape[0];

    let kshape = table_keys.shape();
    let vshape = table_values.shape();
    if kshape.len() != 1 || vshape.len() != 1 || kshape[0] != vshape[0] {
        return Err(Error::InvalidArgument {
            arg: "table_keys / table_values",
            reason: format!(
                "both must be 1D with the same length; got keys {:?} values {:?}",
                kshape, vshape
            ),
        });
    }
    let capacity = kshape[0];

    // Capacity must be a power of two (invariant enforced by GpuRadixTree).
    if capacity == 0 || (capacity & (capacity - 1)) != 0 {
        return Err(Error::InvalidArgument {
            arg: "table_keys",
            reason: format!("capacity {} is not a power of two", capacity),
        });
    }

    let device = query_hashes.device();
    let device_index = device.id();

    let out_block_ids = Tensor::<CudaRuntime>::empty(&[num_queries], DType::I32, device);

    let module =
        kernels::get_or_load_module(client.context(), device_index, PREFIX_CACHE_LOOKUP_MODULE)?;
    let func = kernels::get_kernel_function(&module, "prefix_cache_lookup")?;

    let grid_size = (num_queries as u32).div_ceil(BLOCK_SIZE);
    let cfg = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };

    let q_ptr = query_hashes.ptr();
    let k_ptr = table_keys.ptr();
    let v_ptr = table_values.ptr();
    let out_ptr = out_block_ids.ptr();
    let cap_i32 = capacity as i32;
    let nq_i32 = num_queries as i32;

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&q_ptr);
        builder.arg(&k_ptr);
        builder.arg(&v_ptr);
        builder.arg(&out_ptr);
        builder.arg(&cap_i32);
        builder.arg(&nq_i32);
        builder.launch(cfg).map_err(|e| Error::KernelError {
            reason: format!("prefix_cache_lookup kernel launch failed: {:?}", e),
        })?;
    }

    Ok(out_block_ids)
}

/// Convert a flat `I32` device result tensor to `Vec<Option<BlockId>>`.
///
/// Copies from device to host; use only when the result must be consumed on
/// the CPU (e.g. the scheduler building a block table for a single request).
pub fn result_to_options(result: &Tensor<CudaRuntime>) -> Vec<Option<BlockId>> {
    let host_vec: Vec<i32> = result.to_vec::<i32>();
    host_vec
        .into_iter()
        .map(|v| if v < 0 { None } else { Some(v as BlockId) })
        .collect()
}
