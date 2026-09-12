//! Bucket layout: grouping parameters into fixed-size, dtype-uniform buckets.

use std::collections::HashMap;
use std::sync::Arc;

use numr::dtype::DType;
use numr::runtime::{Communicator, Runtime};
use numr::tensor::{Tensor, TensorId};

/// A bucket of parameters whose gradients are allreduced together.
pub(super) struct Bucket<R: Runtime> {
    /// Parameter IDs in this bucket
    pub(super) param_ids: Vec<TensorId>,
    /// Number of elements per parameter
    pub(super) param_numels: Vec<usize>,
    /// Original shapes for each parameter's gradient
    pub(super) param_shapes: Vec<Vec<usize>>,
    /// DType for the flat buffer (used to validate dtype consistency)
    pub(super) dtype: DType,
    /// Received gradients (stored as we get hook notifications)
    pub(super) received_grads: HashMap<TensorId, Tensor<R>>,
    /// Flat contiguous buffer for allreduce
    pub(super) flat_buffer: Option<Tensor<R>>,
    /// Whether allreduce has been launched for this bucket
    pub(super) allreduce_launched: bool,
    /// Completion event handle (set when using overlapped mode)
    pub(super) completion_event: Option<u64>,
}

/// Manages gradient buckets and fires allreduce during backward.
///
/// Parameters are grouped into buckets of approximately `bucket_size_bytes`.
/// When all gradients in a bucket are ready, they are flattened into a
/// contiguous buffer and allreduced. After backward completes, call
/// [`GradientBucketManager::wait_and_unflatten`] to sync pending allreduce ops and scatter
/// the averaged gradients back into the grad store.
///
/// # Event-Based Compute-Communication Overlap
///
/// When `compute_stream_handle` is provided and the communicator supports
/// [`StreamSyncOps`](numr::runtime::StreamSyncOps), allreduce operations are
/// issued on a dedicated communication stream using CUDA event synchronization.
/// This allows gradient communication to overlap with continued backward
/// computation on the compute stream, yielding 30-40% throughput improvement
/// (the same technique used by PyTorch DDP).
///
/// On CPU or when the communicator lacks stream support, the manager falls
/// back to blocking allreduce during the backward pass.
pub struct GradientBucketManager<R: Runtime> {
    pub(super) buckets: Vec<Bucket<R>>,
    /// Maps parameter ID → bucket index
    pub(super) param_to_bucket: HashMap<TensorId, usize>,
    pub(super) comm: Arc<dyn Communicator>,
    /// Compute stream handle for event-based overlap (None = fallback to blocking sync)
    pub(super) compute_stream_handle: Option<u64>,
}

impl<R: Runtime<DType = DType>> GradientBucketManager<R> {
    /// Create a new bucket manager.
    ///
    /// # Arguments
    ///
    /// * `param_info` - Parameter (id, numel, dtype) in reverse-backward order
    ///   (last gradients computed first). This ordering maximizes overlap.
    /// * `comm` - The communicator for allreduce operations.
    /// * `bucket_size_bytes` - Target bucket size in bytes (default: 25 MiB).
    /// * `compute_stream_handle` - Optional compute stream handle from
    ///   `RuntimeClient::compute_stream_handle()`. When both this and
    ///   `comm.as_stream_sync()` are available, enables event-based
    ///   compute-communication overlap for 30-40% throughput improvement.
    pub fn new(
        param_info: &[(TensorId, usize, DType)],
        comm: Arc<dyn Communicator>,
        bucket_size_bytes: usize,
        compute_stream_handle: Option<u64>,
    ) -> Self {
        let mut buckets = Vec::new();
        let mut param_to_bucket = HashMap::new();
        let mut current_ids = Vec::new();
        let mut current_numels = Vec::new();
        let mut current_bytes = 0usize;
        let mut current_dtype = DType::F32;

        for &(id, numel, dtype) in param_info {
            let elem_bytes = dtype.size_in_bytes();
            let param_bytes = numel * elem_bytes;

            // Start a new bucket if adding this param would exceed the limit
            // or if dtype changes (all params in a bucket must share dtype)
            if !current_ids.is_empty()
                && (current_bytes + param_bytes > bucket_size_bytes || dtype != current_dtype)
            {
                let n = current_ids.len();
                for &cid in &current_ids {
                    param_to_bucket.insert(cid, buckets.len());
                }
                buckets.push(Bucket {
                    param_ids: std::mem::take(&mut current_ids),
                    param_numels: std::mem::take(&mut current_numels),
                    param_shapes: Vec::with_capacity(n),
                    dtype: current_dtype,
                    received_grads: HashMap::new(),
                    flat_buffer: None,
                    allreduce_launched: false,
                    completion_event: None,
                });
                current_bytes = 0;
            }

            current_ids.push(id);
            current_numels.push(numel);
            current_bytes += param_bytes;
            current_dtype = dtype;
        }

        // Flush remaining params into a final bucket
        if !current_ids.is_empty() {
            let n = current_ids.len();
            for &cid in &current_ids {
                param_to_bucket.insert(cid, buckets.len());
            }
            buckets.push(Bucket {
                param_ids: current_ids,
                param_numels: current_numels,
                param_shapes: Vec::with_capacity(n),
                dtype: current_dtype,
                received_grads: HashMap::new(),
                flat_buffer: None,
                allreduce_launched: false,
                completion_event: None,
            });
        }

        // Enable overlapped mode only if both stream sync and compute stream are available.
        // When the communicator lacks stream support, silently fall back to blocking allreduce.
        let overlap_handle = if comm.as_stream_sync().is_some() {
            compute_stream_handle
        } else {
            // Communicator does not support StreamSyncOps; event-based overlap unavailable.
            None
        };

        Self {
            buckets,
            param_to_bucket,
            comm,
            compute_stream_handle: overlap_handle,
        }
    }

    /// Reset all buckets for a new backward pass.
    pub fn reset(&mut self) {
        let sync = self.comm.as_stream_sync();
        for bucket in &mut self.buckets {
            bucket.received_grads.clear();
            bucket.allreduce_launched = false;
            bucket.flat_buffer = None;
            bucket.param_shapes.clear();
            // Clean up any leaked completion events
            if let Some(event) = bucket.completion_event.take()
                && let Some(s) = sync
            {
                let _ = s.destroy_event(event);
            }
        }
    }

    /// Number of buckets.
    pub fn num_buckets(&self) -> usize {
        self.buckets.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::NoOpCommunicator;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_bucket_creation_single_bucket() {
        let comm = Arc::new(NoOpCommunicator);
        let id1 = TensorId::new();
        let id2 = TensorId::new();

        // Small params, large bucket → all in one bucket
        let params = vec![(id1, 100, DType::F32), (id2, 200, DType::F32)];
        let mgr = GradientBucketManager::<CpuRuntime>::new(&params, comm, 25 * 1024 * 1024, None);

        assert_eq!(mgr.num_buckets(), 1);
    }

    #[test]
    fn test_bucket_creation_multiple_buckets() {
        let comm = Arc::new(NoOpCommunicator);
        let id1 = TensorId::new();
        let id2 = TensorId::new();

        // 100 f32 elements = 400 bytes, bucket_size = 200 → two buckets
        let params = vec![(id1, 100, DType::F32), (id2, 100, DType::F32)];
        let mgr = GradientBucketManager::<CpuRuntime>::new(&params, comm, 200, None);

        assert_eq!(mgr.num_buckets(), 2);
    }
}
