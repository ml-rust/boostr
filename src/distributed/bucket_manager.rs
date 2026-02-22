//! Gradient bucket manager for overlapping allreduce with backward pass
//!
//! Groups model parameters into fixed-size buckets and fires allreduce
//! on each bucket as soon as all its gradients are ready, enabling
//! communication/computation overlap during the backward pass.

use std::collections::HashMap;
use std::sync::Arc;

use crate::distributed::comm_utils::all_reduce_tensor;
use crate::error::{Error, Result};
use numr::autograd::{GradStore, Var};
use numr::dtype::DType;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Communicator, ReduceOp, Runtime, RuntimeClient};
use numr::tensor::{Tensor, TensorId};

/// A bucket of parameters whose gradients are allreduced together.
struct Bucket<R: Runtime> {
    /// Parameter IDs in this bucket
    param_ids: Vec<TensorId>,
    /// Number of elements per parameter
    param_numels: Vec<usize>,
    /// Original shapes for each parameter's gradient
    param_shapes: Vec<Vec<usize>>,
    /// DType for the flat buffer (used to validate dtype consistency)
    dtype: DType,
    /// Received gradients (stored as we get hook notifications)
    received_grads: HashMap<TensorId, Tensor<R>>,
    /// Flat contiguous buffer for allreduce
    flat_buffer: Option<Tensor<R>>,
    /// Whether allreduce has been launched for this bucket
    allreduce_launched: bool,
}

/// Manages gradient buckets and fires allreduce during backward.
///
/// Parameters are grouped into buckets of approximately `bucket_size_bytes`.
/// When all gradients in a bucket are ready, they are flattened into a
/// contiguous buffer and allreduced. After backward completes, call
/// [`wait_and_unflatten`] to sync pending allreduce ops and scatter
/// the averaged gradients back into the grad store.
pub struct GradientBucketManager<R: Runtime> {
    buckets: Vec<Bucket<R>>,
    /// Maps parameter ID → bucket index
    param_to_bucket: HashMap<TensorId, usize>,
    comm: Arc<dyn Communicator>,
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
    pub fn new(
        param_info: &[(TensorId, usize, DType)],
        comm: Arc<dyn Communicator>,
        bucket_size_bytes: usize,
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
            });
        }

        Self {
            buckets,
            param_to_bucket,
            comm,
        }
    }

    /// Mark a gradient as ready. When all grads in a bucket are ready,
    /// flatten them into a contiguous buffer and launch allreduce.
    pub fn mark_grad_ready<C>(&mut self, id: TensorId, grad: &Tensor<R>, client: &C) -> Result<()>
    where
        C: RuntimeClient<R> + TensorOps<R>,
    {
        let bucket_idx = match self.param_to_bucket.get(&id) {
            Some(&idx) => idx,
            None => return Ok(()), // Not a tracked parameter
        };

        let bucket = &mut self.buckets[bucket_idx];
        if bucket.allreduce_launched {
            return Ok(()); // Already launched
        }

        // Clone required: the hook borrows grad from the backward pass, but we
        // need to own it until flatten_and_allreduce runs. Temporary 2x memory
        // per gradient until the bucket is flattened.
        bucket.received_grads.insert(id, grad.clone());

        // Check if all grads in this bucket are ready
        if bucket.received_grads.len() < bucket.param_ids.len() {
            return Ok(());
        }

        // All grads ready — flatten into contiguous buffer and launch allreduce
        self.flatten_and_allreduce(bucket_idx, client)
    }

    /// Flatten all gradients in a bucket into a contiguous buffer and launch allreduce.
    fn flatten_and_allreduce<C>(&mut self, bucket_idx: usize, client: &C) -> Result<()>
    where
        C: RuntimeClient<R> + TensorOps<R>,
    {
        let bucket = &mut self.buckets[bucket_idx];

        // Validate dtype consistency
        for &pid in &bucket.param_ids {
            if let Some(g) = bucket.received_grads.get(&pid) {
                if g.dtype() != bucket.dtype {
                    return Err(Error::DistributedError {
                        reason: format!(
                            "dtype mismatch in bucket {bucket_idx}: expected {:?}, got {:?}",
                            bucket.dtype,
                            g.dtype()
                        ),
                    });
                }
            }
        }

        // Save original shapes and collect flattened gradient tensors
        bucket.param_shapes.clear();
        let mut flat_grads: Vec<Tensor<R>> = Vec::with_capacity(bucket.param_ids.len());
        for &pid in &bucket.param_ids {
            let g = bucket
                .received_grads
                .get(&pid)
                .ok_or_else(|| Error::DistributedError {
                    reason: format!("gradient missing for param in bucket {bucket_idx}"),
                })?;
            bucket.param_shapes.push(g.shape().to_vec());
            let flat = g.flatten().map_err(|e| Error::DistributedError {
                reason: format!("flatten gradient failed: {e}"),
            })?;
            flat_grads.push(flat);
        }

        // Concatenate into one contiguous buffer
        let refs: Vec<&Tensor<R>> = flat_grads.iter().collect();
        let flat_buffer = client.cat(&refs, 0).map_err(|e| Error::DistributedError {
            reason: format!("cat gradients failed: {e}"),
        })?;

        // Launch allreduce on the flat buffer
        all_reduce_tensor(self.comm.as_ref(), &flat_buffer, ReduceOp::Sum)?;

        bucket.flat_buffer = Some(flat_buffer);
        bucket.allreduce_launched = true;

        Ok(())
    }

    /// After backward completes: sync all pending allreduce ops, unflatten
    /// buffers back into individual gradients, and divide by world_size.
    ///
    /// Writes the averaged gradients into the provided `GradStore`.
    pub fn wait_and_unflatten<C>(&mut self, client: &C, grads: &mut GradStore<R>) -> Result<()>
    where
        C: RuntimeClient<R> + TensorOps<R> + ScalarOps<R>,
    {
        // Sync all outstanding allreduce ops
        self.comm.sync().map_err(|e| Error::DistributedError {
            reason: format!("sync after allreduce failed: {e}"),
        })?;

        let world_size = self.comm.world_size();
        let scale = 1.0 / world_size as f64;

        // Unflatten each bucket's flat buffer back into individual gradients
        for bucket in &mut self.buckets {
            let flat_buffer = match bucket.flat_buffer.take() {
                Some(buf) => buf,
                None => continue,
            };

            // Slice the flat buffer to extract each param's gradient
            let mut offset = 0usize;
            for (i, &pid) in bucket.param_ids.iter().enumerate() {
                let numel = bucket.param_numels[i];
                let shape = &bucket.param_shapes[i];

                // Extract this param's slice from the flat buffer
                let flat_grad =
                    flat_buffer
                        .narrow(0, offset, numel)
                        .map_err(|e| Error::DistributedError {
                            reason: format!("narrow failed during unflatten: {e}"),
                        })?;

                // Reshape to match original gradient shape
                let reshaped = flat_grad
                    .reshape(shape)
                    .map_err(|e| Error::DistributedError {
                        reason: format!("reshape failed during unflatten: {e}"),
                    })?;

                // Scale by 1/world_size to average
                let averaged = if world_size > 1 {
                    client.mul_scalar(&reshaped, scale)?
                } else {
                    reshaped
                };

                grads.insert(pid, averaged);
                offset += numel;
            }
        }

        Ok(())
    }

    /// Reset all buckets for a new backward pass.
    pub fn reset(&mut self) {
        for bucket in &mut self.buckets {
            bucket.received_grads.clear();
            bucket.allreduce_launched = false;
            bucket.flat_buffer = None;
            bucket.param_shapes.clear();
        }
    }

    /// Number of buckets.
    pub fn num_buckets(&self) -> usize {
        self.buckets.len()
    }
}

/// Extract leaf parameter IDs from a computation graph in backward traversal order.
///
/// Performs a topological sort of the graph (same as backward), collects
/// leaf node IDs (those with no `grad_fn`), and returns them in the order
/// they are encountered during backward (reverse topological order).
///
/// This ordering is optimal for bucket construction: gradients computed
/// first during backward should be in the same bucket so the bucket fills
/// quickly and allreduce can start early.
pub fn param_order_from_graph<R: Runtime>(loss: &Var<R>) -> Vec<TensorId> {
    use std::collections::HashSet;
    use std::sync::Arc;

    let mut topo = Vec::new();
    let mut visited = HashSet::new();

    fn dfs<R: Runtime>(
        id: TensorId,
        grad_fn: Option<Arc<dyn numr::autograd::GradFn<R>>>,
        visited: &mut HashSet<TensorId>,
        topo: &mut Vec<(TensorId, bool)>, // (id, is_leaf)
    ) {
        if visited.contains(&id) {
            return;
        }
        visited.insert(id);

        let input_ids: Vec<TensorId> = grad_fn
            .as_ref()
            .map(|gf| gf.inputs().to_vec())
            .unwrap_or_default();

        if let Some(gf) = &grad_fn {
            for (input_id, input_grad_fn) in input_ids.iter().zip(gf.input_grad_fns()) {
                dfs(*input_id, input_grad_fn, visited, topo);
            }
        }

        topo.push((id, grad_fn.is_none()));
    }

    dfs(loss.id(), loss.grad_fn().cloned(), &mut visited, &mut topo);

    // Reverse topological order, keep only leaves
    topo.into_iter()
        .rev()
        .filter(|(_, is_leaf)| *is_leaf)
        .map(|(id, _)| id)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::NoOpCommunicator;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_bucket_creation_single_bucket() {
        let comm = Arc::new(NoOpCommunicator);
        let id1 = TensorId::new();
        let id2 = TensorId::new();

        // Small params, large bucket → all in one bucket
        let params = vec![(id1, 100, DType::F32), (id2, 200, DType::F32)];
        let mgr = GradientBucketManager::<CpuRuntime>::new(&params, comm, 25 * 1024 * 1024);

        assert_eq!(mgr.num_buckets(), 1);
    }

    #[test]
    fn test_bucket_creation_multiple_buckets() {
        let comm = Arc::new(NoOpCommunicator);
        let id1 = TensorId::new();
        let id2 = TensorId::new();

        // 100 f32 elements = 400 bytes, bucket_size = 200 → two buckets
        let params = vec![(id1, 100, DType::F32), (id2, 100, DType::F32)];
        let mgr = GradientBucketManager::<CpuRuntime>::new(&params, comm, 200);

        assert_eq!(mgr.num_buckets(), 2);
    }

    #[test]
    fn test_flatten_unflatten_roundtrip() {
        let (client, device) = cpu_setup();
        let comm = Arc::new(NoOpCommunicator);

        let id1 = TensorId::new();
        let id2 = TensorId::new();

        let params = vec![(id1, 3, DType::F32), (id2, 2, DType::F32)];
        let mut mgr = GradientBucketManager::<CpuRuntime>::new(&params, comm, 25 * 1024 * 1024);

        let g1 = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);
        let g2 = Tensor::<CpuRuntime>::from_slice(&[4.0f32, 5.0], &[2], &device);

        // Mark both ready — should flatten and launch allreduce
        mgr.mark_grad_ready(id1, &g1, &client).unwrap();
        mgr.mark_grad_ready(id2, &g2, &client).unwrap();

        // Wait and unflatten — with NoOp comm (world_size=1), values unchanged
        let mut grads = GradStore::new();
        mgr.wait_and_unflatten(&client, &mut grads).unwrap();

        let r1: Vec<f32> = grads.get(id1).expect("grad for id1 should exist").to_vec();
        let r2: Vec<f32> = grads.get(id2).expect("grad for id2 should exist").to_vec();
        assert_eq!(r1, vec![1.0, 2.0, 3.0]);
        assert_eq!(r2, vec![4.0, 5.0]);
    }

    #[test]
    fn test_untracked_param_ignored() {
        let (client, device) = cpu_setup();
        let comm = Arc::new(NoOpCommunicator);

        let id1 = TensorId::new();
        let untracked = TensorId::new();

        let params = vec![(id1, 2, DType::F32)];
        let mut mgr = GradientBucketManager::<CpuRuntime>::new(&params, comm, 25 * 1024 * 1024);

        let g = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device);

        // Marking an untracked param should be a no-op
        mgr.mark_grad_ready(untracked, &g, &client).unwrap();
    }

    #[test]
    fn test_multidim_gradient_shape_preserved() {
        let (client, device) = cpu_setup();
        let comm = Arc::new(NoOpCommunicator);

        let id1 = TensorId::new();
        let params = vec![(id1, 6, DType::F32)];
        let mut mgr = GradientBucketManager::<CpuRuntime>::new(&params, comm, 25 * 1024 * 1024);

        // 2x3 gradient
        let g1 =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);

        mgr.mark_grad_ready(id1, &g1, &client).unwrap();

        let mut grads = GradStore::new();
        mgr.wait_and_unflatten(&client, &mut grads).unwrap();

        let result = grads.get(id1).expect("grad for id1 should exist");
        assert_eq!(result.shape(), &[2, 3]);
        let data: Vec<f32> = result.to_vec();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }
}
