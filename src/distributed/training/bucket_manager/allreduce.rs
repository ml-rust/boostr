//! Gradient readiness tracking, flatten + allreduce launch, and unflatten after backward.

use crate::distributed::comm_utils::all_reduce_tensor;
use crate::error::{Error, Result};
use numr::autograd::GradStore;
use numr::dtype::DType;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{ReduceOp, Runtime, RuntimeClient};
use numr::tensor::{Tensor, TensorId};

use super::bucket::GradientBucketManager;

impl<R: Runtime<DType = DType>> GradientBucketManager<R> {
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
            if let Some(g) = bucket.received_grads.get(&pid)
                && g.dtype() != bucket.dtype
            {
                return Err(Error::DistributedError {
                    reason: format!(
                        "dtype mismatch in bucket {bucket_idx}: expected {:?}, got {:?}",
                        bucket.dtype,
                        g.dtype()
                    ),
                });
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

        // Launch allreduce — with event-based overlap if available
        if let Some(compute_stream) = self.compute_stream_handle {
            let sync = self
                .comm
                .as_stream_sync()
                .expect("compute_stream_handle is Some only when as_stream_sync() is Some");

            // 1. Record event on compute stream (gradient data is ready)
            let ready_event = sync.create_event().map_err(|e| Error::DistributedError {
                reason: format!("create ready event failed: {e}"),
            })?;

            // Use a closure to ensure ready_event cleanup on any error path.
            let overlap_result = (|| -> Result<u64> {
                sync.record_on_stream(ready_event, compute_stream)
                    .map_err(|e| Error::DistributedError {
                        reason: format!("record ready event failed: {e}"),
                    })?;

                // 2. Make comm stream wait for gradient data to be ready.
                // After comm_stream_wait_event returns, the CUDA driver has captured
                // the event dependency in the comm stream's work queue. The event
                // handle is safe to destroy: CUDA events are reference-counted
                // internally and the driver keeps the dependency alive until the
                // stream has executed past the wait point.
                sync.comm_stream_wait_event(ready_event)
                    .map_err(|e| Error::DistributedError {
                        reason: format!("comm stream wait for ready event failed: {e}"),
                    })?;

                // 3. Launch allreduce (runs on comm stream, non-blocking to compute)
                all_reduce_tensor(self.comm.as_ref(), &flat_buffer, ReduceOp::Sum)?;

                // 4. Create and record completion event on comm stream
                let completion_event =
                    sync.create_event().map_err(|e| Error::DistributedError {
                        reason: format!("create completion event failed: {e}"),
                    })?;

                if let Err(e) = sync.record_on_comm_stream(completion_event) {
                    let _ = sync.destroy_event(completion_event);
                    return Err(Error::DistributedError {
                        reason: format!("record completion event failed: {e}"),
                    });
                }

                Ok(completion_event)
            })();

            // Always destroy the ready event — safe because CUDA events are
            // reference-counted; the driver holds the dependency until the comm
            // stream executes past its wait point.
            let _ = sync.destroy_event(ready_event);

            bucket.completion_event = Some(overlap_result?);
        } else {
            // Fallback: blocking allreduce (no overlap)
            all_reduce_tensor(self.comm.as_ref(), &flat_buffer, ReduceOp::Sum)?;
        }

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
        let world_size = self.comm.world_size();
        let scale = 1.0 / world_size as f64;

        if let Some(compute_stream) = self.compute_stream_handle {
            // Event-based: make compute stream wait on each bucket's completion event
            let sync = self
                .comm
                .as_stream_sync()
                .expect("compute_stream_handle is Some only when as_stream_sync() is Some");
            for bucket in &mut self.buckets {
                if let Some(event) = bucket.completion_event.take() {
                    if let Err(e) = sync.stream_wait_event(compute_stream, event) {
                        let _ = sync.destroy_event(event);
                        return Err(Error::DistributedError {
                            reason: format!("compute stream wait for completion event failed: {e}"),
                        });
                    }
                    let _ = sync.destroy_event(event);
                }
            }
        } else {
            // Fallback: blocking sync
            self.comm.sync().map_err(|e| Error::DistributedError {
                reason: format!("sync after allreduce failed: {e}"),
            })?;
        }

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::NoOpCommunicator;
    use numr::runtime::cpu::CpuRuntime;
    use std::sync::Arc;

    #[test]
    fn test_flatten_unflatten_roundtrip() {
        let (client, device) = cpu_setup();
        let comm = Arc::new(NoOpCommunicator);

        let id1 = TensorId::new();
        let id2 = TensorId::new();

        let params = vec![(id1, 3, DType::F32), (id2, 2, DType::F32)];
        let mut mgr =
            GradientBucketManager::<CpuRuntime>::new(&params, comm, 25 * 1024 * 1024, None);

        let g1 = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device).unwrap();
        let g2 = Tensor::<CpuRuntime>::from_slice(&[4.0f32, 5.0], &[2], &device).unwrap();

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
        let mut mgr =
            GradientBucketManager::<CpuRuntime>::new(&params, comm, 25 * 1024 * 1024, None);

        let g = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device).unwrap();

        // Marking an untracked param should be a no-op
        mgr.mark_grad_ready(untracked, &g, &client).unwrap();
    }

    #[test]
    fn test_multidim_gradient_shape_preserved() {
        let (client, device) = cpu_setup();
        let comm = Arc::new(NoOpCommunicator);

        let id1 = TensorId::new();
        let params = vec![(id1, 6, DType::F32)];
        let mut mgr =
            GradientBucketManager::<CpuRuntime>::new(&params, comm, 25 * 1024 * 1024, None);

        // 2x3 gradient
        let g1 =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)
                .unwrap();

        mgr.mark_grad_ready(id1, &g1, &client).unwrap();

        let mut grads = GradStore::new();
        mgr.wait_and_unflatten(&client, &mut grads).unwrap();

        let result = grads.get(id1).expect("grad for id1 should exist");
        assert_eq!(result.shape(), &[2, 3]);
        let data: Vec<f32> = result.to_vec();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }
}
