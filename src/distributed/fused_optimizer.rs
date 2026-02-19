//! Fused distributed optimizer
//!
//! Overlaps gradient all-reduce with optimizer step by processing
//! parameter buckets: all-reduce one bucket while stepping the previous.
//! Follows PyTorch DDP's bucketing strategy (default 25MB buckets).

use std::collections::HashMap;
use std::sync::Arc;

use crate::distributed::comm_utils::all_reduce_tensor;
use crate::error::{Error, Result};
use crate::optimizer::adamw::{AdamW, AdamWConfig};
use numr::autograd::GradStore;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ScalarOps, UnaryOps};
use numr::runtime::{Communicator, ReduceOp, Runtime, RuntimeClient};
use numr::tensor::{Tensor, TensorId};

/// Configuration for fused distributed optimizer
#[derive(Debug, Clone)]
pub struct FusedOptimizerConfig {
    /// AdamW optimizer configuration
    pub adamw: AdamWConfig,
    /// Bucket size in bytes for gradient bucketing (default 25MB, matching PyTorch DDP)
    pub bucket_size_bytes: usize,
}

impl Default for FusedOptimizerConfig {
    fn default() -> Self {
        Self {
            adamw: AdamWConfig::default(),
            bucket_size_bytes: 25 * 1024 * 1024, // 25MB
        }
    }
}

/// Fused distributed optimizer that overlaps gradient all-reduce with optimizer step.
///
/// Instead of all-reducing all gradients then stepping all params sequentially,
/// it partitions parameters into buckets by total size. For each bucket:
/// 1. Kick off all-reduce for current bucket's gradients
/// 2. Sync and apply optimizer to the previous bucket
///
/// This overlaps communication with computation, reducing wall-clock time
/// proportionally to the number of buckets.
///
/// Falls back to simple sequential all-reduce + optimize when world_size=1.
pub struct FusedDistributedOptimizer<R: Runtime> {
    optimizer: AdamW<R>,
    comm: Arc<dyn Communicator>,
    bucket_size_bytes: usize,
}

/// A bucket of parameter IDs grouped by total gradient size
struct Bucket {
    param_ids: Vec<TensorId>,
}

impl<R: Runtime<DType = DType>> FusedDistributedOptimizer<R> {
    pub fn new(config: FusedOptimizerConfig, comm: Arc<dyn Communicator>) -> Self {
        Self {
            optimizer: AdamW::new(config.adamw),
            comm,
            bucket_size_bytes: config.bucket_size_bytes,
        }
    }

    /// Perform one fused optimization step.
    ///
    /// All-reduces gradients across ranks and applies AdamW in a pipelined fashion.
    /// Parameters without gradients are skipped entirely.
    pub fn step<C>(
        &mut self,
        client: &C,
        params: &mut HashMap<TensorId, Tensor<R>>,
        grads: &mut GradStore<R>,
    ) -> Result<()>
    where
        C: RuntimeClient<R> + BinaryOps<R> + UnaryOps<R> + ScalarOps<R>,
    {
        let world_size = self.comm.world_size();

        // Single device: skip all-reduce, just optimize
        if world_size <= 1 {
            return self.optimizer.step(client, params, grads);
        }

        // Collect param IDs that have gradients
        let param_ids_with_grads: Vec<TensorId> = params
            .keys()
            .copied()
            .filter(|id| grads.get(*id).is_some())
            .collect();

        if param_ids_with_grads.is_empty() {
            return Ok(());
        }

        // Partition into buckets by gradient size
        let buckets = self.build_buckets(&param_ids_with_grads, grads);

        if buckets.is_empty() {
            return Ok(());
        }

        // Pipeline: all-reduce bucket[i] while optimizing bucket[i-1]
        // Step 1: kick off all-reduce for bucket 0
        self.all_reduce_bucket(&buckets[0], grads)?;

        for i in 1..buckets.len() {
            // Sync previous bucket's all-reduce
            self.comm.sync().map_err(|e| Error::DistributedError {
                reason: format!("sync failed: {e}"),
            })?;

            // Average previous bucket's gradients
            self.average_bucket(client, &buckets[i - 1], grads, world_size)?;

            // Apply optimizer to previous bucket
            let prev_grad_store = self.extract_bucket_grads(&buckets[i - 1], grads);
            self.optimizer.step(client, params, &prev_grad_store)?;

            // Kick off all-reduce for current bucket
            self.all_reduce_bucket(&buckets[i], grads)?;
        }

        // Sync + optimize the last bucket
        self.comm.sync().map_err(|e| Error::DistributedError {
            reason: format!("sync failed: {e}"),
        })?;

        let last = buckets.len() - 1;
        self.average_bucket(client, &buckets[last], grads, world_size)?;

        let last_grad_store = self.extract_bucket_grads(&buckets[last], grads);
        self.optimizer.step(client, params, &last_grad_store)?;

        Ok(())
    }

    fn build_buckets(&self, param_ids: &[TensorId], grads: &GradStore<R>) -> Vec<Bucket> {
        let mut buckets = Vec::new();
        let mut current_ids = Vec::new();
        let mut current_bytes: usize = 0;

        for &id in param_ids {
            let grad = match grads.get(id) {
                Some(g) => g,
                None => continue,
            };
            let grad_bytes = grad.numel() * grad.dtype().size_in_bytes();

            if !current_ids.is_empty() && current_bytes + grad_bytes > self.bucket_size_bytes {
                buckets.push(Bucket {
                    param_ids: std::mem::take(&mut current_ids),
                });
                current_bytes = 0;
            }

            current_ids.push(id);
            current_bytes += grad_bytes;
        }

        if !current_ids.is_empty() {
            buckets.push(Bucket {
                param_ids: current_ids,
            });
        }

        buckets
    }

    fn all_reduce_bucket(&self, bucket: &Bucket, grads: &GradStore<R>) -> Result<()> {
        for &id in &bucket.param_ids {
            let grad = grads.get(id).ok_or_else(|| Error::DistributedError {
                reason: "gradient disappeared during bucket all-reduce".to_string(),
            })?;

            all_reduce_tensor(self.comm.as_ref(), grad, ReduceOp::Sum)?;
        }
        Ok(())
    }

    fn average_bucket<C>(
        &self,
        client: &C,
        bucket: &Bucket,
        grads: &mut GradStore<R>,
        world_size: usize,
    ) -> Result<()>
    where
        C: ScalarOps<R>,
    {
        let scale = 1.0 / world_size as f64;
        for &id in &bucket.param_ids {
            let grad = grads.get(id).ok_or_else(|| Error::DistributedError {
                reason: "gradient disappeared during averaging".to_string(),
            })?;
            let scaled = client.mul_scalar(grad, scale)?;
            grads.insert(id, scaled);
        }
        Ok(())
    }

    fn extract_bucket_grads(&self, bucket: &Bucket, grads: &GradStore<R>) -> GradStore<R> {
        let mut store = GradStore::new();
        for &id in &bucket.param_ids {
            if let Some(grad) = grads.get(id) {
                store.insert(id, grad.clone());
            }
        }
        store
    }

    pub fn set_lr(&mut self, lr: f64) {
        self.optimizer.set_lr(lr);
    }

    pub fn timestep(&self) -> u64 {
        self.optimizer.timestep()
    }

    pub fn communicator(&self) -> &dyn Communicator {
        self.comm.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::NoOpCommunicator;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_fused_optimizer_single_device() {
        let (client, device) = cpu_setup();
        let comm = Arc::new(NoOpCommunicator);
        let config = FusedOptimizerConfig::default();
        let mut opt = FusedDistributedOptimizer::<CpuRuntime>::new(config, comm);

        let id = TensorId::new();
        let param = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);
        let mut params = HashMap::new();
        params.insert(id, param);

        let grad = Tensor::<CpuRuntime>::from_slice(&[0.1f32, 0.2, 0.3], &[3], &device);
        let mut grads = GradStore::new();
        grads.insert(id, grad);

        opt.step(&client, &mut params, &mut grads).unwrap();
        assert_eq!(opt.timestep(), 1);

        let updated = params.get(&id).unwrap().to_vec::<f32>();
        assert_ne!(updated, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_fused_optimizer_no_grads() {
        let (client, _device) = cpu_setup();
        let comm = Arc::new(NoOpCommunicator);
        let config = FusedOptimizerConfig::default();
        let mut opt = FusedDistributedOptimizer::<CpuRuntime>::new(config, comm);

        let mut params = HashMap::new();
        let mut grads = GradStore::new();

        opt.step(&client, &mut params, &mut grads).unwrap();
        // AdamW increments timestep even with no matching grads
        assert_eq!(opt.timestep(), 1);
    }

    #[test]
    fn test_fused_optimizer_bucket_building() {
        let (_client, device) = cpu_setup();
        let comm = Arc::new(NoOpCommunicator);
        // Tiny bucket size to force multiple buckets
        let config = FusedOptimizerConfig {
            bucket_size_bytes: 16, // 4 floats = 16 bytes
            ..Default::default()
        };
        let opt = FusedDistributedOptimizer::<CpuRuntime>::new(config, comm);

        let id1 = TensorId::new();
        let id2 = TensorId::new();
        let g1 = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 4], &[4], &device);
        let g2 = Tensor::<CpuRuntime>::from_slice(&[2.0f32; 4], &[4], &device);

        let mut grads = GradStore::new();
        grads.insert(id1, g1);
        grads.insert(id2, g2);

        let ids = vec![id1, id2];
        let buckets = opt.build_buckets(&ids, &grads);
        // Each grad is 16 bytes, bucket_size is 16, so first fits, second creates new bucket
        assert_eq!(buckets.len(), 2);
    }

    #[test]
    fn test_fused_optimizer_set_lr() {
        let comm = Arc::new(NoOpCommunicator);
        let mut opt =
            FusedDistributedOptimizer::<CpuRuntime>::new(FusedOptimizerConfig::default(), comm);
        opt.set_lr(0.01);
        assert_eq!(opt.optimizer.config().lr, 0.01);
    }
}
