//! Distributed trainer for multi-GPU training
//!
//! Wraps `SimpleTrainer` with gradient synchronization via a `Communicator`,
//! enabling data-parallel training across multiple devices.

use std::collections::HashMap;
use std::sync::Arc;

use crate::distributed::grad_sync::{all_reduce_grads, broadcast_params};
use crate::error::Result;
use crate::trainer::SimpleTrainer;
use crate::trainer::config::{TrainingConfig, TrainingMetrics};
use numr::autograd::GradStore;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ReduceOps, ScalarOps, UnaryOps};
use numr::runtime::Communicator;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::{Tensor, TensorId};

/// Distributed trainer for data-parallel multi-GPU training
///
/// Wraps a `SimpleTrainer` and adds gradient synchronization after each
/// accumulation cycle. Uses numr's `Communicator` trait for all collective
/// operations, so it works with any backend (NCCL, MPI, NoOp).
///
/// # Usage
///
/// ```ignore
/// let comm = Arc::new(NcclCommunicator::new(...)?);
/// let mut trainer = DistributedTrainer::new(config, comm.clone())?;
///
/// // Sync initial parameters from rank 0
/// trainer.broadcast_params(&params)?;
///
/// for micro_batch in local_data {
///     let loss = forward(micro_batch, &params);
///     let grads = backward(&loss, &client)?;
///
///     if let Some(metrics) = trainer.step(&client, &mut params, grads, loss_val)? {
///         println!("rank {} step {} loss={:.4}", comm.rank(), metrics.step, metrics.loss);
///     }
/// }
/// ```
pub struct DistributedTrainer<R: Runtime<DType = DType>> {
    inner: SimpleTrainer<R>,
    comm: Arc<dyn Communicator>,
}

impl<R: Runtime<DType = DType>> DistributedTrainer<R> {
    /// Create a new distributed trainer.
    pub fn new(config: TrainingConfig, comm: Arc<dyn Communicator>) -> Result<Self> {
        let inner = SimpleTrainer::new(config)?;
        Ok(Self { inner, comm })
    }

    /// Broadcast parameters from rank 0 to all other ranks.
    ///
    /// Call this before the first training step to ensure all ranks
    /// start with identical parameters.
    pub fn broadcast_params(&self, params: &HashMap<TensorId, Tensor<R>>) -> Result<()> {
        broadcast_params(self.comm.as_ref(), params, 0)
    }

    /// Process one micro-batch of gradients.
    ///
    /// Delegates to `SimpleTrainer::step()`. When the inner trainer completes
    /// a full accumulation cycle, the accumulated gradients are all-reduced
    /// across ranks before clipping and the optimizer step.
    ///
    /// Returns `None` if still accumulating, `Some(metrics)` after a full step.
    pub fn step<C>(
        &mut self,
        client: &C,
        params: &mut HashMap<TensorId, Tensor<R>>,
        grads: GradStore<R>,
        loss_value: f64,
    ) -> Result<Option<TrainingMetrics>>
    where
        C: RuntimeClient<R> + BinaryOps<R> + UnaryOps<R> + ScalarOps<R> + ReduceOps<R>,
    {
        // We need to intercept the gradient accumulation to insert all_reduce.
        // Since SimpleTrainer handles accumulation + clipping + optimizer internally,
        // and we need to all_reduce between accumulation and clipping, we accumulate
        // manually, all_reduce, then feed into the inner trainer.
        //
        // However, SimpleTrainer's step() is monolithic. Instead, we track
        // accumulation ourselves and delegate the full step to inner, which
        // will re-accumulate (with accum_steps=1 effectively). But that changes
        // semantics.
        //
        // The cleanest approach: just call inner.step() and let it handle
        // accumulation. The all_reduce happens on the averaged grads that
        // inner.step() returns. But inner.step() also applies the optimizer.
        //
        // Actually, the simplest correct approach for data-parallel:
        // all-reduce the raw micro-batch grads BEFORE feeding them to the
        // trainer. This way, each rank's accumulator sums already-synchronized
        // gradients. This is semantically equivalent and simpler.

        let mut synced_grads = grads;
        all_reduce_grads(self.comm.as_ref(), client, &mut synced_grads)?;

        self.inner.step(client, params, synced_grads, loss_value)
    }

    /// Current global step count.
    pub fn global_step(&self) -> u64 {
        self.inner.global_step()
    }

    /// Reference to the underlying communicator.
    pub fn communicator(&self) -> &dyn Communicator {
        self.comm.as_ref()
    }
}

// Safety: DistributedTrainer is Send+Sync if its fields are.
// SimpleTrainer<R> is Send+Sync when R: Send+Sync (Runtime requires it).
// Arc<dyn Communicator> is Send+Sync because Communicator: Send+Sync.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use crate::trainer::config::TrainingConfig;
    use numr::runtime::NoOpCommunicator;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_distributed_trainer_creation() {
        let comm = Arc::new(NoOpCommunicator);
        let config = TrainingConfig {
            learning_rate: 1e-3,
            weight_decay: 0.01,
            grad_accum_steps: 1,
            max_grad_norm: Some(1.0),
        };
        let trainer = DistributedTrainer::<CpuRuntime>::new(config, comm).unwrap();
        assert_eq!(trainer.global_step(), 0);
    }

    #[test]
    fn test_distributed_trainer_broadcast_noop() {
        let comm = Arc::new(NoOpCommunicator);
        let config = TrainingConfig {
            learning_rate: 1e-3,
            weight_decay: 0.01,
            grad_accum_steps: 1,
            max_grad_norm: None,
        };
        let trainer = DistributedTrainer::<CpuRuntime>::new(config, comm).unwrap();

        let (_client, device) = cpu_setup();
        let id = TensorId::new();
        let t = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device);
        let mut params = HashMap::new();
        params.insert(id, t);

        trainer.broadcast_params(&params).unwrap();
    }

    #[test]
    fn test_distributed_trainer_step() {
        let (client, device) = cpu_setup();
        let comm = Arc::new(NoOpCommunicator);
        let config = TrainingConfig {
            learning_rate: 1e-4,
            weight_decay: 0.0,
            grad_accum_steps: 1,
            max_grad_norm: None,
        };
        let mut trainer = DistributedTrainer::<CpuRuntime>::new(config, comm).unwrap();

        // Create a param and matching grad
        let param_id = TensorId::new();
        let param = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device);
        let mut params = HashMap::new();
        params.insert(param_id, param);

        let grad = Tensor::<CpuRuntime>::from_slice(&[0.1f32, 0.2], &[2], &device);
        let mut grads = GradStore::new();
        grads.insert(param_id, grad);

        let result = trainer.step(&client, &mut params, grads, 0.5).unwrap();
        assert!(result.is_some());
        let metrics = result.unwrap();
        assert_eq!(metrics.step, 1);
    }

    #[test]
    fn test_distributed_trainer_grad_accumulation() {
        let (client, device) = cpu_setup();
        let comm = Arc::new(NoOpCommunicator);
        let config = TrainingConfig {
            learning_rate: 1e-4,
            weight_decay: 0.0,
            grad_accum_steps: 2,
            max_grad_norm: None,
        };
        let mut trainer = DistributedTrainer::<CpuRuntime>::new(config, comm).unwrap();

        let param_id = TensorId::new();
        let param = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device);
        let mut params = HashMap::new();
        params.insert(param_id, param);

        // First micro-batch: should return None (still accumulating)
        let grad1 = Tensor::<CpuRuntime>::from_slice(&[0.1f32, 0.2], &[2], &device);
        let mut grads1 = GradStore::new();
        grads1.insert(param_id, grad1);
        let result = trainer.step(&client, &mut params, grads1, 0.5).unwrap();
        assert!(result.is_none());

        // Second micro-batch: should return Some (accumulation complete)
        let grad2 = Tensor::<CpuRuntime>::from_slice(&[0.3f32, 0.4], &[2], &device);
        let mut grads2 = GradStore::new();
        grads2.insert(param_id, grad2);
        let result = trainer.step(&client, &mut params, grads2, 0.6).unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().step, 1);
    }

    #[test]
    fn test_distributed_trainer_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<DistributedTrainer<CpuRuntime>>();
    }
}
