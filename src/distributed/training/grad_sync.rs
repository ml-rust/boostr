//! Tensor-level wrappers around numr's raw-pointer `Communicator`
//!
//! Provides safe gradient synchronization and parameter broadcasting
//! for multi-GPU training, built on numr's `Communicator` trait.

use std::collections::HashMap;

use crate::distributed::comm_utils::{all_reduce_tensor, broadcast_tensor};
use crate::error::{Error, Result};
use numr::autograd::GradStore;
use numr::dtype::DType;
use numr::ops::ScalarOps;
use numr::runtime::{Communicator, ReduceOp, Runtime, RuntimeClient};
use numr::tensor::{Tensor, TensorId};

/// AllReduce all gradient tensors in a `GradStore` (in-place sum, then average by world_size).
///
/// Each gradient tensor is all-reduced with `ReduceOp::Sum` across all ranks,
/// then divided by `world_size` to produce the average gradient.
///
/// # Errors
///
/// Returns `DistributedError` if any tensor is non-contiguous or if a
/// communicator operation fails.
pub fn all_reduce_grads<R, C>(
    comm: &dyn Communicator,
    client: &C,
    grads: &mut GradStore<R>,
) -> Result<()>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + ScalarOps<R>,
{
    let world_size = comm.world_size();
    if world_size <= 1 {
        return Ok(());
    }

    let ids: Vec<TensorId> = grads.keys().copied().collect();

    for id in &ids {
        let tensor = grads.get(*id).ok_or_else(|| Error::DistributedError {
            reason: "gradient disappeared during iteration".to_string(),
        })?;

        all_reduce_tensor(comm, tensor, ReduceOp::Sum)?;
    }

    comm.sync().map_err(|e| Error::DistributedError {
        reason: format!("sync after all_reduce failed: {e}"),
    })?;

    // Average by world_size
    let scale = 1.0 / world_size as f64;
    for id in &ids {
        let tensor = grads.get(*id).ok_or_else(|| Error::DistributedError {
            reason: "gradient disappeared during averaging iteration".to_string(),
        })?;
        let scaled = client.mul_scalar(tensor, scale)?;
        grads.insert(*id, scaled);
    }

    Ok(())
}

/// Broadcast all parameters from the root rank to all other ranks.
///
/// Ensures all ranks start with identical parameters before training begins.
///
/// # Errors
///
/// Returns `DistributedError` if any tensor is non-contiguous or if a
/// communicator operation fails.
pub fn broadcast_params<R: Runtime<DType = DType>>(
    comm: &dyn Communicator,
    params: &HashMap<TensorId, Tensor<R>>,
    root: usize,
) -> Result<()> {
    if comm.world_size() <= 1 {
        return Ok(());
    }

    for tensor in params.values() {
        broadcast_tensor(comm, tensor, root)?;
    }

    comm.sync().map_err(|e| Error::DistributedError {
        reason: format!("sync after broadcast failed: {e}"),
    })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::NoOpCommunicator;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_all_reduce_grads_noop_single_device() {
        let (client, device) = cpu_setup();
        let comm = NoOpCommunicator;

        let id = TensorId::new();
        let t = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 4.0, 6.0], &[3], &device);
        let mut grads = GradStore::new();
        grads.insert(id, t);

        // world_size=1 → early return, grads unchanged
        all_reduce_grads(&comm, &client, &mut grads).unwrap();

        let data = grads.get(id).unwrap().to_vec::<f32>();
        assert!((data[0] - 2.0).abs() < 1e-6);
        assert!((data[1] - 4.0).abs() < 1e-6);
        assert!((data[2] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_all_reduce_grads_empty() {
        let (client, _device) = cpu_setup();
        let comm = NoOpCommunicator;
        let mut grads = GradStore::<CpuRuntime>::new();

        all_reduce_grads(&comm, &client, &mut grads).unwrap();
        assert!(grads.is_empty());
    }

    #[test]
    fn test_broadcast_params_noop() {
        let (_client, device) = cpu_setup();
        let comm = NoOpCommunicator;

        let id = TensorId::new();
        let t = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device);
        let mut params = HashMap::new();
        params.insert(id, t);

        // world_size=1 → early return, no-op
        broadcast_params(&comm, &params, 0).unwrap();

        let data = params[&id].to_vec::<f32>();
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_broadcast_params_empty() {
        let comm = NoOpCommunicator;
        let params: HashMap<TensorId, Tensor<CpuRuntime>> = HashMap::new();
        broadcast_params(&comm, &params, 0).unwrap();
    }
}
