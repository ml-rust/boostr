//! Ordered broadcast of named tensors from a root rank.

use super::buckets::{
    BucketPlan, DEFAULT_ORDERED_BUCKET_BYTES, dist_err, order_after_compute, plan_buckets,
    sync_comm,
};
use crate::distributed::comm_utils::broadcast_tensor;
use crate::error::Result;
use numr::dtype::DType;
use numr::ops::TensorOps;
use numr::runtime::{Communicator, Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Copy the root rank's `tensors` into every rank's `tensors`, in slice order.
///
/// Calls [`broadcast_params_ordered_bucketed`] with [`DEFAULT_ORDERED_BUCKET_BYTES`].
///
/// # Errors
///
/// See [`broadcast_params_ordered_bucketed`].
pub fn broadcast_params_ordered<R, C>(
    comm: &dyn Communicator,
    client: &C,
    tensors: &[(String, Tensor<R>)],
    root: usize,
) -> Result<()>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R>,
{
    broadcast_params_ordered_bucketed(comm, client, tensors, root, DEFAULT_ORDERED_BUCKET_BYTES)
}

/// Copy the root rank's `tensors` into every rank's `tensors`, in slice order.
///
/// # Order contract
///
/// Every rank must pass the same tensors, with the same shapes and dtypes, in the same order.
/// Sort by name on every rank, or use the model's own stable order.
/// The names serve error messages only.
///
/// # Algorithm
///
/// 1. Consecutive tensors of one dtype pack into buckets of at most `bucket_bytes`.
///    A dtype change starts a new bucket, so no tensor is cast.
/// 2. The root concatenates each bucket into a new flat buffer.
///    Other ranks allocate an uninitialized buffer of the same size.
/// 3. Each buffer is broadcast from `root`, then the communicator syncs.
/// 4. Non-root ranks copy each buffer slice back into its tensor's storage.
///
/// The copy writes through the tensor's storage. Frozen and trainable tensors both work.
/// Every tensor sharing that storage sees the new values.
/// The copy addresses device memory by pointer offset, as the `Communicator` contract does.
///
/// # World size 1
///
/// With one rank the function returns at once and copies nothing.
///
/// # Errors
///
/// Returns `DistributedError` before any collective when:
/// - `root` is not below the world size
/// - a tensor is not contiguous
///
/// Returns an error when a device op or a collective fails.
/// An error on one rank leaves its peers waiting in the collective. The caller must abort the group.
pub fn broadcast_params_ordered_bucketed<R, C>(
    comm: &dyn Communicator,
    client: &C,
    tensors: &[(String, Tensor<R>)],
    root: usize,
    bucket_bytes: usize,
) -> Result<()>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R>,
{
    let world = comm.world_size();
    if world <= 1 {
        return Ok(());
    }
    if root >= world {
        return Err(dist_err(format!(
            "broadcast_params_ordered: root {root} is not below the world size {world}. \
             Pass a rank in 0..{world}"
        )));
    }
    for (name, tensor) in tensors {
        if !tensor.is_contiguous() {
            return Err(dist_err(format!(
                "broadcast_params_ordered: tensor `{name}` is not contiguous. \
                 Pass a contiguous tensor so the broadcast can write into it"
            )));
        }
    }

    let segments: Vec<(usize, DType)> = tensors
        .iter()
        .map(|(_, tensor)| (tensor.numel(), tensor.dtype()))
        .collect();
    let plans = plan_buckets(&segments, bucket_bytes);
    let is_root = comm.rank() == root;

    let mut buffers = Vec::with_capacity(plans.len());
    for plan in &plans {
        let buffer = if is_root {
            pack(client, bucket_tensors(tensors, plan)?)?
        } else {
            Tensor::<R>::empty(&[plan.elems], plan.dtype, client.device())?
        };
        buffers.push(buffer);
    }

    order_after_compute::<R, C>(comm, client)?;
    for buffer in &buffers {
        broadcast_tensor(comm, buffer, root)?;
    }
    sync_comm(comm, "broadcast_params_ordered")?;

    if is_root {
        return Ok(());
    }
    for (plan, buffer) in plans.iter().zip(&buffers) {
        unpack(buffer, bucket_tensors(tensors, plan)?, plan.dtype)?;
    }
    Ok(())
}

/// The tensors of one bucket.
fn bucket_tensors<'a, R: Runtime>(
    tensors: &'a [(String, Tensor<R>)],
    plan: &BucketPlan,
) -> Result<&'a [(String, Tensor<R>)]> {
    tensors.get(plan.segments.clone()).ok_or_else(|| {
        dist_err(format!(
            "broadcast_params_ordered: bucket {:?} lies outside {} tensors",
            plan.segments,
            tensors.len()
        ))
    })
}

/// Concatenate one bucket's tensors into a new flat buffer.
fn pack<R, C>(client: &C, group: &[(String, Tensor<R>)]) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R>,
{
    let mut pieces = Vec::with_capacity(group.len());
    for (_, tensor) in group {
        pieces.push(tensor.flatten()?);
    }
    let refs: Vec<&Tensor<R>> = pieces.iter().collect();
    Ok(client.cat(&refs, 0)?)
}

/// Copy consecutive slices of `buffer` into the storage of each tensor in `group`.
fn unpack<R: Runtime<DType = DType>>(
    buffer: &Tensor<R>,
    group: &[(String, Tensor<R>)],
    dtype: DType,
) -> Result<()> {
    let elem_bytes = dtype.size_in_bytes();
    let mut offset = 0usize;
    for (name, tensor) in group {
        let numel = tensor.numel();
        if numel > 0 {
            let src = buffer.narrow(0, offset, numel)?;
            R::copy_within_device(src.ptr(), tensor.ptr(), numel * elem_bytes, tensor.device())
                .map_err(|e| {
                    dist_err(format!(
                        "broadcast_params_ordered: copy into tensor `{name}` failed: {e}"
                    ))
                })?;
        }
        offset += numel;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::LocalCommunicator;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    /// One rank's view after the broadcast: names in slice order, then the values.
    type View = (Vec<String>, Vec<f32>, Vec<f32>, Vec<i64>);

    /// Tensors in an unsorted name order, with mixed dtypes.
    ///
    /// The default bucket packs `w` and `b` together. A 4-byte bucket splits every tensor.
    fn tensors(rank: usize, device: &CpuDevice) -> Vec<(String, Tensor<CpuRuntime>)> {
        let base = rank as f32 * 10.0;
        let w: Vec<f32> = (0..4).map(|i| base + i as f32 + 0.25).collect();
        let steps: Vec<i64> = (0..3).map(|i| rank as i64 * 100 + i).collect();
        let b: Vec<f32> = (0..5).map(|i| -base - i as f32).collect();
        vec![
            (
                "w".to_string(),
                Tensor::<CpuRuntime>::from_slice(&w, &[2, 2], device).unwrap(),
            ),
            (
                "b".to_string(),
                Tensor::<CpuRuntime>::from_slice(&b, &[5], device).unwrap(),
            ),
            (
                "steps".to_string(),
                Tensor::<CpuRuntime>::from_slice(&steps, &[3], device).unwrap(),
            ),
        ]
    }

    fn view(tensors: &[(String, Tensor<CpuRuntime>)]) -> View {
        (
            tensors.iter().map(|(name, _)| name.clone()).collect(),
            tensors[0].1.to_vec::<f32>(),
            tensors[1].1.to_vec::<f32>(),
            tensors[2].1.to_vec::<i64>(),
        )
    }

    fn broadcast_on_ranks(world: usize, root: usize, bucket_bytes: usize) -> Vec<View> {
        let comms = LocalCommunicator::group(world);
        std::thread::scope(|s| {
            let handles: Vec<_> = comms
                .iter()
                .map(|comm| {
                    s.spawn(move || {
                        let _guard = comm.abort_on_panic();
                        let device = CpuDevice::new();
                        let client = CpuClient::new(device.clone());
                        let tensors = tensors(comm.rank(), &device);
                        broadcast_params_ordered_bucketed(
                            comm,
                            &client,
                            &tensors,
                            root,
                            bucket_bytes,
                        )
                        .unwrap();
                        assert_eq!(tensors[0].1.shape(), &[2, 2]);
                        assert_eq!(tensors[2].1.dtype(), DType::I64);
                        view(&tensors)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        })
    }

    #[test]
    fn test_root_values_reach_every_rank_in_order() {
        let device = CpuDevice::new();
        for (world, root) in [(2, 0), (3, 1), (3, 2)] {
            let expected = view(&tensors(root, &device));
            for got in broadcast_on_ranks(world, root, DEFAULT_ORDERED_BUCKET_BYTES) {
                assert_eq!(got, expected, "world {world}, root {root}");
            }
        }
    }

    #[test]
    fn test_tiny_buckets_match_one_bucket() {
        let one = broadcast_on_ranks(3, 1, DEFAULT_ORDERED_BUCKET_BYTES);
        let many = broadcast_on_ranks(3, 1, 4);
        assert_eq!(one, many);
    }

    #[test]
    fn test_world_one_changes_nothing() {
        let comms = LocalCommunicator::group(1);
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let tensors = tensors(0, &device);
        let before = view(&tensors);
        broadcast_params_ordered(&comms[0], &client, &tensors, 0).unwrap();
        assert_eq!(view(&tensors), before);
    }

    #[test]
    fn test_bad_root_and_strided_tensor_are_errors() {
        let comms = LocalCommunicator::group(2);
        std::thread::scope(|s| {
            for comm in &comms {
                s.spawn(move || {
                    let _guard = comm.abort_on_panic();
                    let device = CpuDevice::new();
                    let client = CpuClient::new(device.clone());
                    let good = tensors(comm.rank(), &device);
                    assert!(broadcast_params_ordered(comm, &client, &good, 2).is_err());
                    let strided = good[0].1.transpose(0, 1).unwrap();
                    let bad = vec![("w.t".to_string(), strided)];
                    assert!(broadcast_params_ordered(comm, &client, &bad, 0).is_err());
                    assert!(!comm.is_aborted());
                });
            }
        });
    }
}
