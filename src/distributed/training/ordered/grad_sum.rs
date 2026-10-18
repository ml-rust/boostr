//! Ordered flat all-reduce of a gradient store for data-parallel training.

use std::collections::HashSet;

use super::buckets::{
    BucketPlan, DEFAULT_ORDERED_BUCKET_BYTES, dist_err, order_after_compute, plan_buckets,
    sync_comm,
};
use crate::distributed::comm_utils::all_reduce_tensor;
use crate::error::Result;
use crate::optimizer::optimizer_state_dtype;
use numr::autograd::GradStore;
use numr::dtype::DType;
use numr::ops::TensorOps;
use numr::runtime::{Communicator, ReduceOp, Runtime, RuntimeClient};
use numr::tensor::{Tensor, TensorId};

/// Sum the gradients of `params` across ranks, in slice order, through flat f32 buffers.
///
/// Calls [`all_reduce_flat_sum_bucketed`] with [`DEFAULT_ORDERED_BUCKET_BYTES`].
///
/// # Errors
///
/// See [`all_reduce_flat_sum_bucketed`].
pub fn all_reduce_flat_sum<R, C>(
    comm: &dyn Communicator,
    client: &C,
    grads: &mut GradStore<R>,
    params: &[(TensorId, &[usize])],
    extras: &mut [f64],
) -> Result<()>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R>,
{
    all_reduce_flat_sum_bucketed(
        comm,
        client,
        grads,
        params,
        extras,
        DEFAULT_ORDERED_BUCKET_BYTES,
    )
}

/// Sum the gradients of `params` across ranks, in slice order, through flat f32 buffers.
///
/// Each entry of `params` is a parameter id and the parameter's shape.
///
/// # Order contract
///
/// Every rank must pass the same parameters, with the same shapes, in the same order.
/// A `TensorId` is local to its process, so ids differ across ranks.
/// Pass the parameters in model order, such as the order the model lists them.
/// Never sort by raw id: the sorted orders differ across ranks.
///
/// # Algorithm
///
/// 1. Each present gradient widens to f32, the optimizer state dtype, and flattens.
/// 2. A missing gradient contributes zeros of the parameter's element count.
/// 3. The buffer tail holds one presence flag per parameter, 1.0 or 0.0, then `extras`.
/// 4. Segments pack into new buffers of at most `bucket_bytes`, in slice order.
///    A segment larger than `bucket_bytes` gets a buffer of its own.
/// 5. Each buffer is all-reduced with `Sum`, then the communicator syncs.
///
/// The function never reduces over the caller's gradient storage.
/// The sum is never divided by the world size. The caller's loss weighting must include it.
///
/// # Result
///
/// - A parameter whose summed presence is above 0 gets the summed f32 gradient, with its shape.
///   This holds on every rank, including ranks that had no gradient for it.
/// - A parameter no rank produced a gradient for stays without one. An unrouted MoE expert is one case.
/// - Each `extras` scalar becomes its sum across ranks. Extras travel as f32.
/// - Gradients of ids outside `params` stay unchanged and are not summed.
/// - The summed gradients are views into the reduced buffers.
///
/// # Device traffic
///
/// All gradient data stays on the client's device. The presence and extras tail is built
/// on the host with `from_slice`. The host reads back that one small tail after the sum.
/// The read happens only when this rank lacks a gradient or `extras` is not empty.
///
/// # World size 1
///
/// With one rank the function returns at once. Gradients keep their storage, dtype and bits.
/// `extras` stays unchanged.
///
/// # Errors
///
/// Returns `DistributedError` before any collective when:
/// - an id repeats in `params`
/// - a gradient's shape differs from its declared shape
/// - a gradient's dtype has a non-f32 optimizer state dtype, such as F64 or an integer
///
/// Returns an error when a device op or a collective fails.
/// An error on one rank leaves its peers waiting in the collective. The caller must abort the group.
pub fn all_reduce_flat_sum_bucketed<R, C>(
    comm: &dyn Communicator,
    client: &C,
    grads: &mut GradStore<R>,
    params: &[(TensorId, &[usize])],
    extras: &mut [f64],
    bucket_bytes: usize,
) -> Result<()>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R>,
{
    if comm.world_size() <= 1 {
        return Ok(());
    }
    check_params(grads, params)?;

    let tail = tail_values(grads, params, extras);
    let mut segments: Vec<(usize, DType)> = params
        .iter()
        .map(|(_, shape)| (shape.iter().product(), DType::F32))
        .collect();
    segments.push((tail.len(), DType::F32));
    let plans = plan_buckets(&segments, bucket_bytes);

    let mut buffers = Vec::with_capacity(plans.len());
    for plan in &plans {
        buffers.push(build_bucket(client, grads, params, &tail, plan)?);
    }

    order_after_compute::<R, C>(comm, client)?;
    for buffer in &buffers {
        all_reduce_tensor(comm, buffer, ReduceOp::Sum)?;
    }
    sync_comm(comm, "all_reduce_flat_sum")?;

    write_back(grads, params, extras, &plans, &buffers)
}

/// Check ids, shapes and dtypes on the host before any collective.
fn check_params<R: Runtime<DType = DType>>(
    grads: &GradStore<R>,
    params: &[(TensorId, &[usize])],
) -> Result<()> {
    let mut seen = HashSet::with_capacity(params.len());
    for (index, (id, shape)) in params.iter().enumerate() {
        if !seen.insert(*id) {
            return Err(dist_err(format!(
                "all_reduce_flat_sum: parameter {index} repeats id {id:?}. Pass each parameter once"
            )));
        }
        let Some(grad) = grads.get(*id) else {
            continue;
        };
        if grad.shape() != *shape {
            return Err(dist_err(format!(
                "all_reduce_flat_sum: parameter {index} has gradient shape {:?}, declared {shape:?}. \
                 Declare the parameter's own shape",
                grad.shape()
            )));
        }
        let dtype = grad.dtype();
        if optimizer_state_dtype(dtype) != DType::F32 {
            return Err(dist_err(format!(
                "all_reduce_flat_sum: parameter {index} has gradient dtype {dtype:?}. \
                 The flat sum carries F32, F16, BF16 and FP8 gradients only"
            )));
        }
    }
    Ok(())
}

/// Presence flags in parameter order, then the extras, as f32.
fn tail_values<R: Runtime<DType = DType>>(
    grads: &GradStore<R>,
    params: &[(TensorId, &[usize])],
    extras: &[f64],
) -> Vec<f32> {
    let flags = params
        .iter()
        .map(|(id, _)| if grads.contains(*id) { 1.0 } else { 0.0 });
    flags.chain(extras.iter().map(|&x| x as f32)).collect()
}

/// Concatenate one bucket's segments into a new f32 buffer on the client's device.
fn build_bucket<R, C>(
    client: &C,
    grads: &GradStore<R>,
    params: &[(TensorId, &[usize])],
    tail: &[f32],
    plan: &BucketPlan,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R>,
{
    let device = client.device();
    let mut pieces = Vec::with_capacity(plan.segments.len());
    for index in plan.segments.clone() {
        let piece = match params.get(index) {
            Some((id, shape)) => match grads.get(*id) {
                Some(grad) => flat_f32(client, grad)?,
                None => Tensor::<R>::zeros(&[shape.iter().product()], DType::F32, device)?,
            },
            None => Tensor::<R>::from_slice(tail, &[tail.len()], device)?,
        };
        pieces.push(piece);
    }
    let refs: Vec<&Tensor<R>> = pieces.iter().collect();
    Ok(client.cat(&refs, 0)?)
}

/// Widen `grad` to f32 and view it as 1-D.
fn flat_f32<R, C>(client: &C, grad: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R>,
{
    let wide = client.cast(grad, DType::F32)?;
    let wide = if wide.is_contiguous() {
        wide
    } else {
        wide.contiguous()?
    };
    Ok(wide.flatten()?)
}

/// Split the reduced buffers back into gradients and write the reduced extras.
fn write_back<R: Runtime<DType = DType>>(
    grads: &mut GradStore<R>,
    params: &[(TensorId, &[usize])],
    extras: &mut [f64],
    plans: &[BucketPlan],
    buffers: &[Tensor<R>],
) -> Result<()> {
    let locally_missing = params.iter().any(|(id, _)| !grads.contains(*id));
    let tail = if locally_missing || !extras.is_empty() {
        Some(read_tail(buffers, params.len() + extras.len())?)
    } else {
        None
    };

    for (plan, buffer) in plans.iter().zip(buffers) {
        let mut offset = 0usize;
        for index in plan.segments.clone() {
            let Some((id, shape)) = params.get(index) else {
                break;
            };
            let numel: usize = shape.iter().product();
            let produced = tail
                .as_ref()
                .is_none_or(|flags| flags.get(index).is_some_and(|&flag| flag > 0.5));
            if produced {
                let grad = buffer.narrow(0, offset, numel)?.reshape(shape)?;
                grads.insert(*id, grad);
            } else {
                grads.remove(*id);
            }
            offset += numel;
        }
    }

    if let Some(tail) = &tail {
        for (slot, &sum) in extras.iter_mut().zip(tail.iter().skip(params.len())) {
            *slot = f64::from(sum);
        }
    }
    Ok(())
}

/// Read the presence and extras tail, the last segment of the last buffer, to the host.
fn read_tail<R: Runtime<DType = DType>>(
    buffers: &[Tensor<R>],
    tail_len: usize,
) -> Result<Vec<f32>> {
    let Some(last) = buffers.last() else {
        return Err(dist_err(
            "all_reduce_flat_sum: no buffer holds the presence tail".to_string(),
        ));
    };
    let start = last.numel().checked_sub(tail_len).ok_or_else(|| {
        dist_err(format!(
            "all_reduce_flat_sum: last buffer has {} elements, below the tail length {tail_len}",
            last.numel()
        ))
    })?;
    Ok(last.narrow(0, start, tail_len)?.try_to_vec::<f32>()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::LocalCommunicator;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    const SHAPES: [&[usize]; 3] = [&[2, 3], &[4], &[1, 2]];

    /// Per parameter: shape, dtype and f32 bits of the gradient, or `None` when absent.
    type Grads = Vec<Option<(Vec<usize>, DType, Vec<u32>)>>;

    fn value(rank: usize, param: usize, i: usize) -> f32 {
        (rank as f32 + 1.0) * 0.1 + param as f32 * 0.37 + i as f32 * 0.013
    }

    fn values(rank: usize, param: usize) -> Vec<f32> {
        let numel: usize = SHAPES[param].iter().product();
        (0..numel).map(|i| value(rank, param, i)).collect()
    }

    fn grad(rank: usize, param: usize, device: &CpuDevice) -> Tensor<CpuRuntime> {
        Tensor::<CpuRuntime>::from_slice(&values(rank, param), SHAPES[param], device).unwrap()
    }

    /// The left fold `v0 + v1 + ...` in rank order, the order `LocalCommunicator` sums in.
    fn fold(world: usize, param: usize, present: impl Fn(usize) -> bool) -> Vec<u32> {
        let numel: usize = SHAPES[param].iter().product();
        (0..numel)
            .map(|i| {
                let part = |r: usize| if present(r) { value(r, param, i) } else { 0.0 };
                (1..world).fold(part(0), |acc, r| acc + part(r)).to_bits()
            })
            .collect()
    }

    fn snapshot(grads: &GradStore<CpuRuntime>, ids: &[TensorId]) -> Grads {
        ids.iter()
            .map(|id| {
                grads.get(*id).map(|g| {
                    let bits = g.to_vec::<f32>().iter().map(|v| v.to_bits()).collect();
                    (g.shape().to_vec(), g.dtype(), bits)
                })
            })
            .collect()
    }

    /// Run one reduction per rank on its own thread, client, ids and store.
    ///
    /// `make` builds a rank's store and extras from the rank and its ids.
    fn reduce_on_ranks<F>(world: usize, bucket_bytes: usize, make: F) -> Vec<(Grads, Vec<f64>)>
    where
        F: Fn(usize, &[TensorId], &CpuDevice) -> (GradStore<CpuRuntime>, Vec<f64>) + Sync,
    {
        let comms = LocalCommunicator::group(world);
        std::thread::scope(|s| {
            let make = &make;
            let handles: Vec<_> = comms
                .iter()
                .map(|comm| {
                    s.spawn(move || {
                        let _guard = comm.abort_on_panic();
                        let device = CpuDevice::new();
                        let client = CpuClient::new(device.clone());
                        let ids: Vec<TensorId> = SHAPES.iter().map(|_| TensorId::new()).collect();
                        let params: Vec<(TensorId, &[usize])> =
                            ids.iter().copied().zip(SHAPES).collect();
                        let (mut grads, mut extras) = make(comm.rank(), &ids, &device);
                        all_reduce_flat_sum_bucketed(
                            comm,
                            &client,
                            &mut grads,
                            &params,
                            &mut extras,
                            bucket_bytes,
                        )
                        .unwrap();
                        (snapshot(&grads, &ids), extras)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        })
    }

    fn full_store(rank: usize, ids: &[TensorId], device: &CpuDevice) -> GradStore<CpuRuntime> {
        let mut grads = GradStore::new();
        for (param, id) in ids.iter().enumerate() {
            grads.insert(*id, grad(rank, param, device));
        }
        grads
    }

    fn expected_full(world: usize) -> Grads {
        (0..SHAPES.len())
            .map(|p| Some((SHAPES[p].to_vec(), DType::F32, fold(world, p, |_| true))))
            .collect()
    }

    #[test]
    fn test_sum_matches_rank_order_fold() {
        for world in [2, 3] {
            let ranks = reduce_on_ranks(world, DEFAULT_ORDERED_BUCKET_BYTES, |rank, ids, dev| {
                (full_store(rank, ids, dev), Vec::new())
            });
            for (grads, _) in &ranks {
                assert_eq!(grads, &expected_full(world), "world {world}");
            }
        }
    }

    #[test]
    fn test_missing_on_one_rank_is_summed_and_kept() {
        let ranks = reduce_on_ranks(2, DEFAULT_ORDERED_BUCKET_BYTES, |rank, ids, dev| {
            let mut grads = full_store(rank, ids, dev);
            if rank == 1 {
                grads.remove(ids[1]);
            }
            (grads, Vec::new())
        });
        let kept = Some((SHAPES[1].to_vec(), DType::F32, fold(2, 1, |r| r == 0)));
        for (grads, _) in &ranks {
            assert_eq!(grads[1], kept);
            assert_eq!(grads[0], expected_full(2)[0]);
        }
    }

    #[test]
    fn test_missing_on_all_ranks_is_removed() {
        let ranks = reduce_on_ranks(3, DEFAULT_ORDERED_BUCKET_BYTES, |rank, ids, dev| {
            let mut grads = full_store(rank, ids, dev);
            grads.remove(ids[2]);
            (grads, Vec::new())
        });
        for (grads, _) in &ranks {
            assert_eq!(grads[2], None);
            assert_eq!(grads[..2], expected_full(3)[..2]);
        }
    }

    #[test]
    fn test_extras_are_summed() {
        let ranks = reduce_on_ranks(3, DEFAULT_ORDERED_BUCKET_BYTES, |rank, ids, dev| {
            let extras = vec![rank as f64 + 0.5, 2.0 * rank as f64];
            (full_store(rank, ids, dev), extras)
        });
        for (_, extras) in &ranks {
            assert_eq!(extras, &vec![4.5, 6.0]);
        }
    }

    #[test]
    fn test_tiny_buckets_match_one_bucket() {
        let make = |rank: usize, ids: &[TensorId], dev: &CpuDevice| {
            let mut grads = full_store(rank, ids, dev);
            if rank == 0 {
                grads.remove(ids[0]);
            }
            (grads, vec![rank as f64])
        };
        let one = reduce_on_ranks(2, DEFAULT_ORDERED_BUCKET_BYTES, make);
        let many = reduce_on_ranks(2, 4, make);
        assert_eq!(one, many);
    }

    #[test]
    fn test_insertion_order_does_not_change_the_sum() {
        let ranks = reduce_on_ranks(2, DEFAULT_ORDERED_BUCKET_BYTES, |rank, ids, dev| {
            let mut grads = GradStore::new();
            let mut order: Vec<usize> = (0..ids.len()).collect();
            if rank == 1 {
                order.reverse();
            }
            for param in order {
                grads.insert(ids[param], grad(rank, param, dev));
            }
            (grads, Vec::new())
        });
        assert_eq!(ranks[0], ranks[1]);
        assert_eq!(ranks[0].0, expected_full(2));
    }

    #[test]
    fn test_caller_storage_is_never_reduced_in_place() {
        let comms = LocalCommunicator::group(2);
        std::thread::scope(|s| {
            for comm in &comms {
                s.spawn(move || {
                    let _guard = comm.abort_on_panic();
                    let device = CpuDevice::new();
                    let client = CpuClient::new(device.clone());
                    let id = TensorId::new();
                    let input = grad(comm.rank(), 1, &device);
                    let mut grads = GradStore::new();
                    grads.insert(id, input.clone());
                    let params = [(id, SHAPES[1])];
                    all_reduce_flat_sum(comm, &client, &mut grads, &params, &mut []).unwrap();
                    assert_eq!(input.to_vec::<f32>(), values(comm.rank(), 1));
                    let summed = grads.get(id).unwrap();
                    assert!(!summed.shares_storage_with(&input));
                });
            }
        });
    }

    #[test]
    fn test_world_one_is_identity() {
        let comms = LocalCommunicator::group(1);
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let ids: Vec<TensorId> = SHAPES.iter().map(|_| TensorId::new()).collect();
        let params: Vec<(TensorId, &[usize])> = ids.iter().copied().zip(SHAPES).collect();
        let mut grads = full_store(0, &ids, &device);
        grads.remove(ids[2]);
        let before: Vec<Tensor<CpuRuntime>> = ids[..2]
            .iter()
            .map(|id| grads.get(*id).unwrap().clone())
            .collect();
        let mut extras = [1.25, -3.0];
        all_reduce_flat_sum(&comms[0], &client, &mut grads, &params, &mut extras).unwrap();
        for (id, old) in ids.iter().zip(&before) {
            assert!(grads.get(*id).unwrap().shares_storage_with(old));
        }
        assert!(!grads.contains(ids[2]));
        assert_eq!(extras, [1.25, -3.0]);
    }

    #[test]
    fn test_shape_mismatch_is_an_error_before_any_collective() {
        let comms = LocalCommunicator::group(2);
        std::thread::scope(|s| {
            for comm in &comms {
                s.spawn(move || {
                    let _guard = comm.abort_on_panic();
                    let device = CpuDevice::new();
                    let client = CpuClient::new(device.clone());
                    let id = TensorId::new();
                    let mut grads = GradStore::new();
                    grads.insert(id, grad(comm.rank(), 1, &device));
                    let params = [(id, &[2usize, 2][..])];
                    let result = all_reduce_flat_sum(comm, &client, &mut grads, &params, &mut []);
                    assert!(result.is_err());
                    assert!(!comm.is_aborted());
                });
            }
        });
    }

    #[test]
    fn test_repeated_id_is_an_error() {
        let comms = LocalCommunicator::group(2);
        std::thread::scope(|s| {
            for comm in &comms {
                s.spawn(move || {
                    let _guard = comm.abort_on_panic();
                    let device = CpuDevice::new();
                    let client = CpuClient::new(device.clone());
                    let id = TensorId::new();
                    let mut grads = GradStore::new();
                    grads.insert(id, grad(comm.rank(), 1, &device));
                    let params = [(id, SHAPES[1]), (id, SHAPES[1])];
                    let result = all_reduce_flat_sum(comm, &client, &mut grads, &params, &mut []);
                    assert!(result.is_err());
                });
            }
        });
    }

    #[cfg(feature = "f16")]
    #[test]
    fn test_bf16_gradients_widen_to_f32() {
        use half::bf16;
        let ranks = reduce_on_ranks(2, DEFAULT_ORDERED_BUCKET_BYTES, |rank, ids, dev| {
            let mut grads = GradStore::new();
            for (param, id) in ids.iter().enumerate() {
                let narrow: Vec<bf16> = values(rank, param)
                    .into_iter()
                    .map(bf16::from_f32)
                    .collect();
                let t = Tensor::<CpuRuntime>::from_slice(&narrow, SHAPES[param], dev).unwrap();
                grads.insert(*id, t);
            }
            (grads, Vec::new())
        });
        let widened = |r: usize, p: usize, i: usize| bf16::from_f32(value(r, p, i)).to_f32();
        for (grads, _) in &ranks {
            for (param, got) in grads.iter().enumerate() {
                let numel: usize = SHAPES[param].iter().product();
                let bits = (0..numel)
                    .map(|i| (widened(0, param, i) + widened(1, param, i)).to_bits())
                    .collect();
                assert_eq!(got, &Some((SHAPES[param].to_vec(), DType::F32, bits)));
            }
        }
    }
}
