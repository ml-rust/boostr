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

#[test]
fn test_flatten_unflatten_roundtrip() {
    let (client, device) = cpu_setup();
    let comm = Arc::new(NoOpCommunicator);

    let id1 = TensorId::new();
    let id2 = TensorId::new();

    let params = vec![(id1, 3, DType::F32), (id2, 2, DType::F32)];
    let mut mgr = GradientBucketManager::<CpuRuntime>::new(&params, comm, 25 * 1024 * 1024, None);

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
    let mut mgr = GradientBucketManager::<CpuRuntime>::new(&params, comm, 25 * 1024 * 1024, None);

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
    let mut mgr = GradientBucketManager::<CpuRuntime>::new(&params, comm, 25 * 1024 * 1024, None);

    // 2x3 gradient
    let g1 = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);

    mgr.mark_grad_ready(id1, &g1, &client).unwrap();

    let mut grads = GradStore::new();
    mgr.wait_and_unflatten(&client, &mut grads).unwrap();

    let result = grads.get(id1).expect("grad for id1 should exist");
    assert_eq!(result.shape(), &[2, 3]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}
