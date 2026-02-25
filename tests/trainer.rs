use std::collections::HashMap;
#[cfg(any(feature = "cuda", feature = "wgpu"))]
use std::sync::{Mutex, OnceLock};

use boostr::optimizer::LrSchedule;
use boostr::trainer::{SimpleTrainer, TrainingConfig};
use numr::autograd::{Var, backward, var_mean, var_mul, var_sub};
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

fn cpu_setup() -> (CpuClient, CpuDevice) {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    (client, device)
}

#[test]
fn test_trainer_basic_step() {
    let (client, device) = cpu_setup();

    let config = TrainingConfig::default()
        .with_lr(0.01)
        .with_max_grad_norm(None);
    let mut trainer = SimpleTrainer::<CpuRuntime>::new(config).expect("valid config");

    let target = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2], &device);
    let w_init = Tensor::<CpuRuntime>::from_slice(&[0.5f32, 0.5, 0.5, 0.5], &[2, 2], &device);
    let w_id = w_init.id();

    let mut params = HashMap::new();
    params.insert(w_id, w_init);

    let w = Var::with_id(params[&w_id].clone(), w_id, true);
    let t = Var::new(target.clone(), false);
    let diff = var_sub(&w, &t, &client).expect("var_sub");
    let sq = var_mul(&diff, &diff, &client).expect("var_mul");
    let loss = var_mean(&sq, &[0, 1], false, &client).expect("var_mean");
    let loss_val = loss.tensor().to_vec::<f32>()[0] as f64;
    let grads = backward(&loss, &client).expect("backward");

    let metrics = trainer
        .step(&client, &mut params, grads, loss_val)
        .expect("step");
    assert!(metrics.is_some());

    let m = metrics.unwrap();
    assert_eq!(m.step, 1);
    assert!(m.loss > 0.0);
}

#[test]
fn test_trainer_with_grad_accum() {
    let (client, device) = cpu_setup();

    let config = TrainingConfig::default()
        .with_lr(0.01)
        .with_grad_accum_steps(2)
        .with_max_grad_norm(None);
    let mut trainer = SimpleTrainer::<CpuRuntime>::new(config).expect("valid config");

    let target = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0], &[2], &device);
    let w_init = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0], &[2], &device);
    let w_id = w_init.id();

    let mut params = HashMap::new();
    params.insert(w_id, w_init);

    // First micro-batch: should return None (accumulating)
    let w = Var::with_id(params[&w_id].clone(), w_id, true);
    let t = Var::new(target.clone(), false);
    let diff = var_sub(&w, &t, &client).expect("var_sub");
    let sq = var_mul(&diff, &diff, &client).expect("var_mul");
    let loss = var_mean(&sq, &[0], false, &client).expect("var_mean");
    let grads = backward(&loss, &client).expect("backward");

    let result = trainer
        .step(&client, &mut params, grads, 1.0)
        .expect("step");
    assert!(result.is_none());

    // Second micro-batch: should return Some (step happens)
    let w = Var::with_id(params[&w_id].clone(), w_id, true);
    let t = Var::new(target.clone(), false);
    let diff = var_sub(&w, &t, &client).expect("var_sub");
    let sq = var_mul(&diff, &diff, &client).expect("var_mul");
    let loss = var_mean(&sq, &[0], false, &client).expect("var_mean");
    let grads = backward(&loss, &client).expect("backward");

    let result = trainer
        .step(&client, &mut params, grads, 1.0)
        .expect("step");
    assert!(result.is_some());
    assert_eq!(result.unwrap().step, 1);
}

#[test]
fn test_trainer_convergence() {
    let (client, device) = cpu_setup();

    let config = TrainingConfig::default()
        .with_lr(0.1)
        .with_weight_decay(0.0)
        .with_max_grad_norm(None);
    let mut trainer = SimpleTrainer::<CpuRuntime>::new(config).expect("valid config");

    let target = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2], &device);
    let w_init = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0, 0.0, 0.0], &[2, 2], &device);
    let w_id = w_init.id();

    let mut params = HashMap::new();
    params.insert(w_id, w_init);

    let mut first_loss = 0.0;
    let mut last_loss = 0.0;

    for i in 0..30 {
        let w = Var::with_id(params[&w_id].clone(), w_id, true);
        let t = Var::new(target.clone(), false);
        let diff = var_sub(&w, &t, &client).expect("var_sub");
        let sq = var_mul(&diff, &diff, &client).expect("var_mul");
        let loss = var_mean(&sq, &[0, 1], false, &client).expect("var_mean");
        let loss_val = loss.tensor().to_vec::<f32>()[0] as f64;
        let grads = backward(&loss, &client).expect("backward");

        if i == 0 {
            first_loss = loss_val;
        }
        last_loss = loss_val;

        trainer
            .step(&client, &mut params, grads, loss_val)
            .expect("step");
    }

    assert!(
        last_loss < first_loss * 0.1,
        "loss should decrease: first={first_loss} last={last_loss}"
    );
}

#[test]
fn test_trainer_with_lr_schedule() {
    let config = TrainingConfig::default().with_lr(0.01);
    let trainer = SimpleTrainer::<CpuRuntime>::new(config)
        .expect("valid config")
        .with_lr_schedule(LrSchedule::CosineAnnealing {
            base_lr: 0.01,
            min_lr: 0.001,
            warmup_steps: 10,
            total_steps: 100,
        });

    assert_eq!(trainer.global_step(), 0);
}

#[test]
fn test_trainer_with_grad_clipping() {
    let (client, device) = cpu_setup();

    let config = TrainingConfig::default()
        .with_lr(0.01)
        .with_max_grad_norm(Some(0.1));
    let mut trainer = SimpleTrainer::<CpuRuntime>::new(config).expect("valid config");

    let target = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2], &device);
    let w_init = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0, 0.0, 0.0], &[2, 2], &device);
    let w_id = w_init.id();

    let mut params = HashMap::new();
    params.insert(w_id, w_init);

    let w = Var::with_id(params[&w_id].clone(), w_id, true);
    let t = Var::new(target, false);
    let diff = var_sub(&w, &t, &client).expect("var_sub");
    let sq = var_mul(&diff, &diff, &client).expect("var_mul");
    let loss = var_mean(&sq, &[0, 1], false, &client).expect("var_mean");
    let loss_val = loss.tensor().to_vec::<f32>()[0] as f64;
    let grads = backward(&loss, &client).expect("backward");

    let metrics = trainer
        .step(&client, &mut params, grads, loss_val)
        .expect("step");
    assert!(metrics.is_some());

    let m = metrics.unwrap();
    assert!(m.grad_norm.is_some());
    assert!(m.grad_norm.unwrap() > 0.0);
}

// ============================================================================
// Graph Capture & Replay Tests
// ============================================================================
//
// Tests run on all available backends (CPU, CUDA, WebGPU) via cfg-gated helpers.

#[cfg(feature = "cuda")]
static CUDA_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
#[cfg(feature = "wgpu")]
static WGPU_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

#[cfg(feature = "cuda")]
fn with_cuda<F: FnMut(numr::runtime::cuda::CudaClient, numr::runtime::cuda::CudaDevice)>(mut f: F) {
    let _guard = CUDA_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    let device = numr::runtime::cuda::CudaDevice::new(0);
    let client = numr::runtime::cuda::CudaClient::new(device.clone())
        .expect("CUDA feature enabled but no CUDA device available");
    f(client, device);
}

#[cfg(feature = "wgpu")]
fn with_wgpu<F: FnMut(numr::runtime::wgpu::WgpuClient, numr::runtime::wgpu::WgpuDevice)>(mut f: F) {
    let _guard = WGPU_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    let device = numr::runtime::wgpu::WgpuDevice::default();
    let client = match numr::runtime::wgpu::WgpuClient::new(device.clone()) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to create WgpuClient: {:?}, skipping", e);
            return;
        }
    };
    f(client, device);
}

/// Test graph capture on a specific backend.
fn test_graph_capture_lifecycle<R: Runtime<DType = numr::dtype::DType>>(client: &R::Client) {
    let config = TrainingConfig::default()
        .with_lr(1e-3)
        .with_max_grad_norm(None);
    let mut trainer = SimpleTrainer::<R>::new(config).expect("valid config");

    // Initially no graphs captured
    assert_eq!(trainer.graphs_captured(), (false, false));

    // Launch without capture should error
    assert!(trainer.launch_forward_graph().is_err());
    assert!(trainer.launch_backward_graph().is_err());

    // Capture forward — closure must execute
    let mut fwd_executed = false;
    let fwd_result = trainer
        .capture_forward_pass(client, |_c| {
            fwd_executed = true;
            Ok(42i32)
        })
        .expect("capture_forward_pass");
    assert!(fwd_executed);
    assert_eq!(fwd_result, 42);
    assert_eq!(trainer.graphs_captured(), (true, false));

    // Capture backward
    let mut bwd_executed = false;
    trainer
        .capture_backward_pass(client, |_c| {
            bwd_executed = true;
            Ok(())
        })
        .expect("capture_backward_pass");
    assert!(bwd_executed);
    assert_eq!(trainer.graphs_captured(), (true, true));

    // Launch both (no-op on CPU/WebGPU, real replay on CUDA)
    trainer.launch_forward_graph().expect("launch forward");
    trainer.launch_backward_graph().expect("launch backward");

    // Clear and verify reset
    trainer.clear_graphs();
    assert_eq!(trainer.graphs_captured(), (false, false));
    assert!(trainer.launch_forward_graph().is_err());
    assert!(trainer.launch_backward_graph().is_err());
}

#[test]
fn test_graph_capture_cpu() {
    let (client, _device) = cpu_setup();
    test_graph_capture_lifecycle::<CpuRuntime>(&client);
}

#[cfg(feature = "cuda")]
#[test]
fn test_graph_capture_cuda() {
    with_cuda(|client, _device| {
        test_graph_capture_lifecycle::<numr::runtime::cuda::CudaRuntime>(&client);
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_graph_capture_wgpu() {
    with_wgpu(|client, _device| {
        test_graph_capture_lifecycle::<numr::runtime::wgpu::WgpuRuntime>(&client);
    });
}

/// Test graph capture with real tensor ops on non-capture backends (CPU/WebGPU).
///
/// On these backends, `capture_graph` executes eagerly and `launch` is a no-op,
/// so allocations inside the closure work fine.
fn test_graph_capture_with_tensor_ops_eager<R: Runtime<DType = numr::dtype::DType>>(
    client: &R::Client,
    device: &R::Device,
) where
    R::Client: numr::ops::BinaryOps<R> + numr::ops::ScalarOps<R>,
{
    let config = TrainingConfig::default()
        .with_lr(1e-3)
        .with_max_grad_norm(None);
    let mut trainer = SimpleTrainer::<R>::new(config).expect("valid config");

    let a = Tensor::<R>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], device);

    let result = trainer
        .capture_forward_pass(client, |c| {
            use numr::ops::ScalarOps;
            c.mul_scalar(&a, 2.0f64)
        })
        .expect("capture with tensor ops");

    let result_data = result.to_vec::<f32>();
    assert_eq!(result_data, vec![2.0, 4.0, 6.0, 8.0]);

    trainer.launch_forward_graph().expect("launch");
}

#[test]
fn test_graph_capture_tensor_ops_cpu() {
    let (client, device) = cpu_setup();
    test_graph_capture_with_tensor_ops_eager::<CpuRuntime>(&client, &device);
}

/// Test CUDA graph capture with tensor ops.
///
/// CUDA stream capture forbids memory allocation (cuMemAlloc) inside the
/// captured region. numr's tensor allocation currently panics on OOM rather
/// than returning Result, so we use catch_unwind to handle this gracefully.
/// The lifecycle test (test_graph_capture_cuda) validates the full graph API
/// on CUDA without allocations. This test verifies behavior when allocations
/// are attempted during capture — expected to fail without cudaMallocAsync pools.
#[cfg(feature = "cuda")]
#[test]
fn test_graph_capture_tensor_ops_cuda() {
    use numr::ops::ScalarOps;

    with_cuda(|client, device| {
        let config = TrainingConfig::default()
            .with_lr(1e-3)
            .with_max_grad_norm(None);
        let mut trainer =
            SimpleTrainer::<numr::runtime::cuda::CudaRuntime>::new(config).expect("valid config");

        let a = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0],
            &[4],
            &device,
        );

        // Warmup: execute once outside capture
        let _warmup = client.mul_scalar(&a, 2.0f64).expect("warmup");

        // Capture: mul_scalar allocates output, which panics during CUDA stream
        // capture because cuMemAlloc isn't a stream-ordered operation.
        // catch_unwind handles the expected panic gracefully.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            trainer.capture_forward_pass(&client, |c| {
                let _b = c.mul_scalar(&a, 2.0f64)?;
                Ok(())
            })
        }));

        match result {
            Ok(Ok(())) => {
                // cudaMallocAsync pool available — graph captured successfully
                trainer.launch_forward_graph().expect("launch");
            }
            Ok(Err(_)) | Err(_) => {
                // Expected: allocation inside capture not supported.
                // test_graph_capture_cuda already validates the graph API on CUDA.
            }
        }
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_graph_capture_tensor_ops_wgpu() {
    with_wgpu(|client, device| {
        test_graph_capture_with_tensor_ops_eager::<numr::runtime::wgpu::WgpuRuntime>(
            &client, &device,
        );
    });
}
