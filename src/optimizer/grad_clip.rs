//! Gradient clipping utilities
//!
//! Clip gradients by global norm to prevent exploding gradients during training.

use crate::error::{Error, Result};
use numr::autograd::GradStore;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ReduceOps, ScalarOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::TensorId;

/// Clip gradients in-place by global L2 norm.
///
/// Computes the global norm across all gradients, then scales each gradient
/// so the global norm does not exceed `max_norm`.
///
/// Returns the original global norm (before clipping).
pub fn clip_grad_norm<R, C>(client: &C, grads: &mut GradStore<R>, max_norm: f64) -> Result<f64>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + ReduceOps<R> + ScalarOps<R> + UnaryOps<R> + BinaryOps<R>,
{
    if max_norm <= 0.0 {
        return Err(Error::TrainingError {
            reason: format!("max_norm must be positive, got {max_norm}"),
        });
    }

    // Compute global norm: sqrt(sum of all grad element squares)
    let ids: Vec<TensorId> = grads.keys().copied().collect();

    let mut total_norm_sq = 0.0f64;
    for &id in &ids {
        if let Some(grad) = grads.get(id) {
            // Flatten then sum all elements: sum(grad^2)
            let flat = grad.reshape(&[grad.numel()])?;
            let sq = client.mul(&flat, &flat)?;
            let sum = client.sum(&sq, &[0], false)?;
            let val = sum.to_vec::<f32>();
            total_norm_sq += val[0] as f64;
        }
    }

    let total_norm = total_norm_sq.sqrt();

    if total_norm > max_norm {
        let scale = max_norm / (total_norm + 1e-6);
        for id in ids {
            if let Some(grad) = grads.get(id) {
                let clipped = client.mul_scalar(grad, scale)?;
                grads.insert(id, clipped);
            }
        }
    }

    Ok(total_norm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::autograd::GradStore;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

    #[test]
    fn test_clip_no_op_when_under_max() {
        let (client, device) = cpu_setup();

        let id = TensorId::new();
        // grad = [1, 0] → norm = 1.0
        let t = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0], &[2], &device);
        let mut grads = GradStore::new();
        grads.insert(id, t);

        let norm = clip_grad_norm(&client, &mut grads, 5.0).unwrap();
        assert!((norm - 1.0).abs() < 1e-6);

        // Grads should be unchanged
        let data = grads.get(id).unwrap().to_vec::<f32>();
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_clip_scales_when_over_max() {
        let (client, device) = cpu_setup();

        let id = TensorId::new();
        // grad = [3, 4] → norm = 5.0, clip to max_norm=1.0
        let t = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 4.0], &[2], &device);
        let mut grads = GradStore::new();
        grads.insert(id, t);

        let norm = clip_grad_norm(&client, &mut grads, 1.0).unwrap();
        assert!((norm - 5.0).abs() < 1e-4);

        // After clipping, norm should be ~1.0
        let data = grads.get(id).unwrap().to_vec::<f32>();
        let clipped_norm = (data[0] * data[0] + data[1] * data[1]).sqrt();
        assert!((clipped_norm - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_clip_multi_param_global_norm() {
        let (client, device) = cpu_setup();

        let id1 = TensorId::new();
        let id2 = TensorId::new();
        // grad1 = [3, 0], grad2 = [0, 4] → global norm = sqrt(9+16) = 5.0
        let t1 = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 0.0], &[2], &device);
        let t2 = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 4.0], &[2], &device);
        let mut grads = GradStore::new();
        grads.insert(id1, t1);
        grads.insert(id2, t2);

        let norm = clip_grad_norm(&client, &mut grads, 2.5).unwrap();
        assert!((norm - 5.0).abs() < 1e-4);

        // Both grads should be scaled by 2.5/5.0 = 0.5
        let d1 = grads.get(id1).unwrap().to_vec::<f32>();
        let d2 = grads.get(id2).unwrap().to_vec::<f32>();
        assert!((d1[0] - 1.5).abs() < 1e-4);
        assert!((d2[1] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_clip_empty_grads() {
        let (client, _device) = cpu_setup();
        let mut grads = GradStore::<CpuRuntime>::new();

        let norm = clip_grad_norm(&client, &mut grads, 1.0).unwrap();
        assert!((norm - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_clip_rejects_non_positive_max_norm() {
        let (client, _device) = cpu_setup();
        let mut grads = GradStore::<CpuRuntime>::new();

        assert!(clip_grad_norm(&client, &mut grads, 0.0).is_err());
        assert!(clip_grad_norm(&client, &mut grads, -1.0).is_err());
    }
}
