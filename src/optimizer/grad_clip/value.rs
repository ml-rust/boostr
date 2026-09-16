//! Element-wise gradient clamping.

use super::norm::unique_ids;
use crate::error::{Error, Result};
use numr::autograd::GradStore;
use numr::dtype::DType;
use numr::ops::UtilityOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::TensorId;

/// Clamp every parameter gradient element to `[-clip_value, clip_value]`.
///
/// Unlike norm-based clipping, this operates element-wise and does not
/// preserve gradient direction. Like the norm clips, it acts on `param_ids`
/// only — clamping activation gradients that no optimizer reads is wasted work
/// on a graph two orders of magnitude larger than the parameter set.
pub fn clip_grad_value<R, C>(
    client: &C,
    grads: &mut GradStore<R>,
    param_ids: &[TensorId],
    clip_value: f64,
) -> Result<()>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + UtilityOps<R>,
{
    if clip_value <= 0.0 {
        return Err(Error::TrainingError {
            reason: format!("clip_value must be positive, got {clip_value}"),
        });
    }

    let ids = unique_ids(param_ids);

    for id in ids {
        let grad = match grads.get(id) {
            Some(g) => g,
            None => continue,
        };
        let clamped = client.clamp(grad, -clip_value, clip_value)?;
        grads.insert(id, clamped);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::grad_clip::test_support::{GraphStore, graph_store};
    use crate::test_utils::cpu_setup;
    use numr::autograd::GradStore;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

    #[test]
    fn test_clip_value() {
        let (client, device) = cpu_setup();

        let id = TensorId::new();
        let t =
            Tensor::<CpuRuntime>::from_slice(&[-5.0f32, 3.0, 0.5, -0.1], &[4], &device).unwrap();
        let mut grads = GradStore::new();
        grads.insert(id, t);

        clip_grad_value(&client, &mut grads, &[id], 1.0).unwrap();

        let data = grads.get(id).unwrap().to_vec::<f32>();
        assert!((data[0] - (-1.0)).abs() < 1e-6); // clamped from -5
        assert!((data[1] - 1.0).abs() < 1e-6); // clamped from 3
        assert!((data[2] - 0.5).abs() < 1e-6); // unchanged
        assert!((data[3] - (-0.1)).abs() < 1e-6); // unchanged
    }

    /// `clip_grad_value` carries the same contract as the norm clips: a
    /// non-parameter gradient is never touched.
    #[test]
    fn test_clip_value_ignores_non_parameter_gradients() {
        let GraphStore {
            client,
            mut grads,
            params,
            activation,
            ..
        } = graph_store();

        clip_grad_value(&client, &mut grads, &params, 1.0).unwrap();

        let p2 = grads.get(params[1]).unwrap().to_vec::<f32>();
        assert_eq!(p2, vec![0.0f32, 1.0], "the parameter was not clamped");

        let act_grad = grads.get(activation).unwrap().to_vec::<f32>();
        assert_eq!(
            act_grad,
            vec![0.0f32, 0.0, 5.0, 5.0],
            "an activation gradient was clamped"
        );
    }

    #[test]
    fn test_clip_value_rejects_non_positive() {
        let (client, _device) = cpu_setup();
        let mut grads = GradStore::<CpuRuntime>::new();

        assert!(clip_grad_value(&client, &mut grads, &[], 0.0).is_err());
        assert!(clip_grad_value(&client, &mut grads, &[], -1.0).is_err());
    }
}
