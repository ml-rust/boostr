//! Global- and per-parameter L2 norm gradient clipping.

use crate::error::{Error, Result};
use crate::readback::scalar_f32;
use numr::autograd::GradStore;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ReduceOps, ScalarOps, TypeConversionOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::TensorId;
use std::collections::HashSet;

/// Deduplicate a caller's parameter list, preserving order.
///
/// A repeated id would be squared twice into the norm and, worse, scaled
/// twice by `mul_scalar` in the clip loop. Callers build their lists from
/// maps and are unique in practice; this makes that a guarantee rather than
/// an assumption.
pub(super) fn unique_ids(param_ids: &[TensorId]) -> Vec<TensorId> {
    let mut seen = HashSet::with_capacity(param_ids.len());
    param_ids
        .iter()
        .copied()
        .filter(|id| seen.insert(*id))
        .collect()
}

/// Clip parameter gradients in-place by global L2 norm.
///
/// Computes the global norm across the gradients of `param_ids`, then scales
/// each of those gradients so the global norm does not exceed `max_norm`.
///
/// Returns the original global norm (before clipping).
///
/// # Why `param_ids` and not the whole store
///
/// `GradStore` holds a gradient for EVERY node in the autograd graph, not just
/// the parameters: the loss node's own `dL/dL = 1` seed, every activation, and
/// every frozen tensor that `MatmulBackward` returns a gradient for regardless
/// of `requires_grad`. On a LoRA run of Llama-3.2-1B that is 11975 entries
/// against 128 optimized parameters, dominated by activations and by the frozen
/// `[128256, 2048]` tied embedding.
///
/// Norming over the store therefore both misreports the gradient norm and
/// misapplies the clip, rescaling the parameters that ARE optimized by a factor
/// derived from tensors that are not. This matches PyTorch's
/// `clip_grad_norm_(model.parameters())`: the norm is over the optimized
/// parameters, and only those are scaled.
///
/// A `param_id` with no gradient in the store is skipped, as PyTorch skips a
/// parameter whose `.grad` is `None`.
///
/// The per-gradient sum of squares carries the gradient's own dtype, so it is
/// read back through [`scalar_f32`] rather than `item::<f32>`. Under BF16 or
/// F16 a direct f32 read over-runs the buffer and the norm comes back NaN.
pub fn clip_grad_norm<R, C>(
    client: &C,
    grads: &mut GradStore<R>,
    param_ids: &[TensorId],
    max_norm: f64,
) -> Result<f64>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + UnaryOps<R>
        + BinaryOps<R>
        + TypeConversionOps<R>,
{
    if max_norm <= 0.0 {
        return Err(Error::TrainingError {
            reason: format!("max_norm must be positive, got {max_norm}"),
        });
    }

    // Compute global norm: sqrt(sum of the PARAMETER grad element squares)
    let ids = unique_ids(param_ids);

    let mut total_norm_sq = 0.0f64;
    for &id in &ids {
        if let Some(grad) = grads.get(id) {
            // Flatten then sum all elements: sum(grad^2).
            // Gradients can arrive strided (e.g. through a transpose or a
            // broadcast reduction), and `reshape` requires a contiguous layout.
            let flat = grad.contiguous()?.reshape(&[grad.numel()])?;
            let sq = client.mul(&flat, &flat)?;
            let sum = client.sum(&sq, &[0], false)?;
            total_norm_sq += scalar_f32(client, &sum)? as f64;
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

/// Clip each parameter gradient independently by its own L2 norm.
///
/// For each gradient of `param_ids`, if its L2 norm exceeds `max_norm`, it is
/// scaled down so its norm equals `max_norm`. Other gradients are left
/// unchanged, and non-parameter entries in the store are never touched — see
/// [`clip_grad_norm`] for why the store is not the parameter set.
///
/// Returns a map of tensor ID → original norm for every gradient that was clipped.
///
/// Each per-gradient sum of squares carries the gradient's own dtype and is read
/// back through [`scalar_f32`], for the same reason as [`clip_grad_norm`].
pub fn clip_grad_norm_per_param<R, C>(
    client: &C,
    grads: &mut GradStore<R>,
    param_ids: &[TensorId],
    max_norm: f64,
) -> Result<Vec<(TensorId, f64)>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + UnaryOps<R>
        + BinaryOps<R>
        + TypeConversionOps<R>,
{
    if max_norm <= 0.0 {
        return Err(Error::TrainingError {
            reason: format!("max_norm must be positive, got {max_norm}"),
        });
    }

    let ids = unique_ids(param_ids);
    let mut clipped = Vec::new();

    for id in ids {
        let grad = match grads.get(id) {
            Some(g) => g,
            None => continue,
        };

        // Gradients can arrive strided; `reshape` requires a contiguous layout.
        let flat = grad.contiguous()?.reshape(&[grad.numel()])?;
        let sq = client.mul(&flat, &flat)?;
        let sum = client.sum(&sq, &[0], false)?;
        let norm_sq: f64 = scalar_f32(client, &sum)? as f64;
        let norm = norm_sq.sqrt();

        if norm > max_norm {
            let scale = max_norm / (norm + 1e-6);
            let scaled = client.mul_scalar(grad, scale)?;
            grads.insert(id, scaled);
            clipped.push((id, norm));
        }
    }

    Ok(clipped)
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
        let t = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0], &[2], &device).unwrap();
        let mut grads = GradStore::new();
        grads.insert(id, t);

        let norm = clip_grad_norm(&client, &mut grads, &[id], 5.0).unwrap();
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
        let t = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 4.0], &[2], &device).unwrap();
        let mut grads = GradStore::new();
        grads.insert(id, t);

        let norm = clip_grad_norm(&client, &mut grads, &[id], 1.0).unwrap();
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
        let t1 = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 0.0], &[2], &device).unwrap();
        let t2 = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 4.0], &[2], &device).unwrap();
        let mut grads = GradStore::new();
        grads.insert(id1, t1);
        grads.insert(id2, t2);

        let norm = clip_grad_norm(&client, &mut grads, &[id1, id2], 2.5).unwrap();
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

        let norm = clip_grad_norm(&client, &mut grads, &[], 1.0).unwrap();
        assert!((norm - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_clip_rejects_non_positive_max_norm() {
        let (client, _device) = cpu_setup();
        let mut grads = GradStore::<CpuRuntime>::new();

        assert!(clip_grad_norm(&client, &mut grads, &[], 0.0).is_err());
        assert!(clip_grad_norm(&client, &mut grads, &[], -1.0).is_err());
    }

    #[test]
    fn test_clip_per_param_only_clips_large() {
        let (client, device) = cpu_setup();

        let id1 = TensorId::new();
        let id2 = TensorId::new();
        // grad1 = [3, 4] → norm = 5.0, should be clipped to norm 2.0
        // grad2 = [1, 0] → norm = 1.0, should NOT be clipped
        let t1 = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 4.0], &[2], &device).unwrap();
        let t2 = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0], &[2], &device).unwrap();
        let mut grads = GradStore::new();
        grads.insert(id1, t1);
        grads.insert(id2, t2);

        let clipped = clip_grad_norm_per_param(&client, &mut grads, &[id1, id2], 2.0).unwrap();

        // Only id1 should have been clipped
        assert_eq!(clipped.len(), 1);
        assert!((clipped[0].1 - 5.0).abs() < 1e-4);

        // id1: norm should now be ~2.0
        let d1 = grads.get(id1).unwrap().to_vec::<f32>();
        let norm1 = (d1[0] * d1[0] + d1[1] * d1[1]).sqrt();
        assert!((norm1 - 2.0).abs() < 1e-3);

        // id2: unchanged
        let d2 = grads.get(id2).unwrap().to_vec::<f32>();
        assert!((d2[0] - 1.0).abs() < 1e-6);
    }

    /// The four values below are exactly representable in F32, BF16 and F16,
    /// and so is their sum of squares (15.5), so every dtype must return the
    /// same norm. Pinning the value is the point: a raw f32 read of a BF16
    /// buffer picks up two uninitialized bytes as the sign and exponent, which
    /// yields NaN, a denormal near zero, or a huge finite number depending on
    /// what the allocator handed back. `is_finite()` would let two of those
    /// three through.
    const NORM_FIXTURE: [f32; 4] = [0.5, -1.5, 2.0, 3.0];
    const NORM_FIXTURE_L2: f64 = 3.937_003_937_005_905; // sqrt(15.5)

    /// F32 is the pre-existing path. This pins it so the dtype-correct readback
    /// is provably a no-op at F32.
    #[test]
    fn test_clip_grad_norm_f32_value_is_unchanged() {
        let (client, device) = cpu_setup();

        let id = TensorId::new();
        let t = Tensor::<CpuRuntime>::from_slice(&NORM_FIXTURE, &[4], &device).unwrap();
        let mut grads = GradStore::new();
        grads.insert(id, t);

        let norm = clip_grad_norm(&client, &mut grads, &[id], f64::MAX).unwrap();
        assert!(
            (norm - NORM_FIXTURE_L2).abs() < 1e-6,
            "expected {NORM_FIXTURE_L2}, got {norm}"
        );
    }

    /// A BF16 or F16 gradient must produce the same norm as its F32 original.
    /// This is the LoRA regression: every adapter gradient reported NaN at once
    /// and the whole step was skipped, while the gradients themselves were fine.
    #[cfg(feature = "f16")]
    #[test]
    fn test_clip_grad_norm_reads_narrow_gradients_at_their_own_dtype() {
        let (client, device) = cpu_setup();
        let f32_grad = Tensor::<CpuRuntime>::from_slice(&NORM_FIXTURE, &[4], &device).unwrap();

        for dtype in [DType::BF16, DType::F16] {
            let id = TensorId::new();
            let mut grads = GradStore::new();
            grads.insert(id, client.cast(&f32_grad, dtype).unwrap());

            let norm = clip_grad_norm(&client, &mut grads, &[id], f64::MAX).unwrap();
            assert!(
                (norm - NORM_FIXTURE_L2).abs() < 1e-2,
                "{dtype:?}: expected {NORM_FIXTURE_L2}, got {norm}"
            );
        }
    }

    /// A narrow gradient over `max_norm` must be scaled by a factor derived from
    /// its true norm, not from a reinterpreted one.
    #[cfg(feature = "f16")]
    #[test]
    fn test_clip_grad_norm_scales_a_narrow_gradient_correctly() {
        let (client, device) = cpu_setup();
        // grad = [3, 4] → norm = 5.0, exact in BF16 and F16.
        let f32_grad = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 4.0], &[2], &device).unwrap();

        for dtype in [DType::BF16, DType::F16] {
            let id = TensorId::new();
            let mut grads = GradStore::new();
            grads.insert(id, client.cast(&f32_grad, dtype).unwrap());

            let norm = clip_grad_norm(&client, &mut grads, &[id], 1.0).unwrap();
            assert!((norm - 5.0).abs() < 1e-2, "{dtype:?}: got norm {norm}");

            let clipped = client
                .cast(grads.get(id).unwrap(), DType::F32)
                .unwrap()
                .to_vec::<f32>();
            let clipped_norm = (clipped[0] * clipped[0] + clipped[1] * clipped[1]).sqrt();
            assert!(
                (clipped_norm - 1.0).abs() < 1e-2,
                "{dtype:?}: clipped norm {clipped_norm}"
            );
        }
    }

    /// F64 is the same defect in the other direction: a fixed f32 read takes the
    /// low half of an F64 scalar's bytes. It needs no feature flag, so this is
    /// the case that guards the fix under a plain `cargo test`; the BF16/F16
    /// cases above need `--features f16`.
    #[test]
    fn test_clip_grad_norm_reads_an_f64_gradient_at_its_own_dtype() {
        let (client, device) = cpu_setup();

        let id = TensorId::new();
        let f32_grad = Tensor::<CpuRuntime>::from_slice(&NORM_FIXTURE, &[4], &device).unwrap();
        let mut grads = GradStore::new();
        grads.insert(id, client.cast(&f32_grad, DType::F64).unwrap());

        let norm = clip_grad_norm(&client, &mut grads, &[id], f64::MAX).unwrap();
        assert!(
            (norm - NORM_FIXTURE_L2).abs() < 1e-6,
            "expected {NORM_FIXTURE_L2}, got {norm}"
        );
    }

    /// `clip_grad_norm_per_param` reports the norm it clipped by, so a
    /// reinterpreted readback corrupts both the report and the scale applied.
    #[cfg(feature = "f16")]
    #[test]
    fn test_clip_per_param_reads_narrow_gradients_at_their_own_dtype() {
        let (client, device) = cpu_setup();
        // grad = [3, 4] → norm = 5.0, clipped to 2.0.
        let f32_grad = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 4.0], &[2], &device).unwrap();

        for dtype in [DType::BF16, DType::F16] {
            let id = TensorId::new();
            let mut grads = GradStore::new();
            grads.insert(id, client.cast(&f32_grad, dtype).unwrap());

            let clipped = clip_grad_norm_per_param(&client, &mut grads, &[id], 2.0).unwrap();
            assert_eq!(clipped.len(), 1, "{dtype:?}: expected one clipped gradient");
            assert!(
                (clipped[0].1 - 5.0).abs() < 1e-2,
                "{dtype:?}: expected reported norm 5.0, got {}",
                clipped[0].1
            );

            let data = client
                .cast(grads.get(id).unwrap(), DType::F32)
                .unwrap()
                .to_vec::<f32>();
            let norm = (data[0] * data[0] + data[1] * data[1]).sqrt();
            assert!((norm - 2.0).abs() < 5e-2, "{dtype:?}: clipped norm {norm}");
        }
    }

    /// The per-param path under a dtype that needs no feature flag.
    #[test]
    fn test_clip_per_param_reads_an_f64_gradient_at_its_own_dtype() {
        let (client, device) = cpu_setup();

        let id = TensorId::new();
        // grad = [3, 4] → norm = 5.0, clipped to 2.0.
        let f32_grad = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 4.0], &[2], &device).unwrap();
        let mut grads = GradStore::new();
        grads.insert(id, client.cast(&f32_grad, DType::F64).unwrap());

        let clipped = clip_grad_norm_per_param(&client, &mut grads, &[id], 2.0).unwrap();
        assert_eq!(clipped.len(), 1, "expected one clipped gradient");
        assert!(
            (clipped[0].1 - 5.0).abs() < 1e-6,
            "expected reported norm 5.0, got {}",
            clipped[0].1
        );

        let data = client
            .cast(grads.get(id).unwrap(), DType::F32)
            .unwrap()
            .to_vec::<f32>();
        let norm = (data[0] * data[0] + data[1] * data[1]).sqrt();
        assert!((norm - 2.0).abs() < 1e-4, "clipped norm {norm}");
    }
}
