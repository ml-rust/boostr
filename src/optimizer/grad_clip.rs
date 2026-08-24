//! Gradient clipping utilities
//!
//! Clip gradients by global norm to prevent exploding gradients during training.

use crate::error::{Error, Result};
use crate::readback::scalar_f32;
use numr::autograd::GradStore;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ReduceOps, ScalarOps, TypeConversionOps, UnaryOps, UtilityOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::TensorId;
use std::collections::HashSet;

/// Deduplicate a caller's parameter list, preserving order.
///
/// A repeated id would be squared twice into the norm and, worse, scaled
/// twice by `mul_scalar` in the clip loop. Callers build their lists from
/// maps and are unique in practice; this makes that a guarantee rather than
/// an assumption.
fn unique_ids(param_ids: &[TensorId]) -> Vec<TensorId> {
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

    // ===== The norm is over the PARAMETERS, not the whole graph =====
    //
    // `backward()` seeds the store with `dL/dL = 1` for the loss node and
    // accumulates a gradient for every intermediate node and every matmul
    // operand, `requires_grad` or not. Only the ids the optimizer steps may
    // enter the norm, and only they may be rescaled by the clip.

    /// A store shaped like a real backward pass: two trainable parameters,
    /// the loss node's `1.0` seed, and one activation gradient.
    ///
    /// - parameters: `[3, 0]` and `[0, 4]` → param norm² = 9 + 16 = 25, norm 5.
    /// - loss seed: `[1.0]` → contributes exactly 1.0, as it does on every
    ///   real run.
    /// - activation: `[0, 0, 5, 5]` → contributes 50.
    ///
    /// Whole-graph norm would be sqrt(25 + 1 + 50) = sqrt(76) = 8.7178, so
    /// the two answers are 3.7 apart and cannot be confused.
    struct GraphStore {
        client: numr::runtime::cpu::CpuClient,
        grads: GradStore<CpuRuntime>,
        params: [TensorId; 2],
        loss_node: TensorId,
        activation: TensorId,
    }

    fn graph_store() -> GraphStore {
        let (client, device) = cpu_setup();

        let p1 = TensorId::new();
        let p2 = TensorId::new();
        let loss_node = TensorId::new();
        let activation = TensorId::new();

        let mut grads = GradStore::new();
        grads.insert(
            p1,
            Tensor::<CpuRuntime>::from_slice(&[3.0f32, 0.0], &[2], &device).unwrap(),
        );
        grads.insert(
            p2,
            Tensor::<CpuRuntime>::from_slice(&[0.0f32, 4.0], &[2], &device).unwrap(),
        );
        grads.insert(
            loss_node,
            Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1], &device).unwrap(),
        );
        grads.insert(
            activation,
            Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0, 5.0, 5.0], &[4], &device).unwrap(),
        );

        GraphStore {
            client,
            grads,
            params: [p1, p2],
            loss_node,
            activation,
        }
    }

    const GRAPH_PARAM_NORM: f64 = 5.0; // sqrt(9 + 16)
    const GRAPH_WHOLE_NORM: f64 = 8.717_797_887_081_348; // sqrt(9 + 16 + 1 + 50)

    /// The reported norm is the parameter-only value, pinned exactly.
    ///
    /// The whole-graph value is asserted to be different in the same test, so
    /// the two numbers are visibly distinct rather than merely "not equal".
    #[test]
    fn test_clip_grad_norm_is_over_parameters_not_the_whole_graph() {
        let GraphStore {
            client,
            mut grads,
            params,
            ..
        } = graph_store();

        let norm = clip_grad_norm(&client, &mut grads, &params, f64::MAX).unwrap();
        assert!(
            (norm - GRAPH_PARAM_NORM).abs() < 1e-5,
            "expected the parameter-only norm {GRAPH_PARAM_NORM}, got {norm}; \
             the whole-graph norm would be {GRAPH_WHOLE_NORM}"
        );
        assert!(
            (norm - GRAPH_WHOLE_NORM).abs() > 3.0,
            "norm {norm} matches the whole-graph value, not the parameters"
        );
    }

    /// The clip SCALE applied to a parameter is derived from the
    /// parameter-only norm. This is the consequence that corrupts training:
    /// a mis-reported norm is a bad log line, a mis-applied scale is a wrong
    /// update.
    ///
    /// max_norm = 1.0 against a parameter norm of 5.0 gives
    /// scale = 1.0 / (5.0 + 1e-6), so p1's leading 3.0 becomes 0.5999999.
    /// Under the whole-graph norm the scale would be 1.0 / 8.7178 and p1
    /// would become 0.34413 — a 1.74x difference, far outside the tolerance.
    #[test]
    fn test_clip_scale_is_derived_from_the_parameter_norm() {
        let GraphStore {
            client,
            mut grads,
            params,
            ..
        } = graph_store();

        clip_grad_norm(&client, &mut grads, &params, 1.0).unwrap();

        let expected = 3.0 / (GRAPH_PARAM_NORM + 1e-6);
        let wrong = 3.0 / (GRAPH_WHOLE_NORM + 1e-6);
        let p1 = grads.get(params[0]).unwrap().to_vec::<f32>();
        assert!(
            (p1[0] as f64 - expected).abs() < 1e-5,
            "p1[0] = {}, expected {expected} from the parameter norm; \
             the whole-graph norm would give {wrong}",
            p1[0]
        );

        // The clipped parameter set has norm 1.0, which is the entire point
        // of asking for max_norm = 1.0.
        let p2 = grads.get(params[1]).unwrap().to_vec::<f32>();
        let clipped_norm =
            ((p1[0] * p1[0] + p1[1] * p1[1] + p2[0] * p2[0] + p2[1] * p2[1]) as f64).sqrt();
        assert!(
            (clipped_norm - 1.0).abs() < 1e-5,
            "clipped parameter norm {clipped_norm}, expected 1.0"
        );
    }

    /// Non-parameter entries are left exactly as they were. Scaling them is
    /// pointless work at best, and on the LoRA path it silently rewrites the
    /// frozen base model's gradient buffers.
    #[test]
    fn test_clip_does_not_touch_non_parameter_gradients() {
        let GraphStore {
            client,
            mut grads,
            params,
            loss_node,
            activation,
        } = graph_store();

        clip_grad_norm(&client, &mut grads, &params, 1.0).unwrap();

        let loss_grad = grads.get(loss_node).unwrap().to_vec::<f32>();
        assert_eq!(loss_grad, vec![1.0f32], "the loss seed was rescaled");
        let act_grad = grads.get(activation).unwrap().to_vec::<f32>();
        assert_eq!(
            act_grad,
            vec![0.0f32, 0.0, 5.0, 5.0],
            "an activation gradient was rescaled"
        );
    }

    /// A FROZEN parameter has a gradient in the store — `MatmulBackward`
    /// returns one for both operands regardless of `requires_grad` — but is
    /// not in the parameter set and is never optimized. It must not enter the
    /// norm.
    ///
    /// The frozen gradient here is `[128, 0]`, whose 16384 alone dwarfs the
    /// parameters' 25. Including it would give sqrt(16409) = 128.0977 rather
    /// than 5.0; that is the shape of the real Llama-3.2-1B LoRA case, where
    /// the frozen `[128256, 2048]` tied embedding dominated everything.
    #[test]
    fn test_frozen_parameter_gradient_does_not_enter_the_norm() {
        let (client, device) = cpu_setup();

        let trainable = TensorId::new();
        let frozen = TensorId::new();

        let mut grads = GradStore::new();
        grads.insert(
            trainable,
            Tensor::<CpuRuntime>::from_slice(&[3.0f32, 4.0], &[2], &device).unwrap(),
        );
        grads.insert(
            frozen,
            Tensor::<CpuRuntime>::from_slice(&[128.0f32, 0.0], &[2], &device).unwrap(),
        );

        // Only `trainable` is optimized, so only it is in the parameter set.
        let norm = clip_grad_norm(&client, &mut grads, &[trainable], f64::MAX).unwrap();
        assert!(
            (norm - 5.0).abs() < 1e-5,
            "expected 5.0 from the trainable gradient alone, got {norm}; \
             128.0977 means the frozen gradient entered the norm"
        );

        // And the frozen gradient is untouched by the clip.
        clip_grad_norm(&client, &mut grads, &[trainable], 1.0).unwrap();
        let frozen_grad = grads.get(frozen).unwrap().to_vec::<f32>();
        assert_eq!(frozen_grad, vec![128.0f32, 0.0]);
    }

    /// A parameter with no gradient in the store is skipped, matching
    /// PyTorch's treatment of a parameter whose `.grad` is `None`. Without
    /// this the first step of any run with a partially-connected graph would
    /// error out.
    #[test]
    fn test_parameter_without_a_gradient_is_skipped() {
        let (client, device) = cpu_setup();

        let present = TensorId::new();
        let missing = TensorId::new();
        let mut grads = GradStore::new();
        grads.insert(
            present,
            Tensor::<CpuRuntime>::from_slice(&[3.0f32, 4.0], &[2], &device).unwrap(),
        );

        let norm = clip_grad_norm(&client, &mut grads, &[present, missing], f64::MAX).unwrap();
        assert!((norm - 5.0).abs() < 1e-5, "got {norm}");
    }

    /// A repeated id must not be squared twice into the norm, nor scaled
    /// twice by the clip loop.
    #[test]
    fn test_duplicate_parameter_ids_are_counted_once() {
        let (client, device) = cpu_setup();

        let id = TensorId::new();
        let mut grads = GradStore::new();
        grads.insert(
            id,
            Tensor::<CpuRuntime>::from_slice(&[3.0f32, 4.0], &[2], &device).unwrap(),
        );

        let norm = clip_grad_norm(&client, &mut grads, &[id, id], f64::MAX).unwrap();
        assert!(
            (norm - 5.0).abs() < 1e-5,
            "expected 5.0, got {norm}; 7.0711 means the gradient was counted twice"
        );
    }

    /// `clip_grad_norm_per_param` carries the same contract: a non-parameter
    /// gradient is neither reported nor scaled.
    #[test]
    fn test_clip_per_param_ignores_non_parameter_gradients() {
        let GraphStore {
            client,
            mut grads,
            params,
            activation,
            ..
        } = graph_store();

        let clipped = clip_grad_norm_per_param(&client, &mut grads, &params, 2.0).unwrap();

        // Only p2 (norm 4.0) exceeds 2.0; p1 (norm 3.0) does too. The
        // activation (norm sqrt(50) = 7.07) is the largest of the three and
        // would be reported first if the store were the parameter set.
        assert_eq!(clipped.len(), 2, "expected exactly the two parameters");
        assert!(
            clipped.iter().all(|(id, _)| params.contains(id)),
            "a non-parameter id was clipped"
        );

        let act_grad = grads.get(activation).unwrap().to_vec::<f32>();
        assert_eq!(act_grad, vec![0.0f32, 0.0, 5.0, 5.0]);
    }

    /// `clip_grad_value` carries it too.
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
