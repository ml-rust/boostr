//! The norm is over the PARAMETERS, not the whole autograd graph.
//!
//! `backward()` seeds the store with `dL/dL = 1` for the loss node and
//! accumulates a gradient for every intermediate node and every matmul
//! operand, `requires_grad` or not. Only the ids the optimizer steps may
//! enter the norm, and only they may be rescaled by the clip.

mod common;

use boostr::optimizer::{clip_grad_norm, clip_grad_norm_per_param};
use common::{
    GRAD_CLIP_GRAPH_PARAM_NORM, GRAD_CLIP_GRAPH_WHOLE_NORM, GradClipGraphStore, cpu_setup,
    grad_clip_graph_store,
};
use numr::autograd::GradStore;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::{Tensor, TensorId};

/// The reported norm is the parameter-only value, pinned exactly.
///
/// The whole-graph value is asserted to be different in the same test, so
/// the two numbers are visibly distinct rather than merely "not equal".
#[test]
fn test_clip_grad_norm_is_over_parameters_not_the_whole_graph() {
    let GradClipGraphStore {
        client,
        mut grads,
        params,
        ..
    } = grad_clip_graph_store();

    let norm = clip_grad_norm(&client, &mut grads, &params, f64::MAX).unwrap();
    assert!(
        (norm - GRAD_CLIP_GRAPH_PARAM_NORM).abs() < 1e-5,
        "expected the parameter-only norm {GRAD_CLIP_GRAPH_PARAM_NORM}, got {norm}; \
         the whole-graph norm would be {GRAD_CLIP_GRAPH_WHOLE_NORM}"
    );
    assert!(
        (norm - GRAD_CLIP_GRAPH_WHOLE_NORM).abs() > 3.0,
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
    let GradClipGraphStore {
        client,
        mut grads,
        params,
        ..
    } = grad_clip_graph_store();

    clip_grad_norm(&client, &mut grads, &params, 1.0).unwrap();

    let expected = 3.0 / (GRAD_CLIP_GRAPH_PARAM_NORM + 1e-6);
    let wrong = 3.0 / (GRAD_CLIP_GRAPH_WHOLE_NORM + 1e-6);
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
    let GradClipGraphStore {
        client,
        mut grads,
        params,
        loss_node,
        activation,
    } = grad_clip_graph_store();

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
    let GradClipGraphStore {
        client,
        mut grads,
        params,
        activation,
        ..
    } = grad_clip_graph_store();

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
