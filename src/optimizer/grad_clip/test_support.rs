//! Shared test fixtures for the gradient-clipping test modules.
//!
//! `GraphStore` models a store shaped like a real backward pass: two trainable
//! parameters, the loss node's `1.0` seed, and one activation gradient. Both
//! the norm-clip and value-clip tests use it to check that only the parameter
//! ids are ever touched.

use numr::autograd::GradStore;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::{Tensor, TensorId};

use crate::test_utils::cpu_setup;

/// - parameters: `[3, 0]` and `[0, 4]` → param norm² = 9 + 16 = 25, norm 5.
/// - loss seed: `[1.0]` → contributes exactly 1.0, as it does on every
///   real run.
/// - activation: `[0, 0, 5, 5]` → contributes 50.
///
/// Whole-graph norm would be sqrt(25 + 1 + 50) = sqrt(76) = 8.7178, so
/// the two answers are 3.7 apart and cannot be confused.
pub(super) struct GraphStore {
    pub(super) client: numr::runtime::cpu::CpuClient,
    pub(super) grads: GradStore<CpuRuntime>,
    pub(super) params: [TensorId; 2],
    #[allow(dead_code)]
    pub(super) loss_node: TensorId,
    pub(super) activation: TensorId,
}

pub(super) fn graph_store() -> GraphStore {
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
