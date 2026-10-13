//! CPU setup and patterned parameters: `Linear`, `Conv1d` and `RmsNorm`
//! built with stable ids, rebuilt from a parameter map, and id checks, plus
//! a gradient store shaped like a real backward pass for gradient-clipping
//! tests.

use std::collections::{HashMap, HashSet};

use boostr::nn::{Conv1d, Linear, RmsNorm};
use numr::autograd::{GradStore, Var};
use numr::ops::PaddingMode;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::{Tensor, TensorId};

#[derive(Clone, Copy)]
pub struct LinearIds {
    pub weight: TensorId,
    pub bias: Option<TensorId>,
}

#[derive(Clone, Copy)]
pub struct ConvIds {
    pub weight: TensorId,
    pub bias: Option<TensorId>,
}

pub fn cpu_setup() -> (CpuClient, CpuDevice) {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    (client, device)
}

/// A gradient store shaped like a real backward pass, for gradient-clipping
/// tests: two trainable parameters, the loss node's `1.0` seed, and one
/// activation gradient. Only the parameter ids are ever meant to be touched
/// by a clip.
///
/// - parameters: `[3, 0]` and `[0, 4]` → param norm² = 9 + 16 = 25, norm 5.
/// - loss seed: `[1.0]` → contributes exactly 1.0, as it does on every real
///   run.
/// - activation: `[0, 0, 5, 5]` → contributes 50.
///
/// Whole-graph norm would be sqrt(25 + 1 + 50) = sqrt(76) = 8.7178, so the
/// two answers are 3.7 apart and cannot be confused.
pub struct GradClipGraphStore {
    pub client: CpuClient,
    pub grads: GradStore<CpuRuntime>,
    pub params: [TensorId; 2],
    pub loss_node: TensorId,
    pub activation: TensorId,
}

pub const GRAD_CLIP_GRAPH_PARAM_NORM: f64 = 5.0; // sqrt(9 + 16)
pub const GRAD_CLIP_GRAPH_WHOLE_NORM: f64 = 8.717_797_887_081_348; // sqrt(9 + 16 + 1 + 50)

pub fn grad_clip_graph_store() -> GradClipGraphStore {
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

    GradClipGraphStore {
        client,
        grads,
        params: [p1, p2],
        loss_node,
        activation,
    }
}

pub fn patterned_tensor(
    shape: &[usize],
    base: f32,
    scale: f32,
    device: &CpuDevice,
) -> Tensor<CpuRuntime> {
    let len: usize = shape.iter().product();
    let data: Vec<f32> = (0..len)
        .map(|i| base + scale * ((i % 11) as f32 - 5.0))
        .collect();
    Tensor::<CpuRuntime>::from_slice(&data, shape, device).unwrap()
}

pub fn linear(
    out_features: usize,
    in_features: usize,
    bias: bool,
    seed: f32,
    device: &CpuDevice,
) -> (Linear<CpuRuntime>, LinearIds) {
    let weight = patterned_tensor(&[out_features, in_features], seed, 0.003, device);
    let weight_id = weight.id();
    let (bias_arg, bias_id) = if bias {
        let bias_tensor = patterned_tensor(&[out_features], seed * 0.1, 0.002, device);
        let bias_id = bias_tensor.id();
        (Some((bias_tensor, bias_id)), Some(bias_id))
    } else {
        (None, None)
    };
    let module = Linear::with_ids(weight, weight_id, bias_arg, true);
    (
        module,
        LinearIds {
            weight: weight_id,
            bias: bias_id,
        },
    )
}

pub fn conv(
    channels: usize,
    kernel: usize,
    bias: bool,
    seed: f32,
    device: &CpuDevice,
) -> (Conv1d<CpuRuntime>, ConvIds) {
    let weight = patterned_tensor(&[channels, 1, kernel], seed, 0.002, device);
    let weight_id = weight.id();
    let (bias_arg, bias_id) = if bias {
        let bias_tensor = patterned_tensor(&[channels], seed * 0.1, 0.001, device);
        let bias_id = bias_tensor.id();
        (Some((bias_tensor, bias_id)), Some(bias_id))
    } else {
        (None, None)
    };
    let module = Conv1d::with_ids(
        weight,
        weight_id,
        bias_arg,
        1,
        PaddingMode::Custom(kernel - 1, 0, 0, 0),
        1,
        channels,
        true,
    );
    (
        module,
        ConvIds {
            weight: weight_id,
            bias: bias_id,
        },
    )
}

pub fn rms(dim: usize, eps: f32, seed: f32, device: &CpuDevice) -> (RmsNorm<CpuRuntime>, TensorId) {
    let weight = patterned_tensor(&[dim], 1.0 + seed, 0.001, device);
    let id = weight.id();
    (RmsNorm::with_id(weight, id, eps, true), id)
}

pub fn param(params: &HashMap<TensorId, Tensor<CpuRuntime>>, id: TensorId) -> Tensor<CpuRuntime> {
    params
        .get(&id)
        .unwrap_or_else(|| panic!("missing parameter tensor for stable id {id}"))
        .clone()
}

pub fn rebuild_linear(
    params: &HashMap<TensorId, Tensor<CpuRuntime>>,
    ids: LinearIds,
) -> Linear<CpuRuntime> {
    Linear::with_ids(
        param(params, ids.weight),
        ids.weight,
        ids.bias.map(|id| (param(params, id), id)),
        true,
    )
}

pub fn rebuild_conv(
    params: &HashMap<TensorId, Tensor<CpuRuntime>>,
    ids: ConvIds,
    kernel: usize,
) -> Conv1d<CpuRuntime> {
    let weight = param(params, ids.weight);
    let groups = weight.shape()[0];
    Conv1d::with_ids(
        weight,
        ids.weight,
        ids.bias.map(|id| (param(params, id), id)),
        1,
        PaddingMode::Custom(kernel - 1, 0, 0, 0),
        1,
        groups,
        true,
    )
}

pub fn trainable_map(
    module_params: Vec<(TensorId, &Var<CpuRuntime>)>,
) -> HashMap<TensorId, Tensor<CpuRuntime>> {
    module_params
        .into_iter()
        .map(|(id, var)| (id, var.tensor().clone()))
        .collect()
}

pub fn ids_from(params: Vec<(TensorId, &Var<CpuRuntime>)>) -> Vec<TensorId> {
    params.into_iter().map(|(id, _)| id).collect()
}

pub fn assert_ids(actual: &[TensorId], expected: &[TensorId]) {
    assert_eq!(actual.len(), expected.len());
    let actual_set: HashSet<TensorId> = actual.iter().copied().collect();
    let expected_set: HashSet<TensorId> = expected.iter().copied().collect();
    assert_eq!(actual_set, expected_set);
}
