use std::collections::{HashMap, HashSet};

use boostr::model::mamba::{Mamba2, Mamba2Config, Mamba2WeightsWithIds};
use boostr::nn::{Conv1d, Linear, MaybeQuantLinear, Mla, MlaConfig, MlaWeights, RmsNorm};
use numr::autograd::Var;
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

#[derive(Clone, Copy)]
pub struct MlaIds {
    pub q_down: LinearIds,
    pub q_up: LinearIds,
    pub q_norm: TensorId,
    pub kv_compress: LinearIds,
    pub kv_norm: TensorId,
    pub kv_decompress: LinearIds,
    pub o_proj: LinearIds,
}

#[derive(Clone, Copy)]
pub struct Mamba2Ids {
    pub in_proj: LinearIds,
    pub conv1d: ConvIds,
    pub out_proj: LinearIds,
    pub a_log: TensorId,
    pub dt_bias: TensorId,
    pub d_param: TensorId,
    pub norm: TensorId,
}

pub fn cpu_setup() -> (CpuClient, CpuDevice) {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    (client, device)
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
    Tensor::<CpuRuntime>::from_slice(&data, shape, device)
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

pub fn build_mla(device: &CpuDevice) -> (Mla<CpuRuntime>, MlaConfig, MlaIds, Vec<TensorId>) {
    let config = MlaConfig::deepseek_v2(4, 1, 2, 2, 2, 4);
    let qk_dim = config.qk_head_dim();
    let (q_down, q_down_ids) = linear(config.q_lora_rank, config.hidden_size, false, 0.01, device);
    let (q_up, q_up_ids) = linear(
        config.num_heads * qk_dim,
        config.q_lora_rank,
        false,
        0.02,
        device,
    );
    let (q_norm, q_norm_id) = rms(config.q_lora_rank, config.norm_eps, 0.03, device);
    let (kv_compress, kv_compress_ids) = linear(
        config.kv_lora_rank + config.rope_head_dim,
        config.hidden_size,
        false,
        0.04,
        device,
    );
    let (kv_norm, kv_norm_id) = rms(config.kv_lora_rank, config.norm_eps, 0.05, device);
    let (kv_decompress, kv_decompress_ids) = linear(
        config.num_heads * (config.head_dim + config.head_dim_v),
        config.kv_lora_rank,
        false,
        0.06,
        device,
    );
    let (o_proj, o_proj_ids) = linear(
        config.hidden_size,
        config.num_heads * config.head_dim_v,
        false,
        0.07,
        device,
    );
    let weights = MlaWeights {
        q_down: Some(MaybeQuantLinear::Standard(q_down)),
        q_up: MaybeQuantLinear::Standard(q_up),
        q_norm: Some(q_norm),
        kv_compress: MaybeQuantLinear::Standard(kv_compress),
        kv_norm: Some(kv_norm),
        kv_decompress: MaybeQuantLinear::Standard(kv_decompress),
        o_proj: MaybeQuantLinear::Standard(o_proj),
    };
    let ids = MlaIds {
        q_down: q_down_ids,
        q_up: q_up_ids,
        q_norm: q_norm_id,
        kv_compress: kv_compress_ids,
        kv_norm: kv_norm_id,
        kv_decompress: kv_decompress_ids,
        o_proj: o_proj_ids,
    };
    let expected = vec![
        ids.q_down.weight,
        ids.q_up.weight,
        ids.q_norm,
        ids.kv_compress.weight,
        ids.kv_norm,
        ids.kv_decompress.weight,
        ids.o_proj.weight,
    ];
    (
        Mla::with_ids(&config, weights, device).expect("valid MLA"),
        config,
        ids,
        expected,
    )
}

pub fn rebuild_mla(
    params: &HashMap<TensorId, Tensor<CpuRuntime>>,
    config: &MlaConfig,
    ids: MlaIds,
    device: &CpuDevice,
) -> Mla<CpuRuntime> {
    let weights = MlaWeights {
        q_down: Some(MaybeQuantLinear::Standard(rebuild_linear(
            params, ids.q_down,
        ))),
        q_up: MaybeQuantLinear::Standard(rebuild_linear(params, ids.q_up)),
        q_norm: Some(RmsNorm::with_id(
            param(params, ids.q_norm),
            ids.q_norm,
            config.norm_eps,
            true,
        )),
        kv_compress: MaybeQuantLinear::Standard(rebuild_linear(params, ids.kv_compress)),
        kv_norm: Some(RmsNorm::with_id(
            param(params, ids.kv_norm),
            ids.kv_norm,
            config.norm_eps,
            true,
        )),
        kv_decompress: MaybeQuantLinear::Standard(rebuild_linear(params, ids.kv_decompress)),
        o_proj: MaybeQuantLinear::Standard(rebuild_linear(params, ids.o_proj)),
    };
    Mla::with_ids(config, weights, device).expect("valid rebuilt MLA")
}

pub fn build_mamba2(
    device: &CpuDevice,
) -> (Mamba2<CpuRuntime>, Mamba2Config, Mamba2Ids, Vec<TensorId>) {
    let config = Mamba2Config::new(4)
        .with_nheads(1)
        .with_d_state(2)
        .with_expand(1)
        .with_d_conv(2)
        .with_use_dt_bias(true)
        .with_use_d(true);
    let (in_proj, in_proj_ids) = linear(config.proj_dim(), config.d_model, true, 0.08, device);
    let (conv1d, conv1d_ids) = conv(config.conv_channels(), config.d_conv, true, 0.09, device);
    let (out_proj, out_proj_ids) = linear(config.d_model, config.d_inner(), true, 0.10, device);
    let a_log = Tensor::<CpuRuntime>::from_slice(&[-0.3f32], &[config.nheads], device);
    let a_log_id = a_log.id();
    let dt_bias = Tensor::<CpuRuntime>::from_slice(&[0.1f32], &[config.nheads], device);
    let dt_bias_id = dt_bias.id();
    let d_param = Tensor::<CpuRuntime>::from_slice(&[0.4f32], &[config.nheads], device);
    let d_param_id = d_param.id();
    let (norm, norm_id) = rms(config.d_inner(), 1e-5, 0.11, device);
    let weights = Mamba2WeightsWithIds {
        in_proj,
        conv1d,
        out_proj,
        a_log: (a_log, a_log_id),
        dt_bias: Some((dt_bias, dt_bias_id)),
        d_param: Some((d_param, d_param_id)),
        norm: Some(norm),
    };
    let ids = Mamba2Ids {
        in_proj: in_proj_ids,
        conv1d: conv1d_ids,
        out_proj: out_proj_ids,
        a_log: a_log_id,
        dt_bias: dt_bias_id,
        d_param: d_param_id,
        norm: norm_id,
    };
    let expected = vec![
        ids.in_proj.weight,
        ids.in_proj.bias.expect("in_proj bias"),
        ids.conv1d.weight,
        ids.conv1d.bias.expect("conv bias"),
        ids.out_proj.weight,
        ids.out_proj.bias.expect("out_proj bias"),
        ids.a_log,
        ids.dt_bias,
        ids.d_param,
        ids.norm,
    ];
    (
        Mamba2::with_ids(config.clone(), weights, true),
        config,
        ids,
        expected,
    )
}

pub fn rebuild_mamba2(
    params: &HashMap<TensorId, Tensor<CpuRuntime>>,
    config: &Mamba2Config,
    ids: Mamba2Ids,
) -> Mamba2<CpuRuntime> {
    let weights = Mamba2WeightsWithIds {
        in_proj: rebuild_linear(params, ids.in_proj),
        conv1d: rebuild_conv(params, ids.conv1d, config.d_conv),
        out_proj: rebuild_linear(params, ids.out_proj),
        a_log: (param(params, ids.a_log), ids.a_log),
        dt_bias: Some((param(params, ids.dt_bias), ids.dt_bias)),
        d_param: Some((param(params, ids.d_param), ids.d_param)),
        norm: Some(RmsNorm::with_id(
            param(params, ids.norm),
            ids.norm,
            1e-5,
            true,
        )),
    };
    Mamba2::with_ids(config.clone(), weights, true)
}
