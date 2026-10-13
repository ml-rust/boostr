//! MLA and Mamba2 blocks built from patterned parameters with stable ids,
//! and rebuilt from a parameter map.

use std::collections::HashMap;

use boostr::model::mamba::{Mamba2, Mamba2Config, Mamba2WeightsWithIds};
use boostr::nn::{MaybeQuantLinear, Mla, MlaConfig, MlaWeights, RmsNorm};
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::{Tensor, TensorId};

use super::params::{ConvIds, LinearIds, conv, linear, param, rebuild_conv, rebuild_linear, rms};

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
    let a_log = Tensor::<CpuRuntime>::from_slice(&[-0.3f32], &[config.nheads], device).unwrap();
    let a_log_id = a_log.id();
    let dt_bias = Tensor::<CpuRuntime>::from_slice(&[0.1f32], &[config.nheads], device).unwrap();
    let dt_bias_id = dt_bias.id();
    let d_param = Tensor::<CpuRuntime>::from_slice(&[0.4f32], &[config.nheads], device).unwrap();
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
