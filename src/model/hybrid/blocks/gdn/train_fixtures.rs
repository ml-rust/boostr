//! F64 fixtures for the training-forward tests: a small block with every
//! weight trainable and separate dense gates, so `ssm_alpha` and `ssm_beta`
//! own their weights.

use super::gate_proj::{GateProjections, SeparateGates};
use super::layer::{GdnBlock, GdnWeights};
use crate::model::config::GdnConfig;
use crate::nn::{Linear, MaybeQuantLinear};
use crate::test_utils::Lcg;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::{Tensor, TensorId};

pub(super) const HIDDEN: usize = 8;
pub(super) const H_V: usize = 4;
pub(super) const S: usize = 4;
pub(super) const KERNEL: usize = 4;
pub(super) const SEQ: usize = 9;
pub(super) const BATCH: usize = 2;
/// 9 tokens in chunks of 4: the last chunk is padded.
pub(super) const CHUNK: usize = 4;

/// Indices into the F64 parameter set.
pub(super) const X: usize = 0;
pub(super) const QKV: usize = 1;
pub(super) const GATE: usize = 2;
pub(super) const ALPHA: usize = 3;
pub(super) const BETA: usize = 4;
pub(super) const OUT: usize = 5;
pub(super) const CONV: usize = 6;
pub(super) const A_LOG: usize = 7;
pub(super) const DT_BIAS: usize = 8;
pub(super) const NORM: usize = 9;
pub(super) const NAMES: [&str; 10] = [
    "x",
    "attn_qkv",
    "attn_gate",
    "ssm_alpha",
    "ssm_beta",
    "ssm_out",
    "conv",
    "a_log",
    "dt_bias",
    "norm",
];

pub(super) type T = Tensor<CpuRuntime>;

pub(super) fn f32_tensor(data: &[f64], shape: &[usize], device: &CpuDevice) -> T {
    let data: Vec<f32> = data.iter().map(|v| *v as f32).collect();
    T::from_slice(&data, shape, device).unwrap()
}

pub(super) fn f64_tensor(data: &[f64], shape: &[usize], device: &CpuDevice) -> T {
    T::from_slice(data, shape, device).unwrap()
}

pub(super) fn cfg64(key_heads: usize, v_grouped: bool) -> GdnConfig {
    GdnConfig {
        hidden_size: HIDDEN,
        conv_kernel: KERNEL,
        state_size: S,
        key_heads,
        value_heads: H_V,
        inner_size: H_V * S,
        rms_eps: 1e-6,
        chunk_size: CHUNK,
        v_grouped,
    }
}

fn shapes(cfg: &GdnConfig) -> [Vec<usize>; 10] {
    let (qkv, value) = (cfg.qkv_dim(), cfg.value_dim());
    [
        vec![BATCH, SEQ, HIDDEN],
        vec![qkv, HIDDEN],
        vec![value, HIDDEN],
        vec![H_V, HIDDEN],
        vec![H_V, HIDDEN],
        vec![HIDDEN, value],
        vec![qkv, KERNEL],
        vec![H_V],
        vec![H_V],
        vec![S],
    ]
}

pub(super) fn init_params(cfg: &GdnConfig, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = Lcg(seed);
    let in_scale = 0.5 / (HIDDEN as f64).sqrt();
    let out_scale = 0.5 / (cfg.value_dim() as f64).sqrt();
    let scales = [
        1.0, in_scale, in_scale, in_scale, in_scale, out_scale, 0.5, 0.0, 0.5, 0.0,
    ];
    shapes(cfg)
        .iter()
        .zip(scales)
        .enumerate()
        .map(|(i, (shape, scale))| {
            let n: usize = shape.iter().product();
            match i {
                A_LOG | NORM => (0..n).map(|_| rng.uniform_f64(0.5, 1.5)).collect(),
                _ => rng.vec(n, scale),
            }
        })
        .collect()
}

/// A trainable F64 block over the parameter set `p`.
pub(super) fn build64(
    client: &CpuClient,
    device: &CpuDevice,
    cfg: &GdnConfig,
    p: &[Vec<f64>],
) -> GdnBlock<CpuRuntime> {
    let sh = shapes(cfg);
    let t = |i: usize| f64_tensor(&p[i], &sh[i], device);
    let lin = |i: usize| MaybeQuantLinear::Standard(Linear::new(t(i), None, true));
    let ssm_a: Vec<f64> = p[A_LOG].iter().map(|a| -a.exp()).collect();
    let weights = GdnWeights {
        attn_qkv: lin(QKV).into(),
        attn_gate: lin(GATE).into(),
        ssm_alpha: lin(ALPHA),
        ssm_beta: lin(BETA),
        ssm_out: lin(OUT).into(),
        ssm_conv1d: t(CONV),
        ssm_a: f64_tensor(&ssm_a, &[H_V], device),
        ssm_dt_bias: t(DT_BIAS),
        ssm_norm: t(NORM),
    };
    let mut block = GdnBlock::new(cfg.clone(), weights).unwrap();
    block.gates = GateProjections::Separate(Box::new(SeparateGates {
        alpha: lin(ALPHA).into(),
        beta: lin(BETA).into(),
    }));
    block.with_trainable(client, true).unwrap()
}

/// Weight ids in parameter-index order, `QKV..=NORM`.
pub(super) fn param_ids(block: &GdnBlock<CpuRuntime>) -> Vec<TensorId> {
    let GateProjections::Separate(gates) = &block.gates else {
        panic!("build64 makes separate gates");
    };
    vec![
        block.attn_qkv.weight().unwrap().id(),
        block.attn_gate.weight().unwrap().id(),
        gates.alpha.weight().unwrap().id(),
        gates.beta.weight().unwrap().id(),
        block.ssm_out.weight().unwrap().id(),
        block.conv_weight.id(),
        block.a_log.as_ref().unwrap().id(),
        block.ssm_dt_bias.id(),
        block.norm.weight().id(),
    ]
}
