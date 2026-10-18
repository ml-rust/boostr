//! Deterministic [`Qwen35AttentionBlock`] fixtures for the training tests.
//!
//! Parameter values live on the host in F64, so one fixture builds an F32
//! block for value parity and an F64 block for finite differences. `q`
//! holds the stored, per-head interleaved `attn_q` rows.

use super::layer::{Qwen35AttentionBlock, Qwen35AttentionWeights};
use crate::model::config::Qwen35AttentionConfig;
use crate::nn::{Linear, MaybeQuantLinear, MaybeRotatedLinear, RoPE};
use numr::autograd::{Var, var_mul, var_sum};
use numr::dtype::DType;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

pub(super) const HIDDEN: usize = 8;
pub(super) const HD: usize = 8;
pub(super) const BATCH: usize = 2;
pub(super) const SEQ: usize = 5;
pub(super) const MAX_POS: usize = 16;

/// Indices into [`Fixture::values`].
pub(super) const X: usize = 0;
pub(super) const Q: usize = 1;
pub(super) const K: usize = 2;
pub(super) const V: usize = 3;
pub(super) const O: usize = 4;
pub(super) const Q_NORM: usize = 5;
pub(super) const K_NORM: usize = 6;
pub(super) const NAMES: [&str; 7] = [
    "x",
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_output",
    "q_norm",
    "k_norm",
];

type T = Tensor<CpuRuntime>;

/// Four query heads over two KV heads: `head_repeat() == 2`.
pub(super) fn cfg32() -> Qwen35AttentionConfig {
    cfg_kv(2)
}

/// [`cfg32`] with `num_kv_heads` KV heads.
pub(super) fn cfg_kv(num_kv_heads: usize) -> Qwen35AttentionConfig {
    Qwen35AttentionConfig {
        hidden_size: HIDDEN,
        num_heads: 4,
        num_kv_heads,
        head_dim: HD,
        rope_dim: 4,
        rope_sections: [1, 1, 0, 0],
        rope_theta: 10_000.0,
        rms_eps: 1e-6,
    }
}

/// Deterministic 64-bit LCG; no rand dependency.
pub(super) struct Lcg(pub(super) u64);

impl Lcg {
    pub(super) fn bits(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 11
    }

    pub(super) fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * (self.bits() as f64) / ((1u64 << 53) as f64)
    }

    pub(super) fn vec(&mut self, n: usize, scale: f64) -> Vec<f64> {
        (0..n).map(|_| self.uniform(-scale, scale)).collect()
    }

    pub(super) fn index(&mut self, n: usize) -> usize {
        (self.bits() % n as u64) as usize
    }
}

/// Host-side input and weights of one block, built at `dtype`.
#[derive(Clone)]
pub(super) struct Fixture {
    pub(super) cfg: Qwen35AttentionConfig,
    pub(super) dtype: DType,
    pub(super) values: Vec<Vec<f64>>,
}

impl Fixture {
    pub(super) fn f32(cfg: &Qwen35AttentionConfig, seed: u64) -> Self {
        Self::new(cfg, seed, DType::F32)
    }

    pub(super) fn f64(cfg: &Qwen35AttentionConfig, seed: u64) -> Self {
        Self::new(cfg, seed, DType::F64)
    }

    fn new(cfg: &Qwen35AttentionConfig, seed: u64, dtype: DType) -> Self {
        let mut rng = Lcg(seed);
        let in_scale = 0.5 / (HIDDEN as f64).sqrt();
        let out_scale = 0.5 / (cfg.q_dim() as f64).sqrt();
        let scales = [1.0, in_scale, in_scale, in_scale, out_scale];
        let shapes = shapes(cfg);
        let mut values: Vec<Vec<f64>> = shapes[..5]
            .iter()
            .zip(scales)
            .map(|(shape, scale)| rng.vec(shape.iter().product(), scale))
            .collect();
        for _ in 0..2 {
            values.push((0..HD).map(|_| rng.uniform(0.5, 1.5)).collect());
        }
        Self {
            cfg: cfg.clone(),
            dtype,
            values,
        }
    }

    pub(super) fn shapes(&self) -> [Vec<usize>; 7] {
        shapes(&self.cfg)
    }

    pub(super) fn tensor(&self, i: usize, device: &CpuDevice) -> T {
        let shapes = self.shapes();
        let shape = &shapes[i];
        match self.dtype {
            DType::F64 => T::from_slice(&self.values[i], shape, device).unwrap(),
            _ => {
                let data: Vec<f32> = self.values[i].iter().map(|v| *v as f32).collect();
                T::from_slice(&data, shape, device).unwrap()
            }
        }
    }

    pub(super) fn x(&self, device: &CpuDevice, requires_grad: bool) -> Var<CpuRuntime> {
        Var::new(self.tensor(X, device), requires_grad)
    }

    /// Every weight with `requires_grad = trainable`.
    pub(super) fn block(
        &self,
        device: &CpuDevice,
        trainable: bool,
    ) -> Qwen35AttentionBlock<CpuRuntime> {
        let linear = |i: usize| -> MaybeRotatedLinear<CpuRuntime> {
            let dense = Linear::new(self.tensor(i, device), None, trainable);
            MaybeQuantLinear::Standard(dense).into()
        };
        let weights = Qwen35AttentionWeights {
            attn_q: linear(Q),
            attn_k: linear(K),
            attn_v: linear(V),
            attn_output: linear(O),
            attn_q_norm: self.tensor(Q_NORM, device),
            attn_k_norm: self.tensor(K_NORM, device),
        };
        Qwen35AttentionBlock::new(self.cfg.clone(), weights)
            .unwrap()
            .with_trainable(trainable)
    }

    pub(super) fn frozen_block(&self, device: &CpuDevice) -> Qwen35AttentionBlock<CpuRuntime> {
        self.block(device, false)
    }

    pub(super) fn stored_q_f32(&self) -> Vec<f32> {
        self.values[Q].iter().map(|v| *v as f32).collect()
    }
}

fn shapes(cfg: &Qwen35AttentionConfig) -> [Vec<usize>; 7] {
    [
        vec![BATCH, SEQ, HIDDEN],
        vec![cfg.q_gate_dim(), HIDDEN],
        vec![cfg.kv_dim(), HIDDEN],
        vec![cfg.kv_dim(), HIDDEN],
        vec![HIDDEN, cfg.q_dim()],
        vec![HD],
        vec![HD],
    ]
}

pub(super) fn rope(device: &CpuDevice) -> RoPE<CpuRuntime> {
    cfg32().rope_table::<CpuRuntime>(MAX_POS, device).unwrap()
}

/// `[4, seq]` i32 text positions from 0: `t = h = w = i`, `e = 0`.
pub(super) fn positions(seq: usize, device: &CpuDevice) -> T {
    let mut data = Vec::with_capacity(4 * seq);
    for _ in 0..3 {
        data.extend((0..seq).map(|i| i as i32));
    }
    data.resize(4 * seq, 0);
    T::from_slice(&data, &[4, seq], device).unwrap()
}

/// `sum(y * w)`.
pub(super) fn weighted_loss(
    client: &CpuClient,
    y: &Var<CpuRuntime>,
    w: &Var<CpuRuntime>,
) -> Var<CpuRuntime> {
    let prod = var_mul(y, w, client).unwrap();
    var_sum(&prod, &[0, 1, 2], false, client).unwrap()
}

pub(super) fn to_f64(t: &T) -> Vec<f64> {
    let t = t.contiguous().unwrap();
    match t.dtype() {
        DType::F64 => t.to_vec::<f64>(),
        _ => t.to_vec::<f32>().iter().map(|v| f64::from(*v)).collect(),
    }
}
