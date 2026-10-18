//! CPU-vs-CUDA parity of the training stack: [`GdnBlock::forward_train`]
//! and its checkpointed form, [`Qwen35Model::forward_train`] with LoRA on a
//! frozen base, and the trainable LSTM and GRU stacks.
//!
//! Each case builds the same module twice from the same host weights, once
//! per backend, runs identical inputs, backpropagates `sum(y * w)` with a
//! fixed `w`, and compares the output and every gradient.
//!
//! # Tolerances
//!
//! Every check bounds `max |cuda - cpu|` by `rtol * max |cpu|` over the
//! tensor. Both backends run F32.
//!
//! - Forward outputs: `rtol = 1e-4`. The CUDA matmul and reductions sum in
//!   another order than the CPU kernels.
//! - Gradients: `rtol = 1e-3`. The chunked GDN backward and the recurrent
//!   backward chain many sums per chunk and per step, so the order gap grows.
//! - Checkpointed against plain, both on CUDA: `rtol = 1e-6`. The same
//!   kernels run in the same order. The bound stays above zero because the
//!   CUDA kernels are not checked for bit-stable reductions across launches.
//!
//! Run with:
//!   cd boostr && cargo nextest run --features f16,cuda --test gdn_training_cuda

#![cfg(feature = "cuda")]

mod common;

use std::sync::{Mutex, MutexGuard, OnceLock};

use boostr::model::ModelClient;
use boostr::model::config::GdnConfig;
use boostr::model::hybrid::GdnBlock;
use boostr::model::qwen35::Qwen35Model;
use boostr::nn::rnn::{CellKind, GruKind, LstmKind, RnnStack};
use boostr::nn::{RnnConfig, VarBuilder, VarMap};
use common::qwen35_tiny::{VOCAB, tiny_lora_model};
use numr::autograd::{GradStore, Var, backward, var_add, var_mul, var_sum};
use numr::dtype::DType;
use numr::ops::{RandomOps, TypeConversionOps};
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::runtime::cuda::{CudaDevice, CudaRuntime, is_cuda_available};
use numr::tensor::{Tensor, TensorId};

const FWD_RTOL: f32 = 1e-4;
const GRAD_RTOL: f32 = 1e-3;
const CKPT_RTOL: f32 = 1e-6;

/// Client ops every case here needs, on both backends.
trait TrainClient<R: Runtime>: ModelClient<R> + TypeConversionOps<R> + RandomOps<R> + 'static {}

impl<R: Runtime, C> TrainClient<R> for C where
    C: ModelClient<R> + TypeConversionOps<R> + RandomOps<R> + 'static
{
}

static CUDA_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

/// Serializes the CUDA cases: the device client is process-wide with one
/// stream.
fn cuda_lock() -> MutexGuard<'static, ()> {
    CUDA_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|p| p.into_inner())
}

/// The lock guard and both devices, or `None` without a CUDA device.
fn devices() -> Option<(MutexGuard<'static, ()>, CpuDevice, CudaDevice)> {
    let guard = cuda_lock();
    if !is_cuda_available() {
        eprintln!("no CUDA device: skipping the CPU-vs-CUDA training parity case");
        return None;
    }
    Some((guard, CpuDevice::new(), CudaDevice::new(0)))
}

/// Deterministic 64-bit LCG. One host stream feeds both backends.
struct Lcg(u64);

impl Lcg {
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let unit = ((self.0 >> 40) as f32) / ((1u64 << 24) as f32);
        lo + (hi - lo) * unit
    }

    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.uniform(-scale, scale)).collect()
    }
}

fn upload<R>(data: &[f32], shape: &[usize], device: &R::Device) -> Tensor<R>
where
    R: Runtime<DType = DType>,
{
    Tensor::<R>::from_slice(data, shape, device).expect("upload a host tensor")
}

fn host<R: Runtime>(t: &Tensor<R>) -> Vec<f32> {
    t.contiguous().expect("contiguous copy").to_vec::<f32>()
}

fn grad_host<R: Runtime>(grads: &GradStore<R>, id: TensorId, name: &str) -> Vec<f32> {
    let grad = grads
        .get(id)
        .unwrap_or_else(|| panic!("{name}: no gradient"));
    host(grad)
}

/// `sum(y * w)` with `w` drawn on the host from `seed`, so every output
/// element gets its own upstream gradient.
fn weighted_loss<R>(client: &R::Client, y: &Var<R>, seed: u64) -> Var<R>
where
    R: Runtime<DType = DType>,
    R::Client: TrainClient<R>,
{
    let shape = y.shape().to_vec();
    let n: usize = shape.iter().product();
    let w = upload::<R>(&Lcg(seed).vec(n, 1.0), &shape, y.tensor().device());
    let prod = var_mul(y, &Var::new(w, false), client).expect("y * w");
    let dims: Vec<usize> = (0..shape.len()).collect();
    var_sum(&prod, &dims, false, client).expect("sum(y * w)")
}

/// Asserts `max |got - want| <= rtol * max |want|`, a finite `got`, and a
/// reference that is not all zero.
fn assert_close(label: &str, got: &[f32], want: &[f32], rtol: f32) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    assert!(
        scale.is_finite() && scale > 0.0,
        "{label}: reference scale {scale}"
    );
    let tol = rtol * scale;
    let mut worst = (0usize, 0f32);
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert!(a.is_finite(), "{label}: non-finite value at {i}: {a}");
        let d = (a - b).abs();
        if d > worst.1 {
            worst = (i, d);
        }
    }
    assert!(
        worst.1 <= tol,
        "{label}: max abs diff {} at {} exceeds {tol} (got {}, want {})",
        worst.1,
        worst.0,
        got[worst.0],
        want[worst.0]
    );
}

/// Output values and named gradients of one forward and backward.
struct Pass {
    y: Vec<f32>,
    grads: Vec<(String, Vec<f32>)>,
}

/// Compares two passes: outputs at `fwd_rtol`, each gradient at `grad_rtol`.
fn assert_passes(case: &str, got: &Pass, want: &Pass, fwd_rtol: f32, grad_rtol: f32) {
    assert_close(&format!("{case} output"), &got.y, &want.y, fwd_rtol);
    let names = |p: &Pass| p.grads.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>();
    assert_eq!(names(got), names(want), "{case}: gradient names");
    for ((name, g), (_, w)) in got.grads.iter().zip(&want.grads) {
        assert_close(&format!("{case} grad {name}"), g, w, grad_rtol);
    }
}

// ---- GdnBlock ----

const HIDDEN: usize = 16;
const S: usize = 8;
const H_V: usize = 4;
const KERNEL: usize = 4;
const BATCH: usize = 2;
/// Trainable tensors of a block from `GdnBlock::load`.
const GDN_PARAMS: usize = 9;

/// `key_heads = 2` gives head repeat 2, `key_heads = 4` repeat 1.
fn gdn_cfg(key_heads: usize, chunk_size: usize) -> GdnConfig {
    GdnConfig {
        hidden_size: HIDDEN,
        conv_kernel: KERNEL,
        state_size: S,
        key_heads,
        value_heads: H_V,
        inner_size: H_V * S,
        rms_eps: 1e-6,
        chunk_size,
        v_grouped: false,
    }
}

type HostWeights = Vec<(String, Vec<usize>, Vec<f32>)>;

/// Host weights under the names `GdnBlock::load` reads. `A_log` in
/// `[-0.5, 0.5]` keeps the decay `-exp(A_log)` near -1, and the norm weight
/// stays near 1, so no gate saturates.
fn gdn_weights(cfg: &GdnConfig, seed: u64) -> HostWeights {
    let mut rng = Lcg(seed);
    let (qkv, value, heads) = (cfg.qkv_dim(), cfg.value_dim(), cfg.value_heads);
    let in_scale = 1.0 / (HIDDEN as f32).sqrt();
    let out_scale = 1.0 / (value as f32).sqrt();
    let mut w = |name: &str, shape: Vec<usize>, lo: f32, hi: f32| {
        let n: usize = shape.iter().product();
        let data = (0..n).map(|_| rng.uniform(lo, hi)).collect();
        (name.to_string(), shape, data)
    };
    vec![
        w("in_proj_qkv.weight", vec![qkv, HIDDEN], -in_scale, in_scale),
        w("in_proj_z.weight", vec![value, HIDDEN], -in_scale, in_scale),
        w(
            "alpha_proj.weight",
            vec![heads, HIDDEN],
            -in_scale,
            in_scale,
        ),
        w("beta_proj.weight", vec![heads, HIDDEN], -in_scale, in_scale),
        w(
            "out_proj.weight",
            vec![HIDDEN, value],
            -out_scale,
            out_scale,
        ),
        w("conv1d.weight", vec![qkv, 1, KERNEL], -0.5, 0.5),
        w("A_log", vec![heads], -0.5, 0.5),
        w("dt_bias", vec![heads], -0.5, 0.5),
        w("norm.weight", vec![S], 0.5, 1.5),
    ]
}

/// A `VarMap` on `device` holding `weights`.
fn varmap<R: Runtime<DType = DType>>(weights: &HostWeights, device: &R::Device) -> VarMap<R> {
    let mut map = VarMap::<R>::new();
    for (name, shape, data) in weights {
        map.insert(name.clone(), upload::<R>(data, shape, device));
    }
    map
}

/// One forward and backward of a trainable block over `weights`. The
/// gradients are `x` first, then every trainable parameter by name.
fn gdn_pass<R>(
    client: &R::Client,
    device: &R::Device,
    cfg: &GdnConfig,
    weights: &HostWeights,
    seq: usize,
    checkpointed: bool,
) -> Pass
where
    R: Runtime<DType = DType>,
    R::Client: TrainClient<R>,
{
    let mut map = varmap::<R>(weights, device);
    let mut vb = VarBuilder::new(&mut map, device);
    let block = GdnBlock::<R>::load(cfg, &mut vb, client, true).expect("GdnBlock::load");
    let x_data = Lcg(0x6d4e_7000 + seq as u64).vec(BATCH * seq * HIDDEN, 1.0);
    let x = Var::new(upload::<R>(&x_data, &[BATCH, seq, HIDDEN], device), true);
    let y = if checkpointed {
        block.forward_train_checkpointed(client, &x)
    } else {
        block.forward_train(client, &x)
    }
    .expect("GDN training forward");
    let grads = backward(&weighted_loss::<R>(client, &y, 0x6d4e_7001), client).expect("backward");
    let mut named = vec![("x".to_string(), grad_host(&grads, x.id(), "x"))];
    for (name, var) in block.named_parameters() {
        if var.requires_grad() {
            let values = grad_host(&grads, var.id(), &name);
            named.push((name, values));
        }
    }
    assert_eq!(named.len(), GDN_PARAMS + 1, "x plus every block weight");
    Pass {
        y: host(y.tensor()),
        grads: named,
    }
}

/// CUDA against CPU for one geometry.
fn gdn_parity(key_heads: usize, chunk_size: usize, seq: usize) {
    let Some((_guard, cpu, cuda)) = devices() else {
        return;
    };
    let cpu_client = CpuRuntime::default_client(&cpu);
    let cuda_client = CudaRuntime::default_client(&cuda);
    let cfg = gdn_cfg(key_heads, chunk_size);
    let weights = gdn_weights(&cfg, 0x6d4e_7100 + key_heads as u64);
    let want = gdn_pass::<CpuRuntime>(&cpu_client, &cpu, &cfg, &weights, seq, false);
    let got = gdn_pass::<CudaRuntime>(&cuda_client, &cuda, &cfg, &weights, seq, false);
    let case = format!("gdn rep={} chunk={chunk_size} seq={seq}", H_V / key_heads);
    assert_passes(&case, &got, &want, FWD_RTOL, GRAD_RTOL);
}

/// 9 tokens in chunks of 4: the last chunk holds one token and is padded.
#[test]
fn gdn_chunk4_padded_rep2_matches_cpu() {
    gdn_parity(2, 4, 9);
}

#[test]
fn gdn_chunk4_padded_rep1_matches_cpu() {
    gdn_parity(4, 4, 9);
}

/// 9 tokens in one padded chunk of 64.
#[test]
fn gdn_chunk64_rep2_matches_cpu() {
    gdn_parity(2, 64, 9);
}

#[test]
fn gdn_chunk64_rep1_matches_cpu() {
    gdn_parity(4, 64, 9);
}

/// 70 tokens in chunks of 64: one full chunk and one padded chunk, so the
/// state carries across a chunk boundary.
#[test]
fn gdn_chunk64_two_chunks_rep2_matches_cpu() {
    gdn_parity(2, 64, 70);
}

/// On CUDA, `forward_train_checkpointed` gives the `forward_train` output
/// and gradients.
#[test]
fn gdn_checkpointed_matches_plain_on_cuda() {
    let Some((_guard, _, cuda)) = devices() else {
        return;
    };
    let client = CudaRuntime::default_client(&cuda);
    for (key_heads, chunk_size) in [(2, 4), (4, 64)] {
        let cfg = gdn_cfg(key_heads, chunk_size);
        let weights = gdn_weights(&cfg, 0x6d4e_7200 + key_heads as u64);
        let plain = gdn_pass::<CudaRuntime>(&client, &cuda, &cfg, &weights, 9, false);
        let ckpt = gdn_pass::<CudaRuntime>(&client, &cuda, &cfg, &weights, 9, true);
        let case = format!(
            "gdn checkpointed rep={} chunk={chunk_size}",
            H_V / key_heads
        );
        assert_passes(&case, &ckpt, &plain, CKPT_RTOL, CKPT_RTOL);
    }
}

// ---- Qwen35Model ----

const QWEN_SEQ: usize = 9;

/// Logits and LoRA-factor gradients of the tiny hybrid on one backend.
fn qwen_pass<R>(client: &R::Client, device: &R::Device, checkpoint: bool) -> Pass
where
    R: Runtime<DType = DType>,
    R::Client: TrainClient<R>,
{
    let model: Qwen35Model<R> = tiny_lora_model::<R>(client, device, 0x3a7e_7000, 0x3a7e_7001);
    let ids: Vec<i64> = (0..QWEN_SEQ)
        .map(|i| ((i * 5 + 3) % VOCAB) as i64)
        .collect();
    let tokens = Tensor::<R>::from_slice(&ids, &[1, QWEN_SEQ], device).expect("token ids");
    let logits = model
        .forward_train(client, &tokens, checkpoint)
        .expect("Qwen35Model::forward_train");
    assert_eq!(logits.shape(), &[1, QWEN_SEQ, VOCAB]);
    let grads =
        backward(&weighted_loss::<R>(client, &logits, 0x3a7e_7002), client).expect("backward");
    let mut named = Vec::new();
    for (name, var) in model.named_parameters() {
        if var.requires_grad() {
            assert!(
                name.ends_with("lora_a") || name.ends_with("lora_b"),
                "only LoRA factors train: {name}"
            );
            let values = grad_host(&grads, var.id(), &name);
            named.push((name, values));
        }
    }
    assert!(!named.is_empty(), "the model carries trainable factors");
    Pass {
        y: host(logits.tensor()),
        grads: named,
    }
}

#[test]
fn qwen35_lora_forward_train_matches_cpu() {
    let Some((_guard, cpu, cuda)) = devices() else {
        return;
    };
    let cpu_client = CpuRuntime::default_client(&cpu);
    let cuda_client = CudaRuntime::default_client(&cuda);
    let want = qwen_pass::<CpuRuntime>(&cpu_client, &cpu, false);
    let got = qwen_pass::<CudaRuntime>(&cuda_client, &cuda, false);
    assert_passes("qwen35 lora", &got, &want, FWD_RTOL, GRAD_RTOL);
}

/// Per-layer checkpointing on CUDA gives the plain logits and gradients.
#[test]
fn qwen35_checkpointed_layers_match_plain_on_cuda() {
    let Some((_guard, _, cuda)) = devices() else {
        return;
    };
    let client = CudaRuntime::default_client(&cuda);
    let plain = qwen_pass::<CudaRuntime>(&client, &cuda, false);
    let ckpt = qwen_pass::<CudaRuntime>(&client, &cuda, true);
    assert_passes("qwen35 checkpointed", &ckpt, &plain, CKPT_RTOL, CKPT_RTOL);
}

// ---- TrainableLstm and TrainableGru ----

const RNN_BATCH: usize = 2;
const RNN_STEPS: usize = 7;

/// Two bidirectional batch-first layers with bias.
fn rnn_cfg() -> RnnConfig {
    RnnConfig {
        input_size: 5,
        hidden_size: 6,
        num_layers: 2,
        bias: true,
        batch_first: true,
        dropout: 0.0,
        bidirectional: true,
    }
}

/// Host weights of a seeded CPU `init`, under the PyTorch names `load` reads.
fn rnn_weights<K: CellKind>(cfg: &RnnConfig, seed: u64) -> HostWeights {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let mut map = VarMap::<CpuRuntime>::new();
    let mut vb = VarBuilder::new(&mut map, &device).with_seed(seed);
    let stack = RnnStack::<CpuRuntime, K>::init(&mut vb, cfg.clone(), DType::F32, &client, true)
        .expect("RnnStack::init");
    stack
        .named_parameters()
        .into_iter()
        .map(|(name, var)| (name, var.shape().to_vec(), host(var.tensor())))
        .collect()
}

fn load_rnn<R, K>(cfg: &RnnConfig, weights: &HostWeights, device: &R::Device) -> RnnStack<R, K>
where
    R: Runtime<DType = DType>,
    K: CellKind,
{
    let mut map = varmap::<R>(weights, device);
    let vb = VarBuilder::new(&mut map, device);
    RnnStack::<R, K>::load(&vb, cfg.clone(), true).expect("RnnStack::load")
}

/// Input `[B, T, I]` and one initial state `[layers * dirs, B, H]` per
/// entry of `states`, each requiring grad.
fn rnn_inputs<R: Runtime<DType = DType>>(
    cfg: &RnnConfig,
    states: usize,
    device: &R::Device,
) -> (Var<R>, Vec<Var<R>>) {
    let x_shape = [RNN_BATCH, RNN_STEPS, cfg.input_size];
    let x_data = Lcg(0x4c53_7000).vec(x_shape.iter().product(), 1.0);
    let x = Var::new(upload::<R>(&x_data, &x_shape, device), true);
    let s_shape = [
        cfg.num_layers * cfg.num_directions(),
        RNN_BATCH,
        cfg.hidden_size,
    ];
    let s = (0..states)
        .map(|k| {
            let data = Lcg(0x4c53_7010 + k as u64).vec(s_shape.iter().product(), 0.5);
            Var::new(upload::<R>(&data, &s_shape, device), true)
        })
        .collect();
    (x, s)
}

/// Outputs concatenated, and the gradients of `x`, each initial state and
/// every parameter, for the loss `sum` of `sum(out_k * w_k)`.
fn rnn_grads<R>(
    client: &R::Client,
    outputs: &[&Var<R>],
    inputs: &[(&str, &Var<R>)],
    params: Vec<(String, &Var<R>)>,
) -> Pass
where
    R: Runtime<DType = DType>,
    R::Client: TrainClient<R>,
{
    let mut loss = weighted_loss::<R>(client, outputs[0], 0x4c53_7100);
    for (k, out) in outputs.iter().enumerate().skip(1) {
        let term = weighted_loss::<R>(client, out, 0x4c53_7100 + k as u64);
        loss = var_add(&loss, &term, client).expect("sum the loss terms");
    }
    let grads = backward(&loss, client).expect("backward");
    let mut named: Vec<(String, Vec<f32>)> = inputs
        .iter()
        .map(|(name, var)| (name.to_string(), grad_host(&grads, var.id(), name)))
        .collect();
    for (name, var) in params {
        let values = grad_host(&grads, var.id(), &name);
        named.push((name, values));
    }
    let y = outputs.iter().flat_map(|out| host(out.tensor())).collect();
    Pass { y, grads: named }
}

fn lstm_pass<R>(client: &R::Client, device: &R::Device, weights: &HostWeights) -> Pass
where
    R: Runtime<DType = DType>,
    R::Client: TrainClient<R>,
{
    let cfg = rnn_cfg();
    let lstm = load_rnn::<R, LstmKind>(&cfg, weights, device);
    let (x, s) = rnn_inputs::<R>(&cfg, 2, device);
    let (out, (h_n, c_n)) = lstm
        .forward(client, &x, Some((&s[0], &s[1])))
        .expect("TrainableLstm::forward");
    let inputs = [("x", &x), ("h0", &s[0]), ("c0", &s[1])];
    rnn_grads::<R>(
        client,
        &[&out, &h_n, &c_n],
        &inputs,
        lstm.named_parameters(),
    )
}

fn gru_pass<R>(client: &R::Client, device: &R::Device, weights: &HostWeights) -> Pass
where
    R: Runtime<DType = DType>,
    R::Client: TrainClient<R>,
{
    let cfg = rnn_cfg();
    let gru = load_rnn::<R, GruKind>(&cfg, weights, device);
    let (x, s) = rnn_inputs::<R>(&cfg, 1, device);
    let (out, h_n) = gru
        .forward(client, &x, Some(&s[0]))
        .expect("TrainableGru::forward");
    let inputs = [("x", &x), ("h0", &s[0])];
    rnn_grads::<R>(client, &[&out, &h_n], &inputs, gru.named_parameters())
}

#[test]
fn lstm_forward_and_backward_match_cpu() {
    let Some((_guard, cpu, cuda)) = devices() else {
        return;
    };
    let weights = rnn_weights::<LstmKind>(&rnn_cfg(), 0x4c53_7200);
    let want = lstm_pass::<CpuRuntime>(&CpuRuntime::default_client(&cpu), &cpu, &weights);
    let got = lstm_pass::<CudaRuntime>(&CudaRuntime::default_client(&cuda), &cuda, &weights);
    assert_passes("lstm", &got, &want, FWD_RTOL, GRAD_RTOL);
}

#[test]
fn gru_forward_and_backward_match_cpu() {
    let Some((_guard, cpu, cuda)) = devices() else {
        return;
    };
    let weights = rnn_weights::<GruKind>(&rnn_cfg(), 0x4c53_7300);
    let want = gru_pass::<CpuRuntime>(&CpuRuntime::default_client(&cpu), &cpu, &weights);
    let got = gru_pass::<CudaRuntime>(&CudaRuntime::default_client(&cuda), &cuda, &weights);
    assert_passes("gru", &got, &want, FWD_RTOL, GRAD_RTOL);
}
