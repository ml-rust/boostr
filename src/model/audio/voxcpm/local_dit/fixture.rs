//! The tiny synthetic estimator shared by the sampler's, the bidirectional
//! layer's, the orchestrator's and the trainer's tests, so none of them
//! rebuilds one. Reachable crate-wide as `local_dit::tests`.
//!
//! Every builder has a runtime-generic `*_on::<R>` form and a `CpuRuntime`
//! wrapper: the CUDA graph tests need the SAME weights on a CUDA device, and
//! the CPU wrappers keep every existing call site's inference intact (a
//! `&CpuDevice` argument alone does not pin `R`).

use super::loader::LocalDit;
use crate::model::audio::voxcpm::bidirectional::attention::BidirectionalAttention;
use crate::model::audio::voxcpm::bidirectional::layer::BidirectionalLayer;
use crate::model::audio::voxcpm::bidirectional::mlp::BidirectionalMlp;
use crate::nn::{
    MaybeLoraLinear, MaybeQuantLinear, RmsNorm, RoPE, SinusoidalPosEmb, TimestepEmbedding, Weight,
};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

pub(crate) const FEAT_DIM: usize = 3;
pub(crate) const PATCH_SIZE: usize = 2;
pub(crate) const HIDDEN_DIM: usize = 8;
const FFN_DIM: usize = 8;
pub(crate) const NUM_HEADS: usize = 2;
pub(crate) const NUM_KV_HEADS: usize = 1;
/// The smallest head size the CUDA flash kernels instantiate, so the CUDA
/// graph tests run the same attention entry the production block runs.
pub(crate) const HEAD_DIM: usize = 32;
/// `mu(2) + t(1) + cond(2) + x(2)` — the same derivation as
/// `LocalDitConfig::sequence_len`, at `PATCH_SIZE = 2`.
const SEQUENCE_LEN: usize = 2 + 1 + PATCH_SIZE + PATCH_SIZE;
pub(crate) const MU_TOKENS: usize = 2;

/// Deterministic non-degenerate values: a constant fill would make every
/// position identical and hide a wrong slice window.
pub(crate) fn t_on<R: Runtime<DType = DType>>(
    shape: &[usize],
    seed: f32,
    device: &R::Device,
) -> Tensor<R> {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| 0.4 * ((i as f32) * 0.37 + seed).sin())
        .collect();
    Tensor::<R>::from_slice(&data, shape, device).unwrap()
}

pub(crate) fn t(shape: &[usize], seed: f32, device: &CpuDevice) -> Tensor<CpuRuntime> {
    t_on::<CpuRuntime>(shape, seed, device)
}

/// `q_proj`/`k_proj` for the fixture's attention: [`linear_on`] values
/// scaled by `1/sqrt(HEAD_DIM)`. At `HEAD_DIM` 32 the unscaled `0.4 * sin`
/// weights push every logit past the softmax's flat region, each token
/// attends only itself, and the gradient tests see no signal through
/// attention.
fn qk_proj_on<R: Runtime<DType = DType>>(
    out: usize,
    seed: f32,
    device: &R::Device,
) -> MaybeLoraLinear<R> {
    let scale = (HEAD_DIM as f32).sqrt().recip();
    let n = out * HIDDEN_DIM;
    let data: Vec<f32> = (0..n)
        .map(|i| scale * 0.4 * ((i as f32) * 0.37 + seed).sin())
        .collect();
    let w = Tensor::<R>::from_slice(&data, &[out, HIDDEN_DIM], device).unwrap();
    MaybeQuantLinear::from_weight(Weight::Standard(w), None).into()
}

/// Always the `Standard` variant: a safetensors checkpoint yields exactly
/// this, and it is the arm every assertion is written against.
pub(crate) fn linear_on<R: Runtime<DType = DType>>(
    out: usize,
    inp: usize,
    seed: f32,
    bias: bool,
    device: &R::Device,
) -> MaybeLoraLinear<R> {
    let b = bias.then(|| t_on::<R>(&[out], seed + 5.0, device));
    MaybeQuantLinear::from_weight(Weight::Standard(t_on::<R>(&[out, inp], seed, device)), b).into()
}

pub(crate) fn linear(
    out: usize,
    inp: usize,
    seed: f32,
    bias: bool,
    device: &CpuDevice,
) -> MaybeLoraLinear<CpuRuntime> {
    linear_on::<CpuRuntime>(out, inp, seed, bias, device)
}

pub(crate) fn norm_on<R: Runtime<DType = DType>>(device: &R::Device) -> RmsNorm<R> {
    let ones = Tensor::<R>::from_slice(&[1.0f32; HIDDEN_DIM], &[HIDDEN_DIM], device).unwrap();
    RmsNorm::new(ones, 1e-5, false)
}

pub(crate) fn norm(device: &CpuDevice) -> RmsNorm<CpuRuntime> {
    norm_on::<CpuRuntime>(device)
}

pub(crate) fn layer_on<R: Runtime<DType = DType>>(
    seed: f32,
    device: &R::Device,
) -> BidirectionalLayer<R> {
    BidirectionalLayer {
        input_layernorm: norm_on::<R>(device),
        self_attn: BidirectionalAttention {
            q_proj: qk_proj_on::<R>(NUM_HEADS * HEAD_DIM, seed + 1.0, device),
            k_proj: qk_proj_on::<R>(NUM_KV_HEADS * HEAD_DIM, seed + 2.0, device),
            v_proj: linear_on::<R>(
                NUM_KV_HEADS * HEAD_DIM,
                HIDDEN_DIM,
                seed + 3.0,
                false,
                device,
            ),
            o_proj: linear_on::<R>(HIDDEN_DIM, NUM_HEADS * HEAD_DIM, seed + 4.0, false, device),
            num_heads: NUM_HEADS,
            num_kv_heads: NUM_KV_HEADS,
            head_dim: HEAD_DIM,
        },
        post_attention_layernorm: norm_on::<R>(device),
        mlp: BidirectionalMlp {
            gate_proj: linear_on::<R>(FFN_DIM, HIDDEN_DIM, seed + 6.0, false, device),
            up_proj: linear_on::<R>(FFN_DIM, HIDDEN_DIM, seed + 7.0, false, device),
            down_proj: linear_on::<R>(HIDDEN_DIM, FFN_DIM, seed + 8.0, false, device),
        },
    }
}

pub(crate) fn layer(seed: f32, device: &CpuDevice) -> BidirectionalLayer<CpuRuntime> {
    layer_on::<CpuRuntime>(seed, device)
}

/// `num_layers = 0` builds the same model minus the transformer stack — the
/// only way to observe the slice window in isolation (see
/// `slice_window_keeps_only_the_trailing_x_positions`).
pub(crate) fn model_on<R: Runtime<DType = DType>>(
    num_layers: usize,
    device: &R::Device,
) -> LocalDit<R> {
    let rope = RoPE::<R>::precompute_freqs(32, HEAD_DIM, 10000.0, None, device)
        .unwrap()
        .narrow_positions(SEQUENCE_LEN)
        .unwrap();
    LocalDit {
        in_proj: linear_on::<R>(HIDDEN_DIM, FEAT_DIM, 0.1, true, device),
        cond_proj: linear_on::<R>(HIDDEN_DIM, FEAT_DIM, 0.2, true, device),
        out_proj: linear_on::<R>(FEAT_DIM, HIDDEN_DIM, 0.3, true, device),
        time_mlp: TimestepEmbedding::new(
            linear_on::<R>(HIDDEN_DIM, HIDDEN_DIM, 0.4, true, device),
            linear_on::<R>(HIDDEN_DIM, HIDDEN_DIM, 0.5, true, device),
        ),
        delta_time_mlp: TimestepEmbedding::new(
            linear_on::<R>(HIDDEN_DIM, HIDDEN_DIM, 0.6, true, device),
            linear_on::<R>(HIDDEN_DIM, HIDDEN_DIM, 0.7, true, device),
        ),
        layers: (0..num_layers)
            .map(|i| layer_on::<R>(i as f32, device))
            .collect(),
        norm: norm_on::<R>(device),
        rope,
        time_embeddings: SinusoidalPosEmb::<R>::new(HIDDEN_DIM, device).unwrap(),
        hidden_dim: HIDDEN_DIM,
        feat_dim: FEAT_DIM,
        patch_size: PATCH_SIZE,
        activation_checkpointing: false,
        #[cfg(feature = "cuda")]
        euler_graphs: Default::default(),
    }
}

pub(crate) fn model(num_layers: usize, device: &CpuDevice) -> LocalDit<CpuRuntime> {
    model_on::<CpuRuntime>(num_layers, device)
}
