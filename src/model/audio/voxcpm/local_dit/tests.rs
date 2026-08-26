//! Tests for [`LocalDit::forward`] — the estimator forward pass.
//!
//! Weights are tiny and synthetic; these pin SHAPE and the output SLICE
//! WINDOW, which are the two things the reference makes easy to get wrong.
//!
//! [`model`], [`t`] and the dimension constants are `pub(super)` so the
//! sampler's tests integrate this same tiny estimator instead of rebuilding
//! one.

use crate::model::audio::voxcpm::bidirectional::attention::BidirectionalAttention;
use crate::model::audio::voxcpm::bidirectional::layer::BidirectionalLayer;
use crate::model::audio::voxcpm::bidirectional::mlp::BidirectionalMlp;
use crate::model::audio::voxcpm::local_dit::loader::LocalDit;
use crate::nn::{Linear, RmsNorm, RoPE, SinusoidalPosEmb, TimestepEmbedding};
use crate::test_utils::cpu_setup;
use numr::autograd::Var;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

pub(super) const FEAT_DIM: usize = 3;
pub(super) const PATCH_SIZE: usize = 2;
pub(super) const HIDDEN_DIM: usize = 8;
const FFN_DIM: usize = 8;
const NUM_HEADS: usize = 2;
const NUM_KV_HEADS: usize = 1;
const HEAD_DIM: usize = 4;
/// `mu(2) + t(1) + cond(2) + x(2)` — the same derivation as
/// `LocalDitConfig::sequence_len`, at `PATCH_SIZE = 2`.
const SEQUENCE_LEN: usize = 2 + 1 + PATCH_SIZE + PATCH_SIZE;
pub(super) const MU_TOKENS: usize = 2;

/// Deterministic non-degenerate values: a constant fill would make every
/// position identical and hide a wrong slice window.
pub(super) fn t(shape: &[usize], seed: f32, device: &CpuDevice) -> Tensor<CpuRuntime> {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| 0.4 * ((i as f32) * 0.37 + seed).sin())
        .collect();
    Tensor::<CpuRuntime>::from_slice(&data, shape, device).unwrap()
}

fn linear(out: usize, inp: usize, seed: f32, bias: bool, device: &CpuDevice) -> Linear<CpuRuntime> {
    let b = bias.then(|| t(&[out], seed + 5.0, device));
    Linear::new(t(&[out, inp], seed, device), b, false)
}

fn norm(device: &CpuDevice) -> RmsNorm<CpuRuntime> {
    let ones =
        Tensor::<CpuRuntime>::from_slice(&[1.0f32; HIDDEN_DIM], &[HIDDEN_DIM], device).unwrap();
    RmsNorm::new(ones, 1e-5, false)
}

fn layer(seed: f32, device: &CpuDevice) -> BidirectionalLayer<CpuRuntime> {
    BidirectionalLayer {
        input_layernorm: norm(device),
        self_attn: BidirectionalAttention {
            q_proj: linear(NUM_HEADS * HEAD_DIM, HIDDEN_DIM, seed + 1.0, false, device),
            k_proj: linear(
                NUM_KV_HEADS * HEAD_DIM,
                HIDDEN_DIM,
                seed + 2.0,
                false,
                device,
            ),
            v_proj: linear(
                NUM_KV_HEADS * HEAD_DIM,
                HIDDEN_DIM,
                seed + 3.0,
                false,
                device,
            ),
            o_proj: linear(HIDDEN_DIM, NUM_HEADS * HEAD_DIM, seed + 4.0, false, device),
            num_heads: NUM_HEADS,
            num_kv_heads: NUM_KV_HEADS,
            head_dim: HEAD_DIM,
        },
        post_attention_layernorm: norm(device),
        mlp: BidirectionalMlp {
            gate_proj: linear(FFN_DIM, HIDDEN_DIM, seed + 6.0, false, device),
            up_proj: linear(FFN_DIM, HIDDEN_DIM, seed + 7.0, false, device),
            down_proj: linear(HIDDEN_DIM, FFN_DIM, seed + 8.0, false, device),
        },
    }
}

/// `num_layers = 0` builds the same model minus the transformer stack — the
/// only way to observe the slice window in isolation (see
/// `slice_window_keeps_only_the_trailing_x_positions`).
pub(super) fn model(num_layers: usize, device: &CpuDevice) -> LocalDit<CpuRuntime> {
    let rope = RoPE::<CpuRuntime>::precompute_freqs(32, HEAD_DIM, 10000.0, None, device)
        .unwrap()
        .narrow_positions(SEQUENCE_LEN)
        .unwrap();
    LocalDit {
        in_proj: linear(HIDDEN_DIM, FEAT_DIM, 0.1, true, device),
        cond_proj: linear(HIDDEN_DIM, FEAT_DIM, 0.2, true, device),
        out_proj: linear(FEAT_DIM, HIDDEN_DIM, 0.3, true, device),
        time_mlp: TimestepEmbedding::new(
            linear(HIDDEN_DIM, HIDDEN_DIM, 0.4, true, device),
            linear(HIDDEN_DIM, HIDDEN_DIM, 0.5, true, device),
        ),
        delta_time_mlp: TimestepEmbedding::new(
            linear(HIDDEN_DIM, HIDDEN_DIM, 0.6, true, device),
            linear(HIDDEN_DIM, HIDDEN_DIM, 0.7, true, device),
        ),
        layers: (0..num_layers).map(|i| layer(i as f32, device)).collect(),
        norm: norm(device),
        rope,
        time_embeddings: SinusoidalPosEmb::<CpuRuntime>::new(HIDDEN_DIM, device).unwrap(),
        hidden_dim: HIDDEN_DIM,
        feat_dim: FEAT_DIM,
        patch_size: PATCH_SIZE,
    }
}

struct Inputs {
    x: Var<CpuRuntime>,
    mu: Var<CpuRuntime>,
    t: Var<CpuRuntime>,
    cond: Var<CpuRuntime>,
    dt: Var<CpuRuntime>,
}

fn inputs(batch: usize, x_seed: f32, cond_seed: f32, device: &CpuDevice) -> Inputs {
    Inputs {
        x: Var::new(t(&[batch, FEAT_DIM, PATCH_SIZE], x_seed, device), false),
        mu: Var::new(t(&[batch, MU_TOKENS * HIDDEN_DIM], 1.3, device), false),
        t: Var::new(t(&[batch], 2.1, device), false),
        cond: Var::new(t(&[batch, FEAT_DIM, PATCH_SIZE], cond_seed, device), false),
        // `dt = 0` is the inference value, and it is NOT a no-op branch.
        dt: Var::new(
            Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; batch], &[batch], device).unwrap(),
            false,
        ),
    }
}

fn run(client: &CpuClient, model: &LocalDit<CpuRuntime>, i: &Inputs) -> Vec<f32> {
    let out = model
        .forward(client, &i.x, &i.mu, &i.t, &i.cond, &i.dt)
        .unwrap();
    assert_eq!(out.shape(), &[i.x.shape()[0], FEAT_DIM, PATCH_SIZE]);
    out.tensor().contiguous().unwrap().to_vec()
}

#[test]
fn output_shape_is_batch_feat_dim_patch_size() {
    let (client, device) = cpu_setup();
    let m = model(2, &device);
    let out = run(&client, &m, &inputs(3, 0.9, 1.7, &device));
    assert_eq!(out.len(), 3 * FEAT_DIM * PATCH_SIZE);
}

/// The slice window is `prefix + mu_tokens + 1 ..`, i.e. exactly the trailing
/// `x` positions. With NO transformer layers nothing mixes across positions,
/// so the returned window must depend on `x` alone: change `x` and the output
/// moves, change `cond` and it does not. A wrong window (e.g. starting at the
/// `cond` block, or including `mu`/`t`) flips both assertions.
#[test]
fn slice_window_keeps_only_the_trailing_x_positions() {
    let (client, device) = cpu_setup();
    let m = model(0, &device);

    let base = run(&client, &m, &inputs(2, 0.9, 1.7, &device));
    let other_x = run(&client, &m, &inputs(2, 4.5, 1.7, &device));
    let other_cond = run(&client, &m, &inputs(2, 0.9, 6.2, &device));

    let max_delta = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b.iter())
            .map(|(p, q)| (p - q).abs())
            .fold(0.0f32, f32::max)
    };
    assert!(
        max_delta(&base, &other_x) > 1e-4,
        "output must respond to x: base={base:?} other={other_x:?}"
    );
    assert!(
        max_delta(&base, &other_cond) < 1e-6,
        "with no layers the x window cannot see cond: base={base:?} other={other_cond:?}"
    );
}

/// With the bidirectional stack in place every position attends every other,
/// so `cond` DOES reach the `x` window. Guards against a "fix" that drops
/// `cond` (or `mu`/`t`) from the assembled sequence entirely.
#[test]
fn cond_reaches_the_x_window_through_the_bidirectional_stack() {
    let (client, device) = cpu_setup();
    let m = model(2, &device);

    let base = run(&client, &m, &inputs(2, 0.9, 1.7, &device));
    let other_cond = run(&client, &m, &inputs(2, 0.9, 6.2, &device));
    let max_delta = base
        .iter()
        .zip(other_cond.iter())
        .map(|(p, q)| (p - q).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_delta > 1e-5,
        "cond must influence the x window: base={base:?} other={other_cond:?}"
    );
}

/// `dt = 0` is not a dead branch: `SinusoidalPosEmb(0) = [0..0, 1..1]`, so
/// `delta_time_mlp` adds a real constant. Changing `dt` must change the
/// output.
#[test]
fn dt_branch_contributes() {
    let (client, device) = cpu_setup();
    let m = model(1, &device);

    let mut i = inputs(2, 0.9, 1.7, &device);
    let base = run(&client, &m, &i);
    i.dt = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[0.25f32, 0.5], &[2], &device).unwrap(),
        false,
    );
    let shifted = run(&client, &m, &i);
    let max_delta = base
        .iter()
        .zip(shifted.iter())
        .map(|(p, q)| (p - q).abs())
        .fold(0.0f32, f32::max);
    assert!(max_delta > 1e-5, "dt must change the output");
}

#[test]
fn rejects_wrong_shapes() {
    let (client, device) = cpu_setup();
    let m = model(1, &device);
    let good = inputs(2, 0.9, 1.7, &device);

    // x is 2D, not [batch, feat_dim, patch_size].
    let bad_x = Var::new(t(&[2, FEAT_DIM], 0.9, &device), false);
    assert!(
        m.forward(&client, &bad_x, &good.mu, &good.t, &good.cond, &good.dt)
            .is_err()
    );

    // cond's patch axis is wrong.
    let bad_cond = Var::new(t(&[2, FEAT_DIM, PATCH_SIZE + 1], 1.7, &device), false);
    assert!(
        m.forward(&client, &good.x, &good.mu, &good.t, &bad_cond, &good.dt)
            .is_err()
    );

    // mu's width is not a multiple of hidden_dim.
    let bad_mu = Var::new(t(&[2, MU_TOKENS * HIDDEN_DIM + 1], 1.3, &device), false);
    assert!(
        m.forward(&client, &good.x, &bad_mu, &good.t, &good.cond, &good.dt)
            .is_err()
    );

    // t has the wrong batch.
    let bad_t = Var::new(t(&[3], 2.1, &device), false);
    assert!(
        m.forward(&client, &good.x, &good.mu, &bad_t, &good.cond, &good.dt)
            .is_err()
    );

    // dt is 2D, not [batch].
    let bad_dt = Var::new(t(&[2, 1], 0.0, &device), false);
    assert!(
        m.forward(&client, &good.x, &good.mu, &good.t, &good.cond, &bad_dt)
            .is_err()
    );
}
