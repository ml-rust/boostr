//! Parity of the fused CUDA Gated DeltaNet decode step against the generic
//! `gdn_step_impl` run on the same CUDA client, plus the `S_k = 96` fallback
//! that the fused kernel does not cover.
//!
//! Run with:
//!   cd boostr && cargo test --features cuda --test gdn_step_fused_cuda

#![cfg(feature = "cuda")]

use std::sync::{Mutex, OnceLock};

use boostr::ops::cuda::architecture::gdn_step_fused;
use boostr::ops::impl_generic::architecture::gated_delta_net::gdn_step_impl;
use boostr::ops::traits::architecture::gated_delta_net::GatedDeltaNetOps;
use numr::runtime::Runtime;
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
use numr::tensor::Tensor;

static CUDA_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn cuda_lock() -> std::sync::MutexGuard<'static, ()> {
    CUDA_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|p| p.into_inner())
}

fn cuda_setup() -> (CudaClient, CudaDevice) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    (client, device)
}

/// Deterministic pseudo-random values in about `[-amp, amp]`, distinct per
/// index and per seed.
fn values(len: usize, seed: f32, amp: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = (i as f32) * 0.013 + seed;
            (x.sin() * 0.7 + (x * 3.1).cos() * 0.3) * amp
        })
        .collect()
}

struct Inputs {
    q: Tensor<CudaRuntime>,
    k: Tensor<CudaRuntime>,
    v: Tensor<CudaRuntime>,
    g: Tensor<CudaRuntime>,
    beta: Tensor<CudaRuntime>,
    state: Tensor<CudaRuntime>,
}

fn inputs(b: usize, h: usize, s_k: usize, s_v: usize, device: &CudaDevice) -> Inputs {
    let t = |data: &[f32], shape: &[usize]| {
        Tensor::<CudaRuntime>::from_slice(data, shape, device).expect("fixture tensor")
    };
    // g is a log decay, so keep it non-positive; beta is a gate in (0, 1).
    let g: Vec<f32> = values(b * h, 0.3, 1.0)
        .into_iter()
        .map(|x| -x.abs())
        .collect();
    let beta: Vec<f32> = values(b * h, 0.9, 1.0)
        .into_iter()
        .map(|x| 0.5 + 0.45 * x)
        .collect();
    Inputs {
        q: t(&values(b * h * s_k, 0.1, 1.0), &[b, 1, h, s_k]),
        k: t(&values(b * h * s_k, 0.2, 1.0), &[b, 1, h, s_k]),
        v: t(&values(b * h * s_v, 0.4, 1.0), &[b, 1, h, s_v]),
        g: t(&g, &[b, 1, h]),
        beta: t(&beta, &[b, 1, h]),
        state: t(&values(b * h * s_k * s_v, 0.7, 0.5), &[b, h, s_k, s_v]),
    }
}

/// Asserts `max |got - want| <= 1e-5 * (1 + max |want|)`.
fn assert_parity(name: &str, got: &Tensor<CudaRuntime>, want: &Tensor<CudaRuntime>) {
    assert_eq!(got.shape(), want.shape(), "{name}: shape");
    let got = got.to_vec::<f32>();
    let want = want.to_vec::<f32>();
    let max_abs = want.iter().fold(0f32, |m, x| m.max(x.abs()));
    let tol = 1e-5 * (1.0 + max_abs);
    let mut worst = (0usize, 0f32);
    for (i, (a, b)) in got.iter().zip(&want).enumerate() {
        assert!(a.is_finite(), "{name}: non-finite at {i}: {a}");
        let d = (a - b).abs();
        if d > worst.1 {
            worst = (i, d);
        }
    }
    assert!(
        worst.1 <= tol,
        "{name}: max abs diff {} at {} exceeds {tol} (got {}, want {})",
        worst.1,
        worst.0,
        got[worst.0],
        want[worst.0]
    );
}

fn run_case(b: usize, h: usize, s_k: usize, s_v: usize) {
    let _guard = cuda_lock();
    let (client, device) = cuda_setup();
    let x = inputs(b, h, s_k, s_v, &device);

    let (o_ref, s_ref) = gdn_step_impl(&client, &x.q, &x.k, &x.v, &x.g, &x.beta, &x.state)
        .expect("generic gdn_step_impl");
    let (o_fused, s_fused) =
        gdn_step_fused(&client, &x.q, &x.k, &x.v, &x.g, &x.beta, &x.state).expect("fused gdn_step");
    let case = format!("({b},{h},{s_k},{s_v})");
    assert_parity(&format!("{case} fused o"), &o_fused, &o_ref);
    assert_parity(&format!("{case} fused state"), &s_fused, &s_ref);

    // The trait entry point routes to the same kernel.
    let (o_trait, s_trait) = client
        .gdn_step(&x.q, &x.k, &x.v, &x.g, &x.beta, &x.state)
        .expect("trait gdn_step");
    assert_parity(&format!("{case} trait o"), &o_trait, &o_ref);
    assert_parity(&format!("{case} trait state"), &s_trait, &s_ref);
}

#[test]
fn fused_matches_generic_b1_h48_sk128_sv128() {
    run_case(1, 48, 128, 128);
}

#[test]
fn fused_matches_generic_b2_h4_sk64_sv32() {
    run_case(2, 4, 64, 32);
}

#[test]
fn fused_matches_generic_b1_h3_sk32_sv256() {
    run_case(1, 3, 32, 256);
}

/// `S_v = 512` spans two column chunks per (batch, head).
#[test]
fn fused_matches_generic_b1_h2_sk128_sv512() {
    run_case(1, 2, 128, 512);
}

/// `S_k = 96` has no fused instantiation: the fused entry refuses it and
/// the trait entry falls back to the generic path with correct results.
#[test]
fn sk96_falls_back_to_generic() {
    let _guard = cuda_lock();
    let (client, device) = cuda_setup();
    let x = inputs(2, 3, 96, 64, &device);

    assert!(gdn_step_fused(&client, &x.q, &x.k, &x.v, &x.g, &x.beta, &x.state).is_err());

    let (o_ref, s_ref) = gdn_step_impl(&client, &x.q, &x.k, &x.v, &x.g, &x.beta, &x.state)
        .expect("generic gdn_step_impl");
    let (o, s) = client
        .gdn_step(&x.q, &x.k, &x.v, &x.g, &x.beta, &x.state)
        .expect("trait gdn_step fallback");
    assert_parity("sk96 o", &o, &o_ref);
    assert_parity("sk96 state", &s, &s_ref);
}

/// Non-contiguous inputs are made contiguous before the launch.
#[test]
fn fused_accepts_strided_inputs() {
    let _guard = cuda_lock();
    let (client, device) = cuda_setup();
    let (b, h, s_k, s_v) = (1, 4, 64, 64);
    let x = inputs(b, h, s_k, s_v, &device);
    // A permuted-then-permuted-back view: same shape, non-contiguous strides.
    let state_view = x
        .state
        .transpose(2, 3)
        .expect("transpose")
        .contiguous()
        .expect("contiguous")
        .transpose(2, 3)
        .expect("transpose back");
    assert_eq!(state_view.shape(), x.state.shape());

    let (o_ref, s_ref) = gdn_step_impl(&client, &x.q, &x.k, &x.v, &x.g, &x.beta, &x.state)
        .expect("generic gdn_step_impl");
    let (o, s) = gdn_step_fused(&client, &x.q, &x.k, &x.v, &x.g, &x.beta, &state_view)
        .expect("fused gdn_step on strided state");
    assert_parity("strided o", &o, &o_ref);
    assert_parity("strided state", &s, &s_ref);
}
