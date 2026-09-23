//! A row of a batched Gated DeltaNet decode step must carry the same bits as
//! that row decoded alone. Batching a request must never change what a
//! sequence produces, so the fused kernel's per-row reduction has to be
//! independent of how many rows ride along with it.
//!
//! Run with:
//!   cd boostr && cargo test --features cuda --test gdn_step_batch_invariance

#![cfg(feature = "cuda")]

use std::sync::{Mutex, OnceLock};

use boostr::ops::cuda::architecture::gdn_step_from_conv_fused;
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

struct ConvInputs {
    qkv: Tensor<CudaRuntime>,
    alpha: Tensor<CudaRuntime>,
    beta: Tensor<CudaRuntime>,
    dt_bias: Tensor<CudaRuntime>,
    ssm_a: Tensor<CudaRuntime>,
    state: Tensor<CudaRuntime>,
}

/// Raw conv output and gate projections for `b` rows. `ssm_a` is
/// `-exp(A_log)`, so the decay stays non-positive.
fn conv_inputs(
    b: usize,
    h_v: usize,
    h_k: usize,
    s_k: usize,
    s_v: usize,
    device: &CudaDevice,
) -> ConvInputs {
    let t = |data: &[f32], shape: &[usize]| {
        Tensor::<CudaRuntime>::from_slice(data, shape, device).expect("fixture tensor")
    };
    let (key_dim, value_dim) = (h_k * s_k, h_v * s_v);
    let ssm_a: Vec<f32> = values(h_v, 1.3, 1.0)
        .into_iter()
        .map(|x| -(0.5 + x.abs()).exp())
        .collect();
    ConvInputs {
        qkv: t(
            &values(b * (2 * key_dim + value_dim), 0.1, 1.5),
            &[b, 1, 2 * key_dim + value_dim],
        ),
        alpha: t(&values(b * h_v, 0.3, 3.0), &[b, 1, h_v]),
        beta: t(&values(b * h_v, 0.9, 3.0), &[b, 1, h_v]),
        dt_bias: t(&values(h_v, 1.1, 1.0), &[h_v]),
        ssm_a: t(&ssm_a, &[h_v]),
        state: t(&values(b * h_v * s_k * s_v, 0.7, 0.5), &[b, h_v, s_k, s_v]),
    }
}

/// Row `row` of a batched tensor, as its own one-row tensor.
fn row(t: &Tensor<CudaRuntime>, row: usize) -> Tensor<CudaRuntime> {
    t.narrow(0, row, 1)
        .expect("narrow row")
        .contiguous()
        .expect("contiguous row")
}

/// Asserts bit equality, naming the first differing element.
fn assert_bits(name: &str, got: &Tensor<CudaRuntime>, want: &Tensor<CudaRuntime>) {
    assert_eq!(got.shape(), want.shape(), "{name}: shape");
    let got = got.to_vec::<f32>();
    let want = want.to_vec::<f32>();
    for (i, (a, b)) in got.iter().zip(&want).enumerate() {
        assert!(
            a.to_bits() == b.to_bits(),
            "{name}: element {i} differs: got {a} ({:#010x}), want {b} ({:#010x})",
            a.to_bits(),
            b.to_bits()
        );
    }
}

fn run_case(b: usize, h_v: usize, h_k: usize, s_k: usize, s_v: usize) {
    let _guard = cuda_lock();
    let (client, device) = cuda_setup();
    let x = conv_inputs(b, h_v, h_k, s_k, s_v, &device);
    let (key_dim, value_dim) = (h_k * s_k, h_v * s_v);
    let eps = 1e-6f32;

    let (o_batched, s_batched) = gdn_step_from_conv_fused(
        &client, &x.qkv, &x.alpha, &x.beta, &x.dt_bias, &x.ssm_a, &x.state, h_k, key_dim,
        value_dim, eps,
    )
    .expect("batched gdn_step_from_conv");

    for r in 0..b {
        let (o_solo, s_solo) = gdn_step_from_conv_fused(
            &client,
            &row(&x.qkv, r),
            &row(&x.alpha, r),
            &row(&x.beta, r),
            &x.dt_bias,
            &x.ssm_a,
            &row(&x.state, r),
            h_k,
            key_dim,
            value_dim,
            eps,
        )
        .expect("solo gdn_step_from_conv");
        let case = format!("({b},{h_v},{h_k},{s_k},{s_v}) row {r}");
        assert_bits(&format!("{case} o"), &row(&o_batched, r), &o_solo);
        assert_bits(&format!("{case} state"), &row(&s_batched, r), &s_solo);
    }
}

/// The head count a hybrid model decodes with, at the batch sizes a server
/// admits.
#[test]
fn rows_match_solo_hv48_hk16_sk128_sv128() {
    for b in [2, 5, 8] {
        run_case(b, 48, 16, 128, 128);
    }
}

/// `S_v = 512` spans two column chunks per (row, head), so a row's output
/// is written by more than one block.
#[test]
fn rows_match_solo_multi_chunk_sv512() {
    run_case(3, 2, 2, 128, 512);
}

/// A value head count that does not divide the batch, so `b = bh / h_v`
/// has to be what separates the rows.
#[test]
fn rows_match_solo_hv3_hk3_sk32_sv256() {
    run_case(4, 3, 3, 32, 256);
}
