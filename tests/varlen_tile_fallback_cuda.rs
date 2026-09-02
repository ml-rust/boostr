//! Regression + numerical-parity tests for the varlen attention large/small
//! tile fallback added in `src/ops/cuda/attention/varlen_attention_block_config.rs`.
//!
//! Before this file, `varlen_attention.rs::block_config` always returned the
//! large tile (`BLOCK_M=128, BLOCK_N=64`) for head_dim 64 and 128, with no
//! smaller kernel compiled as a fallback. Calling the public
//! `varlen_attention_fwd` with `head_dim=128, dtype=F32` therefore failed on
//! any device whose opt-in shared-memory limit is under ~132KB (forward) /
//! ~198KB (backward) — for example, "shared memory 129KB exceeds device
//! limit 99KB" on a 99KB device. `head_dim_128_f32_fwd_and_bwd_now_succeed`
//! is exactly that call.
//!
//! The remaining tests prove the new `_small` kernel instantiations compute
//! the same thing as the proven large-tile kernels, not merely that they
//! launch: `head_dim=64` F32 and `head_dim=128` F16 both fit the large tile
//! on a ~99KB device, so both tiles can be forced and compared directly.
//! Tile forcing goes through `varlen_attention_{fwd,bwd}_with_tile_for_test`,
//! explicit-parameter entry points that bottom out in the same inner
//! function the production `VarLenAttentionOps` trait impl calls — NOT a
//! process-wide env var, since Rust test binaries run every `#[test]`
//! function on its own thread within one process and setting such a var from
//! a test would race with every other test reading or setting it.
//!
//! Run with:
//!   cd boostr && cargo test --features cuda --test varlen_tile_fallback_cuda

#![cfg(feature = "cuda")]

use std::sync::{Mutex, OnceLock};

use boostr::ops::cuda::attention::varlen_attention_bwd::varlen_attention_bwd_with_tile_for_test;
use boostr::ops::cuda::attention::varlen_attention_fwd::varlen_attention_fwd_with_tile_for_test;
use boostr::ops::traits::attention::varlen_attention::VarLenAttentionOps;
use numr::dtype::DType;
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

fn cuda_available() -> bool {
    numr::runtime::cuda::is_cuda_available()
}

fn cuda_setup() -> (CudaClient, CudaDevice) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    (client, device)
}

/// Deterministic pseudo-random values, distinct per index and per seed.
fn values(len: usize, seed: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = (i as f32) * 0.017 + seed;
            x.sin() * 0.9 + (x * 2.3).cos() * 0.4
        })
        .collect()
}

/// F32 tile-parity tolerance: both tiles run the identical online-softmax
/// formula in `varlen_flash_attention_{fwd,bwd}_fp32_impl` on the same
/// device, differing only in how the K/V loop is chunked (BLOCK_N) and how
/// many Q rows a block covers (BLOCK_M). Each K-tile boundary rescales the
/// running max/sum by roughly one `f32::EPSILON` of relative error; with
/// `seq_len_k` in the low hundreds and BLOCK_N as small as 16, that is at
/// most a few hundred boundary rescalings, so 1000x `f32::EPSILON` is a
/// generous bound with headroom, not a widened-to-pass number.
const F32_TOL: f32 = 1000.0 * f32::EPSILON;

/// F16 tile-parity tolerance: Q/K/V are quantized to `half` once, identically
/// for both tile runs (input storage does not depend on the tile), and the
/// dot-product/softmax accumulation inside the kernel is entirely `float`
/// (see `varlen_flash_attention_fwd_fp16_impl`'s `float O_local[HEAD_DIM]`,
/// `float m_local`, `float l_local`) — only the final `__float2half` write
/// rounds to half. So the accumulated cross-tile floating-point-order error
/// is the same small `f32`-scale noise as the F32 case, plus at most one
/// half-ULP if that noise pushes a value across a rounding boundary. Half's
/// ULP at unit magnitude is `f16::EPSILON` (2^-10); 4x that covers a
/// boundary-adjacent rounding flip plus the same margin used for F32_TOL.
const F16_TOL: f32 = F32_TOL + 4.0 * 0.0009765625;

fn assert_close(a: &[f32], b: &[f32], label: &str, tol: f32) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let threshold = tol + tol * y.abs();
        assert!(
            diff <= threshold,
            "{label} at index {i}: {x} vs {y} (diff={diff}, tol={threshold})"
        );
    }
}

// ---------------------------------------------------------------------------
// Regression: head_dim=128, F32 fwd AND bwd now succeed.
// ---------------------------------------------------------------------------

#[test]
fn head_dim_128_f32_fwd_and_bwd_now_succeed() {
    if !cuda_available() {
        eprintln!("head_dim_128_f32_fwd_and_bwd_now_succeed: CUDA not available, skipping");
        return;
    }
    let _lock = cuda_lock();
    let (client, device) = cuda_setup();

    let batch_size = 2usize;
    let num_heads = 4usize;
    let num_kv_heads = 4usize;
    let head_dim = 128usize;
    let total_tokens = 10usize;
    let max_seqlen = 6usize;
    let cu_seqlens_data = vec![0i32, 4, 10];

    let n = total_tokens * num_heads * head_dim;
    let q_data = values(n, 0.3);
    let k_data = values(n, 1.1);
    let v_data = values(n, 2.7);
    let do_data = values(n, 4.4);

    let q =
        Tensor::<CudaRuntime>::from_slice(&q_data, &[total_tokens, num_heads, head_dim], &device)
            .expect("build Q");
    let k = Tensor::<CudaRuntime>::from_slice(
        &k_data,
        &[total_tokens, num_kv_heads, head_dim],
        &device,
    )
    .expect("build K");
    let v = Tensor::<CudaRuntime>::from_slice(
        &v_data,
        &[total_tokens, num_kv_heads, head_dim],
        &device,
    )
    .expect("build V");
    let dout =
        Tensor::<CudaRuntime>::from_slice(&do_data, &[total_tokens, num_heads, head_dim], &device)
            .expect("build dOut");
    let cu = Tensor::<CudaRuntime>::from_slice(&cu_seqlens_data, &[batch_size + 1], &device)
        .expect("build cu_seqlens");

    // This is the exact call that failed with "shared memory 129KB exceeds
    // device limit 99KB" before the tile fallback existed.
    let (out, lse) = client
        .varlen_attention_fwd(
            &q,
            &k,
            &v,
            &cu,
            &cu,
            batch_size,
            num_heads,
            num_kv_heads,
            max_seqlen,
            max_seqlen,
            head_dim,
            true,
        )
        .expect("head_dim=128 F32 varlen forward must now succeed via the small-tile fallback");

    let (dq, dk, dv) = client
        .varlen_attention_bwd(
            &dout,
            &q,
            &k,
            &v,
            &out,
            &lse,
            &cu,
            &cu,
            batch_size,
            num_heads,
            num_kv_heads,
            max_seqlen,
            max_seqlen,
            head_dim,
            true,
        )
        .expect("head_dim=128 F32 varlen backward must now succeed via the small-tile fallback");

    assert_eq!(out.shape(), &[total_tokens, num_heads, head_dim]);
    assert_eq!(dq.shape(), &[total_tokens, num_heads, head_dim]);
    assert_eq!(dk.shape(), &[total_tokens, num_kv_heads, head_dim]);
    assert_eq!(dv.shape(), &[total_tokens, num_kv_heads, head_dim]);
    for &x in out.to_vec::<f32>().iter() {
        assert!(x.is_finite(), "forward output has a non-finite value: {x}");
    }
    for &x in dq
        .to_vec::<f32>()
        .iter()
        .chain(dk.to_vec::<f32>().iter())
        .chain(dv.to_vec::<f32>().iter())
    {
        assert!(
            x.is_finite(),
            "backward gradient has a non-finite value: {x}"
        );
    }
}

// ---------------------------------------------------------------------------
// Parity: head_dim=64, F32 — both tiles fit the large kernel on a ~99KB
// device, so both sides can be forced and compared directly.
// ---------------------------------------------------------------------------

#[test]
fn head_dim_64_f32_large_and_small_tile_agree() {
    if !cuda_available() {
        eprintln!("head_dim_64_f32_large_and_small_tile_agree: CUDA not available, skipping");
        return;
    }
    let _lock = cuda_lock();
    let (client, device) = cuda_setup();

    let batch_size = 2usize;
    let num_heads = 4usize;
    let num_kv_heads = 4usize;
    let head_dim = 64usize;
    let total_tokens = 10usize;
    let max_seqlen = 6usize;
    let cu_seqlens_data = vec![0i32, 4, 10];

    let n = total_tokens * num_heads * head_dim;
    let q = Tensor::<CudaRuntime>::from_slice(
        &values(n, 0.3),
        &[total_tokens, num_heads, head_dim],
        &device,
    )
    .expect("build Q");
    let k = Tensor::<CudaRuntime>::from_slice(
        &values(n, 1.1),
        &[total_tokens, num_kv_heads, head_dim],
        &device,
    )
    .expect("build K");
    let v = Tensor::<CudaRuntime>::from_slice(
        &values(n, 2.7),
        &[total_tokens, num_kv_heads, head_dim],
        &device,
    )
    .expect("build V");
    let dout = Tensor::<CudaRuntime>::from_slice(
        &values(n, 4.4),
        &[total_tokens, num_heads, head_dim],
        &device,
    )
    .expect("build dOut");
    let cu = Tensor::<CudaRuntime>::from_slice(&cu_seqlens_data, &[batch_size + 1], &device)
        .expect("build cu_seqlens");

    let (small_out, small_lse) = varlen_attention_fwd_with_tile_for_test(
        &client,
        &q,
        &k,
        &v,
        &cu,
        &cu,
        batch_size,
        num_heads,
        num_kv_heads,
        max_seqlen,
        max_seqlen,
        head_dim,
        true,
        false,
    )
    .expect("F32 small-tile forward must succeed at head_dim=64");
    let (large_out, large_lse) = varlen_attention_fwd_with_tile_for_test(
        &client,
        &q,
        &k,
        &v,
        &cu,
        &cu,
        batch_size,
        num_heads,
        num_kv_heads,
        max_seqlen,
        max_seqlen,
        head_dim,
        true,
        true,
    )
    .expect("F32 large-tile forward must succeed at head_dim=64");

    assert_close(
        &small_out.to_vec::<f32>(),
        &large_out.to_vec::<f32>(),
        "fwd O (hd64 f32)",
        F32_TOL,
    );
    assert_close(
        &small_lse.to_vec::<f32>(),
        &large_lse.to_vec::<f32>(),
        "fwd LSE (hd64 f32)",
        F32_TOL,
    );

    let (dq_small, dk_small, dv_small) = varlen_attention_bwd_with_tile_for_test(
        &client,
        &dout,
        &q,
        &k,
        &v,
        &small_out,
        &small_lse,
        &cu,
        &cu,
        batch_size,
        num_heads,
        num_kv_heads,
        max_seqlen,
        max_seqlen,
        head_dim,
        true,
        false,
    )
    .expect("F32 small-tile backward must succeed at head_dim=64");
    let (dq_large, dk_large, dv_large) = varlen_attention_bwd_with_tile_for_test(
        &client,
        &dout,
        &q,
        &k,
        &v,
        &large_out,
        &large_lse,
        &cu,
        &cu,
        batch_size,
        num_heads,
        num_kv_heads,
        max_seqlen,
        max_seqlen,
        head_dim,
        true,
        true,
    )
    .expect("F32 large-tile backward must succeed at head_dim=64");

    assert_close(
        &dq_small.to_vec::<f32>(),
        &dq_large.to_vec::<f32>(),
        "dQ (hd64 f32)",
        F32_TOL,
    );
    assert_close(
        &dk_small.to_vec::<f32>(),
        &dk_large.to_vec::<f32>(),
        "dK (hd64 f32)",
        F32_TOL,
    );
    assert_close(
        &dv_small.to_vec::<f32>(),
        &dv_large.to_vec::<f32>(),
        "dV (hd64 f32)",
        F32_TOL,
    );
}

// ---------------------------------------------------------------------------
// Parity: head_dim=128, F16 — both tiles fit the large kernel on a ~99KB
// device, and head_dim=128 was untested at every dtype before this file.
// ---------------------------------------------------------------------------

#[test]
fn head_dim_128_f16_large_and_small_tile_agree() {
    if !cuda_available() {
        eprintln!("head_dim_128_f16_large_and_small_tile_agree: CUDA not available, skipping");
        return;
    }
    let _lock = cuda_lock();
    let (client, device) = cuda_setup();

    let batch_size = 2usize;
    let num_heads = 4usize;
    let num_kv_heads = 4usize;
    let head_dim = 128usize;
    let total_tokens = 10usize;
    let max_seqlen = 6usize;
    let cu_seqlens_data = vec![0i32, 4, 10];

    let n = total_tokens * num_heads * head_dim;
    let to_f16 = |data: Vec<f32>, shape: &[usize]| -> Tensor<CudaRuntime> {
        Tensor::<CudaRuntime>::from_slice(&data, shape, &device)
            .expect("build F32 fixture")
            .to_dtype(DType::F16)
            .expect("cast fixture to F16")
    };
    let q = to_f16(values(n, 0.3), &[total_tokens, num_heads, head_dim]);
    let k = to_f16(values(n, 1.1), &[total_tokens, num_kv_heads, head_dim]);
    let v = to_f16(values(n, 2.7), &[total_tokens, num_kv_heads, head_dim]);
    let dout = to_f16(values(n, 4.4), &[total_tokens, num_heads, head_dim]);
    let cu = Tensor::<CudaRuntime>::from_slice(&cu_seqlens_data, &[batch_size + 1], &device)
        .expect("build cu_seqlens");

    let (small_out, small_lse) = varlen_attention_fwd_with_tile_for_test(
        &client,
        &q,
        &k,
        &v,
        &cu,
        &cu,
        batch_size,
        num_heads,
        num_kv_heads,
        max_seqlen,
        max_seqlen,
        head_dim,
        true,
        false,
    )
    .expect("F16 small-tile forward must succeed at head_dim=128");
    let (large_out, large_lse) = varlen_attention_fwd_with_tile_for_test(
        &client,
        &q,
        &k,
        &v,
        &cu,
        &cu,
        batch_size,
        num_heads,
        num_kv_heads,
        max_seqlen,
        max_seqlen,
        head_dim,
        true,
        true,
    )
    .expect("F16 large-tile forward must succeed at head_dim=128");

    let small_out_f32 = small_out
        .to_dtype(DType::F32)
        .expect("cast small-tile output to F32")
        .to_vec::<f32>();
    let large_out_f32 = large_out
        .to_dtype(DType::F32)
        .expect("cast large-tile output to F32")
        .to_vec::<f32>();
    assert_close(&small_out_f32, &large_out_f32, "fwd O (hd128 f16)", F16_TOL);
    assert_close(
        &small_lse.to_vec::<f32>(),
        &large_lse.to_vec::<f32>(),
        "fwd LSE (hd128 f16)",
        F16_TOL,
    );

    let (dq_small, dk_small, dv_small) = varlen_attention_bwd_with_tile_for_test(
        &client,
        &dout,
        &q,
        &k,
        &v,
        &small_out,
        &small_lse,
        &cu,
        &cu,
        batch_size,
        num_heads,
        num_kv_heads,
        max_seqlen,
        max_seqlen,
        head_dim,
        true,
        false,
    )
    .expect("F16 small-tile backward must succeed at head_dim=128");
    let (dq_large, dk_large, dv_large) = varlen_attention_bwd_with_tile_for_test(
        &client,
        &dout,
        &q,
        &k,
        &v,
        &large_out,
        &large_lse,
        &cu,
        &cu,
        batch_size,
        num_heads,
        num_kv_heads,
        max_seqlen,
        max_seqlen,
        head_dim,
        true,
        true,
    )
    .expect("F16 large-tile backward must succeed at head_dim=128");

    let cast = |t: Tensor<CudaRuntime>, label: &str| -> Vec<f32> {
        t.to_dtype(DType::F32)
            .unwrap_or_else(|e| panic!("cast {label} to F32: {e:?}"))
            .to_vec::<f32>()
    };
    assert_close(
        &cast(dq_small, "dQ small"),
        &cast(dq_large, "dQ large"),
        "dQ (hd128 f16)",
        F16_TOL,
    );
    assert_close(
        &cast(dk_small, "dK small"),
        &cast(dk_large, "dK large"),
        "dK (hd128 f16)",
        F16_TOL,
    );
    assert_close(
        &cast(dv_small, "dV small"),
        &cast(dv_large, "dV large"),
        "dV (hd128 f16)",
        F16_TOL,
    );
}

// ---------------------------------------------------------------------------
// Coverage: EVERY (head_dim, dtype) combination the public guards accept
// resolves to a kernel name that actually exists and loads. This is the
// regression class behind the head_dim=256 bug: the selector reported only
// "did the large tile fit" as a bool, so head_dim=256 (which has no large
// tile at all) fell into the same branch as a genuine large-tile miss and
// got an unexisting `_256_..._small` name appended. Every accepted head_dim
// (64/128/256) x dtype (F32/F16) combination is exercised here, through the
// default (unforced) production tile-selection path — including the two
// combinations the parity tests above do not touch: head_dim=256 (both
// dtypes, `TileVariant::Base256`, unsuffixed name, no `_small` sibling) and
// head_dim=64 F16 (no `_small` kernel compiled for it at all).
// ---------------------------------------------------------------------------

#[test]
fn every_accepted_head_dim_dtype_resolves_a_kernel_that_loads() {
    if !cuda_available() {
        eprintln!(
            "every_accepted_head_dim_dtype_resolves_a_kernel_that_loads: CUDA not available, skipping"
        );
        return;
    }
    let _lock = cuda_lock();
    let (client, device) = cuda_setup();

    let batch_size = 1usize;
    let num_heads = 1usize;
    let num_kv_heads = 1usize;
    let total_tokens = 2usize;
    let max_seqlen = 2usize;
    let cu_seqlens_data = vec![0i32, 2];

    for &head_dim in &[64usize, 128, 256] {
        for &dtype in &[DType::F32, DType::F16] {
            let n = total_tokens * num_heads * head_dim;

            let build_fixture = |data: Vec<f32>| -> Tensor<CudaRuntime> {
                let t = Tensor::<CudaRuntime>::from_slice(
                    &data,
                    &[total_tokens, num_heads, head_dim],
                    &device,
                )
                .expect("build F32 fixture");
                if matches!(dtype, DType::F16) {
                    t.to_dtype(DType::F16).expect("cast fixture to F16")
                } else {
                    t
                }
            };

            let q = build_fixture(values(n, 0.3));
            let k = build_fixture(values(n, 1.1));
            let v = build_fixture(values(n, 2.7));
            let dout = build_fixture(values(n, 4.4));
            let cu =
                Tensor::<CudaRuntime>::from_slice(&cu_seqlens_data, &[batch_size + 1], &device)
                    .expect("build cu_seqlens");

            let (out, lse) = client
                .varlen_attention_fwd(
                    &q,
                    &k,
                    &v,
                    &cu,
                    &cu,
                    batch_size,
                    num_heads,
                    num_kv_heads,
                    max_seqlen,
                    max_seqlen,
                    head_dim,
                    true,
                )
                .unwrap_or_else(|e| {
                    panic!(
                        "head_dim={head_dim} dtype={dtype:?}: forward kernel failed to \
                         resolve/load/launch: {e:?}"
                    )
                });

            let (dq, dk, dv) = client
                .varlen_attention_bwd(
                    &dout,
                    &q,
                    &k,
                    &v,
                    &out,
                    &lse,
                    &cu,
                    &cu,
                    batch_size,
                    num_heads,
                    num_kv_heads,
                    max_seqlen,
                    max_seqlen,
                    head_dim,
                    true,
                )
                .unwrap_or_else(|e| {
                    panic!(
                        "head_dim={head_dim} dtype={dtype:?}: backward kernel failed to \
                         resolve/load/launch: {e:?}"
                    )
                });

            assert_eq!(out.shape(), &[total_tokens, num_heads, head_dim]);
            assert_eq!(dq.shape(), &[total_tokens, num_heads, head_dim]);
            assert_eq!(dk.shape(), &[total_tokens, num_kv_heads, head_dim]);
            assert_eq!(dv.shape(), &[total_tokens, num_kv_heads, head_dim]);
        }
    }
}
