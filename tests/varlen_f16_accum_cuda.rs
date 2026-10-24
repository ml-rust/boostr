//! F16 varlen attention backward: dK/dV accumulate in F32 across GQA heads
//! and Q blocks.
//!
//! Every Q head of a KV group and every Q block of a sequence adds one partial
//! into the same dK/dV row. The kernel adds these partials into a zeroed F32
//! buffer. The launcher rounds the result to F16 once. These tests feed the CPU
//! reference the exact inputs the kernel reads: Q, K, V and dO rounded to F16,
//! plus the CUDA forward O (F16) and LSE (F32). The reference then differs from
//! the kernel only by F32 summation order, `__expf`, and the final F16 store.
//!
//! Bounds, with `u = 2^-11` (F16 unit roundoff):
//! - Per element: `|cuda - ref| <= 2u * |ref| + 2^-12 * ref_rms + 2^-25`. The
//!   final round-to-nearest F16 store costs at most `u * |ref|`, or `2^-25` for
//!   an F16 subnormal. The factor 2 and the `ref_rms` term cover F32 order and
//!   `__expf` noise.
//! - Aggregate: `rms(cuda - ref) / ref_rms <= u`. One F16 rounding gives an
//!   error near `u / sqrt(3)` relative.
//!
//! A half accumulator rounds the running sum after each of `P` partials, and
//! rounds each partial too. Its error is near `(u / sqrt(3)) * sqrt(P)`
//! relative. The hd128 case forces the `BLOCK_M=32` tile with 512 Q rows and 8
//! Q heads per KV head, so `P = 16 * 8 = 128`. That error is several times both
//! bounds above.
//!
//! Run with:
//!   cd boostr && cargo test --features cuda,f16 --test varlen_f16_accum_cuda

#![cfg(feature = "cuda")]

use std::sync::{Mutex, OnceLock};

use boostr::ops::cuda::attention::varlen_attention_bwd::varlen_attention_bwd_with_tile_for_test;
use boostr::ops::traits::attention::varlen_attention::VarLenAttentionOps;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
use numr::tensor::Tensor;

static CUDA_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn cuda_lock() -> std::sync::MutexGuard<'static, ()> {
    CUDA_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|p| p.into_inner())
}

/// F16 unit roundoff.
const U_F16: f32 = 1.0 / 2048.0;
/// Per-element relative bound: twice one F16 rounding.
const ELEM_RTOL: f32 = 2.0 * U_F16;
/// Per-element absolute bound, as a fraction of the reference RMS.
const ELEM_ATOL_FRAC: f32 = 1.0 / 4096.0;
/// Largest round-to-nearest error of an F16 subnormal: half its step 2^-24.
const F16_SUBNORMAL_RN: f32 = 1.0 / 33_554_432.0;
/// Aggregate bound: `rms(diff) / ref_rms`.
const RMS_RTOL: f32 = U_F16;

/// Deterministic pseudo-random values, distinct per index and per seed.
fn values(len: usize, seed: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = (i as f32) * 0.017 + seed;
            x.sin() * 0.9 + (x * 2.3).cos() * 0.4
        })
        .collect()
}

/// Reads a CUDA tensor of any float dtype back as F32 values.
fn read_f32(t: &Tensor<CudaRuntime>, what: &str) -> Vec<f32> {
    if t.dtype() == DType::F32 {
        return t.to_vec::<f32>();
    }
    t.to_dtype(DType::F32)
        .unwrap_or_else(|e| panic!("cast {what} to F32 for read-back: {e}"))
        .to_vec::<f32>()
}

fn cpu_tensor(data: &[f32], shape: &[usize], dev: &CpuDevice) -> Tensor<CpuRuntime> {
    Tensor::<CpuRuntime>::from_slice(data, shape, dev).expect("build CPU tensor")
}

/// Uploads F32 data and rounds it to F16 on the device.
fn f16_tensor(data: &[f32], shape: &[usize], dev: &CudaDevice, what: &str) -> Tensor<CudaRuntime> {
    Tensor::<CudaRuntime>::from_slice(data, shape, dev)
        .unwrap_or_else(|e| panic!("upload {what}: {e}"))
        .to_dtype(DType::F16)
        .unwrap_or_else(|e| panic!("cast {what} to F16: {e}"))
}

fn assert_f16_accum(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: element count mismatch"
    );
    let n = expected.len() as f64;
    let ref_rms = (expected.iter().map(|&e| (e as f64).powi(2)).sum::<f64>() / n).sqrt() as f32;
    assert!(
        ref_rms > 1e-4,
        "{label}: reference RMS {ref_rms:.3e} is degenerate; fix the fixture"
    );

    let mut diff_sq = 0.0f64;
    let mut worst = (0usize, 0.0f32, 0.0f32);
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(a.is_finite(), "{label}: non-finite {a} at index {i}");
        let diff = (a - e).abs();
        diff_sq += (diff as f64).powi(2);
        let bound = ELEM_RTOL * e.abs() + ELEM_ATOL_FRAC * ref_rms + F16_SUBNORMAL_RN;
        if diff / bound > worst.1 {
            worst = (i, diff / bound, diff);
        }
    }
    let rel_rms = ((diff_sq / n).sqrt() as f32) / ref_rms;
    let (wi, wratio, wdiff) = worst;
    println!(
        "VARLEN_F16_ACCUM {label}: rms_rel={rel_rms:.3e} (bound {RMS_RTOL:.3e}) \
         worst_elem idx={wi} diff={wdiff:.3e} ratio_to_bound={wratio:.3} ref_rms={ref_rms:.3e}"
    );
    assert!(
        wratio <= 1.0,
        "{label}: index {wi}: cuda {} vs reference {} (diff {wdiff:.3e}) exceeds \
         {ELEM_RTOL:.3e}*|ref| + {ELEM_ATOL_FRAC:.3e}*ref_rms; dK/dV no longer reach \
         F16 through one rounding",
        actual[wi],
        expected[wi]
    );
    assert!(
        rel_rms <= RMS_RTOL,
        "{label}: rms(diff)/ref_rms {rel_rms:.3e} exceeds {RMS_RTOL:.3e}; the partials \
         round to F16 before the reduction ends"
    );
}

struct Case {
    label: &'static str,
    cu_q: Vec<i32>,
    cu_k: Vec<i32>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    causal: bool,
    /// `Some(false)` forces the small tile; `None` uses the production tile.
    force_large: Option<bool>,
}

fn max_len(cu: &[i32]) -> usize {
    cu.windows(2)
        .map(|w| (w[1] - w[0]) as usize)
        .max()
        .unwrap_or(0)
}

fn run(case: &Case) {
    if !cfg!(feature = "f16") {
        eprintln!("SKIPPED: {}: built without the `f16` feature", case.label);
        return;
    }
    if !numr::runtime::cuda::is_cuda_available() {
        eprintln!("SKIPPED: {}: CUDA not available", case.label);
        return;
    }
    let _lock = cuda_lock();

    let (nh, nkv, hd) = (case.num_heads, case.num_kv_heads, case.head_dim);
    let batch = case.cu_q.len() - 1;
    let tq = *case.cu_q.last().expect("cu_q is non-empty") as usize;
    let tk = *case.cu_k.last().expect("cu_k is non-empty") as usize;
    let (msq, msk) = (max_len(&case.cu_q), max_len(&case.cu_k));
    let q_shape = [tq, nh, hd];
    let kv_shape = [tk, nkv, hd];

    let cuda_dev = CudaDevice::new(0);
    let cuda: CudaClient = CudaRuntime::default_client(&cuda_dev);
    let q = f16_tensor(&values(tq * nh * hd, 0.1), &q_shape, &cuda_dev, "q");
    let k = f16_tensor(&values(tk * nkv * hd, 1.3), &kv_shape, &cuda_dev, "k");
    let v = f16_tensor(&values(tk * nkv * hd, 2.7), &kv_shape, &cuda_dev, "v");
    let dout = f16_tensor(&values(tq * nh * hd, 3.9), &q_shape, &cuda_dev, "dout");
    let cu_q = Tensor::<CudaRuntime>::from_slice(&case.cu_q, &[batch + 1], &cuda_dev)
        .expect("upload cu_seqlens_q");
    let cu_k = Tensor::<CudaRuntime>::from_slice(&case.cu_k, &[batch + 1], &cuda_dev)
        .expect("upload cu_seqlens_k");

    let (out, lse) = cuda
        .varlen_attention_fwd(
            &q,
            &k,
            &v,
            &cu_q,
            &cu_k,
            batch,
            nh,
            nkv,
            msq,
            msk,
            hd,
            case.causal,
        )
        .expect("CUDA F16 varlen forward");
    let (dq, dk, dv) = match case.force_large {
        Some(force_large) => varlen_attention_bwd_with_tile_for_test(
            &cuda,
            &dout,
            &q,
            &k,
            &v,
            &out,
            &lse,
            &cu_q,
            &cu_k,
            batch,
            nh,
            nkv,
            msq,
            msk,
            hd,
            case.causal,
            force_large,
        ),
        None => cuda.varlen_attention_bwd(
            &dout,
            &q,
            &k,
            &v,
            &out,
            &lse,
            &cu_q,
            &cu_k,
            &case.cu_q,
            &case.cu_k,
            batch,
            nh,
            nkv,
            msq,
            msk,
            hd,
            case.causal,
        ),
    }
    .expect("CUDA F16 varlen backward");
    assert_eq!(dk.dtype(), DType::F16, "{}: dK dtype", case.label);
    assert_eq!(dv.dtype(), DType::F16, "{}: dV dtype", case.label);

    // CPU reference in F32 over the exact values the kernel read.
    let cpu_dev = CpuDevice::new();
    let cpu = CpuClient::new(cpu_dev.clone());
    let q_c = cpu_tensor(&read_f32(&q, "q"), &q_shape, &cpu_dev);
    let k_c = cpu_tensor(&read_f32(&k, "k"), &kv_shape, &cpu_dev);
    let v_c = cpu_tensor(&read_f32(&v, "v"), &kv_shape, &cpu_dev);
    let do_c = cpu_tensor(&read_f32(&dout, "dout"), &q_shape, &cpu_dev);
    let out_c = cpu_tensor(&read_f32(&out, "output"), &q_shape, &cpu_dev);
    let lse_c = cpu_tensor(&read_f32(&lse, "lse"), &[tq, nh], &cpu_dev);
    let cu_q_c = Tensor::<CpuRuntime>::from_slice(&case.cu_q, &[batch + 1], &cpu_dev)
        .expect("build CPU cu_seqlens_q");
    let cu_k_c = Tensor::<CpuRuntime>::from_slice(&case.cu_k, &[batch + 1], &cpu_dev)
        .expect("build CPU cu_seqlens_k");
    let (dq_r, dk_r, dv_r) = cpu
        .varlen_attention_bwd(
            &do_c,
            &q_c,
            &k_c,
            &v_c,
            &out_c,
            &lse_c,
            &cu_q_c,
            &cu_k_c,
            &case.cu_q,
            &case.cu_k,
            batch,
            nh,
            nkv,
            msq,
            msk,
            hd,
            case.causal,
        )
        .expect("CPU F32 varlen backward reference");

    assert_f16_accum(
        &read_f32(&dk, "dk"),
        &dk_r.to_vec::<f32>(),
        &format!("{} dK", case.label),
    );
    assert_f16_accum(
        &read_f32(&dv, "dv"),
        &dv_r.to_vec::<f32>(),
        &format!("{} dV", case.label),
    );
    assert_f16_accum(
        &read_f32(&dq, "dq"),
        &dq_r.to_vec::<f32>(),
        &format!("{} dQ", case.label),
    );
}

/// 8 Q heads per KV head, small tile forced (`BLOCK_M=32`). Sequence 0 has 512
/// Q rows, so each of its dK/dV rows takes 16 Q blocks x 8 heads = 128
/// partials. Sequence 1 is ragged: 160 Q rows over 288 keys.
#[test]
fn varlen_bwd_f16_gqa8_hd128_small_tile_many_q_blocks() {
    run(&Case {
        label: "gqa8 hd128 small-tile noncausal",
        cu_q: vec![0, 512, 672],
        cu_k: vec![0, 512, 800],
        num_heads: 8,
        num_kv_heads: 1,
        head_dim: 128,
        causal: false,
        force_large: Some(false),
    });
}

/// Same GQA shape on the production tile path at head_dim 64 (`BLOCK_M=128`),
/// causal with a ragged `key_offset`. Sequence 0 gives up to 4 Q blocks x 8
/// heads = 32 partials per dK/dV row.
#[test]
fn varlen_bwd_f16_gqa8_hd64_production_tile_causal() {
    run(&Case {
        label: "gqa8 hd64 production-tile causal",
        cu_q: vec![0, 512, 672],
        cu_k: vec![0, 512, 800],
        num_heads: 8,
        num_kv_heads: 1,
        head_dim: 64,
        causal: true,
        force_large: None,
    });
}
