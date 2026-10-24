//! F16/BF16 varlen attention: the forward O and dQ round to the input dtype
//! once, and dK/dV accumulate in F32 across GQA heads and Q blocks.
//!
//! Every Q head of a KV group and every Q block of a sequence adds one partial
//! into the same dK/dV row. The kernel adds these partials into a zeroed F32
//! buffer. The launcher rounds the result to the input dtype once. These tests
//! feed the CPU reference the exact inputs the kernel reads: Q, K, V and dO
//! rounded to the dtype, plus the CUDA forward O and LSE (F32). The reference
//! then differs from the kernel only by F32 summation order, `__expf`, and the
//! final store. The forward O is checked the same way against a CPU forward on
//! the rounded Q, K and V.
//!
//! Bounds, with `u` the unit roundoff (`2^-11` F16, `2^-8` BF16):
//! - Per element: `|cuda - ref| <= 2u * |ref| + (u / 2) * ref_rms + s`. The
//!   final round-to-nearest store costs at most `u * |ref|`, or `s` for a
//!   subnormal (half the smallest subnormal step). The factor 2 and the
//!   `ref_rms` term cover F32 order and `__expf` noise, which can push a value
//!   across one rounding boundary.
//! - Aggregate: `rms(cuda - ref) / ref_rms <= u`. One rounding gives an error
//!   near `u / sqrt(3)` relative.
//!
//! A half accumulator rounds the running sum after each of `P` partials, and
//! rounds each partial too. Its error is near `(u / sqrt(3)) * sqrt(P)`
//! relative. The hd128 case forces the `BLOCK_M=32` tile with 512 Q rows and 8
//! Q heads per KV head, so `P = 16 * 8 = 128`. That error is several times both
//! bounds above, at either dtype.
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

/// Rounding constants of one storage dtype.
#[derive(Clone, Copy)]
struct Rounding {
    /// Unit roundoff: `2^-11` F16, `2^-8` BF16.
    u: f32,
    /// Largest round-to-nearest error of a subnormal: half the smallest step.
    /// F16 steps by `2^-24`. BF16 shares F32's exponent range, so its step is
    /// `2^-133`, below any F32 noise term; `0.0` is exact enough.
    subnormal_rn: f32,
}

impl Rounding {
    fn of(dtype: DType) -> Self {
        match dtype {
            DType::F16 => Rounding {
                u: 1.0 / 2048.0,
                subnormal_rn: 1.0 / 33_554_432.0,
            },
            DType::BF16 => Rounding {
                u: 1.0 / 256.0,
                subnormal_rn: 0.0,
            },
            other => panic!("Rounding::of: {other:?} is not a half dtype; use F16 or BF16"),
        }
    }
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

/// Uploads F32 data and rounds it to `dtype` on the device.
fn half_tensor(
    data: &[f32],
    shape: &[usize],
    dev: &CudaDevice,
    dtype: DType,
    what: &str,
) -> Tensor<CudaRuntime> {
    Tensor::<CudaRuntime>::from_slice(data, shape, dev)
        .unwrap_or_else(|e| panic!("upload {what}: {e}"))
        .to_dtype(dtype)
        .unwrap_or_else(|e| panic!("cast {what} to {dtype:?}: {e}"))
}

fn assert_half_once(actual: &[f32], expected: &[f32], r: Rounding, label: &str) {
    let elem_rtol = 2.0 * r.u;
    let elem_atol_frac = r.u / 2.0;
    let rms_rtol = r.u;
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
        let bound = elem_rtol * e.abs() + elem_atol_frac * ref_rms + r.subnormal_rn;
        if diff / bound > worst.1 {
            worst = (i, diff / bound, diff);
        }
    }
    let rel_rms = ((diff_sq / n).sqrt() as f32) / ref_rms;
    let (wi, wratio, wdiff) = worst;
    println!(
        "VARLEN_HALF_ACCUM {label}: rms_rel={rel_rms:.3e} (bound {rms_rtol:.3e}) \
         worst_elem idx={wi} diff={wdiff:.3e} ratio_to_bound={wratio:.3} ref_rms={ref_rms:.3e}"
    );
    assert!(
        wratio <= 1.0,
        "{label}: index {wi}: cuda {} vs reference {} (diff {wdiff:.3e}) exceeds \
         {elem_rtol:.3e}*|ref| + {elem_atol_frac:.3e}*ref_rms; the result no longer \
         reaches the half dtype through one rounding",
        actual[wi],
        expected[wi]
    );
    assert!(
        rel_rms <= rms_rtol,
        "{label}: rms(diff)/ref_rms {rel_rms:.3e} exceeds {rms_rtol:.3e}; the partials \
         round to the half dtype before the reduction ends"
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
    dtype: DType,
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
    let dt = case.dtype;
    let r = Rounding::of(dt);
    let q = half_tensor(&values(tq * nh * hd, 0.1), &q_shape, &cuda_dev, dt, "q");
    let k = half_tensor(&values(tk * nkv * hd, 1.3), &kv_shape, &cuda_dev, dt, "k");
    let v = half_tensor(&values(tk * nkv * hd, 2.7), &kv_shape, &cuda_dev, dt, "v");
    let dout = half_tensor(&values(tq * nh * hd, 3.9), &q_shape, &cuda_dev, dt, "dout");
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
            0,
        )
        .expect("CUDA half varlen forward");
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
            0,
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
            0,
        ),
    }
    .expect("CUDA half varlen backward");
    for (name, t) in [("O", &out), ("dQ", &dq), ("dK", &dk), ("dV", &dv)] {
        assert_eq!(t.dtype(), dt, "{}: {name} dtype", case.label);
    }

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
            0,
        )
        .expect("CPU F32 varlen backward reference");
    let (out_r, _) = cpu
        .varlen_attention_fwd(
            &q_c,
            &k_c,
            &v_c,
            &cu_q_c,
            &cu_k_c,
            batch,
            nh,
            nkv,
            msq,
            msk,
            hd,
            case.causal,
            0,
        )
        .expect("CPU F32 varlen forward reference");

    let checks = [
        ("O", read_f32(&out, "output"), out_r),
        ("dK", read_f32(&dk, "dk"), dk_r),
        ("dV", read_f32(&dv, "dv"), dv_r),
        ("dQ", read_f32(&dq, "dq"), dq_r),
    ];
    for (name, actual, reference) in checks {
        let label = format!("{} {dt:?} {name}", case.label);
        assert_half_once(&actual, &reference.to_vec::<f32>(), r, &label);
    }
}

/// 8 Q heads per KV head, small tile forced (`BLOCK_M=32`). Sequence 0 has 512
/// Q rows, so each of its dK/dV rows takes 16 Q blocks x 8 heads = 128
/// partials. Sequence 1 is ragged: 160 Q rows over 288 keys.
fn gqa8_hd128_small_tile(dtype: DType) -> Case {
    Case {
        label: "gqa8 hd128 small-tile noncausal",
        cu_q: vec![0, 512, 672],
        cu_k: vec![0, 512, 800],
        num_heads: 8,
        num_kv_heads: 1,
        head_dim: 128,
        causal: false,
        force_large: Some(false),
        dtype,
    }
}

/// Same GQA shape on the production tile path at head_dim 64 (`BLOCK_M=128`),
/// causal with a ragged `key_offset`. Sequence 0 gives up to 4 Q blocks x 8
/// heads = 32 partials per dK/dV row.
fn gqa8_hd64_production_tile(dtype: DType) -> Case {
    Case {
        label: "gqa8 hd64 production-tile causal",
        cu_q: vec![0, 512, 672],
        cu_k: vec![0, 512, 800],
        num_heads: 8,
        num_kv_heads: 1,
        head_dim: 64,
        causal: true,
        force_large: None,
        dtype,
    }
}

#[test]
fn varlen_bwd_f16_gqa8_hd128_small_tile_many_q_blocks() {
    run(&gqa8_hd128_small_tile(DType::F16));
}

#[test]
fn varlen_bwd_f16_gqa8_hd64_production_tile_causal() {
    run(&gqa8_hd64_production_tile(DType::F16));
}

#[test]
fn varlen_bwd_bf16_gqa8_hd128_small_tile_many_q_blocks() {
    run(&gqa8_hd128_small_tile(DType::BF16));
}

#[test]
fn varlen_bwd_bf16_gqa8_hd64_production_tile_causal() {
    run(&gqa8_hd64_production_tile(DType::BF16));
}
