//! Sliding-window varlen attention on CUDA against a dense reference.
//!
//! Rule: query `i` of a segment does not attend key `j` when
//! `(causal && j > ko + i) || (window > 0 && j + window <= ko + i)`, with
//! `ko = max(0, seq_len_k - seq_len_q)` per segment. A row with no visible
//! key outputs zeros with LSE = -inf, and its gradients are zero.
//!
//! The reference lives in this file and runs in F64. It does not call the
//! CPU backend. It reads the exact inputs the kernel reads: Q, K, V and dO
//! rounded to the dtype under test. The backward reference takes
//! `D = rowsum(dO * O)` from the CUDA forward O, as the kernel does.
//!
//! Bounds follow `tests/varlen_f16_accum_cuda.rs`, with `u` the unit roundoff
//! of the stored result (`2^-11` F16, `2^-8` BF16):
//! - Per element: `|cuda - ref| <= 2u * |ref| + (u / 2) * ref_rms + s`. The
//!   final store costs at most `u * |ref|`, or `s` for a subnormal. The factor
//!   2 and the `ref_rms` term cover F32 order and `__expf` noise.
//! - Aggregate: `rms(cuda - ref) / ref_rms <= u`.
//!
//! F32 has no final rounding. Its error is F32 order and `__expf` noise over
//! at most a few thousand terms, near `1e-6` relative. `u = 2^-13` bounds it
//! with headroom. A wrong mask moves results by O(1), far past every bound.
//!
//! A reference with RMS under `1e-6` is exactly zero by construction. Window 1
//! gives P = 1 and dS = 0, so dQ and dK vanish. Such a tensor must stay
//! within `1e-5` of zero.
//!
//! Run with:
//!   cd boostr && cargo test --features cuda,f16 --test varlen_window_cuda -j 1

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

/// Windows under test. 2048 exceeds every segment. 64 and 200 span several
/// K tiles at every tile size.
const WINDOWS: [usize; 6] = [1, 3, 17, 64, 200, 2048];

/// LSE is `m + log(l)` in F32, with `l` a sum of at most 700 `__expf` terms.
/// Its absolute error stays near `1e-5`. This bound gives 10x headroom.
const LSE_ATOL: f64 = 1e-4;

/// Reference RMS under this value marks a tensor that is zero by construction.
const DEGENERATE_RMS: f32 = 1e-6;
const DEGENERATE_ATOL: f32 = 1e-5;

#[derive(Clone, Copy)]
struct Rounding {
    u: f32,
    subnormal_rn: f32,
}

impl Rounding {
    fn of(dtype: DType) -> Self {
        match dtype {
            DType::F32 => Rounding {
                u: 1.0 / 8192.0,
                subnormal_rn: 0.0,
            },
            DType::F16 => Rounding {
                u: 1.0 / 2048.0,
                subnormal_rn: 1.0 / 33_554_432.0,
            },
            DType::BF16 => Rounding {
                u: 1.0 / 256.0,
                subnormal_rn: 0.0,
            },
            other => panic!("Rounding::of: {other:?} has no varlen kernel; use F32, F16 or BF16"),
        }
    }
}

/// Tile the launcher runs. `Small` and `Large` go through the
/// `*_with_tile_for_test` entry points.
#[derive(Clone, Copy, Debug)]
enum Tile {
    Production,
    Small,
    Large,
}

impl Tile {
    fn force_large(self) -> Option<bool> {
        match self {
            Tile::Production => None,
            Tile::Small => Some(false),
            Tile::Large => Some(true),
        }
    }
}

struct Layout {
    cu_q: Vec<i32>,
    cu_k: Vec<i32>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl Layout {
    fn batch(&self) -> usize {
        self.cu_q.len() - 1
    }
    fn total_q(&self) -> usize {
        self.cu_q[self.batch()] as usize
    }
    fn total_k(&self) -> usize {
        self.cu_k[self.batch()] as usize
    }
    fn max_len(cu: &[i32]) -> usize {
        cu.windows(2)
            .map(|w| (w[1] - w[0]) as usize)
            .max()
            .unwrap_or(0)
    }
    fn max_q(&self) -> usize {
        Self::max_len(&self.cu_q)
    }
    fn max_k(&self) -> usize {
        Self::max_len(&self.cu_k)
    }
}

/// Segment 0: 700 queries over 700 keys. Segment 1: 300 queries over 420
/// keys (`ko = 120`). Segment 2: 200 queries over 50 keys, so a window
/// leaves late rows with no visible key. 2 Q heads share 1 KV head.
fn long_layout(head_dim: usize) -> Layout {
    Layout {
        cu_q: vec![0, 700, 1000, 1200],
        cu_k: vec![0, 700, 1120, 1170],
        num_heads: 2,
        num_kv_heads: 1,
        head_dim,
    }
}

/// 8 Q heads over 2 KV heads, short ragged segments with the same three
/// shapes as [`long_layout`].
fn gqa_layout(head_dim: usize) -> Layout {
    Layout {
        cu_q: vec![0, 160, 260, 330],
        cu_k: vec![0, 160, 300, 350],
        num_heads: 8,
        num_kv_heads: 2,
        head_dim,
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

fn read_f32(t: &Tensor<CudaRuntime>, what: &str) -> Vec<f32> {
    if t.dtype() == DType::F32 {
        return t.to_vec::<f32>();
    }
    t.to_dtype(DType::F32)
        .unwrap_or_else(|e| panic!("cast {what} to F32 for read-back: {e}"))
        .to_vec::<f32>()
}

fn device_tensor(
    data: &[f32],
    shape: &[usize],
    dev: &CudaDevice,
    dtype: DType,
    what: &str,
) -> Tensor<CudaRuntime> {
    let t = Tensor::<CudaRuntime>::from_slice(data, shape, dev)
        .unwrap_or_else(|e| panic!("upload {what}: {e}"));
    if dtype == DType::F32 {
        return t;
    }
    t.to_dtype(dtype)
        .unwrap_or_else(|e| panic!("cast {what} to {dtype:?}: {e}"))
}

struct Inputs {
    q: Tensor<CudaRuntime>,
    k: Tensor<CudaRuntime>,
    v: Tensor<CudaRuntime>,
    dout: Tensor<CudaRuntime>,
    cu_q: Tensor<CudaRuntime>,
    cu_k: Tensor<CudaRuntime>,
    /// The device values read back as F32: what the kernel reads.
    host_q: Vec<f32>,
    host_k: Vec<f32>,
    host_v: Vec<f32>,
    host_dout: Vec<f32>,
}

fn make_inputs(l: &Layout, dtype: DType, dev: &CudaDevice) -> Inputs {
    let q_shape = [l.total_q(), l.num_heads, l.head_dim];
    let kv_shape = [l.total_k(), l.num_kv_heads, l.head_dim];
    let q_len = q_shape.iter().product();
    let kv_len = kv_shape.iter().product();
    let q = device_tensor(&values(q_len, 0.1), &q_shape, dev, dtype, "q");
    let k = device_tensor(&values(kv_len, 1.3), &kv_shape, dev, dtype, "k");
    let v = device_tensor(&values(kv_len, 2.7), &kv_shape, dev, dtype, "v");
    let dout = device_tensor(&values(q_len, 3.9), &q_shape, dev, dtype, "dout");
    let cu_q = Tensor::<CudaRuntime>::from_slice(&l.cu_q, &[l.batch() + 1], dev)
        .expect("upload cu_seqlens_q");
    let cu_k = Tensor::<CudaRuntime>::from_slice(&l.cu_k, &[l.batch() + 1], dev)
        .expect("upload cu_seqlens_k");
    Inputs {
        host_q: read_f32(&q, "q"),
        host_k: read_f32(&k, "k"),
        host_v: read_f32(&v, "v"),
        host_dout: read_f32(&dout, "dout"),
        q,
        k,
        v,
        dout,
        cu_q,
        cu_k,
    }
}

struct Outputs {
    out: Tensor<CudaRuntime>,
    lse: Tensor<CudaRuntime>,
    dq: Tensor<CudaRuntime>,
    dk: Tensor<CudaRuntime>,
    dv: Tensor<CudaRuntime>,
}

fn run_cuda(
    client: &CudaClient,
    l: &Layout,
    x: &Inputs,
    causal: bool,
    window: usize,
    tile: Tile,
) -> Outputs {
    let (b, nh, nkv, hd) = (l.batch(), l.num_heads, l.num_kv_heads, l.head_dim);
    let (msq, msk) = (l.max_q(), l.max_k());
    let (out, lse) = match tile.force_large() {
        Some(force) => varlen_attention_fwd_with_tile_for_test(
            client, &x.q, &x.k, &x.v, &x.cu_q, &x.cu_k, b, nh, nkv, msq, msk, hd, causal, window,
            force,
        ),
        None => client.varlen_attention_fwd(
            &x.q, &x.k, &x.v, &x.cu_q, &x.cu_k, b, nh, nkv, msq, msk, hd, causal, window,
        ),
    }
    .unwrap_or_else(|e| panic!("CUDA varlen forward ({tile:?}, window {window}): {e}"));
    let (dq, dk, dv) = match tile.force_large() {
        Some(force) => varlen_attention_bwd_with_tile_for_test(
            client, &x.dout, &x.q, &x.k, &x.v, &out, &lse, &x.cu_q, &x.cu_k, b, nh, nkv, msq, msk,
            hd, causal, window, force,
        ),
        None => client.varlen_attention_bwd(
            &x.dout, &x.q, &x.k, &x.v, &out, &lse, &x.cu_q, &x.cu_k, &l.cu_q, &l.cu_k, b, nh, nkv,
            msq, msk, hd, causal, window,
        ),
    }
    .unwrap_or_else(|e| panic!("CUDA varlen backward ({tile:?}, window {window}): {e}"));
    Outputs {
        out,
        lse,
        dq,
        dk,
        dv,
    }
}

/// The rule from the file header, written without overflow.
fn masked(i: usize, j: usize, ko: usize, causal: bool, window: usize) -> bool {
    let q_pos = ko + i;
    (causal && j > q_pos) || (window > 0 && q_pos >= window && j <= q_pos - window)
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum()
}

/// Calls `f(row, keys)` for every (token, head) query row. `row` indexes the
/// `[tokens, heads]` LSE. The row's Q offset is `row * head_dim`. `keys`
/// holds the flat K/V offsets of the visible keys.
fn for_each_row(l: &Layout, causal: bool, window: usize, mut f: impl FnMut(usize, &[usize])) {
    let (nh, nkv, hd) = (l.num_heads, l.num_kv_heads, l.head_dim);
    let group = nh / nkv;
    let mut pairs = Vec::new();
    for s in 0..l.batch() {
        let (q0, k0) = (l.cu_q[s] as usize, l.cu_k[s] as usize);
        let sq = l.cu_q[s + 1] as usize - q0;
        let sk = l.cu_k[s + 1] as usize - k0;
        let ko = sk.saturating_sub(sq);
        for h in 0..nh {
            let kvh = h / group;
            for i in 0..sq {
                pairs.clear();
                for j in 0..sk {
                    if !masked(i, j, ko, causal, window) {
                        pairs.push(((k0 + j) * nkv + kvh) * hd);
                    }
                }
                f((q0 + i) * nh + h, &pairs);
            }
        }
    }
}

struct Forward {
    out: Vec<f32>,
    lse: Vec<f64>,
}

fn reference_forward(l: &Layout, x: &Inputs, causal: bool, window: usize) -> Forward {
    let hd = l.head_dim;
    let scale = 1.0 / (hd as f64).sqrt();
    let mut out = vec![0.0f32; l.total_q() * l.num_heads * hd];
    let mut lse = vec![f64::NEG_INFINITY; l.total_q() * l.num_heads];
    for_each_row(l, causal, window, |row, pairs| {
        if pairs.is_empty() {
            return;
        }
        let qo = row * hd;
        let scores: Vec<f64> = pairs
            .iter()
            .map(|&kr| dot(&x.host_q[qo..qo + hd], &x.host_k[kr..kr + hd]) * scale)
            .collect();
        let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let l_sum: f64 = scores.iter().map(|s| (s - m).exp()).sum();
        let mut acc = vec![0.0f64; hd];
        for (&kr, s) in pairs.iter().zip(&scores) {
            let w = (s - m).exp() / l_sum;
            for (a, &v) in acc.iter_mut().zip(&x.host_v[kr..kr + hd]) {
                *a += w * v as f64;
            }
        }
        for (o, a) in out[qo..qo + hd].iter_mut().zip(&acc) {
            *o = *a as f32;
        }
        lse[row] = m + l_sum.ln();
    });
    Forward { out, lse }
}

struct Backward {
    dq: Vec<f32>,
    dk: Vec<f32>,
    dv: Vec<f32>,
    /// Per KV row (token, kv head): true when some query sees it.
    key_seen: Vec<bool>,
}

fn reference_backward(
    l: &Layout,
    x: &Inputs,
    fwd: &Forward,
    cuda_out: &[f32],
    causal: bool,
    window: usize,
) -> Backward {
    let hd = l.head_dim;
    let scale = 1.0 / (hd as f64).sqrt();
    let mut dq = vec![0.0f64; l.total_q() * l.num_heads * hd];
    let mut dk = vec![0.0f64; l.total_k() * l.num_kv_heads * hd];
    let mut dv = vec![0.0f64; l.total_k() * l.num_kv_heads * hd];
    let mut key_seen = vec![false; l.total_k() * l.num_kv_heads];
    for_each_row(l, causal, window, |row, pairs| {
        let qo = row * hd;
        let q = &x.host_q[qo..qo + hd];
        let dout = &x.host_dout[qo..qo + hd];
        let d_row = dot(dout, &cuda_out[qo..qo + hd]);
        for &kr in pairs {
            key_seen[kr / hd] = true;
            let (k, v) = (&x.host_k[kr..kr + hd], &x.host_v[kr..kr + hd]);
            let prob = (dot(q, k) * scale - fwd.lse[row]).exp();
            let ds = prob * (dot(v, dout) - d_row);
            for (g, &kd) in dq[qo..qo + hd].iter_mut().zip(k) {
                *g += scale * ds * kd as f64;
            }
            for (g, &qd) in dk[kr..kr + hd].iter_mut().zip(q) {
                *g += scale * ds * qd as f64;
            }
            for (g, &od) in dv[kr..kr + hd].iter_mut().zip(dout) {
                *g += prob * od as f64;
            }
        }
    });
    let to_f32 = |v: Vec<f64>| -> Vec<f32> { v.into_iter().map(|e| e as f32).collect() };
    Backward {
        dq: to_f32(dq),
        dk: to_f32(dk),
        dv: to_f32(dv),
        key_seen,
    }
}

fn assert_close(actual: &[f32], expected: &[f32], r: Rounding, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: element count mismatch"
    );
    if let Some(i) = actual.iter().position(|a| !a.is_finite()) {
        panic!("{label}: non-finite {} at index {i}", actual[i]);
    }
    let n = expected.len() as f64;
    let ref_rms = (expected.iter().map(|&e| (e as f64).powi(2)).sum::<f64>() / n).sqrt() as f32;
    if ref_rms < DEGENERATE_RMS {
        let worst = actual
            .iter()
            .zip(expected)
            .map(|(a, e)| (a - e).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst <= DEGENERATE_ATOL,
            "{label}: reference is zero by construction, but cuda differs by {worst:.3e}"
        );
        return;
    }
    let (elem_rtol, elem_atol, rms_rtol) = (2.0 * r.u, r.u / 2.0 * ref_rms, r.u);
    let mut diff_sq = 0.0f64;
    let mut worst = (0usize, 0.0f32);
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let diff = (a - e).abs();
        diff_sq += (diff as f64).powi(2);
        let ratio = diff / (elem_rtol * e.abs() + elem_atol + r.subnormal_rn);
        if ratio > worst.1 {
            worst = (i, ratio);
        }
    }
    let rel_rms = ((diff_sq / n).sqrt() as f32) / ref_rms;
    let (wi, wratio) = worst;
    assert!(
        wratio <= 1.0,
        "{label}: index {wi}: cuda {} vs reference {} exceeds {elem_rtol:.3e}*|ref| + \
         {elem_atol:.3e} (ratio {wratio:.3})",
        actual[wi],
        expected[wi]
    );
    assert!(
        rel_rms <= rms_rtol,
        "{label}: rms(diff)/ref_rms {rel_rms:.3e} exceeds {rms_rtol:.3e}"
    );
}

fn assert_lse(actual: &[f32], expected: &[f64], label: &str) {
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        if e == f64::NEG_INFINITY {
            assert!(
                a == f32::NEG_INFINITY,
                "{label}: row {i} has no visible key; LSE must be -inf, got {a}"
            );
            continue;
        }
        assert!(
            a.is_finite() && (a as f64 - e).abs() <= LSE_ATOL * (1.0 + e.abs()),
            "{label}: LSE row {i}: cuda {a} vs reference {e}"
        );
    }
}

/// Every row flagged in `rows` (one flag per `hd` elements) is exactly zero.
fn assert_zero_rows(actual: &[f32], rows: impl Iterator<Item = bool>, hd: usize, label: &str) {
    for (r, flagged) in rows.enumerate() {
        if flagged && actual[r * hd..(r + 1) * hd].iter().any(|&x| x != 0.0) {
            panic!("{label}: row {r} has no visible pair and must be exactly zero");
        }
    }
}

fn skip_reason(dtype: DType) -> Option<&'static str> {
    if dtype != DType::F32 && !cfg!(feature = "f16") {
        return Some("built without the `f16` feature");
    }
    if !numr::runtime::cuda::is_cuda_available() {
        return Some("CUDA not available");
    }
    None
}

/// Runs every window in [`WINDOWS`], causal and non-causal, against the
/// reference.
fn run_windows(l: &Layout, dtype: DType, tile: Tile, name: &str) {
    if let Some(reason) = skip_reason(dtype) {
        eprintln!("SKIPPED: {name}: {reason}");
        return;
    }
    let _lock = cuda_lock();
    let dev = CudaDevice::new(0);
    let client: CudaClient = CudaRuntime::default_client(&dev);
    let r = Rounding::of(dtype);
    let x = make_inputs(l, dtype, &dev);
    let hd = l.head_dim;
    for causal in [false, true] {
        for window in WINDOWS {
            let label = format!("{name} {dtype:?} {tile:?} causal={causal} window={window}");
            let got = run_cuda(&client, l, &x, causal, window, tile);
            let out = read_f32(&got.out, "output");
            let fwd = reference_forward(l, &x, causal, window);
            assert_close(&out, &fwd.out, r, &format!("{label} O"));
            assert_lse(&got.lse.to_vec::<f32>(), &fwd.lse, &format!("{label} LSE"));

            let bwd = reference_backward(l, &x, &fwd, &out, causal, window);
            let (dq, dk, dv) = (
                read_f32(&got.dq, "dq"),
                read_f32(&got.dk, "dk"),
                read_f32(&got.dv, "dv"),
            );
            assert_close(&dq, &bwd.dq, r, &format!("{label} dQ"));
            assert_close(&dk, &bwd.dk, r, &format!("{label} dK"));
            assert_close(&dv, &bwd.dv, r, &format!("{label} dV"));

            let empty_rows = || fwd.lse.iter().map(|&e| e == f64::NEG_INFINITY);
            assert_zero_rows(&out, empty_rows(), hd, &format!("{label} O"));
            assert_zero_rows(&dq, empty_rows(), hd, &format!("{label} dQ"));
            let unseen = || bwd.key_seen.iter().map(|&s| !s);
            assert_zero_rows(&dk, unseen(), hd, &format!("{label} dK"));
            assert_zero_rows(&dv, unseen(), hd, &format!("{label} dV"));
        }
    }
}

/// A window of at least `max(max_seqlen_q, max_seqlen_k)` masks no key, so it
/// runs the same tiles and arithmetic as `window = 0`. O, LSE and dQ have no
/// atomics and match bit for bit. dK and dV sum through float `atomicAdd`,
/// whose order differs between any two launches. They match within the
/// reference bound of the dtype instead.
fn run_wide_window_equals_none(l: &Layout, dtype: DType, tile: Tile, name: &str) {
    if let Some(reason) = skip_reason(dtype) {
        eprintln!("SKIPPED: {name}: {reason}");
        return;
    }
    let _lock = cuda_lock();
    let dev = CudaDevice::new(0);
    let client: CudaClient = CudaRuntime::default_client(&dev);
    let x = make_inputs(l, dtype, &dev);
    let wide = l.max_q().max(l.max_k());
    for causal in [false, true] {
        let base = run_cuda(&client, l, &x, causal, 0, tile);
        for window in [wide, wide + 1, usize::MAX] {
            let got = run_cuda(&client, l, &x, causal, window, tile);
            let label = format!("{name} {dtype:?} causal={causal} window={window}");
            for (what, a, b) in [("dK", &base.dk, &got.dk), ("dV", &base.dv, &got.dv)] {
                let (a, b) = (read_f32(a, what), read_f32(b, what));
                assert_close(&b, &a, Rounding::of(dtype), &format!("{label} {what}"));
            }
            let exact = [
                ("O", &base.out, &got.out),
                ("LSE", &base.lse, &got.lse),
                ("dQ", &base.dq, &got.dq),
            ];
            for (what, a, b) in exact {
                let (a, b) = (read_f32(a, what), read_f32(b, what));
                let mismatch = a
                    .iter()
                    .zip(&b)
                    .position(|(x, y)| x.to_bits() != y.to_bits());
                if let Some(i) = mismatch {
                    panic!(
                        "{label} {what}: index {i} is {} with the window and {} without it",
                        b[i], a[i]
                    );
                }
            }
        }
    }
}

#[test]
fn varlen_window_f32_hd64_small_tile() {
    run_windows(&long_layout(64), DType::F32, Tile::Small, "hd64");
}

#[test]
fn varlen_window_f32_hd64_large_tile() {
    run_windows(&long_layout(64), DType::F32, Tile::Large, "hd64");
}

#[test]
fn varlen_window_f32_hd128_production_tile() {
    run_windows(&long_layout(128), DType::F32, Tile::Production, "hd128");
}

#[test]
fn varlen_window_f32_hd256() {
    run_windows(&long_layout(256), DType::F32, Tile::Production, "hd256");
}

#[test]
fn varlen_window_f16_hd64_production_tile() {
    run_windows(&long_layout(64), DType::F16, Tile::Production, "hd64");
}

#[test]
fn varlen_window_f16_hd128_small_tile() {
    run_windows(&long_layout(128), DType::F16, Tile::Small, "hd128");
}

#[test]
fn varlen_window_f16_hd128_large_tile() {
    run_windows(&long_layout(128), DType::F16, Tile::Large, "hd128");
}

#[test]
fn varlen_window_f16_hd256() {
    run_windows(&long_layout(256), DType::F16, Tile::Production, "hd256");
}

#[test]
fn varlen_window_bf16_hd128_small_tile() {
    run_windows(&long_layout(128), DType::BF16, Tile::Small, "hd128");
}

#[test]
fn varlen_window_bf16_hd128_large_tile() {
    run_windows(&long_layout(128), DType::BF16, Tile::Large, "hd128");
}

#[test]
fn varlen_window_bf16_hd256() {
    run_windows(&long_layout(256), DType::BF16, Tile::Production, "hd256");
}

#[test]
fn varlen_window_gqa4_f32_hd64() {
    run_windows(&gqa_layout(64), DType::F32, Tile::Production, "gqa4 hd64");
}

#[test]
fn varlen_window_gqa4_bf16_hd128() {
    run_windows(
        &gqa_layout(128),
        DType::BF16,
        Tile::Production,
        "gqa4 hd128",
    );
}

#[test]
fn varlen_wide_window_equals_no_window_f32() {
    run_wide_window_equals_none(&long_layout(64), DType::F32, Tile::Small, "hd64");
    run_wide_window_equals_none(&long_layout(128), DType::F32, Tile::Production, "hd128");
}

#[test]
fn varlen_wide_window_equals_no_window_f16() {
    run_wide_window_equals_none(&long_layout(128), DType::F16, Tile::Production, "hd128");
}

#[test]
fn varlen_wide_window_equals_no_window_bf16() {
    run_wide_window_equals_none(&long_layout(256), DType::BF16, Tile::Production, "hd256");
}
