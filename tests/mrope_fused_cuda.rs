//! Bit equality of the fused CUDA IMROPE kernel (`MRopeOps::mrope_interleaved_fused`)
//! against the composed op `apply_mrope_interleaved_impl` on the same CUDA
//! client: the `qwen35` production shape, the tiny test config, a
//! multi-token prefill, distinct streams, the `e` fall-through sectors,
//! out-of-range positions and a strided input view.
//!
//! Run with:
//!   cd boostr && cargo test --features cuda --test mrope_fused_cuda

#![cfg(feature = "cuda")]

use std::sync::{Mutex, OnceLock};

use boostr::nn::RoPE;
use boostr::ops::impl_generic::position::{apply_mrope_interleaved_impl, mrope_stream_selector};
use boostr::ops::traits::position::MRopeOps;
use numr::autograd::Var;
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

fn cuda_setup() -> Option<(CudaClient, CudaDevice)> {
    if !numr::runtime::cuda::is_cuda_available() {
        return None;
    }
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    Some((client, device))
}

const MAX_POS: usize = 2048;

/// Production `qwen35` attention: `head_dim = 256`, `rope_dim = 64`,
/// `rope_sections = [11, 11, 10, 0]`, `rope_theta = 1e7`.
const BONSAI: Case = Case {
    head_dim: 256,
    n_rot: 64,
    sections: [11, 11, 10, 0],
    theta: 1e7,
};

/// The `tests/common/qwen35_tiny.rs` attention config.
const TINY: Case = Case {
    head_dim: 32,
    n_rot: 16,
    sections: [3, 3, 2, 0],
    theta: 10_000.0,
};

/// Sections whose sector 1 falls through to the `e` stream.
const E_SECTOR: Case = Case {
    head_dim: 8,
    n_rot: 4,
    sections: [1, 0, 1, 0],
    theta: 10_000.0,
};

#[derive(Clone, Copy)]
struct Case {
    head_dim: usize,
    n_rot: usize,
    sections: [usize; 4],
    theta: f32,
}

/// Deterministic values in about `[-1, 1]`, distinct per index and seed.
fn values(len: usize, seed: u32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let h = (i as u32)
                .wrapping_mul(2_654_435_761)
                .wrapping_add(seed.wrapping_mul(97));
            (h % 2001) as f32 / 1000.0 - 1.0
        })
        .collect()
}

fn positions(device: &CudaDevice, streams: [&[i32]; 4]) -> Tensor<CudaRuntime> {
    let seq = streams[0].len();
    let data: Vec<i32> = streams.iter().flat_map(|s| s.iter().copied()).collect();
    Tensor::<CudaRuntime>::from_slice(&data, &[4, seq], device).unwrap()
}

/// Composed op and fused kernel on the same inputs; both as flat `f32`.
fn both(
    client: &CudaClient,
    device: &CudaDevice,
    case: Case,
    x: &Tensor<CudaRuntime>,
    pos: &Tensor<CudaRuntime>,
) -> (Vec<f32>, Vec<f32>) {
    let rope = RoPE::<CudaRuntime>::precompute_freqs(MAX_POS, case.n_rot, case.theta, None, device)
        .unwrap();
    let (cos, sin) = (rope.cos_cache(), rope.sin_cache());
    let sel = mrope_stream_selector::<CudaRuntime>(case.sections, case.n_rot / 2, device).unwrap();
    let xv = Var::new(x.clone(), false);
    let chain = apply_mrope_interleaved_impl(client, &xv, cos, sin, pos, &sel, case.n_rot).unwrap();
    let fused = client
        .mrope_interleaved_fused(&xv, cos, sin, pos, &sel, case.n_rot)
        .unwrap();
    assert_eq!(fused.shape(), chain.shape());
    assert!(fused.tensor().is_contiguous());
    (
        chain.tensor().contiguous().unwrap().to_vec::<f32>(),
        fused.tensor().to_vec::<f32>(),
    )
}

fn assert_bits_equal(chain: &[f32], fused: &[f32], what: &str) {
    assert_eq!(chain.len(), fused.len(), "{what}: length");
    for (i, (c, f)) in chain.iter().zip(fused).enumerate() {
        assert_eq!(
            c.to_bits(),
            f.to_bits(),
            "{what}: element {i} chain {c} fused {f}"
        );
    }
}

fn text_case(case: Case, heads: usize, p: i32, what: &str) {
    let _guard = cuda_lock();
    let Some((client, device)) = cuda_setup() else {
        return;
    };
    let shape = [1, 1, heads, case.head_dim];
    let x = Tensor::<CudaRuntime>::from_slice(&values(heads * case.head_dim, 1), &shape, &device)
        .unwrap();
    let pos = positions(&device, [&[p], &[p], &[p], &[0]]);
    let (chain, fused) = both(&client, &device, case, &x, &pos);
    assert_bits_equal(&chain, &fused, what);
}

#[test]
fn bonsai_query_heads_text_positions() {
    for p in [0, 5, 1000] {
        text_case(BONSAI, 24, p, &format!("bonsai q p={p}"));
    }
}

#[test]
fn bonsai_kv_heads_text_positions() {
    for p in [0, 5, 1000] {
        text_case(BONSAI, 4, p, &format!("bonsai kv p={p}"));
    }
}

#[test]
fn tiny_config_text_positions() {
    for p in [0, 5, 31] {
        text_case(TINY, 2, p, &format!("tiny p={p}"));
    }
}

#[test]
fn distinct_streams_select_per_pair() {
    let _guard = cuda_lock();
    let Some((client, device)) = cuda_setup() else {
        return;
    };
    let shape = [1, 1, 24, BONSAI.head_dim];
    let x = Tensor::<CudaRuntime>::from_slice(&values(24 * BONSAI.head_dim, 2), &shape, &device)
        .unwrap();
    let pos = positions(&device, [&[7], &[19], &[301], &[1]]);
    let (chain, fused) = both(&client, &device, BONSAI, &x, &pos);
    assert_bits_equal(&chain, &fused, "distinct t/h/w/e");
}

#[test]
fn prefill_with_distinct_streams() {
    let _guard = cuda_lock();
    let Some((client, device)) = cuda_setup() else {
        return;
    };
    let (batch, seq, heads) = (2, 7, 4);
    let shape = [batch, seq, heads, BONSAI.head_dim];
    let x = Tensor::<CudaRuntime>::from_slice(
        &values(batch * seq * heads * BONSAI.head_dim, 3),
        &shape,
        &device,
    )
    .unwrap();
    let t: Vec<i32> = (0..seq as i32).map(|i| 100 + i).collect();
    let h: Vec<i32> = t.iter().map(|p| p * 3).collect();
    let w: Vec<i32> = t.iter().map(|p| p + 900).collect();
    let e: Vec<i32> = t.iter().map(|p| p % 4).collect();
    let pos = positions(&device, [&t, &h, &w, &e]);
    let (chain, fused) = both(&client, &device, BONSAI, &x, &pos);
    assert_bits_equal(&chain, &fused, "prefill");
}

#[test]
fn e_sector_fall_through_reads_e_stream() {
    let _guard = cuda_lock();
    let Some((client, device)) = cuda_setup() else {
        return;
    };
    let (seq, heads) = (5, 3);
    let shape = [1, seq, heads, E_SECTOR.head_dim];
    let x = Tensor::<CudaRuntime>::from_slice(
        &values(seq * heads * E_SECTOR.head_dim, 4),
        &shape,
        &device,
    )
    .unwrap();
    let t: Vec<i32> = (0..seq as i32).collect();
    let w: Vec<i32> = t.iter().map(|p| p + 7).collect();
    let e: Vec<i32> = t.iter().map(|p| p + 9).collect();
    let pos = positions(&device, [&t, &t, &w, &e]);
    let (chain, fused) = both(&client, &device, E_SECTOR, &x, &pos);
    assert_bits_equal(&chain, &fused, "e sector");
}

/// A position past the table reads a zero row in the composed op's gather;
/// the kernel must do the same.
#[test]
fn out_of_range_position_reads_zero_row() {
    let _guard = cuda_lock();
    let Some((client, device)) = cuda_setup() else {
        return;
    };
    let shape = [1, 2, 2, TINY.head_dim];
    let x = Tensor::<CudaRuntime>::from_slice(&values(2 * 2 * TINY.head_dim, 5), &shape, &device)
        .unwrap();
    let over = MAX_POS as i32;
    let pos = positions(&device, [&[over, 3], &[3, -1], &[3, 3], &[0, 0]]);
    let (chain, fused) = both(&client, &device, TINY, &x, &pos);
    assert_bits_equal(&chain, &fused, "out of range");
}

/// `x` as a `[B, S, H, D]` view of a `[B, H, S, D]` tensor: unit stride
/// along `D`, swapped strides along `S` and `H`.
#[test]
fn strided_input_view() {
    let _guard = cuda_lock();
    let Some((client, device)) = cuda_setup() else {
        return;
    };
    let (seq, heads) = (3, 4);
    let head_major = Tensor::<CudaRuntime>::from_slice(
        &values(heads * seq * BONSAI.head_dim, 6),
        &[1, heads, seq, BONSAI.head_dim],
        &device,
    )
    .unwrap();
    let x = head_major.permute(&[0, 2, 1, 3]).unwrap();
    assert!(!x.is_contiguous());
    let t: Vec<i32> = (0..seq as i32).map(|i| 40 + i).collect();
    let pos = positions(&device, [&t, &t, &t, &[0; 3]]);
    let (chain, fused) = both(&client, &device, BONSAI, &x, &pos);
    assert_bits_equal(&chain, &fused, "strided view");
}
