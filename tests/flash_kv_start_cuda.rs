//! `kv_start` (per-row left padding) on every CUDA forward tier of
//! `FlashAttentionOps::flash_attention_fwd`, against the sliced per-row
//! reference in `tests/common/kv_start.rs`, run on the same CUDA client.
//!
//! Tiers, by the routing in `src/ops/cuda/attention/flash/impl_ops.rs`:
//!
//! - decode (`decode_attention.cu`): `S_q == 1`, whole-sequence grid at a
//!   short key axis and split-KV at a long one; with a window too.
//! - register-tiled MQA/GQA (`mqa_gqa.cu`): `S_q > 1`, `D = 128`, no window.
//! - general flash v2 (`flash_v2.cu`): `D = 96`, and any `D` with a window.
//! - folded short query (`decode_attention_fwd_folded`): non-causal
//!   `S_q = 11`, `D = 128`, the VoxCPM2 DiT shape, in both output layouts.
//!
//! Every case runs `B = 3` with `kv_start = [0, 5, 17]` unless it exists to
//! leave a row with no valid key, and checks that an all-zero `kv_start`
//! gives the same bytes as `None`.
//!
//! Run with:
//!   cd boostr && cargo test --features cuda,f16 --test flash_kv_start_cuda

#![cfg(feature = "cuda")]

mod common;

use std::sync::{Mutex, OnceLock};

use boostr::ops::AttnOutLayout;
use boostr::ops::traits::attention::flash::FlashAttentionOps;
use common::kv_start::{
    KvStartGeom, max_abs_diff, read_f32, sliced_reference, token_major_to_head_major, values,
};
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

fn cuda_setup() -> (CudaClient, CudaDevice) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    (client, device)
}

const BATCH: usize = 3;
const NUM_HEADS: usize = 16;
const NUM_KV_HEADS: usize = 2;
const STARTS: [i32; 3] = [0, 5, 17];

/// Absolute tolerance against the same kernel run per row: only the
/// accumulation order across tiles or splits differs.
fn tol(dtype: DType) -> f32 {
    match dtype {
        DType::F32 => 1e-5,
        DType::F16 => 4e-3,
        DType::BF16 => 2e-2,
        other => unimplemented!("tol: {other:?}"),
    }
}

fn geom(seq_q: usize, seq_k: usize, head_dim: usize, causal: bool, window: usize) -> KvStartGeom {
    KvStartGeom {
        batch: BATCH,
        num_heads: NUM_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        seq_q,
        seq_k,
        head_dim,
        causal,
        window,
    }
}

fn fixture(g: &KvStartGeom, dtype: DType, device: &CudaDevice) -> [Tensor<CudaRuntime>; 3] {
    let q_shape = [g.batch, g.num_heads, g.seq_q, g.head_dim];
    let kv_shape = [g.batch, g.num_kv_heads, g.seq_k, g.head_dim];
    let make = |data: Vec<f32>, shape: &[usize]| {
        let t = Tensor::<CudaRuntime>::from_slice(&data, shape, device).expect("fixture");
        if dtype == DType::F32 {
            t
        } else {
            t.to_dtype(dtype).expect("cast fixture")
        }
    };
    [
        make(values(q_shape.iter().product(), 0.1), &q_shape),
        make(values(kv_shape.iter().product(), 1.3), &kv_shape),
        make(values(kv_shape.iter().product(), 2.7), &kv_shape),
    ]
}

fn run(
    client: &CudaClient,
    [q, k, v]: &[Tensor<CudaRuntime>; 3],
    g: &KvStartGeom,
    kv_start: Option<&Tensor<CudaRuntime>>,
    layout: AttnOutLayout,
) -> (Tensor<CudaRuntime>, Tensor<CudaRuntime>) {
    client
        .flash_attention_fwd(
            q,
            k,
            v,
            g.num_heads,
            g.num_kv_heads,
            g.head_dim,
            g.causal,
            g.window,
            None,
            kv_start,
            layout,
        )
        .expect("flash_attention_fwd")
}

/// One padded case at `dtype`: parity with the sliced reference in both
/// output layouts, dead rows exactly zero, and `zeros == None` bitwise.
fn check_case(g: KvStartGeom, starts: &[i32], dtype: DType, label: &str) {
    if dtype != DType::F32 && !cfg!(feature = "f16") {
        return;
    }
    let _lock = cuda_lock();
    let (client, device) = cuda_setup();
    let tensors = fixture(&g, dtype, &device);
    let kv_start = Tensor::<CudaRuntime>::from_slice(starts, &[g.batch], &device).expect("starts");
    let (want_out, want_lse) =
        sliced_reference(&client, &tensors[0], &tensors[1], &tensors[2], &g, starts);

    for layout in [AttnOutLayout::HeadMajor, AttnOutLayout::TokenMajor] {
        let (out, lse) = run(&client, &tensors, &g, Some(&kv_start), layout);
        let got = read_f32(&out);
        let got = match layout {
            AttnOutLayout::HeadMajor => got,
            AttnOutLayout::TokenMajor => token_major_to_head_major(&got, &g),
        };
        let tag = format!("{label} {dtype:?} {layout:?}");
        let d_out = max_abs_diff(&got, &want_out, &format!("{tag} out"));
        let d_lse = max_abs_diff(&read_f32(&lse), &want_lse, &format!("{tag} lse"));
        assert!(d_out <= tol(dtype), "{tag}: output diff {d_out:.3e}");
        assert!(d_lse <= tol(dtype), "{tag}: lse diff {d_lse:.3e}");
        // Dead rows are exactly zero, not merely close.
        for (i, e) in want_lse.iter().enumerate() {
            if e.is_infinite() {
                let row = &got[i * g.head_dim..(i + 1) * g.head_dim];
                assert!(
                    row.iter().all(|x| *x == 0.0),
                    "{tag}: dead row {i} is not zero"
                );
            }
        }
    }

    // The padding changes the answer: the padded rows must not equal the
    // unpadded run, or the case would prove nothing.
    let (out_none_hm, _) = run(&client, &tensors, &g, None, AttnOutLayout::HeadMajor);
    let (out_pad_hm, _) = run(
        &client,
        &tensors,
        &g,
        Some(&kv_start),
        AttnOutLayout::HeadMajor,
    );
    assert_ne!(
        read_f32(&out_none_hm),
        read_f32(&out_pad_hm),
        "{label} {dtype:?}: padded output equals the unpadded one"
    );

    let zeros = Tensor::<CudaRuntime>::zeros(&[g.batch], DType::I32, &device).expect("zeros");
    let (out_none, lse_none) = run(&client, &tensors, &g, None, AttnOutLayout::HeadMajor);
    let (out_zero, lse_zero) = run(
        &client,
        &tensors,
        &g,
        Some(&zeros),
        AttnOutLayout::HeadMajor,
    );
    let bits =
        |t: &Tensor<CudaRuntime>| read_f32(t).iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&out_none),
        bits(&out_zero),
        "{label} {dtype:?}: zeros != None (out)"
    );
    assert_eq!(
        bits(&lse_none),
        bits(&lse_zero),
        "{label} {dtype:?}: zeros != None (lse)"
    );
}

fn all_dtypes(g: KvStartGeom, starts: &[i32], label: &str) {
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        check_case(g, starts, dtype, label);
    }
}

// ---- decode tier -----------------------------------------------------------

#[test]
fn decode_whole_sequence_d64() {
    all_dtypes(geom(1, 40, 64, false, 0), &STARTS, "decode D=64 Sk=40");
}

#[test]
fn decode_whole_sequence_d128() {
    all_dtypes(geom(1, 40, 128, true, 0), &STARTS, "decode D=128 Sk=40");
}

/// A long key axis takes the split-KV grid; each slice clamps to the row's
/// start.
#[test]
fn decode_split_kv_d64() {
    all_dtypes(
        geom(1, 700, 64, false, 0),
        &STARTS,
        "decode split D=64 Sk=700",
    );
}

#[test]
fn decode_split_kv_d128() {
    all_dtypes(
        geom(1, 700, 128, false, 0),
        &STARTS,
        "decode split D=128 Sk=700",
    );
}

/// Starts deep inside the key axis leave most slices empty, and row 2 with
/// no key at all: the combine must yield zeros there.
#[test]
fn decode_split_kv_mostly_empty_and_dead_row() {
    all_dtypes(
        geom(1, 700, 128, false, 0),
        &[0, 650, 700],
        "decode split D=128 Sk=700 starts=[0,650,700]",
    );
}

/// Window and start together: row 2's start lies inside the window, row 1's
/// below it.
#[test]
fn decode_window_with_kv_start() {
    all_dtypes(
        geom(1, 700, 128, false, 100),
        &[0, 5, 650],
        "decode window=100 D=128 Sk=700 starts=[0,5,650]",
    );
}

// ---- register-tiled MQA/GQA tier ------------------------------------------

/// Prefill: rows below the start are dead under the causal mask.
#[test]
fn mqa_gqa_prefill_d128() {
    all_dtypes(
        geom(40, 40, 128, true, 0),
        &STARTS,
        "mqa_gqa prefill D=128 S=40",
    );
}

/// Chunked prefill against a longer key axis: no dead rows, the start cuts
/// the first tile and skips none or one whole tile.
#[test]
fn mqa_gqa_chunked_d128() {
    all_dtypes(
        geom(8, 40, 128, true, 0),
        &STARTS,
        "mqa_gqa chunked D=128 Sq=8 Sk=40",
    );
}

/// Starts past one full tile and past the whole key axis.
#[test]
fn mqa_gqa_prefill_tile_skip_and_dead_row() {
    all_dtypes(
        geom(40, 40, 128, true, 0),
        &[0, 33, 40],
        "mqa_gqa prefill D=128 S=40 starts=[0,33,40]",
    );
}

#[test]
fn mqa_gqa_noncausal_long_query_d64() {
    all_dtypes(
        geom(40, 40, 64, false, 0),
        &STARTS,
        "mqa_gqa non-causal D=64 S=40",
    );
}

// ---- general flash v2 tier -------------------------------------------------

#[test]
fn flash_v2_prefill_d96() {
    all_dtypes(
        geom(40, 40, 96, true, 0),
        &STARTS,
        "flash_v2 prefill D=96 S=40",
    );
}

#[test]
fn flash_v2_chunked_d96() {
    all_dtypes(
        geom(8, 40, 96, true, 0),
        &STARTS,
        "flash_v2 chunked D=96 Sq=8 Sk=40",
    );
}

/// Window plus start at the flash_v2 head dims and at D=128 (a window routes
/// every head dim here).
#[test]
fn flash_v2_window_with_kv_start() {
    all_dtypes(
        geom(40, 40, 96, true, 8),
        &STARTS,
        "flash_v2 window=8 D=96 S=40",
    );
    all_dtypes(
        geom(40, 40, 128, true, 8),
        &STARTS,
        "flash_v2 window=8 D=128 S=40",
    );
    all_dtypes(
        geom(40, 64, 128, false, 12),
        &[0, 20, 24],
        "flash_v2 window=12 non-causal D=128 Sq=40 Sk=64 starts=[0,20,24]",
    );
}

// ---- folded short-query tier ----------------------------------------------

/// The VoxCPM2 DiT shape: non-causal, eleven query rows, D=128.
#[test]
fn folded_short_query_d128() {
    all_dtypes(
        geom(11, 40, 128, false, 0),
        &STARTS,
        "folded Sq=11 D=128 Sk=40",
    );
}

/// Same fold with `S_k == S_q`, so row 2's start reaches the end of the key
/// axis and every one of its query rows is dead.
#[test]
fn folded_short_query_dead_row() {
    all_dtypes(
        geom(11, 11, 128, false, 0),
        &[0, 5, 11],
        "folded Sq=11 D=128 Sk=11 starts=[0,5,11]",
    );
}

#[test]
fn folded_short_query_d64() {
    all_dtypes(geom(5, 40, 64, false, 0), &STARTS, "folded Sq=5 D=64 Sk=40");
}
