//! `kv_start` (per-row left padding) on the CPU `flash_attention_fwd`,
//! against the sliced per-row reference in `tests/common/kv_start.rs`.
//!
//! Both CPU paths are covered: the fused F32 decode loop (`S_q == 1`,
//! non-causal, no window) and the composed standard path (everything else,
//! and every dtype). An all-zero `kv_start` must give the same bytes as
//! `None`, and a row with no valid key stores zeros with an LSE of `-inf`.
//!
//! Run with:
//!   cd boostr && cargo test --test flash_kv_start_cpu

mod common;

use boostr::ops::traits::attention::flash::FlashAttentionOps;
use boostr::ops::{AttnOutLayout, KvSeqLen};
use common::cpu_setup;
use common::kv_start::{
    KvStartGeom, max_abs_diff, read_f32, sliced_reference, token_major_to_head_major, values,
};
use numr::dtype::DType;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

const STARTS: [i32; 3] = [0, 5, 17];

fn geom(seq_q: usize, seq_k: usize, head_dim: usize, causal: bool, window: usize) -> KvStartGeom {
    KvStartGeom {
        batch: 3,
        num_heads: 4,
        num_kv_heads: 2,
        seq_q,
        seq_k,
        head_dim,
        causal,
        window,
    }
}

fn fixture(g: &KvStartGeom, dtype: DType, device: &CpuDevice) -> [Tensor<CpuRuntime>; 3] {
    let q_shape = [g.batch, g.num_heads, g.seq_q, g.head_dim];
    let kv_shape = [g.batch, g.num_kv_heads, g.seq_k, g.head_dim];
    let make = |data: Vec<f32>, shape: &[usize]| {
        let t = Tensor::<CpuRuntime>::from_slice(&data, shape, device).expect("fixture");
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
    client: &CpuClient,
    [q, k, v]: &[Tensor<CpuRuntime>; 3],
    g: &KvStartGeom,
    kv_start: Option<&Tensor<CpuRuntime>>,
    layout: AttnOutLayout,
) -> (Tensor<CpuRuntime>, Tensor<CpuRuntime>) {
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

/// Bit identity, element by element, naming the first mismatch: a padded row
/// runs alone over its own keys, so it forms the bits its unpadded run does.
fn assert_bits(got: &[f32], want: &[f32], tag: &str) {
    assert_eq!(got.len(), want.len(), "{tag}: length");
    if let Some(i) = (0..got.len()).find(|&i| got[i].to_bits() != want[i].to_bits()) {
        panic!(
            "{tag}: element {i} is {:e} ({:#010x}), reference {:e} ({:#010x})",
            got[i],
            got[i].to_bits(),
            want[i],
            want[i].to_bits()
        );
    }
}

fn check_case(g: KvStartGeom, starts: &[i32], dtype: DType, label: &str) {
    if dtype != DType::F32 && !cfg!(feature = "f16") {
        return;
    }
    let (client, device) = cpu_setup();
    let tensors = fixture(&g, dtype, &device);
    let kv_start = Tensor::<CpuRuntime>::from_slice(starts, &[g.batch], &device).expect("starts");
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
        assert_bits(
            &got,
            &want_out,
            &format!("{tag} out (max abs diff {d_out:.3e})"),
        );
        assert_bits(
            &read_f32(&lse),
            &want_lse,
            &format!("{tag} lse (max abs diff {d_lse:.3e})"),
        );
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

    let zeros = Tensor::<CpuRuntime>::zeros(&[g.batch], DType::I32, &device).expect("zeros");
    let (out_none, lse_none) = run(&client, &tensors, &g, None, AttnOutLayout::HeadMajor);
    let (out_zero, lse_zero) = run(
        &client,
        &tensors,
        &g,
        Some(&zeros),
        AttnOutLayout::HeadMajor,
    );
    let bits = |t: &Tensor<CpuRuntime>| read_f32(t).iter().map(|x| x.to_bits()).collect::<Vec<_>>();
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

/// The fused F32 decode loop, and the standard path for the other dtypes.
#[test]
fn decode_non_causal() {
    all_dtypes(geom(1, 40, 64, false, 0), &STARTS, "cpu decode D=64 Sk=40");
}

/// Row 2's start reaches the end of the key axis: zeros, no NaN.
#[test]
fn decode_dead_row() {
    all_dtypes(
        geom(1, 40, 64, false, 0),
        &[0, 39, 40],
        "cpu decode dead row",
    );
}

/// Causal decode takes the standard path at every dtype.
#[test]
fn decode_causal_standard_path() {
    all_dtypes(geom(1, 40, 64, true, 0), &STARTS, "cpu decode causal D=64");
}

/// Prefill: the rows below each start are dead under the causal mask.
#[test]
fn prefill_causal() {
    all_dtypes(
        geom(40, 40, 64, true, 0),
        &STARTS,
        "cpu prefill causal S=40",
    );
}

#[test]
fn chunked_causal() {
    all_dtypes(
        geom(8, 40, 64, true, 0),
        &STARTS,
        "cpu chunked causal Sq=8 Sk=40",
    );
}

#[test]
fn window_with_kv_start() {
    all_dtypes(
        geom(40, 40, 64, true, 8),
        &STARTS,
        "cpu window=8 causal S=40",
    );
    all_dtypes(
        geom(16, 64, 64, false, 12),
        &[0, 20, 40],
        "cpu window=12 non-causal Sq=16 Sk=64",
    );
}

/// The DiT shape: non-causal short query, with and without a dead row.
#[test]
fn short_query_non_causal() {
    all_dtypes(
        geom(11, 40, 128, false, 0),
        &STARTS,
        "cpu Sq=11 D=128 Sk=40",
    );
    all_dtypes(
        geom(11, 11, 128, false, 0),
        &[0, 5, 11],
        "cpu Sq=11 D=128 Sk=11 dead row",
    );
}

/// `kv_seq_len` narrows first, then the start applies within the narrowed
/// axis.
#[test]
fn kv_seq_len_with_kv_start() {
    let g = geom(1, 40, 64, false, 0);
    let (client, device) = cpu_setup();
    let [q, k, v] = fixture(&g, DType::F32, &device);
    let kv_start = Tensor::<CpuRuntime>::from_slice(&STARTS, &[3], &device).expect("starts");
    let (out_full, _) = client
        .flash_attention_fwd(
            &q,
            &k,
            &v,
            g.num_heads,
            g.num_kv_heads,
            g.head_dim,
            false,
            0,
            Some(KvSeqLen::Shared(30)),
            Some(&kv_start),
            AttnOutLayout::HeadMajor,
        )
        .expect("with kv_seq_len");
    let k30 = k
        .narrow(2, 0, 30)
        .expect("narrow")
        .contiguous()
        .expect("contiguous");
    let v30 = v
        .narrow(2, 0, 30)
        .expect("narrow")
        .contiguous()
        .expect("contiguous");
    let (out_narrow, _) = client
        .flash_attention_fwd(
            &q,
            &k30,
            &v30,
            g.num_heads,
            g.num_kv_heads,
            g.head_dim,
            false,
            0,
            None,
            Some(&kv_start),
            AttnOutLayout::HeadMajor,
        )
        .expect("narrowed");
    assert_eq!(out_full.to_vec::<f32>(), out_narrow.to_vec::<f32>());
}

/// Shape and dtype checks on the start vector.
#[test]
fn rejects_malformed_kv_start() {
    let g = geom(1, 40, 64, false, 0);
    let (client, device) = cpu_setup();
    let tensors = fixture(&g, DType::F32, &device);
    let wrong_len = Tensor::<CpuRuntime>::from_slice(&[0i32, 1], &[2], &device).expect("t");
    let wrong_dtype =
        Tensor::<CpuRuntime>::from_slice(&[0f32, 1.0, 2.0], &[3], &device).expect("t");
    for bad in [&wrong_len, &wrong_dtype] {
        let err = client.flash_attention_fwd(
            &tensors[0],
            &tensors[1],
            &tensors[2],
            g.num_heads,
            g.num_kv_heads,
            g.head_dim,
            false,
            0,
            None,
            Some(bad),
            AttnOutLayout::HeadMajor,
        );
        assert!(err.is_err(), "malformed kv_start was accepted");
    }
}

/// One row of a per-row-end batch, decoded alone: the same start, and
/// `KvSeqLen::Shared` at the row's own end.
fn row_alone(
    client: &CpuClient,
    [q, k, v]: &[Tensor<CpuRuntime>; 3],
    g: &KvStartGeom,
    device: &CpuDevice,
    b: usize,
    start: i32,
    end: usize,
) -> (Vec<f32>, Vec<f32>) {
    let row = |t: &Tensor<CpuRuntime>| {
        t.narrow(0, b, 1)
            .expect("narrow row")
            .contiguous()
            .expect("row contiguous")
    };
    let start_t = Tensor::<CpuRuntime>::from_slice(&[start], &[1], device).expect("row start");
    let (out, lse) = client
        .flash_attention_fwd(
            &row(q),
            &row(k),
            &row(v),
            g.num_heads,
            g.num_kv_heads,
            g.head_dim,
            g.causal,
            g.window,
            Some(KvSeqLen::Shared(end)),
            Some(&start_t),
            AttnOutLayout::HeadMajor,
        )
        .expect("row alone");
    (read_f32(&out), read_f32(&lse))
}

/// Rows with different ends in one call, each held to the bits its own
/// single-row run forms. A row whose start equals its end sees no key.
fn check_ends(g: KvStartGeom, starts: &[i32], ends: &[i32], label: &str) {
    let (client, device) = cpu_setup();
    let tensors = fixture(&g, DType::F32, &device);
    let kv_start = Tensor::<CpuRuntime>::from_slice(starts, &[g.batch], &device).expect("starts");
    let kv_end = Tensor::<CpuRuntime>::from_slice(ends, &[g.batch], &device).expect("ends");
    let (out, lse) = client
        .flash_attention_fwd(
            &tensors[0],
            &tensors[1],
            &tensors[2],
            g.num_heads,
            g.num_kv_heads,
            g.head_dim,
            g.causal,
            g.window,
            Some(KvSeqLen::PerRow(&kv_end)),
            Some(&kv_start),
            AttnOutLayout::HeadMajor,
        )
        .expect("per-row ends");
    let out = read_f32(&out);
    let lse = read_f32(&lse);
    let row_len = g.num_heads * g.seq_q * g.head_dim;
    let lse_len = g.num_heads * g.seq_q;
    for b in 0..g.batch {
        let end = ends[b] as usize;
        let (want_out, want_lse) = row_alone(&client, &tensors, &g, &device, b, starts[b], end);
        let got_out = &out[b * row_len..(b + 1) * row_len];
        let got_lse = &lse[b * lse_len..(b + 1) * lse_len];
        assert_bits(got_out, &want_out, &format!("{label} row {b} out"));
        assert_bits(got_lse, &want_lse, &format!("{label} row {b} lse"));
        if starts[b].max(0) as usize == end {
            assert!(
                got_out.iter().all(|x| *x == 0.0),
                "{label} row {b}: no valid key, output is not zero"
            );
            assert!(
                got_lse.iter().all(|x| *x == f32::NEG_INFINITY),
                "{label} row {b}: no valid key, LSE is not -inf"
            );
        }
    }
}

/// Decode with ends that differ across rows: one at the buffer's dim-2, the
/// others below it.
#[test]
fn decode_per_row_ends() {
    check_ends(
        geom(1, 40, 64, false, 0),
        &STARTS,
        &[40, 31, 22],
        "cpu decode per-row ends",
    );
}

/// The same axis through the composed path, causal and windowed.
#[test]
fn prefill_per_row_ends() {
    check_ends(
        geom(8, 40, 64, true, 0),
        &STARTS,
        &[40, 33, 25],
        "cpu prefill causal per-row ends",
    );
    check_ends(
        geom(8, 40, 64, true, 8),
        &STARTS,
        &[40, 33, 25],
        "cpu prefill window=8 per-row ends",
    );
}

/// Row 1's start equals its end: zeros and `-inf`, never NaN.
#[test]
fn per_row_end_equal_to_start() {
    check_ends(
        geom(1, 40, 64, false, 0),
        &[0, 12, 17],
        &[40, 12, 30],
        "cpu per-row end equals start",
    );
}

/// A zero start and a nonzero one in the same batch, each with its own end.
#[test]
fn per_row_ends_mixed_starts() {
    check_ends(
        geom(1, 40, 64, false, 0),
        &[0, 9, 0],
        &[40, 28, 17],
        "cpu per-row ends mixed starts",
    );
    check_ends(
        geom(4, 40, 64, true, 0),
        &[0, 9, 0],
        &[40, 28, 17],
        "cpu per-row ends mixed starts causal",
    );
}

/// Shape, dtype and value checks on the end vector.
#[test]
fn rejects_malformed_kv_seq_len_ends() {
    let g = geom(1, 40, 64, false, 0);
    let (client, device) = cpu_setup();
    let tensors = fixture(&g, DType::F32, &device);
    let kv_start = Tensor::<CpuRuntime>::from_slice(&STARTS, &[3], &device).expect("starts");
    let i32s = |data: &[i32], shape: &[usize]| {
        Tensor::<CpuRuntime>::from_slice(data, shape, &device).expect("ends")
    };
    let past_buffer = i32s(&[40, 41, 40], &[3]);
    let below_start = i32s(&[40, 4, 40], &[3]);
    let wrong_len = i32s(&[40, 40], &[2]);
    let wrong_dtype =
        Tensor::<CpuRuntime>::from_slice(&[40f32, 40.0, 40.0], &[3], &device).expect("ends");
    for (bad, case) in [
        (&past_buffer, "end past the buffer"),
        (&below_start, "end below its row's start"),
        (&wrong_len, "wrong shape"),
        (&wrong_dtype, "wrong dtype"),
    ] {
        let err = client.flash_attention_fwd(
            &tensors[0],
            &tensors[1],
            &tensors[2],
            g.num_heads,
            g.num_kv_heads,
            g.head_dim,
            false,
            0,
            Some(KvSeqLen::PerRow(bad)),
            Some(&kv_start),
            AttnOutLayout::HeadMajor,
        );
        assert!(err.is_err(), "{case} was accepted");
    }
}
