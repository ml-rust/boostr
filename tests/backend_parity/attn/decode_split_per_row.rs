//! Backend parity for a per-row key end on the CUDA decode path.
//!
//! `KvSeqLen::PerRow(ends)` gives row `b` its own last key, so one batched KV
//! buffer holds rows of different lengths with no padding. The CPU client
//! walks `[kv_start[b], ends[b])` directly; the CUDA decode kernel derives the
//! same range per block and lets the slices past it fall out empty. The two
//! must agree on the output and on the log-sum-exp, which is the combine
//! pass's only other product.
//!
//! Both decode grid shapes run here: a capacity below the split threshold
//! keeps one block per `(batch, head)`, and a longer one cuts each row's span
//! into slices. `src/ops/cuda/attention/decode_split.rs` picks between them.
//!
//! A sliding window anchors at the row's own end on both backends: the CPU
//! narrows the row's keys to that end before applying the window, and the
//! CUDA kernel counts the window back from the same end.
//!
//! The per-row end reaches the decode kernel alone. At `seq_len_q > 1` the
//! CUDA flash path still refuses it, which the last test pins.

use super::helpers::*;
use boostr::ops::traits::attention::flash::FlashAttentionOps;
use boostr::ops::{AttnOutLayout, KvSeqLen};

/// Runs one decode shape with per-row ends on CPU and CUDA and asserts the
/// output and LSE agree.
///
/// `kv_capacity` is the key tensor's dim-2, which both backends read as the
/// memory stride. `ends[b]` is row `b`'s last key and `starts` its first, or
/// `None` for an unpadded batch.
#[allow(clippy::too_many_arguments)]
fn assert_per_row_decode_parity(
    label: &str,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_capacity: usize,
    ends: &[i32],
    starts: Option<&[i32]>,
    window_size: usize,
) {
    let batch = ends.len();
    let (cpu_client, cpu_device) = setup_cpu();

    let q_shape = [batch, num_heads, 1, head_dim];
    let kv_shape = [batch, num_kv_heads, kv_capacity, head_dim];

    let q_vec = det_tensor(&q_shape, &cpu_device).to_vec::<f32>();
    let k_vec = det_tensor(&kv_shape, &cpu_device).to_vec::<f32>();
    let v_vec = det_tensor(&kv_shape, &cpu_device).to_vec::<f32>();

    let q = numr::tensor::Tensor::from_slice(&q_vec, &q_shape, &cpu_device).unwrap();
    let k = numr::tensor::Tensor::from_slice(&k_vec, &kv_shape, &cpu_device).unwrap();
    let v = numr::tensor::Tensor::from_slice(&v_vec, &kv_shape, &cpu_device).unwrap();
    let cpu_ends = det_i32_tensor(ends, &[batch], &cpu_device);
    let cpu_starts = starts.map(|s| det_i32_tensor(s, &[batch], &cpu_device));

    let (cpu_out, cpu_lse) = cpu_client
        .flash_attention_fwd(
            &q,
            &k,
            &v,
            num_heads,
            num_kv_heads,
            head_dim,
            false,
            window_size,
            Some(KvSeqLen::PerRow(&cpu_ends)),
            cpu_starts.as_ref(),
            AttnOutLayout::HeadMajor,
        )
        .unwrap_or_else(|e| panic!("CPU per-row decode failed for {label}: {e}"));
    let cpu_out_vec = cpu_out.to_vec::<f32>();
    let cpu_lse_vec = cpu_lse.to_vec::<f32>();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use numr::tensor::Tensor;

        let q_c = Tensor::from_slice(&q_vec, &q_shape, &cuda_device).unwrap();
        let k_c = Tensor::from_slice(&k_vec, &kv_shape, &cuda_device).unwrap();
        let v_c = Tensor::from_slice(&v_vec, &kv_shape, &cuda_device).unwrap();
        let cuda_ends = Tensor::from_slice(ends, &[batch], &cuda_device).unwrap();
        let cuda_starts = starts.map(|s| Tensor::from_slice(s, &[batch], &cuda_device).unwrap());

        let (cuda_out, cuda_lse) = cuda_client
            .flash_attention_fwd(
                &q_c,
                &k_c,
                &v_c,
                num_heads,
                num_kv_heads,
                head_dim,
                false,
                window_size,
                Some(KvSeqLen::PerRow(&cuda_ends)),
                cuda_starts.as_ref(),
                AttnOutLayout::HeadMajor,
            )
            .unwrap_or_else(|e| panic!("CUDA per-row decode failed for {label}: {e}"));

        assert_parity_f32_tol(
            &cuda_out.to_vec::<f32>(),
            &cpu_out_vec,
            &format!("{label} output CUDA vs CPU"),
            1e-4,
            1e-6,
        );
        assert_parity_f32_tol(
            &cuda_lse.to_vec::<f32>(),
            &cpu_lse_vec,
            &format!("{label} lse CUDA vs CPU"),
            1e-4,
            1e-6,
        );
    });
}

/// Ragged ends on the whole-sequence grid: the capacity is below the split
/// threshold, so one block per `(batch, head)` walks each row's own range.
#[test]
fn per_row_ends_whole_sequence_parity() {
    assert_per_row_decode_parity("per_row_short", 4, 4, 64, 64, &[64, 37, 8, 1], None, 0);
}

/// Ragged ends on the split grid: each row cuts its own span into slices and
/// the slices past its end come out empty.
#[test]
fn per_row_ends_split_parity() {
    assert_per_row_decode_parity(
        "per_row_split",
        8,
        2,
        128,
        1024,
        &[1024, 777, 320, 33],
        None,
        0,
    );
}

/// Ragged ends with left padding on the whole-sequence grid: both edges of
/// each row's range vary.
#[test]
fn per_row_ends_with_kv_start_whole_sequence_parity() {
    assert_per_row_decode_parity(
        "per_row_short_padded",
        4,
        2,
        64,
        64,
        &[64, 37, 20, 9],
        Some(&[0, 5, 19, 3]),
        0,
    );
}

/// Ragged ends with left padding on the split grid.
#[test]
fn per_row_ends_with_kv_start_split_parity() {
    assert_per_row_decode_parity(
        "per_row_split_padded",
        8,
        2,
        128,
        1024,
        &[1024, 900, 512, 70],
        Some(&[0, 128, 300, 65]),
        0,
    );
}

/// A capacity far above every row's end: the grid is sized to the buffer, so
/// most of its slices do no work and must not reach the combine.
#[test]
fn per_row_ends_far_below_capacity_parity() {
    assert_per_row_decode_parity(
        "per_row_sparse_capacity",
        8,
        2,
        64,
        4096,
        &[60, 41, 17, 3],
        None,
        0,
    );
}

/// Ragged ends under a sliding window on the whole-sequence grid: the window
/// covers part of some rows and all of the shorter ones, so a window anchored
/// at the buffer instead of the row's end would pick different keys.
#[test]
fn per_row_ends_with_window_whole_sequence_parity() {
    assert_per_row_decode_parity(
        "per_row_short_window",
        4,
        4,
        64,
        64,
        &[64, 37, 8, 1],
        None,
        16,
    );
}

/// Ragged ends under a sliding window on the split grid, with left padding so
/// both edges of each row's range and the window's anchor all move.
#[test]
fn per_row_ends_with_window_split_parity() {
    assert_per_row_decode_parity(
        "per_row_split_window",
        8,
        2,
        128,
        1024,
        &[1024, 900, 512, 70],
        Some(&[0, 128, 300, 65]),
        128,
    );
}

/// A sliding window on rows that all end far below the buffer's capacity: the
/// window's first key is counted back from each row's own end.
#[test]
fn per_row_ends_far_below_capacity_with_window_parity() {
    assert_per_row_decode_parity(
        "per_row_sparse_capacity_window",
        8,
        2,
        64,
        4096,
        &[60, 41, 17, 3],
        None,
        16,
    );
}

/// The per-row end reaches the decode kernel alone; every other CUDA flash
/// tier reads one key length for the whole batch and refuses it.
#[cfg(feature = "cuda")]
#[test]
fn per_row_ends_above_one_query_row_is_refused() {
    with_cuda_backend(|client, device| {
        use numr::tensor::Tensor;

        let (_, cpu_device) = setup_cpu();
        let q_shape = [2usize, 4, 2, 64];
        let kv_shape = [2usize, 4, 32, 64];
        let q_vec = det_tensor(&q_shape, &cpu_device).to_vec::<f32>();
        let kv_vec = det_tensor(&kv_shape, &cpu_device).to_vec::<f32>();

        let q = Tensor::from_slice(&q_vec, &q_shape, &device).unwrap();
        let k = Tensor::from_slice(&kv_vec, &kv_shape, &device).unwrap();
        let v = Tensor::from_slice(&kv_vec, &kv_shape, &device).unwrap();
        let ends = Tensor::from_slice(&[32i32, 17], &[2], &device).unwrap();

        let err = client.flash_attention_fwd(
            &q,
            &k,
            &v,
            4,
            4,
            64,
            false,
            0,
            Some(KvSeqLen::PerRow(&ends)),
            None,
            AttnOutLayout::HeadMajor,
        );
        assert!(
            err.is_err(),
            "seq_len_q > 1 with a per-row key end must be refused on CUDA"
        );
    });
}
