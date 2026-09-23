//! A row's decode attention result does not depend on how many rows share
//! the launch.
//!
//! The decode grid cuts the KV span into slices at absolute positions whose
//! length is fixed by the head count, head dimension and device — never by
//! the batch size — and the combine folds the slices in order. So row `b` of
//! a B-row decode must be the same bits, output and LSE, as the decode of row
//! `b` on its own, at every span length: below one slice (the whole-sequence
//! kernel), a few slices, and enough to reach the split cap's regime. Slices
//! are counted from each row's first key, so a left-padded row is cut as its
//! unpadded single-row decode is and must match it the same way.
//!
//! A per-row `KvSeqLen::PerRow(ends)` moves the other edge: row `b` attends
//! over `[kv_start[b], ends[b])`, and the slices past that end come out empty.
//! So a row with its own end must also be the same bits as its own decode over
//! exactly those keys, at any buffer capacity. A sliding window anchors at
//! that same per-row end, so a windowed row with its own end must match its
//! own decode over `[start_b, end_b)` under the same window.
//!
//! Run with:
//!   cd boostr && cargo test --features cuda --test flash_decode_batch_invariance

#![cfg(feature = "cuda")]

use boostr::ops::traits::attention::flash::FlashAttentionOps;
use boostr::ops::{AttnOutLayout, KvSeqLen};
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
use numr::tensor::Tensor;

const NUM_HEADS: usize = 16;
const NUM_KV_HEADS: usize = 2;
const BATCHES: [usize; 4] = [1, 2, 3, 8];
const HEAD_DIMS: [usize; 2] = [64, 128];
/// Under one slice, a handful of slices, and past the reference length.
const KV_LENS: [usize; 3] = [31, 700, 4096];
/// Shorter than every row's span, and long enough to cover some of them.
const WINDOWS: [usize; 2] = [16, 256];

fn cuda() -> Option<(CudaClient, CudaDevice)> {
    let device = CudaDevice::new(0);
    let client = CudaClient::new(device.clone()).ok()?;
    Some((client, device))
}

/// Deterministic values with no repeated rows across the batch, so a row
/// read from the wrong batch slot changes the answer.
fn values(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 * ((i as f32) * 0.731 + seed).sin() + 0.1 * ((i as f32) * 0.017).cos())
        .collect()
}

/// Per-row and window arguments to [`decode`], grouped to keep its own
/// argument list short.
struct DecodeOpts<'a> {
    kv_start: Option<&'a Tensor<CudaRuntime>>,
    kv_seq_len: Option<KvSeqLen<'a, CudaRuntime>>,
    window: usize,
}

fn decode(
    client: &CudaClient,
    q: &Tensor<CudaRuntime>,
    k: &Tensor<CudaRuntime>,
    v: &Tensor<CudaRuntime>,
    head_dim: usize,
    opts: DecodeOpts<'_>,
) -> (Vec<f32>, Vec<f32>) {
    let (out, lse) = client
        .flash_attention_fwd(
            q,
            k,
            v,
            NUM_HEADS,
            NUM_KV_HEADS,
            head_dim,
            false,
            opts.window,
            opts.kv_seq_len,
            opts.kv_start,
            AttnOutLayout::HeadMajor,
        )
        .expect("flash_attention_fwd");
    (out.to_vec::<f32>(), lse.to_vec::<f32>())
}

/// Left-padding starts for a batch: row 0 unpadded, the rest staggered so
/// no two rows share a start, and never past the span.
fn starts(batch: usize, kv_len: usize) -> Vec<i32> {
    (0..batch)
        .map(|b| ((b * 37 + 13 * b * b) % kv_len.max(1)) as i32 * (b > 0) as i32)
        .collect()
}

/// Per-row key ends for a batch: row 0 reads through to `limit`, the rest
/// stop earlier and staggered, and every row keeps at least one key.
fn ends(batch: usize, limit: usize, starts: &[i32]) -> Vec<i32> {
    (0..batch)
        .map(|b| {
            let start = starts[b] as usize;
            let cut = (b * 29 + 7 * b * b) % (limit - start);
            (limit - cut) as i32
        })
        .collect()
}

/// Row `b`'s own keys: the batch row's `[start, end)` span out of a
/// `capacity`-key buffer, as an unpadded `[1, H_kv, end - start, D]` tensor.
fn row_span(
    kv_all: &[f32],
    b: usize,
    start: usize,
    end: usize,
    capacity: usize,
    head_dim: usize,
) -> Vec<f32> {
    let row = NUM_KV_HEADS * capacity * head_dim;
    (0..NUM_KV_HEADS)
        .flat_map(|h| {
            let base = b * row + h * capacity * head_dim;
            kv_all[base + start * head_dim..base + end * head_dim].to_vec()
        })
        .collect()
}

fn check(
    client: &CudaClient,
    device: &CudaDevice,
    batch: usize,
    head_dim: usize,
    kv_len: usize,
    padded: bool,
) {
    let starts = if padded {
        starts(batch, kv_len)
    } else {
        vec![0; batch]
    };
    let kv_start = padded
        .then(|| Tensor::<CudaRuntime>::from_slice(&starts, &[batch], device).expect("kv_start"));
    let q_all = values(batch * NUM_HEADS * head_dim, 0.1);
    let k_all = values(batch * NUM_KV_HEADS * kv_len * head_dim, 0.2);
    let v_all = values(batch * NUM_KV_HEADS * kv_len * head_dim, 0.3);
    let q = Tensor::<CudaRuntime>::from_slice(&q_all, &[batch, NUM_HEADS, 1, head_dim], device)
        .expect("q");
    let k =
        Tensor::<CudaRuntime>::from_slice(&k_all, &[batch, NUM_KV_HEADS, kv_len, head_dim], device)
            .expect("k");
    let v =
        Tensor::<CudaRuntime>::from_slice(&v_all, &[batch, NUM_KV_HEADS, kv_len, head_dim], device)
            .expect("v");
    let (out, lse) = decode(
        client,
        &q,
        &k,
        &v,
        head_dim,
        DecodeOpts {
            kv_start: kv_start.as_ref(),
            kv_seq_len: None,
            window: 0,
        },
    );

    let q_row = NUM_HEADS * head_dim;
    for b in 0..batch {
        let start = starts[b] as usize;
        let own = kv_len - start;
        let q1 = Tensor::<CudaRuntime>::from_slice(
            &q_all[b * q_row..(b + 1) * q_row],
            &[1, NUM_HEADS, 1, head_dim],
            device,
        )
        .expect("q row");
        let k1 = Tensor::<CudaRuntime>::from_slice(
            &row_span(&k_all, b, start, kv_len, kv_len, head_dim),
            &[1, NUM_KV_HEADS, own, head_dim],
            device,
        )
        .expect("k row");
        let v1 = Tensor::<CudaRuntime>::from_slice(
            &row_span(&v_all, b, start, kv_len, kv_len, head_dim),
            &[1, NUM_KV_HEADS, own, head_dim],
            device,
        )
        .expect("v row");
        let (out1, lse1) = decode(
            client,
            &q1,
            &k1,
            &v1,
            head_dim,
            DecodeOpts {
                kv_start: None,
                kv_seq_len: None,
                window: 0,
            },
        );

        for (i, (got, want)) in out[b * q_row..(b + 1) * q_row]
            .iter()
            .zip(&out1)
            .enumerate()
        {
            assert!(
                got.to_bits() == want.to_bits(),
                "B={batch} D={head_dim} kv={kv_len} start={start}: row {b} output element {i} is \
                 {got:e}, alone it is {want:e}"
            );
        }
        for (h, (got, want)) in lse[b * NUM_HEADS..(b + 1) * NUM_HEADS]
            .iter()
            .zip(&lse1)
            .enumerate()
        {
            assert!(
                got.to_bits() == want.to_bits(),
                "B={batch} D={head_dim} kv={kv_len} start={start}: row {b} head {h} LSE is {got:e}, \
                 alone it is {want:e}"
            );
        }
    }
}

#[test]
fn decode_rows_do_not_depend_on_the_batch() {
    let Some((client, device)) = cuda() else {
        eprintln!("CUDA not available, skipping");
        return;
    };
    for &head_dim in &HEAD_DIMS {
        for &kv_len in &KV_LENS {
            for &batch in &BATCHES {
                check(&client, &device, batch, head_dim, kv_len, false);
            }
        }
    }
}

#[test]
fn padded_rows_match_their_own_unpadded_decode() {
    let Some((client, device)) = cuda() else {
        eprintln!("CUDA not available, skipping");
        return;
    };
    for &head_dim in &HEAD_DIMS {
        for &kv_len in &KV_LENS {
            for &batch in &BATCHES[1..] {
                check(&client, &device, batch, head_dim, kv_len, true);
            }
        }
    }
}

/// Shape and padding for one [`check_ends`] case, grouped to keep its
/// argument list short.
///
/// `capacity` is the key tensor's dim-2; `limit` is the largest end any row
/// gets, so a capacity far above it leaves most slices of the grid empty.
struct EndsCase {
    batch: usize,
    head_dim: usize,
    capacity: usize,
    limit: usize,
    padded: bool,
    window: usize,
}

/// Runs one batch with per-row key ends and compares each row against its own
/// decode over exactly `[start_b, end_b)`.
fn check_ends(client: &CudaClient, device: &CudaDevice, case: EndsCase) {
    let EndsCase {
        batch,
        head_dim,
        capacity,
        limit,
        padded,
        window,
    } = case;
    let starts = if padded {
        starts(batch, limit)
    } else {
        vec![0; batch]
    };
    let ends = ends(batch, limit, &starts);
    let kv_start = padded
        .then(|| Tensor::<CudaRuntime>::from_slice(&starts, &[batch], device).expect("kv_start"));
    let kv_end = Tensor::<CudaRuntime>::from_slice(&ends, &[batch], device).expect("kv_end");

    let q_all = values(batch * NUM_HEADS * head_dim, 0.1);
    let k_all = values(batch * NUM_KV_HEADS * capacity * head_dim, 0.2);
    let v_all = values(batch * NUM_KV_HEADS * capacity * head_dim, 0.3);
    let q = Tensor::<CudaRuntime>::from_slice(&q_all, &[batch, NUM_HEADS, 1, head_dim], device)
        .expect("q");
    let k = Tensor::<CudaRuntime>::from_slice(
        &k_all,
        &[batch, NUM_KV_HEADS, capacity, head_dim],
        device,
    )
    .expect("k");
    let v = Tensor::<CudaRuntime>::from_slice(
        &v_all,
        &[batch, NUM_KV_HEADS, capacity, head_dim],
        device,
    )
    .expect("v");
    let (out, lse) = decode(
        client,
        &q,
        &k,
        &v,
        head_dim,
        DecodeOpts {
            kv_start: kv_start.as_ref(),
            kv_seq_len: Some(KvSeqLen::PerRow(&kv_end)),
            window,
        },
    );

    let q_row = NUM_HEADS * head_dim;
    for b in 0..batch {
        let start = starts[b] as usize;
        let end = ends[b] as usize;
        let own = end - start;
        let q1 = Tensor::<CudaRuntime>::from_slice(
            &q_all[b * q_row..(b + 1) * q_row],
            &[1, NUM_HEADS, 1, head_dim],
            device,
        )
        .expect("q row");
        let k1 = Tensor::<CudaRuntime>::from_slice(
            &row_span(&k_all, b, start, end, capacity, head_dim),
            &[1, NUM_KV_HEADS, own, head_dim],
            device,
        )
        .expect("k row");
        let v1 = Tensor::<CudaRuntime>::from_slice(
            &row_span(&v_all, b, start, end, capacity, head_dim),
            &[1, NUM_KV_HEADS, own, head_dim],
            device,
        )
        .expect("v row");
        let (out1, lse1) = decode(
            client,
            &q1,
            &k1,
            &v1,
            head_dim,
            DecodeOpts {
                kv_start: None,
                kv_seq_len: None,
                window,
            },
        );

        for (i, (got, want)) in out[b * q_row..(b + 1) * q_row]
            .iter()
            .zip(&out1)
            .enumerate()
        {
            assert!(
                got.to_bits() == want.to_bits(),
                "B={batch} D={head_dim} cap={capacity} start={start} end={end} win={window}: row \
                 {b} output \
                 element {i} is {got:e}, alone it is {want:e}"
            );
        }
        for (h, (got, want)) in lse[b * NUM_HEADS..(b + 1) * NUM_HEADS]
            .iter()
            .zip(&lse1)
            .enumerate()
        {
            assert!(
                got.to_bits() == want.to_bits(),
                "B={batch} D={head_dim} cap={capacity} start={start} end={end} win={window}: row \
                 {b} head {h} LSE is {got:e}, alone it is {want:e}"
            );
        }
    }
}

#[test]
fn per_row_end_rows_match_their_own_unpadded_decode() {
    let Some((client, device)) = cuda() else {
        eprintln!("CUDA not available, skipping");
        return;
    };
    for &head_dim in &HEAD_DIMS {
        for &kv_len in &KV_LENS {
            for &batch in &BATCHES[1..] {
                check_ends(
                    &client,
                    &device,
                    EndsCase {
                        batch,
                        head_dim,
                        capacity: kv_len,
                        limit: kv_len,
                        padded: false,
                        window: 0,
                    },
                );
            }
        }
    }
}

#[test]
fn per_row_end_with_left_padding_matches_its_own_decode() {
    let Some((client, device)) = cuda() else {
        eprintln!("CUDA not available, skipping");
        return;
    };
    for &head_dim in &HEAD_DIMS {
        for &kv_len in &KV_LENS {
            for &batch in &BATCHES[1..] {
                check_ends(
                    &client,
                    &device,
                    EndsCase {
                        batch,
                        head_dim,
                        capacity: kv_len,
                        limit: kv_len,
                        padded: true,
                        window: 0,
                    },
                );
            }
        }
    }
}

/// Every row ends far below the buffer's capacity, so the grid offers many
/// more slices than any row's span fills and most of them must come out empty.
#[test]
fn per_row_end_far_below_capacity_matches_its_own_decode() {
    let Some((client, device)) = cuda() else {
        eprintln!("CUDA not available, skipping");
        return;
    };
    for &head_dim in &HEAD_DIMS {
        for &padded in &[false, true] {
            for &batch in &BATCHES[1..] {
                check_ends(
                    &client,
                    &device,
                    EndsCase {
                        batch,
                        head_dim,
                        capacity: 4096,
                        limit: 60,
                        padded,
                        window: 0,
                    },
                );
            }
        }
    }
}

/// A sliding window on top of per-row ends: the window's first key is counted
/// back from the row's own end, so the row still sees exactly what its own
/// decode over `[start_b, end_b)` under the same window sees.
#[test]
fn per_row_end_with_window_matches_its_own_decode() {
    let Some((client, device)) = cuda() else {
        eprintln!("CUDA not available, skipping");
        return;
    };
    for &head_dim in &HEAD_DIMS {
        for &kv_len in &KV_LENS {
            for &window in &WINDOWS {
                for &padded in &[false, true] {
                    for &batch in &BATCHES[1..] {
                        check_ends(
                            &client,
                            &device,
                            EndsCase {
                                batch,
                                head_dim,
                                capacity: kv_len,
                                limit: kv_len,
                                padded,
                                window,
                            },
                        );
                    }
                }
            }
        }
    }
}

/// The windowed row also ends far below the buffer's capacity, so anchoring
/// the window at the capacity instead of the row's end would move every
/// surviving key.
#[test]
fn per_row_end_far_below_capacity_with_window_matches_its_own_decode() {
    let Some((client, device)) = cuda() else {
        eprintln!("CUDA not available, skipping");
        return;
    };
    for &head_dim in &HEAD_DIMS {
        for &window in &WINDOWS {
            for &padded in &[false, true] {
                for &batch in &BATCHES[1..] {
                    check_ends(
                        &client,
                        &device,
                        EndsCase {
                            batch,
                            head_dim,
                            capacity: 4096,
                            limit: 60,
                            padded,
                            window,
                        },
                    );
                }
            }
        }
    }
}
