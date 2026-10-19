//! KV cache rollback for speculative verification, end to end on CPU and CUDA.
//!
//! A tiny random GQA Llama runs three ways over the same tokens:
//!
//! - one token per forward from position 0, the reference rows;
//! - a prompt prefill, then ONE forward over M tokens at offset N;
//! - that verify forward, `LayeredKvCache::truncate` back to `N + j`, then
//!   one forward over different tokens at `N + j`.
//!
//! The multi-token forward at an offset must give the one-at-a-time rows.
//! The forward after a rollback must give the rows of a fresh cache that
//! never held the rejected tokens. Both hold to 1e-4 relative. With the
//! `cuda` feature the offset forward also runs on CUDA device 0.

mod common;

use boostr::inference::LayeredKvCache;
use boostr::model::LoadedModel;
use common::llama_tiny::{HEAD_DIM, KV_HEADS, LAYERS, VOCAB, tiny_llama_on};
use numr::dtype::DType;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

const SEED: u64 = 0x5bec_0de5;
/// Small, so the cache grows on the key axis during the runs.
const INITIAL_CAPACITY: usize = 4;
const MAX_SEQ_LEN: usize = 64;
const RTOL: f32 = 1e-4;

type Model = LoadedModel<CpuRuntime>;

fn model(device: &CpuDevice) -> Model {
    tiny_llama_on::<CpuRuntime>(device, SEED, DType::F32)
}

fn cache(device: &CpuDevice) -> LayeredKvCache<CpuRuntime> {
    LayeredKvCache::new_positional(
        LAYERS,
        1,
        KV_HEADS,
        INITIAL_CAPACITY,
        MAX_SEQ_LEN,
        HEAD_DIM,
        DType::F32,
        device,
    )
    .expect("cache")
}

/// `n` token ids in `0..VOCAB`, fixed by `seed`.
fn tokens(seed: u64, n: usize) -> Vec<i64> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 33) % VOCAB as u64) as i64
        })
        .collect()
}

/// One forward over `ids` at `position`; one logits row per token.
fn forward(
    model: &Model,
    kv: &mut LayeredKvCache<CpuRuntime>,
    ids: &[i64],
    position: usize,
    device: &CpuDevice,
) -> Vec<Vec<f32>> {
    let input = Tensor::<CpuRuntime>::from_slice(ids, &[1, ids.len()], device).expect("ids");
    let logits = model
        .forward_with_kv_cache(&input, kv, position)
        .expect("forward");
    assert_eq!(logits.shape(), &[1, ids.len(), VOCAB]);
    let all = logits.contiguous().expect("contiguous").to_vec::<f32>();
    all.chunks(VOCAB).map(<[f32]>::to_vec).collect()
}

/// Every token in its own forward, from position 0 on a fresh cache.
fn one_at_a_time(model: &Model, ids: &[i64], device: &CpuDevice) -> Vec<Vec<f32>> {
    let mut kv = cache(device);
    let mut rows = Vec::with_capacity(ids.len());
    for (pos, &id) in ids.iter().enumerate() {
        rows.extend(forward(model, &mut kv, &[id], pos, device));
    }
    assert_eq!(kv.seq_len(), ids.len());
    rows
}

/// Every token in one forward, from position 0 on a fresh cache.
fn all_at_once(model: &Model, ids: &[i64], device: &CpuDevice) -> Vec<Vec<f32>> {
    let mut kv = cache(device);
    forward(model, &mut kv, ids, 0, device)
}

fn assert_rows_close(got: &[Vec<f32>], want: &[Vec<f32>], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: row count");
    for (r, (g, w)) in got.iter().zip(want).enumerate() {
        let scale = w.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1.0);
        for (i, (a, b)) in g.iter().zip(w).enumerate() {
            assert!(
                a.is_finite() && (a - b).abs() <= RTOL * scale,
                "{what}: row {r} logit {i}: got {a}, want {b} (scale {scale})"
            );
        }
    }
}

/// Tokens of length `n` that differ from `avoid` at the first position.
fn other_tokens(seed: u64, n: usize, avoid: Option<i64>) -> Vec<i64> {
    let mut ids = tokens(seed, n);
    if let (Some(first), Some(a)) = (ids.first_mut(), avoid)
        && *first == a
    {
        *first = (a + 1) % VOCAB as i64;
    }
    ids
}

/// The model is not degenerate: different contexts give different rows.
#[test]
fn tiny_model_logits_depend_on_context() {
    let device = CpuDevice::new();
    let model = model(&device);
    let rows = one_at_a_time(&model, &tokens(1, 4), &device);
    let diff = rows[1]
        .iter()
        .zip(&rows[3])
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        diff > 1e-3,
        "rows 1 and 3 agree to {diff}; the test is vacuous"
    );
}

/// A prefill, then one forward over M tokens at offset N, gives the rows
/// of one token per forward.
#[test]
fn multi_token_forward_at_an_offset_matches_one_at_a_time() {
    let device = CpuDevice::new();
    let model = model(&device);
    for n in [1usize, 3, 7] {
        for m in [1usize, 2, 4, 6] {
            let ids = tokens(100 + (n * 10 + m) as u64, n + m);
            let reference = one_at_a_time(&model, &ids, &device);

            let mut kv = cache(&device);
            let prefill = forward(&model, &mut kv, &ids[..n], 0, &device);
            assert_rows_close(&prefill, &reference[..n], &format!("N={n}: prefill"));
            let verify = forward(&model, &mut kv, &ids[n..], n, &device);
            assert_eq!(kv.seq_len(), n + m);
            assert_rows_close(
                &verify,
                &reference[n..],
                &format!("N={n} M={m}: verify forward"),
            );
        }
    }
}

/// After a verify forward over M tokens, truncation to `N + j` and one
/// forward over different tokens give the rows of a fresh run over
/// `prompt + drafts[..j] + new`.
#[test]
fn rollback_then_refeed_matches_a_fresh_run() {
    let device = CpuDevice::new();
    let model = model(&device);
    for n in [1usize, 4, 9] {
        for m in [2usize, 5] {
            let ids = tokens(300 + (n * 10 + m) as u64, n + m);
            for j in 0..=m {
                for new_len in [1usize, 3] {
                    let mut kv = cache(&device);
                    forward(&model, &mut kv, &ids[..n], 0, &device);
                    forward(&model, &mut kv, &ids[n..], n, &device);
                    kv.truncate(n + j).expect("truncate");
                    assert_eq!(kv.seq_len(), n + j);
                    for idx in 0..kv.num_layers() {
                        assert_eq!(kv.layer(idx).expect("layer").seq_len(), n + j);
                    }

                    let new = other_tokens(
                        900 + (n * 100 + m * 10 + j) as u64,
                        new_len,
                        ids.get(n + j).copied(),
                    );
                    let got = forward(&model, &mut kv, &new, n + j, &device);
                    assert_eq!(kv.seq_len(), n + j + new_len);

                    let mut fresh_ids = ids[..n + j].to_vec();
                    fresh_ids.extend_from_slice(&new);
                    let fresh = all_at_once(&model, &fresh_ids, &device);
                    assert_rows_close(
                        &got,
                        &fresh[n + j..],
                        &format!("N={n} M={m} j={j} new={new_len}: refeed after rollback"),
                    );
                }
            }
        }
    }
}

/// Several speculative rounds on one cache: draft k tokens, verify in one
/// forward, keep `accepted` of them, roll back, repeat. Every verify row
/// matches the one-at-a-time rows of the sequence kept so far.
#[test]
fn chained_rounds_on_one_cache_match_one_at_a_time() {
    let device = CpuDevice::new();
    let model = model(&device);
    let prompt = tokens(7, 5);
    let mut kept = prompt.clone();
    let mut kv = cache(&device);
    forward(&model, &mut kv, &prompt, 0, &device);

    for (round, &(k, accepted)) in [(4usize, 2usize), (3, 0), (5, 5), (2, 1), (4, 3)]
        .iter()
        .enumerate()
    {
        let base = kept.len();
        assert_eq!(kv.seq_len(), base, "round {round}: cache length");
        let drafts = other_tokens(5000 + round as u64, k, None);
        let verify = forward(&model, &mut kv, &drafts, base, &device);

        let mut full = kept.clone();
        full.extend_from_slice(&drafts);
        let reference = one_at_a_time(&model, &full, &device);
        assert_rows_close(
            &verify,
            &reference[base..],
            &format!("round {round}: verify over {k} drafts"),
        );

        kv.truncate(base + accepted).expect("rollback");
        kept.extend_from_slice(&drafts[..accepted]);
    }

    // A last single-token step after the final rollback.
    let base = kept.len();
    let next = other_tokens(77, 1, None);
    let got = forward(&model, &mut kv, &next, base, &device);
    kept.extend_from_slice(&next);
    let reference = one_at_a_time(&model, &kept, &device);
    assert_rows_close(&got, &reference[base..], "final step after rollbacks");
}

/// The multi-token forward at an offset on CUDA, against one token per CUDA forward.
#[cfg(feature = "cuda")]
mod cuda {
    use super::{INITIAL_CAPACITY, MAX_SEQ_LEN, SEED, assert_rows_close, tokens};
    use crate::common::llama_tiny::{HEAD_DIM, KV_HEADS, LAYERS, VOCAB, tiny_llama};
    use boostr::inference::LayeredKvCache;
    use boostr::model::LoadedModel;
    use numr::dtype::DType;
    use numr::runtime::cuda::{CudaDevice, CudaRuntime};
    use numr::tensor::Tensor;

    fn cache(device: &CudaDevice) -> LayeredKvCache<CudaRuntime> {
        LayeredKvCache::new_positional(
            LAYERS,
            1,
            KV_HEADS,
            INITIAL_CAPACITY,
            MAX_SEQ_LEN,
            HEAD_DIM,
            DType::F32,
            device,
        )
        .expect("cache")
    }

    /// One forward over `ids` at `position`; one logits row per token, read back for the check.
    fn forward(
        model: &LoadedModel<CudaRuntime>,
        kv: &mut LayeredKvCache<CudaRuntime>,
        ids: &[i64],
        position: usize,
        device: &CudaDevice,
    ) -> Vec<Vec<f32>> {
        let input = Tensor::<CudaRuntime>::from_slice(ids, &[1, ids.len()], device).expect("ids");
        let logits = model
            .forward_with_kv_cache(&input, kv, position)
            .expect("forward");
        assert_eq!(logits.shape(), &[1, ids.len(), VOCAB]);
        let all = logits.contiguous().expect("contiguous").to_vec::<f32>();
        all.chunks(VOCAB).map(<[f32]>::to_vec).collect()
    }

    #[test]
    fn cuda_multi_token_forward_at_an_offset_matches_one_at_a_time() {
        let device = CudaDevice::new(0);
        let model = tiny_llama(&device, SEED, DType::F32);
        for n in [1usize, 3, 7] {
            for m in [1usize, 2, 4, 6] {
                let ids = tokens(100 + (n * 10 + m) as u64, n + m);
                let mut one = cache(&device);
                let mut reference = Vec::with_capacity(ids.len());
                for (pos, &id) in ids.iter().enumerate() {
                    reference.extend(forward(&model, &mut one, &[id], pos, &device));
                }

                let mut kv = cache(&device);
                let prefill = forward(&model, &mut kv, &ids[..n], 0, &device);
                assert_rows_close(&prefill, &reference[..n], &format!("CUDA N={n}: prefill"));
                let verify = forward(&model, &mut kv, &ids[n..], n, &device);
                assert_eq!(kv.seq_len(), n + m);
                assert_rows_close(
                    &verify,
                    &reference[n..],
                    &format!("CUDA N={n} M={m}: verify forward"),
                );
            }
        }
    }
}

/// A rollback past the cache length is refused and the cache still
/// decodes correctly.
#[test]
fn refused_rollback_leaves_the_cache_usable() {
    let device = CpuDevice::new();
    let model = model(&device);
    let ids = tokens(11, 6);
    let mut kv = cache(&device);
    forward(&model, &mut kv, &ids[..5], 0, &device);

    let err = kv.truncate(6).expect_err("6 is past 5");
    assert!(err.to_string().contains('6'), "{err}");
    assert_eq!(kv.seq_len(), 5);

    let got = forward(&model, &mut kv, &ids[5..], 5, &device);
    let reference = one_at_a_time(&model, &ids, &device);
    assert_rows_close(&got, &reference[5..], "decode after a refused rollback");
}
