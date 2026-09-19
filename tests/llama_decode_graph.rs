//! CUDA graph-mode decode for Llama against the eager KV-cache forward.
//!
//! Run with:
//!   cd boostr && cargo test --release --features cuda --test llama_decode_graph
//!
//! `graph_matches_eager`: a tiny synthetic Llama. Prefill 4 tokens eagerly
//! into a full-capacity cache, build the graph the way a serving session
//! does (stable buffers, device scalars, uncaptured warmup, reset and
//! re-prefill into the same cache, capture, then `pre_replay_and_launch` per
//! step), and replay 6 greedy steps. Eager decode on a fresh cache with the
//! same ids must give the same argmax at every step and logits within
//! tolerance. Runs at F32 for two capacities (one keeps the whole-sequence
//! decode kernel, the other takes the split path), then at F16 and BF16.
//!
//! `eager_is_deterministic`: the eager path alone, twice, bit-identical
//! logits. Separates a graph defect from run-to-run noise.
//!
//! `real_file_graph_matches_eager`: the same comparison against a real
//! checkpoint under `BOOSTR_LLAMA_MODEL` (a GGUF file or a SafeTensors
//! model directory), with a printed per-step diff table. Run with
//! `--nocapture` to see it, and `--release` — it is not tiny.

#![cfg(feature = "cuda")]

mod common;

use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use common::llama_graph::{Dims, eager_decode, full_cache, graph_decode, load_real_model, prefill};
use common::llama_tiny::{HEAD_DIM, KV_HEADS, LAYERS, VOCAB, tiny_llama};
use common::qwen35_cuda::{argmax, cuda_available, cuda_setup, max_abs_diff};
use numr::dtype::DType;
use numr::runtime::cuda::{CudaClient, CudaDevice};

static CUDA_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn cuda_lock() -> std::sync::MutexGuard<'static, ()> {
    CUDA_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|p| p.into_inner())
}

const PROMPT: [i64; 4] = [3, 7, 11, 5];
const DECODE_STEPS: usize = 6;
const SEED: u64 = 0x11a3_a0de;

const TINY_DIMS: Dims = Dims {
    layers: LAYERS,
    kv_heads: KV_HEADS,
    head_dim: HEAD_DIM,
    vocab: VOCAB,
};

/// Logits tolerance per activation dtype.
fn tolerance(dtype: DType) -> f32 {
    match dtype {
        DType::F32 => 1e-4,
        _ => 5e-2,
    }
}

fn compare(client: &CudaClient, device: &CudaDevice, capacity: usize, dtype: DType) {
    let model = tiny_llama(device, SEED, dtype);

    let mut kv = full_cache(device, capacity, dtype, &TINY_DIMS);
    let first_a = argmax(&prefill(client, device, &model, &mut kv, &PROMPT, VOCAB));
    let (rows_a, ids_a) = eager_decode(client, device, &model, first_a, DECODE_STEPS, &mut kv);

    let (first_b, rows_b, ids_b) = graph_decode(
        client,
        device,
        &model,
        &TINY_DIMS,
        &PROMPT,
        capacity,
        dtype,
        DECODE_STEPS,
    );
    assert_eq!(
        first_a, first_b,
        "{dtype:?} cap {capacity}: prefill argmax differs"
    );

    let tol = tolerance(dtype);
    for (step, (a, b)) in rows_a.iter().zip(&rows_b).enumerate() {
        assert!(
            a.iter().all(|v| v.is_finite()),
            "{dtype:?} cap {capacity}: eager step {step} not finite"
        );
        assert!(
            b.iter().all(|v| v.is_finite()),
            "{dtype:?} cap {capacity}: graph step {step} not finite"
        );
        let diff = max_abs_diff(a, b);
        assert!(
            diff < tol,
            "{dtype:?} cap {capacity}: step {step} logits diff {diff} (eager id {}, graph id {})",
            ids_a[step],
            ids_b[step]
        );
    }
    assert_eq!(ids_a, ids_b, "{dtype:?} cap {capacity}: argmax ids diverge");
}

#[test]
fn graph_matches_eager() {
    if !cuda_available() {
        println!("skip: no CUDA device");
        return;
    }
    let _guard = cuda_lock();
    let (client, device) = cuda_setup();

    // Capacity 32 keeps the whole-sequence decode kernel; 64 takes the split path.
    compare(&client, &device, 32, DType::F32);
    compare(&client, &device, 64, DType::F32);
    compare(&client, &device, 32, DType::F16);
    compare(&client, &device, 32, DType::BF16);
    compare(&client, &device, 4096, DType::F32);
    compare(&client, &device, 4096, DType::BF16);
}

#[test]
fn eager_is_deterministic() {
    if !cuda_available() {
        println!("skip: no CUDA device");
        return;
    }
    let _guard = cuda_lock();
    let (client, device) = cuda_setup();
    let model = tiny_llama(&device, SEED, DType::F32);

    let run = || {
        let mut kv = full_cache(&device, 32, DType::F32, &TINY_DIMS);
        let prefill_row = prefill(&client, &device, &model, &mut kv, &PROMPT, VOCAB);
        let first = argmax(&prefill_row);
        let (rows, ids) = eager_decode(&client, &device, &model, first, DECODE_STEPS, &mut kv);
        (prefill_row, rows, ids)
    };
    let (p1, rows1, ids1) = run();
    let (p2, rows2, ids2) = run();
    assert_eq!(p1, p2, "prefill logits differ between runs");
    assert_eq!(ids1, ids2, "eager ids differ between runs");
    assert_eq!(rows1, rows2, "eager logits differ between runs");
}

// ── Real checkpoint ─────────────────────────────────────────────────────

const ENV_MODEL: &str = "BOOSTR_LLAMA_MODEL";
const ENV_IDS: &str = "BOOSTR_LLAMA_IDS";
const REAL_STEPS: usize = 8;
const REAL_CAPACITY: usize = 2048;

/// "The capital of France is" for a Mistral-family GGUF vocabulary.
const DEFAULT_IDS: [i64; 6] = [1, 415, 5565, 302, 4843, 349];

fn prompt_ids_from_env() -> Vec<i64> {
    match std::env::var(ENV_IDS) {
        Ok(raw) => raw
            .split(',')
            .map(|part| {
                part.trim()
                    .parse::<i64>()
                    .unwrap_or_else(|e| panic!("{ENV_IDS}: '{part}' is not an integer id: {e}"))
            })
            .collect(),
        Err(_) => DEFAULT_IDS.to_vec(),
    }
}

/// Relative logits tolerance: F32 activations hold to 1e-2 of the largest
/// magnitude in the pair, 16-bit activations to 5e-2.
fn relative_tolerance(dtype: DType) -> f32 {
    match dtype {
        DType::F32 => 1e-2,
        _ => 5e-2,
    }
}

fn max_abs(row: &[f32]) -> f32 {
    row.iter().fold(0.0f32, |acc, v| acc.max(v.abs()))
}

#[test]
fn real_file_graph_matches_eager() {
    if !cuda_available() {
        println!("skip: no CUDA device");
        return;
    }
    let Some(path) = std::env::var(ENV_MODEL).ok().map(PathBuf::from) else {
        println!("skip: {ENV_MODEL} not set");
        return;
    };
    if !path.exists() {
        println!("skip: {} not found", path.display());
        return;
    }
    let _guard = cuda_lock();
    let (client, device) = cuda_setup();

    let model = load_real_model(&path, &device).unwrap();
    assert_eq!(
        model.model_type(),
        "llama",
        "{}: expected a Llama-family model, got {}",
        path.display(),
        model.model_type()
    );
    let dims = Dims::from_model(&model);
    let prompt = prompt_ids_from_env();

    let (rope_cos, _) = model
        .rope_caches()
        .expect("attention model has RoPE caches");
    let kv_dtype = rope_cos.tensor().dtype();

    let mut kv_a = full_cache(&device, REAL_CAPACITY, kv_dtype, &dims);
    let first_a = argmax(&prefill(
        &client, &device, &model, &mut kv_a, &prompt, dims.vocab,
    ));
    let (rows_a, ids_a) = eager_decode(&client, &device, &model, first_a, REAL_STEPS, &mut kv_a);

    let (first_b, rows_b, ids_b) = graph_decode(
        &client,
        &device,
        &model,
        &dims,
        &prompt,
        REAL_CAPACITY,
        kv_dtype,
        REAL_STEPS,
    );
    assert_eq!(first_a, first_b, "prefill argmax differs");

    let rel_tol = relative_tolerance(kv_dtype);
    for (step, (a, b)) in rows_a.iter().zip(&rows_b).enumerate() {
        let diff = max_abs_diff(a, b);
        let max_mag = max_abs(a).max(max_abs(b));
        println!(
            "step {step}: eager id {}, graph id {}, max abs diff {diff}, max abs logit {max_mag}",
            ids_a[step], ids_b[step]
        );
        assert!(
            diff <= rel_tol * max_mag,
            "step {step}: logits diff {diff} exceeds {rel_tol} * {max_mag} (eager id {}, graph id {})",
            ids_a[step],
            ids_b[step]
        );
    }
    assert_eq!(ids_a, ids_b, "argmax ids diverge between eager and graph");
}
