//! Numerical parity between our `qwen35` CPU forward and the PrismML
//! llama.cpp fork's oracle, on the real Ternary-Bonsai-2-27B `PQ2_0`
//! checkpoint.
//!
//! # Fixture file
//!
//! `BOOSTR_BONSAI2_DIR` must hold `Ternary-Bonsai-2-27B-PQ2_0.gguf` and
//! `fixtures/fork_logits_capital_of_france_pq2_0.bin`. Unset dir, missing
//! model, or missing fixture: every test prints one `skip:` line and passes.
//!
//! The fixture was generated with `tests/tools/prism_dump_logits`:
//!
//! ```bash
//! ./dump_logits Ternary-Bonsai-2-27B-PQ2_0.gguf \
//!   fork_logits_capital_of_france_pq2_0.bin 99 760 6511 314 9338 369
//! ```
//!
//! See `tests/tools/prism_dump_logits/README.md` for the build and the
//! binary layout.
//!
//! ```bash
//! BOOSTR_BONSAI2_DIR=/path/to/dir cargo test --release \
//!   --test qwen35_parity -- --nocapture
//! ```

use std::path::PathBuf;
use std::time::Instant;

use boostr::format::gguf::Gguf;
use boostr::inference::{LayeredGdnState, LayeredKvCache, LayeredKvCacheConfig};
use boostr::model::{LoadedModel, qwen35_config_from_gguf};
use boostr::nn::{VarBuilder, VarMap};
use numr::dtype::DType;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

const ENV_DIR: &str = "BOOSTR_BONSAI2_DIR";
const PQ2_0_FILE: &str = "Ternary-Bonsai-2-27B-PQ2_0.gguf";
const FIXTURE_FILE: &str = "fixtures/fork_logits_capital_of_france_pq2_0.bin";

const VOCAB: usize = 248_320;
/// "The capital of France is", no BOS.
const PROMPT: [i64; 5] = [760, 6511, 314, 9338, 369];
/// Fork's greedy decode ids for the 3 steps after the prefill.
const EXPECTED_DECODE: [i64; 3] = [13, 198, 760];
/// Argmax of the fixture's row 4 (last prefill position).
const ROW4_ARGMAX: u32 = 11751;

const COSINE_MIN: f64 = 0.999;
const MAX_ABS_DIFF_MAX: f64 = 0.5;

/// One fixture row: `n_vocab` `f32` logits.
struct Fixture {
    n_positions: usize,
    n_vocab: usize,
    rows: Vec<Vec<f32>>,
    decoded_ids: [u32; 3],
}

impl Fixture {
    fn load(path: &PathBuf) -> Self {
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let mut off = 0usize;
        let read_u32 = |bytes: &[u8], off: &mut usize| -> u32 {
            let v = u32::from_le_bytes(bytes[*off..*off + 4].try_into().unwrap());
            *off += 4;
            v
        };
        let n_positions = read_u32(&bytes, &mut off) as usize;
        let n_vocab = read_u32(&bytes, &mut off) as usize;
        assert_eq!(n_vocab, VOCAB, "fixture n_vocab mismatch");
        let mut rows = Vec::with_capacity(n_positions);
        for _ in 0..n_positions {
            let mut row = Vec::with_capacity(n_vocab);
            for _ in 0..n_vocab {
                row.push(f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()));
                off += 4;
            }
            rows.push(row);
        }
        let n_decoded = read_u32(&bytes, &mut off);
        assert_eq!(n_decoded, 3, "fixture n_decoded must be 3");
        let decoded_ids = [
            read_u32(&bytes, &mut off),
            read_u32(&bytes, &mut off),
            read_u32(&bytes, &mut off),
        ];
        Self {
            n_positions,
            n_vocab,
            rows,
            decoded_ids,
        }
    }
}

/// The PQ2_0 model and fixture paths, or `None` after one `skip:` line.
fn require_files() -> Option<(PathBuf, PathBuf)> {
    let Some(dir) = std::env::var(ENV_DIR).ok().map(PathBuf::from) else {
        println!("skip: {ENV_DIR} not set");
        return None;
    };
    let model_path = dir.join(PQ2_0_FILE);
    if !model_path.is_file() {
        println!("skip: {} not found", model_path.display());
        return None;
    }
    let fixture_path = dir.join(FIXTURE_FILE);
    if !fixture_path.is_file() {
        println!("skip: {} not found", fixture_path.display());
        return None;
    }
    Some((model_path, fixture_path))
}

/// argmax and the value at it, over an `f32` row.
fn argmax_f32(row: &[f32]) -> (usize, f32) {
    let mut best_idx = 0usize;
    let mut best_val = row[0];
    for (i, &v) in row.iter().enumerate().skip(1) {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    (best_idx, best_val)
}

/// The `k` highest-scoring indices in `row`, descending.
fn top_k(row: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..row.len()).collect();
    idx.sort_unstable_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap());
    idx.truncate(k);
    idx
}

/// Cosine similarity and max/mean absolute diff between two equal-length
/// rows, computed in `f64`.
fn row_stats(ours: &[f32], theirs: &[f32]) -> (f64, f64, f64) {
    let mut dot = 0f64;
    let mut norm_a = 0f64;
    let mut norm_b = 0f64;
    let mut max_abs = 0f64;
    let mut sum_abs = 0f64;
    for (&a, &b) in ours.iter().zip(theirs.iter()) {
        let a = a as f64;
        let b = b as f64;
        dot += a * b;
        norm_a += a * a;
        norm_b += b * b;
        let diff = (a - b).abs();
        max_abs = max_abs.max(diff);
        sum_abs += diff;
    }
    let cosine = dot / (norm_a.sqrt() * norm_b.sqrt());
    let mean_abs = sum_abs / ours.len() as f64;
    (cosine, max_abs, mean_abs)
}

/// Compares one of our rows to one fixture row: argmax equality, cosine
/// similarity, and max abs diff. Panics with the full numbers on failure.
fn assert_row_matches(label: &str, position: usize, ours: &[f32], theirs: &[f32]) {
    let (our_idx, our_val) = argmax_f32(ours);
    let (their_idx, their_val) = argmax_f32(theirs);
    let (cosine, max_abs, mean_abs) = row_stats(ours, theirs);
    println!(
        "{label} pos {position}: our argmax {our_idx} ({our_val:.4}), fork argmax {their_idx} \
         ({their_val:.4}), cosine {cosine:.6}, max_abs_diff {max_abs:.4}, mean_abs_diff \
         {mean_abs:.6}"
    );
    assert_eq!(
        our_idx, their_idx,
        "{label} pos {position}: argmax mismatch (ours {our_idx} vs fork {their_idx})"
    );
    assert!(
        cosine >= COSINE_MIN,
        "{label} pos {position}: cosine {cosine:.6} < {COSINE_MIN}"
    );
    assert!(
        max_abs <= MAX_ABS_DIFF_MAX,
        "{label} pos {position}: max_abs_diff {max_abs:.4} > {MAX_ABS_DIFF_MAX}"
    );
}

/// Loads the model, KV cache, and GDN state exactly like
/// `tests/qwen35_gguf_real_file.rs`.
fn load_model(
    model_path: &PathBuf,
) -> (
    CpuDevice,
    CpuClient,
    boostr::model::qwen35::Qwen35Model<CpuRuntime>,
    LayeredKvCache<CpuRuntime>,
    LayeredGdnState<CpuRuntime>,
) {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());

    let gguf =
        Gguf::open(model_path).unwrap_or_else(|e| panic!("open {}: {e}", model_path.display()));
    let config = qwen35_config_from_gguf(gguf.metadata())
        .unwrap_or_else(|e| panic!("qwen35_config_from_gguf on {}: {e}", model_path.display()));
    drop(gguf);

    let mut varmap = VarMap::<CpuRuntime>::from_gguf(model_path, &device)
        .unwrap_or_else(|e| panic!("VarMap::from_gguf on {}: {e}", model_path.display()));
    let mut vb = VarBuilder::new(&mut varmap, &device);
    let loaded = LoadedModel::<CpuRuntime>::load(&config, &mut vb)
        .unwrap_or_else(|e| panic!("LoadedModel::load: {e}"));
    let num_attention = loaded
        .num_attention_layers()
        .expect("qwen35 attention layers");
    let num_gdn = loaded.num_gdn_layers().expect("qwen35 gdn layers");
    let model = match loaded {
        LoadedModel::Qwen35(model) => model,
        other => panic!("LoadedModel::load returned {other:?}, expected Qwen35"),
    };

    let attn = model.attention_config();
    let kv_config = LayeredKvCacheConfig {
        batch_size: 1,
        num_kv_heads: attn.num_kv_heads,
        initial_capacity: 16,
        max_seq_len: config.max_seq_len,
        head_dim: attn.head_dim,
        dtype: DType::F32,
    };
    let kv = LayeredKvCache::<CpuRuntime>::new(num_attention, &kv_config, &device)
        .unwrap_or_else(|e| panic!("LayeredKvCache::new: {e}"));
    let gdn =
        LayeredGdnState::<CpuRuntime>::zeros(num_gdn, model.gdn_config(), 1, DType::F32, &device)
            .unwrap_or_else(|e| panic!("LayeredGdnState::zeros: {e}"));

    (device, client, *model, kv, gdn)
}

#[test]
fn prefill_logits_match_fork() {
    let Some((model_path, fixture_path)) = require_files() else {
        return;
    };
    let fixture = Fixture::load(&fixture_path);
    assert!(
        fixture.n_positions >= PROMPT.len(),
        "fixture has {} rows, need at least {}",
        fixture.n_positions,
        PROMPT.len()
    );

    let start = Instant::now();
    let (device, client, model, mut kv, mut gdn) = load_model(&model_path);
    println!("model load: {:.1?} elapsed", start.elapsed());

    let ids = Tensor::<CpuRuntime>::from_slice(&PROMPT, &[1, PROMPT.len()], &device)
        .unwrap_or_else(|e| panic!("prompt tensor: {e}"));
    let step = Instant::now();
    let logits = model
        .forward_qwen35(&client, &ids, &mut kv, &mut gdn, 0)
        .unwrap_or_else(|e| panic!("prefill forward: {e}"));
    println!(
        "prefill ({} tokens): {:.1?} elapsed",
        PROMPT.len(),
        step.elapsed()
    );
    assert_eq!(logits.shape(), &[1, PROMPT.len(), VOCAB]);

    let flat = logits.to_vec::<f32>();
    for p in 0..PROMPT.len() {
        let ours = &flat[p * fixture.n_vocab..(p + 1) * fixture.n_vocab];
        assert_row_matches("prefill", p, ours, &fixture.rows[p]);
    }

    let last = PROMPT.len() - 1;
    let ours_last = &flat[last * fixture.n_vocab..(last + 1) * fixture.n_vocab];
    let (our_argmax_last, _) = argmax_f32(ours_last);
    assert_eq!(
        our_argmax_last as u32, ROW4_ARGMAX,
        "row {last} argmax {our_argmax_last} does not match documented fork argmax \
         {ROW4_ARGMAX}"
    );

    let our_top5: std::collections::BTreeSet<usize> = top_k(ours_last, 5).into_iter().collect();
    let fork_top5: std::collections::BTreeSet<usize> =
        top_k(&fixture.rows[last], 5).into_iter().collect();
    assert_eq!(
        our_top5, fork_top5,
        "top-5 id set mismatch at last prefill position: ours {our_top5:?}, fork {fork_top5:?}"
    );
}

#[test]
fn greedy_decode_matches_fork() {
    let Some((model_path, fixture_path)) = require_files() else {
        return;
    };
    let fixture = Fixture::load(&fixture_path);
    assert!(
        fixture.n_positions >= PROMPT.len() + EXPECTED_DECODE.len(),
        "fixture has {} rows, need at least {}",
        fixture.n_positions,
        PROMPT.len() + EXPECTED_DECODE.len()
    );

    let start = Instant::now();
    let (device, client, model, mut kv, mut gdn) = load_model(&model_path);
    println!("model load: {:.1?} elapsed", start.elapsed());

    let ids = Tensor::<CpuRuntime>::from_slice(&PROMPT, &[1, PROMPT.len()], &device)
        .unwrap_or_else(|e| panic!("prompt tensor: {e}"));
    let step = Instant::now();
    let logits = model
        .forward_qwen35(&client, &ids, &mut kv, &mut gdn, 0)
        .unwrap_or_else(|e| panic!("prefill forward: {e}"));
    println!("prefill: {:.1?} elapsed", step.elapsed());

    let flat = logits.to_vec::<f32>();
    let last = PROMPT.len() - 1;
    let mut last_row: Vec<f32> = flat[last * VOCAB..(last + 1) * VOCAB].to_vec();

    let mut chosen = Vec::with_capacity(EXPECTED_DECODE.len());
    let decode_start = Instant::now();
    for step_idx in 0..EXPECTED_DECODE.len() {
        let (next_id, _) = argmax_f32(&last_row);

        let position = kv.seq_len();
        let next = Tensor::<CpuRuntime>::from_slice(&[next_id as i64], &[1, 1], &device)
            .unwrap_or_else(|e| panic!("decode tensor: {e}"));
        let step = Instant::now();
        let decode_logits = model
            .forward_qwen35(&client, &next, &mut kv, &mut gdn, position)
            .unwrap_or_else(|e| panic!("decode forward: {e}"));
        println!("decode step {step_idx}: {:.1?} elapsed", step.elapsed());
        assert_eq!(decode_logits.shape(), &[1, 1, VOCAB]);

        let decode_flat = decode_logits.to_vec::<f32>();
        assert_row_matches(
            "decode",
            PROMPT.len() + step_idx,
            &decode_flat,
            &fixture.rows[PROMPT.len() + step_idx],
        );
        chosen.push(argmax_f32(&decode_flat).0 as i64);
        last_row = decode_flat;
    }
    println!("decode (3 tokens): {:.1?} elapsed", decode_start.elapsed());

    assert_eq!(
        chosen,
        EXPECTED_DECODE.to_vec(),
        "greedy decode ids diverged from fork"
    );
    assert_eq!(
        fixture.decoded_ids,
        [
            EXPECTED_DECODE[0] as u32,
            EXPECTED_DECODE[1] as u32,
            EXPECTED_DECODE[2] as u32
        ],
        "fixture's recorded decoded ids do not match EXPECTED_DECODE"
    );

    println!("total: {:.1?} elapsed", start.elapsed());
}
