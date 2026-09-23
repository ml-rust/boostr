//! CUDA graph-mode decode for the `qwen35` hybrid against the eager forward.
//!
//! Run with:
//!   cd boostr && cargo test --release --features cuda --test qwen35_decode_graph
//!
//! `synthetic_graph_matches_eager`: a tiny hybrid model built from the public
//! constructors. Prefill 5 tokens eagerly, then 6 greedy decode steps twice:
//! eager `forward_qwen35`, and one captured graph replayed 6 times. Argmax
//! ids must agree at every step, logits within `1e-4`, and the GDN buffers
//! after the last step within `1e-5`.
//!
//! `real_file_graph_matches_eager`: the same on the PQ2_0 checkpoint under
//! `BOOSTR_BONSAI2_DIR` for 4 decode tokens, argmax against eager and
//! against the fork's greedy ids.

#![cfg(feature = "cuda")]

mod common;

use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use boostr::inference::LaneTable;
use boostr::inference::decode_graph::Qwen35DecodeGraph;
use boostr::inference::{LayeredGdnState, LayeredKvCache};
use boostr::model::qwen35::Qwen35Model;
use common::qwen35_cuda::{
    caches, cuda_available, cuda_setup, eager_decode, gdn_flat, max_abs_diff, prefill,
};
use common::qwen35_graph::{GraphBufs, capture};
use common::qwen35_tiny::{MAX_POS, VOCAB, tiny_model};
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};

static CUDA_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn cuda_lock() -> std::sync::MutexGuard<'static, ()> {
    CUDA_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|p| p.into_inner())
}

const PREFILL: usize = 5;
const DECODE_STEPS: usize = 6;

/// Graph greedy decode: warm up, re-prefill, capture one step, replay
/// `steps` times. Returns logits rows and ids, and leaves the replayed
/// state in `kv`/`gdn`.
///
/// One lane, claimed at the prompt's length: the prefill wrote straight
/// into the lane the graph reads.
#[allow(clippy::too_many_arguments)]
fn graph_decode(
    client: &CudaClient,
    device: &CudaDevice,
    model: &Qwen35Model<CudaRuntime>,
    prompt: &[i64],
    first: i64,
    steps: usize,
    kv: &mut LayeredKvCache<CudaRuntime>,
    gdn: &mut LayeredGdnState<CudaRuntime>,
) -> (Vec<Vec<f32>>, Vec<i64>) {
    let vocab = model.config().vocab_size;
    let seq_len = kv.seq_len();
    assert_eq!(seq_len, prompt.len());

    // Stable buffers and lane state, all set before capture.
    let mut bufs = GraphBufs::new(device, 1, vocab);
    let mut lanes = LaneTable::<CudaRuntime>::new(1, MAX_POS, device).unwrap();
    lanes.claim(client, 0, 0, seq_len, seq_len).unwrap();

    // Warm-up outside capture: kernels JIT and the tuner probes here. This
    // run writes the KV cache and the GDN buffers, so prefill again after.
    let capacity = kv.layer(0).unwrap().capacity();
    bufs.warmup(client, model, kv, gdn, &lanes, capacity);

    let (fresh_kv, fresh_gdn) = caches(device, model, kv.layer(0).unwrap().capacity());
    *kv = fresh_kv;
    *gdn = fresh_gdn;
    let re_first = prefill(client, device, model, prompt, kv, gdn);
    assert_eq!(re_first, first);

    let mut captures = 0usize;
    let (graph, _) = capture(client, device, model, kv, gdn, &lanes, &bufs, &mut captures);
    assert_eq!(captures, 1, "one decode step, one capture");

    let GraphBufs {
        token_buf,
        next_token_buf,
        logits_buf,
        scalars,
    } = bufs;
    let mut decode =
        Qwen35DecodeGraph::new(graph, scalars, capacity, lanes, token_buf, next_token_buf).unwrap();
    decode.seed_next_token(client, 0, first).unwrap();

    let mut rows = Vec::with_capacity(steps);
    let mut ids = Vec::with_capacity(steps);
    for _ in 0..steps {
        decode.pre_replay_and_launch(client).unwrap();
        rows.push(logits_buf.to_vec::<f32>());
        ids.push(decode.next_token_buf.to_vec::<i64>()[0]);
    }
    (rows, ids)
}

#[test]
fn synthetic_graph_matches_eager() {
    if !cuda_available() {
        println!("skip: no CUDA device");
        return;
    }
    let _guard = cuda_lock();
    let (client, device) = cuda_setup();
    let model = tiny_model::<CudaRuntime>(&device, 0x3535_0900);
    let prompt: Vec<i64> = (0..PREFILL).map(|i| ((i * 5 + 3) % VOCAB) as i64).collect();

    let (mut kv_a, mut gdn_a) = caches(&device, &model, MAX_POS);
    let first_a = prefill(&client, &device, &model, &prompt, &mut kv_a, &mut gdn_a);
    let (rows_a, ids_a) = eager_decode(
        &client,
        &device,
        &model,
        first_a,
        DECODE_STEPS,
        &mut kv_a,
        &mut gdn_a,
    );

    let (mut kv_b, mut gdn_b) = caches(&device, &model, MAX_POS);
    let first_b = prefill(&client, &device, &model, &prompt, &mut kv_b, &mut gdn_b);
    assert_eq!(first_a, first_b);
    let (rows_b, ids_b) = graph_decode(
        &client,
        &device,
        &model,
        &prompt,
        first_b,
        DECODE_STEPS,
        &mut kv_b,
        &mut gdn_b,
    );

    assert_eq!(ids_a, ids_b, "argmax ids diverge between eager and graph");
    for (step, (a, b)) in rows_a.iter().zip(&rows_b).enumerate() {
        assert!(a.iter().all(|v| v.is_finite()));
        let diff = max_abs_diff(a, b);
        assert!(diff < 1e-4, "step {step}: logits diff {diff}");
    }

    let (conv_a, ssm_a) = gdn_flat(&gdn_a);
    let (conv_b, ssm_b) = gdn_flat(&gdn_b);
    let conv_diff = max_abs_diff(&conv_a, &conv_b);
    assert!(conv_diff < 1e-5, "gdn conv window diff {conv_diff}");
    let ssm_diff = max_abs_diff(&ssm_a, &ssm_b);
    assert!(ssm_diff < 1e-5, "gdn ssm state diff {ssm_diff}");
}

// ── Real checkpoint ─────────────────────────────────────────────────────

const ENV_DIR: &str = "BOOSTR_BONSAI2_DIR";
const PQ2_0_FILE: &str = "Ternary-Bonsai-2-27B-PQ2_0.gguf";
/// "The capital of France is", no BOS.
const PROMPT: [i64; 5] = [760, 6511, 314, 9338, 369];
/// Fork's greedy ids for the 3 steps after the prefill (`tests/qwen35_parity.rs`).
const EXPECTED_DECODE: [i64; 3] = [13, 198, 760];
const REAL_STEPS: usize = 4;
const REAL_CAPACITY: usize = 16;

#[test]
fn real_file_graph_matches_eager() {
    if !cuda_available() {
        println!("skip: no CUDA device");
        return;
    }
    let Some(dir) = std::env::var(ENV_DIR).ok().map(PathBuf::from) else {
        println!("skip: {ENV_DIR} not set");
        return;
    };
    let model_path = dir.join(PQ2_0_FILE);
    if !model_path.is_file() {
        println!("skip: {} not found", model_path.display());
        return;
    }
    let _guard = cuda_lock();
    let (client, device) = cuda_setup();

    let gguf = boostr::format::gguf::Gguf::open(&model_path).unwrap();
    let config = boostr::model::qwen35_config_from_gguf(gguf.metadata()).unwrap();
    drop(gguf);
    let mut varmap = boostr::nn::VarMap::<CudaRuntime>::from_gguf(&model_path, &device).unwrap();
    let mut vb = boostr::nn::VarBuilder::new(&mut varmap, &device);
    let model = match boostr::model::LoadedModel::<CudaRuntime>::load(&config, &mut vb).unwrap() {
        boostr::model::LoadedModel::Qwen35(m) => *m,
        other => panic!("expected Qwen35, got {other:?}"),
    };

    let (mut kv_a, mut gdn_a) = caches(&device, &model, REAL_CAPACITY);
    let first = prefill(&client, &device, &model, &PROMPT, &mut kv_a, &mut gdn_a);
    let (_, ids_a) = eager_decode(
        &client, &device, &model, first, REAL_STEPS, &mut kv_a, &mut gdn_a,
    );

    let (mut kv_b, mut gdn_b) = caches(&device, &model, REAL_CAPACITY);
    let first_b = prefill(&client, &device, &model, &PROMPT, &mut kv_b, &mut gdn_b);
    assert_eq!(first, first_b);
    let (_, ids_b) = graph_decode(
        &client, &device, &model, &PROMPT, first_b, REAL_STEPS, &mut kv_b, &mut gdn_b,
    );

    println!("eager ids {ids_a:?}, graph ids {ids_b:?}");
    assert_eq!(ids_a, ids_b, "argmax ids diverge between eager and graph");
    assert_eq!(&ids_b[..EXPECTED_DECODE.len()], &EXPECTED_DECODE[..]);
}
