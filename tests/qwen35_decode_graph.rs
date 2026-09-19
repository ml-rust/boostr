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

/// One decode step records on the order of a thousand intermediate
/// allocations; the arena serves them from one buffer instead of one graph
/// memory node each.
const ARENA_BYTES: usize = 512 << 20;

use boostr::inference::decode_graph::{
    DeviceScalars, MropeScalars, argmax_to_buf, copy_into_stable,
};
use boostr::inference::{LayeredGdnState, LayeredKvCache, LayeredKvCacheConfig};
use boostr::model::qwen35::Qwen35Model;
use common::qwen35_tiny::{MAX_POS, VOCAB, tiny_model};
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

fn cuda_available() -> bool {
    numr::runtime::cuda::is_cuda_available()
}

fn cuda_setup() -> (CudaClient, CudaDevice) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    (client, device)
}

const PREFILL: usize = 5;
const DECODE_STEPS: usize = 6;

// ── Shared harness ──────────────────────────────────────────────────────

/// Full-capacity KV cache (graph mode needs stable k/v addresses) plus zero
/// GDN state.
fn caches(
    device: &CudaDevice,
    model: &Qwen35Model<CudaRuntime>,
    capacity: usize,
) -> (LayeredKvCache<CudaRuntime>, LayeredGdnState<CudaRuntime>) {
    let attn = model.attention_config();
    let kv_config = LayeredKvCacheConfig {
        batch_size: 1,
        num_kv_heads: attn.num_kv_heads,
        initial_capacity: capacity,
        max_seq_len: capacity,
        head_dim: attn.head_dim,
        dtype: DType::F32,
    };
    let kv = LayeredKvCache::<CudaRuntime>::new(model.num_attention_layers(), &kv_config, device)
        .unwrap();
    let gdn = LayeredGdnState::<CudaRuntime>::zeros(
        model.num_gdn_layers(),
        model.gdn_config(),
        1,
        DType::F32,
        device,
    )
    .unwrap();
    (kv, gdn)
}

fn argmax(row: &[f32]) -> i64 {
    let mut best = 0usize;
    for (i, &v) in row.iter().enumerate().skip(1) {
        if v > row[best] {
            best = i;
        }
    }
    best as i64
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// Eager prefill of `prompt`; returns the argmax of the last row.
fn prefill(
    client: &CudaClient,
    device: &CudaDevice,
    model: &Qwen35Model<CudaRuntime>,
    prompt: &[i64],
    kv: &mut LayeredKvCache<CudaRuntime>,
    gdn: &mut LayeredGdnState<CudaRuntime>,
) -> i64 {
    let ids = Tensor::<CudaRuntime>::from_slice(prompt, &[1, prompt.len()], device).unwrap();
    let logits = model.forward_qwen35(client, &ids, kv, gdn, 0).unwrap();
    let vocab = logits.shape()[2];
    let flat = logits.to_vec::<f32>();
    argmax(&flat[(prompt.len() - 1) * vocab..])
}

/// Eager greedy decode: `steps` rows of logits and the ids they produce.
fn eager_decode(
    client: &CudaClient,
    device: &CudaDevice,
    model: &Qwen35Model<CudaRuntime>,
    first: i64,
    steps: usize,
    kv: &mut LayeredKvCache<CudaRuntime>,
    gdn: &mut LayeredGdnState<CudaRuntime>,
) -> (Vec<Vec<f32>>, Vec<i64>) {
    let mut token = first;
    let mut rows = Vec::with_capacity(steps);
    let mut ids = Vec::with_capacity(steps);
    for _ in 0..steps {
        let position = kv.seq_len();
        let x = Tensor::<CudaRuntime>::from_slice(&[token], &[1, 1], device).unwrap();
        let logits = model.forward_qwen35(client, &x, kv, gdn, position).unwrap();
        let row = logits.to_vec::<f32>();
        token = argmax(&row);
        rows.push(row);
        ids.push(token);
    }
    (rows, ids)
}

/// Graph greedy decode: warm up on throwaway state, re-prefill, capture one
/// step, replay `steps` times. Returns logits rows and ids, and leaves the
/// replayed state in `kv`/`gdn`.
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

    // Stable buffers, all allocated before capture.
    let token_buf = Tensor::<CudaRuntime>::from_slice(&[first], &[1, 1], device).unwrap();
    let next_token_buf = Tensor::<CudaRuntime>::zeros(&[1], DType::I64, device).unwrap();
    let logits_buf = Tensor::<CudaRuntime>::zeros(&[1, 1, vocab], DType::F32, device).unwrap();
    let scalars = DeviceScalars::new(seq_len, device).unwrap();
    let mrope = MropeScalars::new(seq_len, device).unwrap();

    // Warm-up outside capture: kernels JIT and the tuner probes here. This
    // run writes the KV cache and the GDN buffers, so prefill again after.
    scalars.update(client, seq_len).unwrap();
    mrope.update(client, seq_len).unwrap();
    let warm = model
        .forward_qwen35_graph_mode(client, &token_buf, &*kv, &*gdn, &scalars, &mrope)
        .unwrap();
    argmax_to_buf(client, &warm, &next_token_buf).unwrap();
    let _ = next_token_buf.to_vec::<i64>();

    let (fresh_kv, fresh_gdn) = caches(device, model, kv.layer(0).unwrap().capacity());
    *kv = fresh_kv;
    *gdn = fresh_gdn;
    let re_first = prefill(client, device, model, prompt, kv, gdn);
    assert_eq!(re_first, first);

    let graph = CudaRuntime::capture_graph_into_with_arena(
        client,
        &[&token_buf],
        &[&next_token_buf],
        ARENA_BYTES,
        |c| {
            let logits = model
                .forward_qwen35_graph_mode(c, &token_buf, &*kv, &*gdn, &scalars, &mrope)
                .map_err(|e| numr::error::Error::Backend(format!("capture forward: {e}")))?;
            copy_into_stable(c, &logits, &logits_buf)?;
            argmax_to_buf(c, &logits, &next_token_buf)
        },
    )
    .unwrap();

    let mut rows = Vec::with_capacity(steps);
    let mut ids = Vec::with_capacity(steps);
    for (step, position) in (seq_len..seq_len + steps).enumerate() {
        if step > 0 {
            copy_into_stable(client, &next_token_buf, &token_buf).unwrap();
        }
        scalars.update(client, position).unwrap();
        mrope.update(client, position).unwrap();
        graph.launch().unwrap();
        rows.push(logits_buf.to_vec::<f32>());
        ids.push(next_token_buf.to_vec::<i64>()[0]);
    }
    (rows, ids)
}

fn gdn_flat(state: &LayeredGdnState<CudaRuntime>) -> (Vec<f32>, Vec<f32>) {
    let mut conv = Vec::new();
    let mut ssm = Vec::new();
    for i in 0..state.num_layers() {
        let layer = state.layer(i).unwrap();
        conv.extend(layer.conv().to_vec::<f32>());
        ssm.extend(layer.ssm().to_vec::<f32>());
    }
    (conv, ssm)
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
