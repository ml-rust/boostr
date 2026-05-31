//! Throughput diagnostic for the varlen embedding path (nomic-768, 12 layers).
//!
//! Measures docs/s and samples GPU free memory. The free-list cap is taken from
//! the inherited NUMR_CUDA_FREE_LIST_CAP_MB env (do NOT set it here) so the same
//! binary can be run under different caps to isolate allocator-churn vs kernel
//! cost:
//!   NUMR_CUDA_FREE_LIST_CAP_MB=1024 cargo test --features cuda --test perf_diag -- --nocapture
//!   NUMR_CUDA_FREE_LIST_CAP_MB=8192 cargo test --features cuda --test perf_diag -- --nocapture

#![cfg(all(feature = "cuda", feature = "f16"))]

use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use boostr::model::encoder::{
    config::{ArchFamily, EncoderConfig, FfnVariant, HiddenAct},
    model::{Encoder, Pooling},
};
use boostr::nn::Weight;
use numr::dtype::DType;
use numr::runtime::cuda::{CudaDevice, CudaRuntime};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

static CUDA_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
fn cuda_lock() -> std::sync::MutexGuard<'static, ()> {
    CUDA_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|p| p.into_inner())
}

const HIDDEN: usize = 768;
const HEADS: usize = 12;
const LAYERS: usize = 12;
const INTER: usize = 3072;
const VOCAB: usize = 256;
const MAX_POS: usize = 512;

type CudaClient = <CudaRuntime as Runtime>::Client;

fn make_encoder(device: &CudaDevice, client: &CudaClient) -> Encoder<CudaRuntime> {
    let config = EncoderConfig {
        vocab_size: VOCAB,
        hidden_size: HIDDEN,
        num_hidden_layers: LAYERS,
        num_attention_heads: HEADS,
        intermediate_size: INTER,
        max_position_embeddings: MAX_POS,
        layer_norm_eps: 1e-12,
        hidden_act: HiddenAct::Gelu,
        type_vocab_size: 2,
        arch_family: ArchFamily::NomicBert,
        padding_token_id: 0,
        compute_dtype: DType::F16,
        rope_freq_base: 10000.0,
        causal: false,
        ffn_variant: FfnVariant::GatedSilu,
        token_type_embed_size: 2,
        num_kv_heads: HEADS,
        head_dim_explicit: None,
        rms_eps: 1e-6,
        sliding_window: None,
        embed_scale: false,
        max_tokens_per_forward: None,
    };
    let d = device;
    let small = |n: usize, shape: &[usize]| {
        Ok(Weight::Standard(Tensor::<CudaRuntime>::from_slice(
            &(0..n).map(|i| (i as f32).sin() * 0.02).collect::<Vec<_>>(),
            shape,
            d,
        )))
    };
    let ones = |n: usize| {
        Ok(Weight::Standard(Tensor::<CudaRuntime>::from_slice(
            &vec![1.0f32; n],
            &[n],
            d,
        )))
    };
    let zeros = |n: usize| {
        Ok(Weight::Standard(Tensor::<CudaRuntime>::from_slice(
            &vec![0.0f32; n],
            &[n],
            d,
        )))
    };
    Encoder::from_weights_nomic(config, Pooling::Mean, client, |name| match name {
        "token_embd.weight" => small(VOCAB * HIDDEN, &[VOCAB, HIDDEN]),
        "token_embd_norm.weight" => ones(HIDDEN),
        "token_embd_norm.bias" => zeros(HIDDEN),
        "token_types.weight" => small(2 * HIDDEN, &[2, HIDDEN]),
        other => {
            let field = other
                .strip_prefix("blk.")
                .and_then(|r| r.split_once('.'))
                .map(|(_, f)| f);
            match field {
                Some("attn_qkv.weight") => small(3 * HIDDEN * HIDDEN, &[3 * HIDDEN, HIDDEN]),
                Some("attn_output.weight") => small(HIDDEN * HIDDEN, &[HIDDEN, HIDDEN]),
                Some("attn_output_norm.weight") => ones(HIDDEN),
                Some("attn_output_norm.bias") => zeros(HIDDEN),
                Some("ffn_gate.weight") => small(INTER * HIDDEN, &[INTER, HIDDEN]),
                Some("ffn_up.weight") => small(INTER * HIDDEN, &[INTER, HIDDEN]),
                Some("ffn_down.weight") => small(HIDDEN * INTER, &[HIDDEN, INTER]),
                Some("layer_output_norm.weight") => ones(HIDDEN),
                Some("layer_output_norm.bias") => zeros(HIDDEN),
                _ => Err(boostr::error::Error::ModelError {
                    reason: format!("unknown: {other}"),
                }),
            }
        }
    })
    .expect("encoder build")
}

/// One token-budgeted batch: `n_docs` docs of `seq` tokens packed together.
fn run_batch(
    encoder: &Encoder<CudaRuntime>,
    client: &CudaClient,
    device: &CudaDevice,
    n_docs: usize,
    seq: usize,
) {
    let mut ids = Vec::new();
    let mut cu = vec![0i32];
    let mut pos = Vec::new();
    let mut seg = Vec::new();
    for b in 0..n_docs {
        for t in 0..seq {
            ids.push(((b + t) % VOCAB) as i64);
            pos.push(t as i64);
            seg.push(b as i32);
        }
        cu.push(((b + 1) * seq) as i32);
    }
    let total = ids.len();
    let input = Tensor::<CudaRuntime>::from_slice(&ids, &[total], device);
    let cu_t = Tensor::<CudaRuntime>::from_slice(&cu, &[n_docs + 1], device);
    let pos_t = Tensor::<CudaRuntime>::from_slice(&pos, &[total], device);
    let seg_t = Tensor::<CudaRuntime>::from_slice(&seg, &[total], device);
    // Full pooled path (scatter_reduce mean) — matches the real embed entrypoint.
    let out = encoder
        .embed_inference_varlen(client, &input, &cu_t, &pos_t, &seg_t, n_docs, seq)
        .expect("fwd");
    let _ = out;
}

#[test]
fn perf_varlen_throughput() {
    let _g = cuda_lock();
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let encoder = make_encoder(&device, &client);

    let cap = std::env::var("NUMR_CUDA_FREE_LIST_CAP_MB").unwrap_or_else(|_| "default".into());
    // UNALIGNED total_tokens (15*500 = 7500, not a multiple of 16) — the real
    // varlen case. M-padding must route this to WMMA, else it falls to the 57
    // GFLOP/s generic kernel (the ma8e regression).
    const SEQ: usize = 500;
    const DOCS_PER_BATCH: usize = 15; // 7500 tokens/forward (unaligned)
    const BATCHES: usize = 8;
    // Pre-fix nomic-768 throughput was 23-40 docs/s (F16/WMMA). The F32 matmul
    // regression dropped it to ~0.5. Assert we stay well above that floor — a
    // conservative bar that catches the ~50x regression class without flaking on
    // debug-build / GPU-load variance. (Measured ~29 docs/s on a 3060.)
    const MIN_DOCS_PER_SEC: f64 = 10.0;
    const MAX_DRIFT_MIB: f64 = 512.0;

    // Warm up (one batch) so kernels are JITed and caches primed.
    run_batch(&encoder, &client, &device, DOCS_PER_BATCH, SEQ);
    client.synchronize();

    let (free0, _) = device.memory_info().expect("meminfo");
    let t0 = Instant::now();
    for _ in 0..BATCHES {
        run_batch(&encoder, &client, &device, DOCS_PER_BATCH, SEQ);
    }
    client.synchronize();
    let elapsed = t0.elapsed().as_secs_f64();
    let (free1, _) = device.memory_info().expect("meminfo");

    let docs = (BATCHES * DOCS_PER_BATCH) as f64;
    let docs_per_sec = docs / elapsed;
    let drift_mib = (free0 as i64 - free1 as i64) as f64 / 1048576.0;
    println!(
        "[perf cap={cap}] {docs} docs in {elapsed:.2}s => {docs_per_sec:.1} docs/s | \
         free {:.0}->{:.0} MiB (drift {drift_mib:.0} MiB)",
        free0 as f64 / 1048576.0,
        free1 as f64 / 1048576.0,
    );

    // Acceptance (per resource/encoder_embed_gpu_memory_fix.md §7): throughput
    // within ~2x of pre-fix AND bounded memory, both at once.
    assert!(
        docs_per_sec >= MIN_DOCS_PER_SEC,
        "throughput {docs_per_sec:.1} docs/s below floor {MIN_DOCS_PER_SEC} — F16/WMMA \
         regression (F32 matmul fallback?)",
    );
    assert!(
        drift_mib.abs() <= MAX_DRIFT_MIB,
        "memory drift {drift_mib:.0} MiB exceeds {MAX_DRIFT_MIB} — unbounded retention",
    );
}

// Regression (§7e): a single short query (batch=1) through the PADDED path that
// `embed_text` uses — input [1, seq] (3D internal hidden states). The pad-to-16
// matmul wrapper must narrow the LAST TWO dims, not dims 0/1; narrowing the size-1
// batch dim produced a degenerate `[0, seq]` → "Shape mismatch: expected [1], got
// [0, 17]". This is the query path that broke; run_batch (varlen) did NOT catch it.
#[test]
fn single_short_query_padded_path() {
    let _g = cuda_lock();
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let encoder = make_encoder(&device, &client);
    // Sizes around the GEMV(16) / WMMA-align(16) boundaries, all batch=1.
    for seq in [1usize, 7, 16, 17, 31, 33, 64, 65] {
        let ids: Vec<i64> = (0..seq as i64).map(|i| i % VOCAB as i64).collect();
        let input = Tensor::<CudaRuntime>::from_slice(&ids, &[1, seq], &device);
        let out = encoder
            .embed_inference(&client, &input, None)
            .unwrap_or_else(|e| panic!("padded embed failed at seq={seq}: {e}"));
        assert_eq!(out.shape(), &[1, HIDDEN], "seq={seq}");
        client.synchronize();
    }
}
