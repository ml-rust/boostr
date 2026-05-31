//! GPU bounded-memory acceptance test for the varlen packed encoder path.
//!
//! Drives many DISTINCT packed shapes through `encode_inference_varlen` and
//! asserts GPU free memory PLATEAUS rather than falling monotonically.
//!
//! The original bug: the CUDA allocator's free list bucketed cached buffers by
//! exact byte size with a per-bucket cap but NO global cap, so variable-length
//! (many-shape) workloads spawned unbounded size buckets — each retaining live
//! device buffers the driver never reclaimed. Free VRAM fell monotonically with
//! the number of distinct shapes until OOM. The fix adds a global byte cap on
//! the free list (allocator.rs), bounding total retention.
//!
//! This test is fast: it issues a few hundred tiny forwards of varying shape
//! with no tokenizer, and samples `cuMemGetInfo` between them.
//!
//! Run with: `cd boostr && cargo test --features cuda --test encoder_varlen_gpu_acceptance`
//!
//! Realistic dims (hidden 768, 12 heads → head_dim 64) because the CUDA varlen
//! kernel only supports head_dim ∈ {64, 128}.

#![cfg(feature = "cuda")]

use std::sync::{Mutex, OnceLock};

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
const HEADS: usize = 12; // head_dim = 64
const LAYERS: usize = 1;
const INTER: usize = 2048;
const VOCAB: usize = 128;
const MAX_POS: usize = 512;

type CudaClient = <CudaRuntime as Runtime>::Client;

fn make_cuda_encoder(device: &CudaDevice, client: &CudaClient) -> Encoder<CudaRuntime> {
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
        compute_dtype: DType::F32,
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
        Ok(Weight::Standard(Tensor::from_slice(
            &vec![0.02f32; n],
            shape,
            d,
        )))
    };
    let ones = |n: usize| {
        Ok(Weight::Standard(Tensor::from_slice(
            &vec![1.0f32; n],
            &[n],
            d,
        )))
    };
    let zeros = |n: usize| {
        Ok(Weight::Standard(Tensor::from_slice(
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
                .and_then(|rest| rest.split_once('.'))
                .map(|(_idx, field)| field);
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
                    reason: format!("unknown weight: {other}"),
                }),
            }
        }
    })
    .expect("Encoder::from_weights_nomic must succeed")
}

/// One varlen forward of a single packed sequence of length `seq_len`.
fn run_one_shape(
    encoder: &Encoder<CudaRuntime>,
    client: &CudaClient,
    device: &CudaDevice,
    seq_len: usize,
) {
    let ids: Vec<i64> = (0..seq_len as i64).map(|i| i % VOCAB as i64).collect();
    let cu: Vec<i32> = vec![0, seq_len as i32];
    let pos: Vec<i64> = (0..seq_len as i64).collect();
    let seg: Vec<i32> = vec![0i32; seq_len];

    let input = Tensor::<CudaRuntime>::from_slice(&ids, &[seq_len], device);
    let cu_t = Tensor::<CudaRuntime>::from_slice(&cu, &[2], device);
    let pos_t = Tensor::<CudaRuntime>::from_slice(&pos, &[seq_len], device);
    let seg_t = Tensor::<CudaRuntime>::from_slice(&seg, &[seg.len()], device);

    let out = encoder
        .encode_inference_varlen(client, &input, &cu_t, &pos_t, &seg_t, 1, seq_len)
        .expect("encode_inference_varlen failed");
    // Drop `out` (and all intermediates) → buffers return to the free list.
    let _ = out;
}

/// Bounded-memory acceptance: many distinct shapes, free memory must plateau.
#[test]
fn gpu_varlen_memory_is_bounded() {
    let _guard = cuda_lock();

    // Force a small global free-list cap so the cache saturates within the
    // warm-up and steady-state retention is comfortably below the slack. The
    // env var is read when the allocator is constructed (first client below).
    // SAFETY: set before any CUDA client/allocator exists in this test process.
    unsafe { std::env::set_var("NUMR_CUDA_FREE_LIST_CAP_MB", "256") };

    const ITERS: usize = 250;
    const WARMUP: usize = 60;
    // Steady-state free-list retention is bounded by the 256 MiB cap (+ driver
    // pool ≤512 MiB + per-forward working set). 768 MiB slack covers that; the
    // original unbounded leak drops multiple GiB over 600 distinct shapes and
    // still trips this.
    const SLACK_BYTES: u64 = 768 * 1024 * 1024;

    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let encoder = make_cuda_encoder(&device, &client);

    let mut baseline_free: Option<u64> = None;
    let mut min_free_after_baseline = u64::MAX;

    for i in 0..ITERS {
        // Sweep sequence length over 16..=512 with a stride that produces many
        // distinct values (hence many distinct allocation sizes / free-list
        // buckets) — exactly the workload that exposed the leak.
        let seq_len = 16 + (i * 16) % (MAX_POS - 16 + 1);
        run_one_shape(&encoder, &client, &device, seq_len);

        if i % 25 == 0 {
            client.synchronize();
            let (free_bytes, _total) = device.memory_info().expect("cuMemGetInfo failed");
            let free_mib = free_bytes as f64 / (1024.0 * 1024.0);

            if i >= WARMUP {
                let base = *baseline_free.get_or_insert(free_bytes);
                min_free_after_baseline = min_free_after_baseline.min(free_bytes);
                println!(
                    "  iter={i:>4} seq_len={seq_len:>4} free={free_mib:8.1} MiB baseline={:8.1} MiB",
                    base as f64 / (1024.0 * 1024.0)
                );
                assert!(
                    free_bytes + SLACK_BYTES >= base,
                    "GPU free memory at iter={i} ({free_mib:.1} MiB) dropped > {:.0} MiB below \
                     the post-warmup baseline ({:.1} MiB) — unbounded allocator retention; the \
                     free-list global cap is not bounding memory.",
                    SLACK_BYTES as f64 / (1024.0 * 1024.0),
                    base as f64 / (1024.0 * 1024.0),
                );
            } else {
                println!("  warm-up iter={i:>4} seq_len={seq_len:>4} free={free_mib:8.1} MiB");
            }
        }
    }

    println!(
        "PASS gpu_varlen_memory_is_bounded: {ITERS} distinct shapes, free memory stayed within \
         {:.0} MiB of baseline (min seen {:.1} MiB).",
        SLACK_BYTES as f64 / (1024.0 * 1024.0),
        min_free_after_baseline as f64 / (1024.0 * 1024.0),
    );
}
