//! GPU bounded-memory acceptance test for the EmbeddingGemma varlen path.
//!
//! Exercises the head_dim-256 + GQA varlen CUDA kernel under the full Gemma
//! forward (embed_scale, QK-norm, sandwich RMSNorm) across many distinct packed
//! shapes, and asserts GPU free memory PLATEAUS (bounded retention) rather than
//! falling monotonically — i.e. Gemma no longer hits the padded-path OOM.
//!
//! Run with:
//!   cd boostr && cargo test --features cuda --test gemma_varlen_gpu_acceptance
//!
//! Uses real EmbeddingGemma dims: hidden 768, 12 heads, 3 kv-heads (GQA),
//! head_dim 256 (explicit).

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

// Real EmbeddingGemma-ish dims. head_dim 256 → exercises the 256 varlen kernel.
const HIDDEN: usize = 768;
const HEADS: usize = 12;
const KV_HEADS: usize = 3; // 4:1 GQA
const HEAD_DIM: usize = 256; // explicit; HEADS*HEAD_DIM = 3072 ≠ HIDDEN
const INTER: usize = 1024;
const VOCAB: usize = 128;
const MAX_POS: usize = 512;

type CudaClient = <CudaRuntime as Runtime>::Client;

fn make_cuda_gemma_encoder(device: &CudaDevice, client: &CudaClient) -> Encoder<CudaRuntime> {
    let config = EncoderConfig {
        vocab_size: VOCAB,
        hidden_size: HIDDEN,
        num_hidden_layers: 1,
        num_attention_heads: HEADS,
        intermediate_size: INTER,
        max_position_embeddings: MAX_POS,
        layer_norm_eps: 1e-6,
        hidden_act: HiddenAct::Gelu,
        type_vocab_size: 0,
        arch_family: ArchFamily::GemmaEmbedding,
        padding_token_id: 0,
        compute_dtype: DType::F32,
        rope_freq_base: 10000.0,
        causal: false,
        ffn_variant: FfnVariant::GatedGelu,
        token_type_embed_size: 0,
        num_kv_heads: KV_HEADS,
        head_dim_explicit: Some(HEAD_DIM),
        rms_eps: 1e-6,
        sliding_window: None,
        embed_scale: true,
        max_tokens_per_forward: None,
    };

    let d = device;
    let small = |n: usize, shape: &[usize]| {
        Tensor::<CudaRuntime>::from_slice(
            &(0..n).map(|i| (i as f32).sin() * 0.01).collect::<Vec<_>>(),
            shape,
            d,
        )
    };
    let ones = |n: usize| Tensor::<CudaRuntime>::from_slice(&vec![1.0f32; n], &[n], d);

    Encoder::from_weights_gemma(config, Pooling::Mean, client, |name| {
        let qd = HEADS * HEAD_DIM;
        let kvd = KV_HEADS * HEAD_DIM;
        let t = match name {
            "token_embd.weight" => small(VOCAB * HIDDEN, &[VOCAB, HIDDEN]),
            "position_embd.weight" => {
                Tensor::<CudaRuntime>::from_slice(&vec![0.0f32; HIDDEN], &[1, HIDDEN], d)
            }
            "blk.0.attn_norm.weight" => ones(HIDDEN),
            "blk.0.attn_q.weight" => small(qd * HIDDEN, &[qd, HIDDEN]),
            "blk.0.attn_k.weight" => small(kvd * HIDDEN, &[kvd, HIDDEN]),
            "blk.0.attn_v.weight" => small(kvd * HIDDEN, &[kvd, HIDDEN]),
            "blk.0.attn_output.weight" => small(HIDDEN * qd, &[HIDDEN, qd]),
            "blk.0.attn_q_norm.weight" => ones(HEAD_DIM),
            "blk.0.attn_k_norm.weight" => ones(HEAD_DIM),
            "blk.0.post_attention_norm.weight" => ones(HIDDEN),
            "blk.0.ffn_norm.weight" => ones(HIDDEN),
            "blk.0.ffn_gate.weight" => small(INTER * HIDDEN, &[INTER, HIDDEN]),
            "blk.0.ffn_up.weight" => small(INTER * HIDDEN, &[INTER, HIDDEN]),
            "blk.0.ffn_down.weight" => small(HIDDEN * INTER, &[HIDDEN, INTER]),
            "blk.0.post_ffw_norm.weight" => ones(HIDDEN),
            "output_norm.weight" => ones(HIDDEN),
            other => {
                return Err(boostr::error::Error::ModelError {
                    reason: format!("unknown weight in test: {other}"),
                });
            }
        };
        Ok(Weight::Standard(t))
    })
    .expect("Encoder::from_weights_gemma must succeed")
}

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
    let seg_t = Tensor::<CudaRuntime>::from_slice(&seg, &[seq_len], device);

    let out = encoder
        .encode_inference_varlen(client, &input, &cu_t, &pos_t, &seg_t, 1, seq_len)
        .expect("encode_inference_varlen failed (Gemma head_dim 256 path)");
    let _ = out;
}

/// Gemma (head_dim 256, GQA) varlen forward: many shapes, free memory bounded.
#[test]
fn gpu_gemma_varlen_memory_is_bounded() {
    let _guard = cuda_lock();

    // SAFETY: set before any CUDA client/allocator exists in this test process.
    unsafe { std::env::set_var("NUMR_CUDA_FREE_LIST_CAP_MB", "256") };

    const ITERS: usize = 140;
    const WARMUP: usize = 40;
    const SLACK_BYTES: u64 = 768 * 1024 * 1024;

    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let encoder = make_cuda_gemma_encoder(&device, &client);

    let mut baseline_free: Option<u64> = None;
    let mut min_free_after_baseline = u64::MAX;

    for i in 0..ITERS {
        let seq_len = 16 + (i * 16) % (MAX_POS - 16 + 1);
        run_one_shape(&encoder, &client, &device, seq_len);

        if i % 20 == 0 {
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
                    "Gemma varlen: GPU free at iter={i} ({free_mib:.1} MiB) dropped > {:.0} MiB \
                     below baseline ({:.1} MiB) — unbounded retention.",
                    SLACK_BYTES as f64 / (1024.0 * 1024.0),
                    base as f64 / (1024.0 * 1024.0),
                );
            } else {
                println!("  warm-up iter={i:>4} seq_len={seq_len:>4} free={free_mib:8.1} MiB");
            }
        }
    }

    println!(
        "PASS gpu_gemma_varlen_memory_is_bounded: {ITERS} head_dim-256 GQA shapes, free memory \
         bounded within {:.0} MiB (min {:.1} MiB).",
        SLACK_BYTES as f64 / (1024.0 * 1024.0),
        min_free_after_baseline as f64 / (1024.0 * 1024.0),
    );
}
