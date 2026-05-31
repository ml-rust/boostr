//! CPU parity test: Gemma-embedding padded path vs. varlen (packed) path.
//!
//! Validates that the four Gemma-specific features all produce consistent
//! results across both forward paths:
//!   - embed_scale: token embeddings × sqrt(hidden_size)
//!   - Sandwich RMSNorm: pre-attn/FFN norm + post-attn/FFN norm before residual
//!   - QK-norm: RmsNorm on Q and K after reshape, before RoPE
//!   - GQA: num_kv_heads < num_heads
//!   - RoPE (applied inside each layer)
//!   - Final output_norm before pooling
//!
//! Architecture: hidden=128, heads=4, kv_heads=2, head_dim=64 (explicit),
//! num_heads*head_dim = 256 ≠ hidden=128 (exercises projection-dim difference),
//! intermediate=256, 1 layer, GatedGelu FFN, embed_scale=true.
//!
//! Two documents of different lengths are embedded via BOTH paths. The
//! per-document mean-pooled embeddings must agree within 1e-4.

use boostr::model::encoder::{
    config::{ArchFamily, EncoderConfig, FfnVariant, HiddenAct},
    model::{Encoder, Pooling},
};
use boostr::nn::Weight;
use numr::dtype::DType;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

// ─────────────────────────────────────────────────────────────────────────────
// Architecture constants
// ─────────────────────────────────────────────────────────────────────────────
const HIDDEN: usize = 128;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64; // explicit; HEADS*HEAD_DIM = 256 ≠ HIDDEN
const INTER: usize = 256;
const VOCAB: usize = 64;
const MAX_POS: usize = 128;
const RMS_EPS: f32 = 1e-6;

fn make_gemma_encoder() -> (Encoder<CpuRuntime>, CpuClient, CpuDevice) {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    let d = &device;

    let config = EncoderConfig {
        vocab_size: VOCAB,
        hidden_size: HIDDEN,
        num_hidden_layers: 1,
        num_attention_heads: HEADS,
        intermediate_size: INTER,
        max_position_embeddings: MAX_POS,
        layer_norm_eps: RMS_EPS as f64,
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
        rms_eps: RMS_EPS as f64,
        sliding_window: None,
        embed_scale: true,
        max_tokens_per_forward: None,
    };

    // Weight factory — mirrors build_gemma.rs naming exactly.
    // Uses non-trivial values (not all identical) so different positions produce
    // different outputs and any path divergence will be detectable.
    let encoder = Encoder::from_weights_gemma(config, Pooling::Mean, &client, |name| {
        let t = match name {
            // Token embedding: [vocab, hidden]
            "token_embd.weight" => {
                let data: Vec<f32> = (0..VOCAB * HIDDEN)
                    .map(|i| (i as f32 + 1.0) * 0.001)
                    .collect();
                Tensor::from_slice(&data, &[VOCAB, HIDDEN], d)
            }
            // Sentinel position embedding (unused by Gemma)
            "position_embd.weight" => Tensor::from_slice(&vec![0.0f32; HIDDEN], &[1, HIDDEN], d),

            // Layer 0: pre-attention RMSNorm
            "blk.0.attn_norm.weight" => Tensor::from_slice(&vec![1.0f32; HIDDEN], &[HIDDEN], d),

            // q_proj: [num_heads*head_dim, hidden] = [256, 128]
            "blk.0.attn_q.weight" => {
                let rows = HEADS * HEAD_DIM;
                let data: Vec<f32> = (0..rows * HIDDEN)
                    .map(|i| (i as f32).sin() * 0.01)
                    .collect();
                Tensor::from_slice(&data, &[rows, HIDDEN], d)
            }
            // k_proj: [kv_heads*head_dim, hidden] = [128, 128]
            "blk.0.attn_k.weight" => {
                let rows = KV_HEADS * HEAD_DIM;
                let data: Vec<f32> = (0..rows * HIDDEN)
                    .map(|i| (i as f32).cos() * 0.01)
                    .collect();
                Tensor::from_slice(&data, &[rows, HIDDEN], d)
            }
            // v_proj: [kv_heads*head_dim, hidden] = [128, 128]
            "blk.0.attn_v.weight" => {
                let rows = KV_HEADS * HEAD_DIM;
                let data: Vec<f32> = (0..rows * HIDDEN)
                    .map(|i| ((i as f32) * 0.7).sin() * 0.01)
                    .collect();
                Tensor::from_slice(&data, &[rows, HIDDEN], d)
            }
            // o_proj: [hidden, num_heads*head_dim] = [128, 256]
            "blk.0.attn_output.weight" => {
                let cols = HEADS * HEAD_DIM;
                let data: Vec<f32> = (0..HIDDEN * cols)
                    .map(|i| ((i as f32) * 1.3).sin() * 0.01)
                    .collect();
                Tensor::from_slice(&data, &[HIDDEN, cols], d)
            }
            // QK-norm weights: shape [head_dim]
            "blk.0.attn_q_norm.weight" => {
                Tensor::from_slice(&vec![1.0f32; HEAD_DIM], &[HEAD_DIM], d)
            }
            "blk.0.attn_k_norm.weight" => {
                Tensor::from_slice(&vec![1.0f32; HEAD_DIM], &[HEAD_DIM], d)
            }
            // Post-attention sandwich RMSNorm
            "blk.0.post_attention_norm.weight" => {
                Tensor::from_slice(&vec![1.0f32; HIDDEN], &[HIDDEN], d)
            }
            // Pre-FFN RMSNorm
            "blk.0.ffn_norm.weight" => Tensor::from_slice(&vec![1.0f32; HIDDEN], &[HIDDEN], d),
            // GeGLU gate: [intermediate, hidden]
            "blk.0.ffn_gate.weight" => {
                let data: Vec<f32> = (0..INTER * HIDDEN)
                    .map(|i| ((i as f32) * 0.5).sin() * 0.01)
                    .collect();
                Tensor::from_slice(&data, &[INTER, HIDDEN], d)
            }
            // FFN up: [intermediate, hidden]
            "blk.0.ffn_up.weight" => {
                let data: Vec<f32> = (0..INTER * HIDDEN)
                    .map(|i| ((i as f32) * 0.3).cos() * 0.01)
                    .collect();
                Tensor::from_slice(&data, &[INTER, HIDDEN], d)
            }
            // FFN down: [hidden, intermediate]
            "blk.0.ffn_down.weight" => {
                let data: Vec<f32> = (0..HIDDEN * INTER)
                    .map(|i| ((i as f32) * 0.2).sin() * 0.01)
                    .collect();
                Tensor::from_slice(&data, &[HIDDEN, INTER], d)
            }
            // Post-FFN sandwich RMSNorm
            "blk.0.post_ffw_norm.weight" => Tensor::from_slice(&vec![1.0f32; HIDDEN], &[HIDDEN], d),
            // Final output_norm (applied before pooling)
            "output_norm.weight" => Tensor::from_slice(&vec![1.0f32; HIDDEN], &[HIDDEN], d),
            other => {
                return Err(boostr::error::Error::ModelError {
                    reason: format!("unknown weight in test: {other}"),
                });
            }
        };
        Ok(Weight::Standard(t))
    })
    .unwrap();

    (encoder, client, device)
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: run padded path for a single document → mean-pooled [hidden] vec.
// ─────────────────────────────────────────────────────────────────────────────
fn embed_padded(
    encoder: &Encoder<CpuRuntime>,
    client: &CpuClient,
    device: &CpuDevice,
    ids: &[i64],
) -> Vec<f32> {
    let seq_len = ids.len();
    let input = Tensor::<CpuRuntime>::from_slice(ids, &[1, seq_len], device);
    // No mask — single doc, no padding.
    let out = encoder
        .embed_inference_standard(client, &input, None)
        .unwrap();
    out.to_vec()
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: run varlen path for a batch of documents → Vec<Vec<f32>>.
// ─────────────────────────────────────────────────────────────────────────────
fn embed_varlen(
    encoder: &Encoder<CpuRuntime>,
    client: &CpuClient,
    device: &CpuDevice,
    all_ids: &[Vec<i64>],
) -> Vec<Vec<f32>> {
    let total: usize = all_ids.iter().map(|v| v.len()).sum();
    let sub_batch = all_ids.len();

    // Build flat_ids, cu_seqlens, pos_ids, seg_ids on the host.
    let mut flat: Vec<i64> = Vec::with_capacity(total);
    let mut cu: Vec<i32> = vec![0i32];
    let mut pos_ids: Vec<i64> = Vec::with_capacity(total);
    let mut seg_ids: Vec<i32> = Vec::with_capacity(total);
    let mut max_seqlen = 0usize;

    for (b, ids) in all_ids.iter().enumerate() {
        let n = ids.len();
        if n > max_seqlen {
            max_seqlen = n;
        }
        flat.extend_from_slice(ids);
        for p in 0..n as i64 {
            pos_ids.push(p); // Gemma: plain 0..n per sequence
        }
        for _ in 0..n {
            seg_ids.push(b as i32);
        }
        let last = *cu.last().unwrap();
        cu.push(last + n as i32);
    }

    let input_t = Tensor::<CpuRuntime>::from_slice(&flat, &[total], device);
    let cu_t = Tensor::<CpuRuntime>::from_slice(&cu, &[sub_batch + 1], device);
    let pos_t = Tensor::<CpuRuntime>::from_slice(&pos_ids, &[total], device);
    let seg_t = Tensor::<CpuRuntime>::from_slice(&seg_ids, &[total], device);

    let out = encoder
        .embed_inference_varlen(
            client, &input_t, &cu_t, &pos_t, &seg_t, sub_batch, max_seqlen,
        )
        .unwrap();

    let data: Vec<f32> = out.to_vec();
    let hidden = HIDDEN;
    data.chunks_exact(hidden).map(|c| c.to_vec()).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Parity test: padded ≈ varlen for each document in a mixed-length batch.
// ─────────────────────────────────────────────────────────────────────────────
#[test]
fn test_gemma_varlen_padded_parity() {
    let (encoder, client, device) = make_gemma_encoder();

    // Three documents of different lengths; all token ids in [1, VOCAB-1].
    let docs: Vec<Vec<i64>> = vec![
        vec![1, 5, 3],        // 3 tokens
        vec![2, 7, 4, 11, 6], // 5 tokens
        vec![9, 2],           // 2 tokens
    ];

    // Reference: padded path, one document at a time.
    let ref_embs: Vec<Vec<f32>> = docs
        .iter()
        .map(|ids| embed_padded(&encoder, &client, &device, ids))
        .collect();

    // Under test: varlen path, all documents packed together.
    let varlen_embs = embed_varlen(&encoder, &client, &device, &docs);

    assert_eq!(
        ref_embs.len(),
        varlen_embs.len(),
        "embedding count mismatch"
    );

    let mut global_max_diff = 0.0f32;

    for (i, (ref_v, var_v)) in ref_embs.iter().zip(varlen_embs.iter()).enumerate() {
        assert_eq!(
            ref_v.len(),
            HIDDEN,
            "doc {i}: padded embedding has wrong length"
        );
        assert_eq!(
            var_v.len(),
            HIDDEN,
            "doc {i}: varlen embedding has wrong length"
        );

        let max_diff = ref_v
            .iter()
            .zip(var_v.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        if max_diff > global_max_diff {
            global_max_diff = max_diff;
        }

        assert!(
            max_diff < 1e-4,
            "doc {i}: padded vs varlen max_diff = {max_diff:.3e} (threshold 1e-4)"
        );
    }

    // Report the actual max diff for visibility.
    println!(
        "gemma_varlen_padded_parity: global max_diff across {} docs = {:.3e}",
        docs.len(),
        global_max_diff
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Additional sanity: varlen output shape is correct.
// ─────────────────────────────────────────────────────────────────────────────
#[test]
fn test_gemma_varlen_output_shape() {
    let (encoder, client, device) = make_gemma_encoder();
    let docs = vec![vec![1i64, 2, 3], vec![4i64, 5]];
    let results = embed_varlen(&encoder, &client, &device, &docs);
    assert_eq!(results.len(), 2, "should return one embedding per doc");
    assert_eq!(
        results[0].len(),
        HIDDEN,
        "doc 0 embedding should be [hidden]"
    );
    assert_eq!(
        results[1].len(),
        HIDDEN,
        "doc 1 embedding should be [hidden]"
    );
}
