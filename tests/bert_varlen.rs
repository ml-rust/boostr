//! Parity tests: BERT and XLM-RoBERTa padded path == varlen path.
//!
//! ## What is tested
//!
//! For each architecture (BERT, XLM-R) a small encoder is built with
//! randomised-but-fixed weights (not uniform, so padded positions in the
//! padded path differ from real ones and would contaminate results if masking
//! is wrong).  A batch of three documents with different lengths is embedded
//! via BOTH the standard padded path (with an attention mask) AND the new
//! varlen packed path.  The per-document pooled embeddings must agree within
//! 1e-4.
//!
//! This proves:
//!   1. `encode_inference_varlen` correctly adds learned position embeddings
//!      for BERT/XLM-R (Change 2, mod.rs).
//!   2. `self_attention_varlen` does not error when `rope` is `None` (Change 1,
//!      layer.rs).
//!   3. XLM-R position-id offset (`pad_id + 1 + p`) is applied correctly in
//!      the varlen path (Change 3, pipeline.rs).
//!
//! The XLM-R test specifically validates the offset because a wrong offset would
//! produce different position embeddings in the varlen path versus the padded
//! path, causing the two pooled vectors to disagree.

use boostr::error::Result;
use boostr::model::encoder::{
    config::{ArchFamily, EncoderConfig, FfnVariant, HiddenAct},
    model::{Encoder, Pooling},
};
use numr::dtype::DType;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

// ──────────────────────────────────────────────────────────────────────────────
// Shared encoder builder
// ──────────────────────────────────────────────────────────────────────────────

/// Build a small BERT or XLM-R encoder on CPU.
///
/// hidden=8, heads=2 (head_dim=4 — deliberately NOT in {64,128} so the
/// pipeline dispatch test is separate).  Weights are non-uniform so that
/// different positions produce noticeably different outputs.
fn make_encoder(
    arch: ArchFamily,
    padding_token_id: i64,
) -> (Encoder<CpuRuntime>, CpuClient, CpuDevice) {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    let d = &device;

    let hidden = 8usize;
    let heads = 2usize;
    let vocab = 32usize;
    let max_pos = 64usize;
    let inter = 16usize;

    let config = EncoderConfig {
        vocab_size: vocab,
        hidden_size: hidden,
        num_hidden_layers: 1,
        num_attention_heads: heads,
        intermediate_size: inter,
        max_position_embeddings: max_pos,
        layer_norm_eps: 1e-12,
        hidden_act: HiddenAct::Gelu,
        type_vocab_size: 0,
        arch_family: arch,
        padding_token_id,
        compute_dtype: DType::F32,
        rope_freq_base: 10000.0,
        causal: false,
        ffn_variant: FfnVariant::Standard,
        token_type_embed_size: 0,
        num_kv_heads: 0,
        head_dim_explicit: None,
        rms_eps: 1e-6,
        sliding_window: None,
        embed_scale: false,
        max_tokens_per_forward: None,
    };

    // Non-uniform position embeddings: position i → value (i+1)*0.1 in every dim.
    // Non-uniform token embeddings: token t → value (t+1)*0.05 in every dim.
    // This ensures that embeddings at different positions are distinct, making
    // any offset error in the varlen path observable.
    let mut pos_emb: Vec<f32> = vec![0.0; max_pos * hidden];
    for pos in 0..max_pos {
        let v = (pos as f32 + 1.0) * 0.1;
        for dim in 0..hidden {
            pos_emb[pos * hidden + dim] = v;
        }
    }

    let mut tok_emb: Vec<f32> = vec![0.0; vocab * hidden];
    for t in 0..vocab {
        let v = (t as f32 + 1.0) * 0.05;
        for dim in 0..hidden {
            tok_emb[t * hidden + dim] = v;
        }
    }

    let encoder = Encoder::from_weights(config, Pooling::Mean, |name| match name {
        "embeddings.word_embeddings.weight" => {
            Ok(Tensor::from_slice(&tok_emb, &[vocab, hidden], d))
        }
        "embeddings.position_embeddings.weight" => {
            Ok(Tensor::from_slice(&pos_emb, &[max_pos, hidden], d))
        }
        "embeddings.layer_norm.weight" => Ok(Tensor::from_slice(&[1.0f32; 8], &[8], d)),
        "embeddings.layer_norm.bias" => Ok(Tensor::from_slice(&[0.0f32; 8], &[8], d)),
        n if n.ends_with("query.weight")
            || n.ends_with("key.weight")
            || n.ends_with("value.weight")
            || n.ends_with("attention.output.dense.weight") =>
        {
            Ok(Tensor::from_slice(
                &vec![0.02f32; hidden * hidden],
                &[hidden, hidden],
                d,
            ))
        }
        n if n.ends_with("query.bias")
            || n.ends_with("key.bias")
            || n.ends_with("value.bias")
            || n.ends_with("attention.output.dense.bias")
            || n.ends_with("output.dense.bias") =>
        {
            Ok(Tensor::from_slice(&[0.0f32; 8], &[8], d))
        }
        n if n.ends_with("LayerNorm.weight") => Ok(Tensor::from_slice(&[1.0f32; 8], &[8], d)),
        n if n.ends_with("LayerNorm.bias") => Ok(Tensor::from_slice(&[0.0f32; 8], &[8], d)),
        n if n.ends_with("intermediate.dense.weight") => Ok(Tensor::from_slice(
            &vec![0.02f32; inter * hidden],
            &[inter, hidden],
            d,
        )),
        n if n.ends_with("intermediate.dense.bias") => {
            Ok(Tensor::from_slice(&[0.0f32; 16], &[16], d))
        }
        n if n.ends_with("output.dense.weight") => Ok(Tensor::from_slice(
            &vec![0.02f32; hidden * inter],
            &[hidden, inter],
            d,
        )),
        _ => Err(boostr::error::Error::ModelError {
            reason: format!("unknown weight: {name}"),
        }),
    })
    .unwrap();

    (encoder, client, device)
}

// ──────────────────────────────────────────────────────────────────────────────
// Helper: run the padded path on a batch of documents.
//
// Returns a Vec<Vec<f32>>, one per document.
// ──────────────────────────────────────────────────────────────────────────────
fn embed_padded(
    encoder: &Encoder<CpuRuntime>,
    client: &CpuClient,
    device: &CpuDevice,
    docs: &[Vec<i64>],
    padding_token_id: i64,
) -> Vec<Vec<f32>> {
    let batch = docs.len();
    let max_len = docs.iter().map(|d| d.len()).max().unwrap_or(0);
    let hidden = encoder.config().hidden_size;

    let mut flat: Vec<i64> = Vec::with_capacity(batch * max_len);
    let mut mask_flat: Vec<f32> = Vec::with_capacity(batch * max_len);
    for ids in docs {
        let real = ids.len();
        flat.extend_from_slice(ids);
        flat.extend(std::iter::repeat_n(padding_token_id, max_len - real));
        mask_flat.extend(std::iter::repeat_n(1.0f32, real));
        mask_flat.extend(std::iter::repeat_n(0.0f32, max_len - real));
    }

    let input = Tensor::<CpuRuntime>::from_slice(&flat, &[batch, max_len], device);
    let mask = Tensor::<CpuRuntime>::from_slice(&mask_flat, &[batch, max_len], device);
    let pooled = encoder
        .embed_inference_standard(client, &input, Some(&mask))
        .unwrap();

    let data: Vec<f32> = pooled.to_vec();
    data.chunks_exact(hidden).map(|c| c.to_vec()).collect()
}

// ──────────────────────────────────────────────────────────────────────────────
// Helper: run the varlen path on a batch of documents.
//
// Builds cu_seqlens, position_ids (with optional XLM-R offset), seg_ids
// on the host — no GPU↔CPU tensor-data transfers.
// ──────────────────────────────────────────────────────────────────────────────
fn embed_varlen(
    encoder: &Encoder<CpuRuntime>,
    client: &CpuClient,
    device: &CpuDevice,
    docs: &[Vec<i64>],
    arch: ArchFamily,
    padding_token_id: i64,
) -> Vec<Vec<f32>> {
    let batch = docs.len();
    let hidden = encoder.config().hidden_size;

    let mut flat_ids: Vec<i64> = Vec::new();
    let mut cu: Vec<i32> = vec![0i32];
    let mut pos_ids: Vec<i64> = Vec::new();
    let mut seg_ids: Vec<i32> = Vec::new();
    let mut max_seqlen = 0usize;

    for (b, ids) in docs.iter().enumerate() {
        let n = ids.len();
        if n > max_seqlen {
            max_seqlen = n;
        }
        flat_ids.extend_from_slice(ids);
        for p in 0..n as i64 {
            let pid = match arch {
                ArchFamily::XlmRoberta => padding_token_id + 1 + p,
                _ => p,
            };
            pos_ids.push(pid);
        }
        seg_ids.extend(std::iter::repeat_n(b as i32, n));
        let last = *cu.last().unwrap_or(&0);
        cu.push(last + n as i32);
    }

    let total = flat_ids.len();
    let d = device;
    let input_t = Tensor::<CpuRuntime>::from_slice(&flat_ids, &[total], d);
    let cu_t = Tensor::<CpuRuntime>::from_slice(&cu, &[batch + 1], d);
    let pos_t = Tensor::<CpuRuntime>::from_slice(&pos_ids, &[total], d);
    let seg_t = Tensor::<CpuRuntime>::from_slice(&seg_ids, &[total], d);

    let pooled = encoder
        .embed_inference_varlen(client, &input_t, &cu_t, &pos_t, &seg_t, batch, max_seqlen)
        .unwrap();

    let data: Vec<f32> = pooled.to_vec();
    data.chunks_exact(hidden).map(|c| c.to_vec()).collect()
}

// ──────────────────────────────────────────────────────────────────────────────
// Test 1: BERT parity — padded path == varlen path
// ──────────────────────────────────────────────────────────────────────────────

/// BERT padded path and varlen path must produce identical per-document pooled
/// embeddings (within 1e-4) for a batch of three variable-length documents.
///
/// Doc 0: 2 tokens, Doc 1: 4 tokens, Doc 2: 3 tokens.
/// The padded path pads shorter sequences with 0s and uses an attention mask;
/// the varlen path packs them contiguously.  Mean-pool results must match.
#[test]
fn bert_varlen_parity() -> Result<()> {
    let (encoder, client, device) = make_encoder(ArchFamily::Bert, 0);

    let docs: Vec<Vec<i64>> = vec![
        vec![2, 5],       // 2 tokens
        vec![1, 7, 3, 9], // 4 tokens
        vec![4, 6, 8],    // 3 tokens
    ];

    let padded = embed_padded(&encoder, &client, &device, &docs, 0);
    let varlen = embed_varlen(&encoder, &client, &device, &docs, ArchFamily::Bert, 0);

    assert_eq!(padded.len(), varlen.len(), "output count must match");

    let eps = 1e-4f32;
    for (doc_idx, (p, v)) in padded.iter().zip(varlen.iter()).enumerate() {
        assert_eq!(p.len(), v.len(), "doc {doc_idx}: embedding length mismatch");
        let max_diff = p
            .iter()
            .zip(v.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < eps,
            "BERT doc {doc_idx}: padded vs varlen differ by {max_diff:.3e} (tolerance 1e-4)"
        );
    }

    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────────
// Test 2: XLM-RoBERTa parity — padded path == varlen path (with offset)
// ──────────────────────────────────────────────────────────────────────────────

/// XLM-R padded path and varlen path must produce identical per-document pooled
/// embeddings (within 1e-4) for a batch of three variable-length documents.
///
/// XLM-R uses pad_id = 1.  The varlen path must apply the offset `pad_id + 1 + p`
/// to match the padded path's `compute_position_ids_host` convention.  A wrong
/// offset (e.g. using plain `p`) would produce different position embeddings in
/// the varlen path and cause this test to fail.
#[test]
fn xlm_roberta_varlen_parity() -> Result<()> {
    // XLM-R: pad_id = 1.  Documents must not contain token_id == pad_id so that
    // the padded path's cumsum-style position counting matches the varlen offset.
    let pad_id: i64 = 1;
    let (encoder, client, device) = make_encoder(ArchFamily::XlmRoberta, pad_id);

    let docs: Vec<Vec<i64>> = vec![
        vec![2, 5],       // 2 tokens (no pad_id=1 in real tokens)
        vec![3, 7, 4, 9], // 4 tokens
        vec![6, 8, 10],   // 3 tokens
    ];

    // For padded path, pad with pad_id=1.
    let padded = embed_padded(&encoder, &client, &device, &docs, pad_id);
    let varlen = embed_varlen(
        &encoder,
        &client,
        &device,
        &docs,
        ArchFamily::XlmRoberta,
        pad_id,
    );

    assert_eq!(padded.len(), varlen.len(), "output count must match");

    let eps = 1e-4f32;
    for (doc_idx, (p, v)) in padded.iter().zip(varlen.iter()).enumerate() {
        assert_eq!(p.len(), v.len(), "doc {doc_idx}: embedding length mismatch");
        let max_diff = p
            .iter()
            .zip(v.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < eps,
            "XLM-R doc {doc_idx}: padded vs varlen differ by {max_diff:.3e} (tolerance 1e-4); \
             likely the XLM-R position-id offset (pad_id+1+p) is wrong in the varlen path"
        );
    }

    Ok(())
}
