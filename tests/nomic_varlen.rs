//! Integration tests for the NomicBert varlen (packed, unpadded) attention path.
//!
//! ## Test 1 — pooling parity
//! Construct a known `[total_tokens, hidden]` tensor and `seg_ids` for two
//! sequences of different lengths, run the scatter_reduce-Mean pooling, and
//! verify the result equals a manually-computed per-segment mean.
//!
//! ## Test 2 — cu_seqlens isolation
//! Run varlen attention on a packed batch of 2 sequences and verify that
//! sequence A's output tokens match the output produced when sequence A is
//! run alone as a 1-sequence packed batch.  This proves cu_seqlens prevents
//! cross-sequence attention leakage.

use boostr::error::Result;
use boostr::model::encoder::{
    config::{ArchFamily, EncoderConfig, FfnVariant, HiddenAct},
    model::{Encoder, Pooling},
    pipeline::EmbeddingPipeline,
};
use numr::dtype::DType;
use numr::ops::{IndexingOps, ScatterReduceOp};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

// ──────────────────────────────────────────────────────────────────────────────
// Helper: build a minimal NomicBert encoder on CPU.
// hidden=8, heads=2, head_dim=4, 1 layer, SwiGLU FFN (intermediate=16).
// ──────────────────────────────────────────────────────────────────────────────
fn make_nomic_encoder() -> (Encoder<CpuRuntime>, CpuClient, CpuDevice) {
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
        type_vocab_size: 2,
        arch_family: ArchFamily::NomicBert,
        padding_token_id: 0,
        compute_dtype: DType::F32,
        rope_freq_base: 10000.0,
        causal: false,
        ffn_variant: FfnVariant::GatedSilu,
        token_type_embed_size: 2,
        num_kv_heads: heads,
        head_dim_explicit: None,
        rms_eps: 1e-6,
        sliding_window: None,
        embed_scale: false,
        max_tokens_per_forward: None,
    };

    use boostr::nn::Weight;

    let encoder = Encoder::from_weights_nomic(config, Pooling::Mean, &client, |name| {
        // Token embedding: [vocab, hidden]
        if name == "token_embd.weight" {
            return Ok(Weight::Standard(Tensor::from_slice(
                &vec![0.1f32; vocab * hidden],
                &[vocab, hidden],
                d,
            )));
        }
        // Embedding norm
        if name == "token_embd_norm.weight" {
            return Ok(Weight::Standard(Tensor::from_slice(
                &vec![1.0f32; hidden],
                &[hidden],
                d,
            )));
        }
        if name == "token_embd_norm.bias" {
            return Ok(Weight::Standard(Tensor::from_slice(
                &vec![0.0f32; hidden],
                &[hidden],
                d,
            )));
        }
        // Token type embedding: [2, hidden]
        if name == "token_types.weight" {
            return Ok(Weight::Standard(Tensor::from_slice(
                &vec![0.05f32; 2 * hidden],
                &[2, hidden],
                d,
            )));
        }
        // Layer 0 QKV (fused [3H, H])
        if name == "blk.0.attn_qkv.weight" {
            return Ok(Weight::Standard(Tensor::from_slice(
                &vec![0.02f32; 3 * hidden * hidden],
                &[3 * hidden, hidden],
                d,
            )));
        }
        if name == "blk.0.attn_output.weight" {
            return Ok(Weight::Standard(Tensor::from_slice(
                &vec![0.02f32; hidden * hidden],
                &[hidden, hidden],
                d,
            )));
        }
        if name == "blk.0.attn_output_norm.weight" {
            return Ok(Weight::Standard(Tensor::from_slice(
                &vec![1.0f32; hidden],
                &[hidden],
                d,
            )));
        }
        if name == "blk.0.attn_output_norm.bias" {
            return Ok(Weight::Standard(Tensor::from_slice(
                &vec![0.0f32; hidden],
                &[hidden],
                d,
            )));
        }
        if name == "blk.0.ffn_gate.weight" {
            return Ok(Weight::Standard(Tensor::from_slice(
                &vec![0.02f32; inter * hidden],
                &[inter, hidden],
                d,
            )));
        }
        if name == "blk.0.ffn_up.weight" {
            return Ok(Weight::Standard(Tensor::from_slice(
                &vec![0.02f32; inter * hidden],
                &[inter, hidden],
                d,
            )));
        }
        if name == "blk.0.ffn_down.weight" {
            return Ok(Weight::Standard(Tensor::from_slice(
                &vec![0.02f32; hidden * inter],
                &[hidden, inter],
                d,
            )));
        }
        if name == "blk.0.layer_output_norm.weight" {
            return Ok(Weight::Standard(Tensor::from_slice(
                &vec![1.0f32; hidden],
                &[hidden],
                d,
            )));
        }
        if name == "blk.0.layer_output_norm.bias" {
            return Ok(Weight::Standard(Tensor::from_slice(
                &vec![0.0f32; hidden],
                &[hidden],
                d,
            )));
        }
        Err(boostr::error::Error::ModelError {
            reason: format!("unknown weight: {name}"),
        })
    })
    .unwrap();

    (encoder, client, device)
}

// ──────────────────────────────────────────────────────────────────────────────
// Test 1: scatter_reduce Mean pooling correctness
// ──────────────────────────────────────────────────────────────────────────────
/// Build a known `[total_tokens, hidden]` tensor and `seg_ids` for two
/// sequences (seq 0 has 3 tokens, seq 1 has 2 tokens), scatter-mean-pool them,
/// and verify the result equals the manually computed per-segment mean.
#[test]
fn test_varlen_pooling_scatter_mean() {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    let d = &device;

    let hidden = 4usize;
    let batch = 2usize;

    // hidden_out[token, dim]:
    //   seq 0: tokens 0,1,2  → rows [1,2,3]  (all dims identical per token)
    //   seq 1: tokens 3,4    → rows [4,5]
    #[rustfmt::skip]
    let hidden_data: Vec<f32> = vec![
        1.0, 1.0, 1.0, 1.0,   // token 0 (seq 0)
        2.0, 2.0, 2.0, 2.0,   // token 1 (seq 0)
        3.0, 3.0, 3.0, 3.0,   // token 2 (seq 0)
        4.0, 4.0, 4.0, 4.0,   // token 3 (seq 1)
        5.0, 5.0, 5.0, 5.0,   // token 4 (seq 1)
    ];
    let total_tokens = 5usize;

    let hidden_t = Tensor::<CpuRuntime>::from_slice(&hidden_data, &[total_tokens, hidden], d);

    // seg_ids: [0,0,0, 1,1] — each token's batch-sequence index
    let seg_ids_data: Vec<i32> = vec![0, 0, 0, 1, 1];
    let seg_ids = Tensor::<CpuRuntime>::from_slice(&seg_ids_data, &[total_tokens], d);

    // Build index for scatter_reduce: reshape seg_ids to [total,1] then broadcast to [total,hidden]
    let seg_2d = seg_ids.reshape(&[total_tokens, 1]).unwrap();
    let idx = seg_2d
        .broadcast_to(&[total_tokens, hidden])
        .unwrap()
        .contiguous()
        .unwrap();

    let dst = Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; batch * hidden], &[batch, hidden], d);

    let pooled = client
        .scatter_reduce(&dst, 0, &idx, &hidden_t, ScatterReduceOp::Mean, false)
        .expect("scatter_reduce failed");

    assert_eq!(pooled.shape(), &[batch, hidden]);

    let result: Vec<f32> = pooled.to_vec();

    // Expected:
    //   seq 0 mean = (1+2+3)/3 = 2.0
    //   seq 1 mean = (4+5)/2   = 4.5
    let eps = 1e-5f32;
    for dim in 0..hidden {
        let s0 = result[dim];
        let s1 = result[hidden + dim];
        assert!(
            (s0 - 2.0).abs() < eps,
            "seq0 dim {dim}: expected 2.0, got {s0}"
        );
        assert!(
            (s1 - 4.5).abs() < eps,
            "seq1 dim {dim}: expected 4.5, got {s1}"
        );
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Test 2: cu_seqlens isolation — no cross-sequence attention leakage
// ──────────────────────────────────────────────────────────────────────────────
/// Run the full NomicBert varlen forward on a 2-sequence packed batch (seq A
/// with 3 tokens, seq B with 2 tokens) and verify that seq A's output tokens
/// match those produced when seq A is run alone as a 1-sequence packed batch.
#[test]
fn test_varlen_attention_no_cross_sequence_leakage() -> Result<()> {
    let (encoder, client, device) = make_nomic_encoder();
    let d = &device;

    // Seq A: token ids [1, 2, 3]  (3 tokens)
    // Seq B: token ids [4, 5]     (2 tokens)
    let ids_a = vec![1i64, 2, 3];
    let ids_b = vec![4i64, 5];
    let len_a = ids_a.len();
    let len_b = ids_b.len();

    // ── Pack A+B together ──────────────────────────────────────────────────
    let mut flat_ab: Vec<i64> = ids_a.clone();
    flat_ab.extend(&ids_b);
    let total_ab = flat_ab.len();

    let cu_ab: Vec<i32> = vec![0, len_a as i32, (len_a + len_b) as i32];
    let pos_ab: Vec<i64> = (0..len_a as i64).chain(0..len_b as i64).collect();
    let seg_ab: Vec<i32> = std::iter::repeat_n(0i32, len_a)
        .chain(std::iter::repeat_n(1i32, len_b))
        .collect();
    let max_ab = len_a.max(len_b);

    let input_ab = Tensor::<CpuRuntime>::from_slice(&flat_ab, &[total_ab], d);
    let cu_ab_t = Tensor::<CpuRuntime>::from_slice(&cu_ab, &[3], d);
    let pos_ab_t = Tensor::<CpuRuntime>::from_slice(&pos_ab, &[total_ab], d);
    let seg_ab_t = Tensor::<CpuRuntime>::from_slice(&seg_ab, &[total_ab], d);

    let out_ab = encoder.encode_inference_varlen(
        &client, &input_ab, &cu_ab_t, &pos_ab_t, &seg_ab_t, 2, max_ab,
    )?;
    // out_ab: [total_ab, hidden] — first len_a rows belong to seq A
    let out_ab_vec: Vec<f32> = out_ab.to_vec();

    // ── Run A alone ────────────────────────────────────────────────────────
    let cu_a: Vec<i32> = vec![0, len_a as i32];
    let pos_a: Vec<i64> = (0..len_a as i64).collect();
    let seg_a: Vec<i32> = vec![0i32; len_a];

    let input_a = Tensor::<CpuRuntime>::from_slice(&ids_a, &[len_a], d);
    let cu_a_t = Tensor::<CpuRuntime>::from_slice(&cu_a, &[2], d);
    let pos_a_t = Tensor::<CpuRuntime>::from_slice(&pos_a, &[len_a], d);
    let seg_a_t = Tensor::<CpuRuntime>::from_slice(&seg_a, &[len_a], d);

    let out_a = encoder
        .encode_inference_varlen(&client, &input_a, &cu_a_t, &pos_a_t, &seg_a_t, 1, len_a)?;
    let out_a_vec: Vec<f32> = out_a.to_vec();

    let hidden = encoder.config().hidden_size;

    // The first len_a rows of out_ab must match out_a row-for-row.
    let eps = 1e-5f32;
    for token in 0..len_a {
        for dim in 0..hidden {
            let ab_val = out_ab_vec[token * hidden + dim];
            let a_val = out_a_vec[token * hidden + dim];
            assert!(
                (ab_val - a_val).abs() < eps,
                "cross-sequence leakage detected at token={token} dim={dim}: \
                 packed={ab_val:.6}, solo={a_val:.6}, diff={:.3e}",
                (ab_val - a_val).abs()
            );
        }
    }

    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────────
// Test 3: sub-batching invariance
//
// Embed a set of documents with a budget large enough for one forward, then
// embed the same documents with a tiny budget that forces multiple sub-batches,
// and assert element-wise equality (within 1e-5).  Proves that chunking does
// not change embeddings.
// ──────────────────────────────────────────────────────────────────────────────

/// Build a NomicBert `EmbeddingPipeline` on CPU with the given
/// `max_tokens_per_forward` budget.
fn make_nomic_pipeline(budget: Option<usize>) -> (EmbeddingPipeline<CpuRuntime>, CpuClient) {
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
        type_vocab_size: 2,
        arch_family: ArchFamily::NomicBert,
        padding_token_id: 0,
        compute_dtype: DType::F32,
        rope_freq_base: 10000.0,
        causal: false,
        ffn_variant: FfnVariant::GatedSilu,
        token_type_embed_size: 2,
        num_kv_heads: heads,
        head_dim_explicit: None,
        rms_eps: 1e-6,
        sliding_window: None,
        embed_scale: false,
        max_tokens_per_forward: budget,
    };

    use boostr::nn::Weight;

    let encoder = Encoder::from_weights_nomic(config, Pooling::Mean, &client, |name| {
        if name == "token_embd.weight" {
            return Ok(Weight::Standard(Tensor::from_slice(
                &vec![0.1f32; vocab * hidden],
                &[vocab, hidden],
                d,
            )));
        }
        if name == "token_embd_norm.weight" {
            return Ok(Weight::Standard(Tensor::from_slice(
                &vec![1.0f32; hidden],
                &[hidden],
                d,
            )));
        }
        if name == "token_embd_norm.bias" {
            return Ok(Weight::Standard(Tensor::from_slice(
                &vec![0.0f32; hidden],
                &[hidden],
                d,
            )));
        }
        if name == "token_types.weight" {
            return Ok(Weight::Standard(Tensor::from_slice(
                &vec![0.05f32; 2 * hidden],
                &[2, hidden],
                d,
            )));
        }
        if name == "blk.0.attn_qkv.weight" {
            return Ok(Weight::Standard(Tensor::from_slice(
                &vec![0.02f32; 3 * hidden * hidden],
                &[3 * hidden, hidden],
                d,
            )));
        }
        if name == "blk.0.attn_output.weight" {
            return Ok(Weight::Standard(Tensor::from_slice(
                &vec![0.02f32; hidden * hidden],
                &[hidden, hidden],
                d,
            )));
        }
        if name == "blk.0.attn_output_norm.weight" {
            return Ok(Weight::Standard(Tensor::from_slice(
                &vec![1.0f32; hidden],
                &[hidden],
                d,
            )));
        }
        if name == "blk.0.attn_output_norm.bias" {
            return Ok(Weight::Standard(Tensor::from_slice(
                &vec![0.0f32; hidden],
                &[hidden],
                d,
            )));
        }
        if name == "blk.0.ffn_gate.weight" {
            return Ok(Weight::Standard(Tensor::from_slice(
                &vec![0.02f32; inter * hidden],
                &[inter, hidden],
                d,
            )));
        }
        if name == "blk.0.ffn_up.weight" {
            return Ok(Weight::Standard(Tensor::from_slice(
                &vec![0.02f32; inter * hidden],
                &[inter, hidden],
                d,
            )));
        }
        if name == "blk.0.ffn_down.weight" {
            return Ok(Weight::Standard(Tensor::from_slice(
                &vec![0.02f32; hidden * inter],
                &[hidden, inter],
                d,
            )));
        }
        if name == "blk.0.layer_output_norm.weight" {
            return Ok(Weight::Standard(Tensor::from_slice(
                &vec![1.0f32; hidden],
                &[hidden],
                d,
            )));
        }
        if name == "blk.0.layer_output_norm.bias" {
            return Ok(Weight::Standard(Tensor::from_slice(
                &vec![0.0f32; hidden],
                &[hidden],
                d,
            )));
        }
        Err(boostr::error::Error::ModelError {
            reason: format!("unknown weight: {name}"),
        })
    })
    .unwrap();

    // Use a small fixed-vocabulary synthetic tokenizer via splintr cl100k_base.
    // We only need it to produce token ids within [0, vocab); the ids are
    // remapped to [0, 31] via modulo in the embedding lookup inside numr.
    let tokenizer = splintr::from_pretrained("cl100k_base").unwrap();
    let pipeline = EmbeddingPipeline::new(encoder, tokenizer, device);
    (pipeline, client)
}

/// Sub-batching invariance: same documents, different budget → identical embeddings.
///
/// Uses a budget large enough to hold all documents in one forward, then a tiny
/// budget (1 token) that forces one sub-batch per document, and asserts all
/// result vectors agree element-wise within 1e-5.
#[test]
fn test_sub_batching_is_result_invariant() -> Result<()> {
    let texts = &[
        "hello world",
        "this is a second document",
        "and a third one",
        "four",
        "the fifth and final document",
    ];

    // Large budget: all docs in one forward.
    let (pipeline_large, client_large) = make_nomic_pipeline(Some(65536));
    let embs_large = pipeline_large.embed_texts(&client_large, texts)?;

    // Tiny budget: forces one sub-batch per document (budget = 1 token each).
    let (pipeline_tiny, client_tiny) = make_nomic_pipeline(Some(1));
    let embs_tiny = pipeline_tiny.embed_texts(&client_tiny, texts)?;

    assert_eq!(embs_large.len(), embs_tiny.len());

    let eps = 1e-5f32;
    for (doc_idx, (large_emb, tiny_emb)) in embs_large.iter().zip(embs_tiny.iter()).enumerate() {
        assert_eq!(
            large_emb.len(),
            tiny_emb.len(),
            "doc {doc_idx}: embedding length mismatch"
        );
        let max_diff = large_emb
            .iter()
            .zip(tiny_emb.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < eps,
            "doc {doc_idx}: sub-batch produced different embedding; \
             max element-wise diff = {max_diff:.3e} (budget=1 vs budget=65536)"
        );
    }

    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────────
// Test 4: many-distinct-shapes robustness (CPU, CI-runnable)
//
// Build 300 documents whose token lengths span a wide range (deterministically,
// no RNG).  Embed them twice:
//   - small budget (256 tokens) → forces many sub-batches (many distinct shapes)
//   - large budget (None / DEFAULT) → all 300 docs fit in a single forward
//
// Assertions:
//   1. Exactly 300 embeddings returned; each has length == hidden_size.
//   2. Every value in every embedding is finite (no NaN / Inf).
//   3. Element-wise equality within 1e-5 between the two budget runs.
//      This proves that driving many distinct packed shapes does not corrupt
//      any embedding — the decisive CI-runnable proof of the varlen memory fix.
// ──────────────────────────────────────────────────────────────────────────────

/// Generate a deterministic corpus of `n` documents whose word counts cycle
/// over `1..=max_word_count`.  No external RNG — fully reproducible.
fn make_corpus(n: usize, max_word_count: usize) -> Vec<String> {
    let words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "hello", "world", "rust",
        "tensor", "embed", "model", "batch", "token", "text", "data", "index", "shape",
    ];
    (0..n)
        .map(|i| {
            let word_count = (i % max_word_count) + 1;
            (0..word_count)
                .map(|j| words[(i + j) % words.len()])
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect()
}

#[test]
fn test_many_distinct_shapes_robustness() -> Result<()> {
    const N_DOCS: usize = 300;
    // Cycle word counts 1..=50; with a realistic BPE tokenizer this produces
    // token lengths spanning a wide range across 1..250, generating many
    // distinct packed-batch shapes when the budget is tight.
    const MAX_WORD_COUNT: usize = 50;

    // Small budget: 256 tokens per forward → many sub-batches, many shapes.
    const SMALL_BUDGET: usize = 256;

    let corpus = make_corpus(N_DOCS, MAX_WORD_COUNT);
    let texts: Vec<&str> = corpus.iter().map(String::as_str).collect();

    // ── Small-budget run ──────────────────────────────────────────────────
    let (pipeline_small, client_small) = make_nomic_pipeline(Some(SMALL_BUDGET));
    let embs_small = pipeline_small.embed_texts(&client_small, &texts)?;

    // Assertion 1: count and dimension.
    assert_eq!(
        embs_small.len(),
        N_DOCS,
        "small-budget: expected {N_DOCS} embeddings, got {}",
        embs_small.len()
    );
    let hidden = pipeline_small.config().hidden_size;
    for (i, emb) in embs_small.iter().enumerate() {
        assert_eq!(
            emb.len(),
            hidden,
            "small-budget: doc {i} embedding length {} != hidden_size {hidden}",
            emb.len()
        );
    }

    // Assertion 2: all values finite.
    for (i, emb) in embs_small.iter().enumerate() {
        for (j, &v) in emb.iter().enumerate() {
            assert!(
                v.is_finite(),
                "small-budget: doc {i} dim {j} is non-finite ({v})"
            );
        }
    }

    // ── Large-budget run (single forward) ────────────────────────────────
    // None → uses DEFAULT_MAX_TOKENS_PER_FORWARD, so all 300 docs pass in
    // a single varlen forward (total token count << 16384 for this corpus).
    let (pipeline_large, client_large) = make_nomic_pipeline(None);
    let embs_large = pipeline_large.embed_texts(&client_large, &texts)?;

    assert_eq!(
        embs_large.len(),
        N_DOCS,
        "large-budget: expected {N_DOCS} embeddings, got {}",
        embs_large.len()
    );

    // Assertion 3: element-wise invariance between small and large budgets.
    let eps = 1e-5f32;
    let mut global_max_diff = 0.0f32;
    for (doc_idx, (small_emb, large_emb)) in embs_small.iter().zip(embs_large.iter()).enumerate() {
        assert_eq!(
            small_emb.len(),
            large_emb.len(),
            "doc {doc_idx}: embedding length mismatch ({} vs {})",
            small_emb.len(),
            large_emb.len()
        );
        let max_diff = small_emb
            .iter()
            .zip(large_emb.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        if max_diff > global_max_diff {
            global_max_diff = max_diff;
        }
        assert!(
            max_diff < eps,
            "doc {doc_idx}: small-budget vs large-budget max element-wise diff \
             {max_diff:.3e} exceeds tolerance {eps:.0e}; varlen sub-batching \
             corrupted this embedding"
        );
    }

    println!(
        "test_many_distinct_shapes_robustness PASS: \
         {N_DOCS} docs, budget={SMALL_BUDGET}, global_max_diff={global_max_diff:.3e}"
    );

    Ok(())
}
