use super::super::config::HiddenAct;
use super::*;
use crate::error::Error;
use crate::model::Pooling;
use crate::test_utils::cpu_setup;
use numr::runtime::cpu::CpuRuntime;

#[test]
fn cpu_prefers_f32_compute() {
    assert_eq!(preferred_compute_dtype::<CpuRuntime>(), DType::F32);
}

// from_gguf must select F16 on CUDA (WMMA) when the f16 feature is on, so the
// embed forward uses tensor cores instead of the ~50-100x slower F32 matmul.
// No GPU needed: R::name() is a pure identifier.
#[cfg(all(feature = "cuda", feature = "f16"))]
#[test]
fn cuda_prefers_f16_compute() {
    assert_eq!(
        preferred_compute_dtype::<numr::runtime::cuda::CudaRuntime>(),
        DType::F16
    );
}

fn make_test_pipeline() -> (EmbeddingPipeline<CpuRuntime>, numr::runtime::cpu::CpuClient) {
    let (client, device) = cpu_setup();

    // Use cl100k_base vocab size for a realistic tokenizer
    let tokenizer = splintr::from_pretrained("cl100k_base").unwrap();
    let vocab_size = tokenizer.vocab_size();

    let config = EncoderConfig {
        vocab_size,
        hidden_size: 8,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        intermediate_size: 16,
        max_position_embeddings: 64,
        layer_norm_eps: 1e-12,
        hidden_act: HiddenAct::Gelu,
        type_vocab_size: 0,
        arch_family: crate::model::encoder::config::ArchFamily::Bert,
        padding_token_id: 0,
        compute_dtype: numr::dtype::DType::F32,
        rope_freq_base: 10000.0,
        causal: false,
        ffn_variant: crate::model::encoder::config::FfnVariant::Standard,
        token_type_embed_size: 0,
        num_kv_heads: 0,
        head_dim_explicit: None,
        rms_eps: 1e-6,
        sliding_window: None,
        embed_scale: false,
        max_tokens_per_forward: None,
    };

    let d = &device;
    let encoder = Encoder::from_weights(config, Pooling::Mean, |name| match name {
        "embeddings.word_embeddings.weight" => Ok(Tensor::from_slice(
            &vec![0.1f32; vocab_size * 8],
            &[vocab_size, 8],
            d,
        )),
        "embeddings.position_embeddings.weight" => {
            Ok(Tensor::from_slice(&vec![0.01f32; 64 * 8], &[64, 8], d))
        }
        "embeddings.layer_norm.weight" => Ok(Tensor::from_slice(&[1.0f32; 8], &[8], d)),
        "embeddings.layer_norm.bias" => Ok(Tensor::from_slice(&[0.0f32; 8], &[8], d)),
        n if n.ends_with("query.weight")
            || n.ends_with("key.weight")
            || n.ends_with("value.weight")
            || n.ends_with("attention.output.dense.weight") =>
        {
            Ok(Tensor::from_slice(&vec![0.02f32; 8 * 8], &[8, 8], d))
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
        n if n.ends_with("intermediate.dense.weight") => {
            Ok(Tensor::from_slice(&vec![0.02f32; 16 * 8], &[16, 8], d))
        }
        n if n.ends_with("intermediate.dense.bias") => {
            Ok(Tensor::from_slice(&[0.0f32; 16], &[16], d))
        }
        n if n.ends_with("output.dense.weight") => {
            Ok(Tensor::from_slice(&vec![0.02f32; 8 * 16], &[8, 16], d))
        }
        _ => Err(Error::ModelError {
            reason: format!("unknown weight: {name}"),
        }),
    })
    .unwrap();

    let pipeline = EmbeddingPipeline::new(encoder, tokenizer, device);
    (pipeline, client)
}

#[test]
fn test_embed_text_returns_hidden_size() {
    let (pipeline, client) = make_test_pipeline();
    let emb = pipeline.embed_text(&client, "hello").unwrap();
    assert_eq!(emb.len(), 8);
}

#[test]
fn test_embed_texts_batch() {
    let (pipeline, client) = make_test_pipeline();
    let embs = pipeline.embed_texts(&client, &["hello", "world"]).unwrap();
    assert_eq!(embs.len(), 2);
    assert_eq!(embs[0].len(), 8);
    assert_eq!(embs[1].len(), 8);
}

#[test]
fn test_embed_texts_empty() {
    let (pipeline, client) = make_test_pipeline();
    let embs = pipeline.embed_texts(&client, &[]).unwrap();
    assert!(embs.is_empty());
}

/// Build a pipeline whose position embeddings are non-uniform so that
/// unmasked padding produces a detectably different mean-pool output.
fn make_pipeline_with_distinct_positions()
-> (EmbeddingPipeline<CpuRuntime>, numr::runtime::cpu::CpuClient) {
    let (client, device) = cpu_setup();

    let tokenizer = splintr::from_pretrained("cl100k_base").unwrap();
    let vocab_size = tokenizer.vocab_size();

    let config = EncoderConfig {
        vocab_size,
        hidden_size: 8,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        intermediate_size: 16,
        max_position_embeddings: 64,
        layer_norm_eps: 1e-12,
        hidden_act: HiddenAct::Gelu,
        type_vocab_size: 0,
        arch_family: crate::model::encoder::config::ArchFamily::Bert,
        padding_token_id: 0,
        compute_dtype: numr::dtype::DType::F32,
        rope_freq_base: 10000.0,
        causal: false,
        ffn_variant: crate::model::encoder::config::FfnVariant::Standard,
        token_type_embed_size: 0,
        num_kv_heads: 0,
        head_dim_explicit: None,
        rms_eps: 1e-6,
        sliding_window: None,
        embed_scale: false,
        max_tokens_per_forward: None,
    };

    let d = &device;
    // Position embeddings: position i has value (i+1) * 0.1 in all 8 dims,
    // so different positions produce distinctly different hidden states.
    let mut pos_emb_data = vec![0.0f32; 64 * 8];
    for pos in 0..64usize {
        let v = (pos + 1) as f32 * 0.1;
        for dim in 0..8usize {
            pos_emb_data[pos * 8 + dim] = v;
        }
    }

    let encoder = Encoder::from_weights(config, Pooling::Mean, |name| match name {
        "embeddings.word_embeddings.weight" => Ok(Tensor::from_slice(
            &vec![0.1f32; vocab_size * 8],
            &[vocab_size, 8],
            d,
        )),
        "embeddings.position_embeddings.weight" => {
            Ok(Tensor::from_slice(&pos_emb_data, &[64, 8], d))
        }
        "embeddings.layer_norm.weight" => Ok(Tensor::from_slice(&[1.0f32; 8], &[8], d)),
        "embeddings.layer_norm.bias" => Ok(Tensor::from_slice(&[0.0f32; 8], &[8], d)),
        n if n.ends_with("query.weight")
            || n.ends_with("key.weight")
            || n.ends_with("value.weight")
            || n.ends_with("attention.output.dense.weight") =>
        {
            Ok(Tensor::from_slice(&vec![0.02f32; 8 * 8], &[8, 8], d))
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
        n if n.ends_with("intermediate.dense.weight") => {
            Ok(Tensor::from_slice(&vec![0.02f32; 16 * 8], &[16, 8], d))
        }
        n if n.ends_with("intermediate.dense.bias") => {
            Ok(Tensor::from_slice(&[0.0f32; 16], &[16], d))
        }
        n if n.ends_with("output.dense.weight") => {
            Ok(Tensor::from_slice(&vec![0.02f32; 8 * 16], &[8, 16], d))
        }
        _ => Err(Error::ModelError {
            reason: format!("unknown weight: {name}"),
        }),
    })
    .unwrap();

    let pipeline = EmbeddingPipeline::new(encoder, tokenizer, device);
    (pipeline, client)
}

/// Core correctness test: embedding a short sequence alone (no padding)
/// must produce the same vector as embedding it in a batch alongside a
/// longer sequence (where it is padded on the right).
///
/// Without an attention mask the pad tokens contribute to the mean-pool
/// output, causing V1 != V1'.  With the mask they are excluded and
/// V1 == V1' (within float epsilon).
#[test]
fn embed_texts_with_padding_excludes_pad_contamination() {
    let (pipeline, client) = make_pipeline_with_distinct_positions();

    // Embed "hello" alone — no padding, no mask needed.
    let solo = pipeline.embed_texts(&client, &["hello"]).unwrap();
    let v1 = &solo[0];

    // Embed "hello" together with a longer text — "hello" gets padded.
    let batch = pipeline
        .embed_texts(&client, &["hello", "this is a longer input sequence"])
        .unwrap();
    let v1_prime = &batch[0];

    assert_eq!(v1.len(), v1_prime.len());

    // Both should agree to within a small epsilon; if masking is broken
    // they will differ by the contribution of pad-token hidden states.
    let max_diff = v1
        .iter()
        .zip(v1_prime.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_diff < 1e-5,
        "pad contamination detected: max element-wise diff = {max_diff:.3e}"
    );
}

/// Regression guard: a single-text call (no padding path) must not regress.
#[test]
fn embed_text_single_is_stable() {
    let (pipeline, client) = make_pipeline_with_distinct_positions();
    let e1 = pipeline.embed_text(&client, "hello world").unwrap();
    let e2 = pipeline.embed_text(&client, "hello world").unwrap();
    assert_eq!(e1, e2);
}
