//! XLM-RoBERTa encoder integration test.
//!
//! Loads a real GGUF file and verifies that a single forward pass completes
//! without error and returns finite values of the correct shape.
//!
//! Gated behind `BOOSTR_XLM_ROBERTA_MODEL` so CI does not need the model file.
//!
//! Run with:
//!   BOOSTR_XLM_ROBERTA_MODEL=/path/to/bge-reranker-v2-m3-Q4_K_M.gguf \
//!     cargo nextest run --test xlm_roberta --run-ignored=ignored-only

use boostr::format::Gguf;
use boostr::model::encoder::{Encoder, EncoderConfig, Pooling};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

/// Map the HF embedding and encoder layer weight names to GGUF names
/// (same mapping used by EmbeddingPipeline::from_gguf and ma8e's cross-encoder loader).
fn hf_to_gguf(hf: &str) -> String {
    if hf == "embeddings.word_embeddings.weight" {
        return "token_embd.weight".into();
    }
    if hf == "embeddings.position_embeddings.weight" {
        return "position_embd.weight".into();
    }
    if hf == "embeddings.layer_norm.weight" {
        return "token_embd_norm.weight".into();
    }
    if hf == "embeddings.layer_norm.bias" {
        return "token_embd_norm.bias".into();
    }

    if let Some(rest) = hf.strip_prefix("encoder.layer.") {
        if let Some(dot) = rest.find('.') {
            let layer = &rest[..dot];
            let suffix = &rest[dot + 1..];
            let mapped = match suffix {
                "attention.self.query.weight" => "attn_q.weight",
                "attention.self.query.bias" => "attn_q.bias",
                "attention.self.key.weight" => "attn_k.weight",
                "attention.self.key.bias" => "attn_k.bias",
                "attention.self.value.weight" => "attn_v.weight",
                "attention.self.value.bias" => "attn_v.bias",
                "attention.output.dense.weight" => "attn_output.weight",
                "attention.output.dense.bias" => "attn_output.bias",
                "attention.output.LayerNorm.weight" => "attn_output_norm.weight",
                "attention.output.LayerNorm.bias" => "attn_output_norm.bias",
                "intermediate.dense.weight" => "ffn_up.weight",
                "intermediate.dense.bias" => "ffn_up.bias",
                "output.dense.weight" => "ffn_down.weight",
                "output.dense.bias" => "ffn_down.bias",
                "output.LayerNorm.weight" => "layer_output_norm.weight",
                "output.LayerNorm.bias" => "layer_output_norm.bias",
                _ => return hf.to_string(),
            };
            return format!("blk.{layer}.{mapped}");
        }
    }

    hf.to_string()
}

/// Smoke test: load bge-reranker-v2-m3 (xlm-roberta backbone) from GGUF and
/// run a single forward pass.  Verifies the CLS-pool output is finite and
/// the right shape.
#[ignore]
#[test]
fn xlm_roberta_gguf_forward_pass_produces_finite_cls_embedding() {
    let model_path_str = std::env::var("BOOSTR_XLM_ROBERTA_MODEL")
        .expect("set BOOSTR_XLM_ROBERTA_MODEL=/path/to/model.gguf to run this test");

    let path = std::path::Path::new(&model_path_str);
    assert!(path.exists(), "model file not found: {}", path.display());

    let mut gguf = Gguf::open(path).expect("GGUF open failed");

    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());

    let config =
        EncoderConfig::from_gguf_metadata(gguf.metadata()).expect("config from GGUF metadata");

    let hidden_size = config.hidden_size;

    let d = &device;
    let encoder = Encoder::<CpuRuntime>::from_weights(config, Pooling::Cls, |hf_name| {
        let gguf_name = hf_to_gguf(hf_name);
        gguf.load_tensor_f32::<CpuRuntime>(&gguf_name, d)
    })
    .expect("encoder weights loaded");

    // Three-token input: [CLS=0, tok=4, SEP=2] (using xlm-roberta special token IDs)
    let input_ids = Tensor::<CpuRuntime>::from_slice(&[0i64, 4, 2], &[1, 3], &device);

    let output = encoder
        .embed(&client, &input_ids, None)
        .expect("forward pass must succeed");

    assert_eq!(
        output.shape(),
        &[1, hidden_size],
        "output shape must be [1, hidden_size]"
    );

    let values: Vec<f32> = output.tensor().to_vec();
    assert!(
        values.iter().all(|v| v.is_finite()),
        "all output values must be finite"
    );
}
