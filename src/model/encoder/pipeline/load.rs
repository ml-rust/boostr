//! Loading a pipeline from GGUF: compute-dtype choice, architecture dispatch,
//! and HF → GGUF weight-name mapping.

use super::super::config::{ArchFamily, EncoderConfig};
use super::super::model::{Encoder, Pooling};
use super::embed::EmbeddingPipeline;
use crate::error::Result;
use crate::format::{Gguf, extract_gguf_vocab};
use crate::nn::Weight;
use crate::quant::traits::DequantOps;
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::Runtime;

/// Preferred forward compute dtype for the embedding model.
///
/// F16 on CUDA (when the `f16` feature is enabled) so matmul uses WMMA tensor
/// cores; the F32 matmul kernel has no tensor cores and is ~50–100× slower for
/// these shapes. CPU and WGPU keep F32 (no host-side F16 WMMA; WGPU is 32-bit).
fn preferred_compute_dtype<R: Runtime>() -> DType {
    // `R::name()` is the dtype selector; bind it unconditionally so the type
    // parameter is always used (the F16 result is gated behind the feature).
    let is_cuda = R::name() == "cuda";
    #[cfg(feature = "f16")]
    if is_cuda {
        return DType::F16;
    }
    #[cfg(not(feature = "f16"))]
    let _ = is_cuda;
    DType::F32
}

impl<R: Runtime<DType = DType>> EmbeddingPipeline<R> {
    /// Load a complete sentence embedding model from a GGUF file.
    ///
    /// Extracts config, weights, and tokenizer from the single file.
    /// Dispatches on `general.architecture`:
    /// - `"nomic-bert"` → `Encoder::from_weights_nomic` with direct GGUF tensor names.
    /// - All others → standard BERT/XLM-RoBERTa path via `hf_name_to_gguf`.
    ///
    /// Compute dtype defaults to F16 on CUDA (when built with the `f16` feature)
    /// so the forward uses WMMA tensor-core matmul — the F32 matmul kernel has no
    /// tensor cores and runs ~50–100× slower for these shapes (profiled: 0.5 vs
    /// ~29 docs/s for nomic-768 on a 3060). CPU/WGPU keep F32. GGUF weights are
    /// loaded dequantized to F32 then cast to the compute dtype by the builders.
    pub fn from_gguf(gguf: &mut Gguf, device: R::Device) -> Result<Self>
    where
        R::Client: Clone + TypeConversionOps<R> + DequantOps<R>,
    {
        // The container is boostr's to read; what the vocabulary *means* is
        // splintr's, so the metadata is lifted into a plain struct and handed
        // straight over rather than interpreted here.
        let tokenizer =
            splintr::from_gguf_vocab(extract_gguf_vocab(gguf.metadata())?).map_err(|e| {
                crate::error::Error::ModelError {
                    reason: format!("GGUF tokenizer: {e}"),
                }
            })?;
        let mut config = EncoderConfig::from_gguf_metadata(gguf.metadata())?;
        config.compute_dtype = preferred_compute_dtype::<R>();
        let d = &device;

        let pooling = Pooling::from_config(&config);

        let encoder = match config.arch_family {
            ArchFamily::NomicBert => {
                // Obtain a default client to satisfy from_weights_nomic's C bound.
                // compute_dtype remains F32 on this path so no casts are issued.
                let client = R::default_client(d);
                Encoder::from_weights_nomic(config, pooling, &client, |gguf_name| {
                    gguf.load_tensor_f32::<R>(gguf_name, d)
                        .map(Weight::Standard)
                })?
            }
            ArchFamily::GemmaEmbedding => {
                // Obtain a default client to satisfy from_weights_gemma's C bound.
                // compute_dtype remains F32 on this path so no casts are issued.
                let client = R::default_client(d);
                Encoder::from_weights_gemma(config, pooling, &client, |gguf_name| {
                    gguf.load_tensor_f32::<R>(gguf_name, d)
                        .map(Weight::Standard)
                })?
            }
            ArchFamily::Qwen3 => {
                let client = R::default_client(d);
                Encoder::from_weights_qwen3(config, pooling, &client, |gguf_name| {
                    gguf.load_tensor_f32::<R>(gguf_name, d)
                        .map(Weight::Standard)
                })?
            }
            ArchFamily::JinaBertV2 => {
                let client = R::default_client(d);
                Encoder::from_weights_jina_v2(config, pooling, &client, |gguf_name| {
                    gguf.load_tensor_f32::<R>(gguf_name, d)
                        .map(Weight::Standard)
                })?
            }
            ArchFamily::JinaBertV3 => {
                let client = R::default_client(d);
                Encoder::from_weights_jina_v3(config, pooling, &client, |gguf_name| {
                    gguf.load_tensor_f32::<R>(gguf_name, d)
                        .map(Weight::Standard)
                })?
            }
            _ => Encoder::from_weights(config, pooling, |hf_name| {
                let gguf_name = hf_name_to_gguf(hf_name);
                gguf.load_tensor_f32::<R>(&gguf_name, d)
            })?,
        };

        Ok(Self::new(encoder, tokenizer, device))
    }
}

/// Map HuggingFace BERT weight names to GGUF standard names.
///
/// GGUF (llama.cpp) uses a flat naming scheme for all converted models:
/// - `token_embd.weight` / `position_embd.weight`
/// - `blk.{i}.attn_q.weight` / `.bias`, `attn_k`, `attn_v`, `attn_output`
/// - `blk.{i}.attn_output_norm.weight` / `.bias`
/// - `blk.{i}.ffn_up.weight` / `.bias` (intermediate.dense)
/// - `blk.{i}.ffn_down.weight` / `.bias` (output.dense)
/// - `blk.{i}.layer_output_norm.weight` / `.bias`
fn hf_name_to_gguf(hf: &str) -> String {
    // Embeddings
    if hf == "embeddings.word_embeddings.weight" {
        return "token_embd.weight".into();
    }
    if hf == "embeddings.position_embeddings.weight" {
        return "position_embd.weight".into();
    }
    if hf == "embeddings.token_type_embeddings.weight" {
        return "token_types.weight".into();
    }
    if hf == "embeddings.layer_norm.weight" {
        return "token_embd_norm.weight".into();
    }
    if hf == "embeddings.layer_norm.bias" {
        return "token_embd_norm.bias".into();
    }

    // Encoder layers: encoder.layer.{i}.{rest}
    if let Some(rest) = hf.strip_prefix("encoder.layer.")
        && let Some(dot) = rest.find('.')
    {
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

    hf.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
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
}
