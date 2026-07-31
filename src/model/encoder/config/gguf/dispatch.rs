//! Architecture dispatch for GGUF metadata parsing.

use crate::error::{Error, Result};
use crate::format::GgufMetadata;
use crate::model::encoder::config::EncoderConfig;

/// Architectures this encoder can load, for the unsupported-architecture error.
const SUPPORTED: &str = "bert, nomic-bert, gemma-embedding, qwen3, jina-bert-v2, jina-bert-v3 \
     (XLM-RoBERTa loads via the bert namespace)";

impl EncoderConfig {
    /// Build an `EncoderConfig` from GGUF metadata keys.
    ///
    /// Dispatches on `general.architecture`. Anything without a dedicated
    /// namespace falls through to the BERT path, which also serves XLM-RoBERTa
    /// (distinguished by its SentencePiece tokenizer).
    pub fn from_gguf_metadata(metadata: &GgufMetadata) -> Result<Self> {
        match metadata.get_string("general.architecture") {
            Some("gemma-embedding") => Self::from_gguf_metadata_gemma(metadata),
            Some("nomic-bert") => Self::from_gguf_metadata_nomic(metadata),
            Some("qwen3") => Self::from_gguf_metadata_qwen3(metadata),
            Some("jina-bert-v2") => Self::from_gguf_metadata_jina_v2(metadata),
            Some("jina-bert-v3") => Self::from_gguf_metadata_jina_v3(metadata),
            _ => {
                // Report the architecture by name rather than letting this fail
                // on a missing `bert.*` key — that names the wrong architecture
                // and reads like a corrupt file rather than an unsupported one.
                let arch = metadata
                    .get_string("general.architecture")
                    .unwrap_or("<unset>");
                if metadata.get_u32("bert.embedding_length").is_none() {
                    return Err(Error::ModelError {
                        reason: format!(
                            "unsupported encoder architecture '{arch}': no `bert.*` \
                             metadata found. Supported: {SUPPORTED}."
                        ),
                    });
                }
                Self::from_gguf_metadata_bert(metadata)
            }
        }
    }
}

/// Read a required `u32` key, naming the key in the error.
pub(super) fn required_u32(metadata: &GgufMetadata, key: &str) -> Result<usize> {
    metadata
        .get_u32(key)
        .map(|v| v as usize)
        .ok_or_else(|| Error::ModelError {
            reason: format!("GGUF missing {key}"),
        })
}

/// Vocabulary size from the tokenizer token array, or `fallback` when absent.
pub(super) fn vocab_size(metadata: &GgufMetadata, fallback: usize) -> usize {
    metadata
        .get_array("tokenizer.ggml.tokens")
        .map(|a| a.len())
        .unwrap_or(fallback)
}

/// Reject a GGUF whose `pooling_type` is not one this encoder implements.
pub(super) fn require_pooling_type(
    metadata: &GgufMetadata,
    key: &str,
    accepted: &[u32],
) -> Result<()> {
    if let Some(pt) = metadata.get_u32(key)
        && !accepted.contains(&pt)
    {
        return Err(Error::ModelError {
            reason: format!(
                "{key} = {pt} is not supported; this architecture supports {accepted:?} \
                 (1 = mean, 2 = cls, 3 = last)"
            ),
        });
    }
    Ok(())
}
