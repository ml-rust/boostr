//! [`VoxCpm2Model::from_gguf`]: load the whole model from a single GGUF file
//! written by `compressr convert <ckpt-dir> --format gguf`.
//!
//! # What a VoxCPM2 GGUF holds
//!
//! 577 tensors — the transformer stack, under their ORIGINAL HuggingFace
//! names. compressr writes the checkpoint's key strings verbatim, so
//! [`gguf_to_hf_name`](crate::format::gguf::gguf_to_hf_name) is deliberately
//! NOT applied here: the sub-loaders ask for exactly the keys they ask a
//! safetensors checkpoint for, and renaming would break every one of them.
//!
//! It does NOT hold the AudioVAE (a separate file that is not part of the
//! checkpoint compressr converts) and it does not hold `tokenizer.json`.
//! `from_gguf` therefore takes the VAE path as its own argument, exactly like
//! [`from_checkpoint`](VoxCpm2Model::from_checkpoint).
//!
//! # What stays quantized in memory, and what does not
//!
//! The MiniCPM4 stack (`base_lm` and `residual_lm` — 327 of the 577 tensors,
//! the bulk of the model) keeps its attention and MLP projections PACKED:
//! its loader reads them through
//! [`WeightSource::load_named_weight`](crate::model::audio::voxcpm::loader::support::WeightSource::load_named_weight),
//! which hands back a `QuantTensor`, and multiplies them with
//! `quant_matmul`. A Q4_K weight there costs Q4_K-sized memory.
//!
//! Everything else still DEQUANTIZES on the way in — `feat_encoder` /
//! `bidirectional`, `local_dit`, the `fsq` aux projections, and, on every
//! stack, the norms, biases and embedding tables, which are row gathers and
//! element-wise ops with no packed kernel to run against. Those sub-models
//! are a later unit.

use crate::error::{Error, Result};
use crate::format::gguf::Gguf;
use crate::model::audio::voxcpm::model::loader::{StackConfigs, VoxCpm2Model};
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::Runtime;
use std::path::Path;

/// GGUF metadata string key holding the verbatim contents of the
/// checkpoint's `config.json`.
///
/// compressr does not write this key yet; a later unit adds it. Reading it
/// now means that unit lands with no boostr change, and a GGUF written today
/// still loads through the `config_json` path argument.
pub const GGUF_CONFIG_JSON_KEY: &str = "voxcpm2.config_json";

impl<R: Runtime<DType = DType>> VoxCpm2Model<R>
where
    R::Client: TypeConversionOps<R>,
{
    /// Load the whole model from a GGUF plus the separate AudioVAE file.
    ///
    /// The architecture config is resolved in this order:
    ///
    /// 1. the GGUF's own [`GGUF_CONFIG_JSON_KEY`] metadata string, when present;
    /// 2. the `config_json` path argument;
    /// 3. neither — an error naming both options.
    ///
    /// `dtype` casts every transformer-stack tensor that arrives dense, same
    /// as [`from_checkpoint`](Self::from_checkpoint). Dequantized tensors
    /// arrive as F32, so `None` here means F32 rather than the BF16 a
    /// safetensors checkpoint would give.
    ///
    /// `Some(BF16)`/`Some(F16)` is REJECTED for a GGUF whose MiniCPM4
    /// projections are quantized: `quant_matmul` requires F32 activations,
    /// so honouring the request would mean dequantizing the very weights
    /// this path keeps packed. The error names the tensor.
    pub fn from_gguf<P: AsRef<Path>, Q: AsRef<Path>>(
        gguf_path: P,
        config_json: Option<&Path>,
        audiovae_path: Q,
        device: &R::Device,
        dtype: Option<DType>,
    ) -> Result<Self> {
        // Opened ONCE for all five transformer-stack sub-models, like
        // `from_checkpoint`'s safetensors source.
        let mut source = Gguf::open(gguf_path.as_ref())?;
        let embedded = source.metadata().get_string(GGUF_CONFIG_JSON_KEY);
        let content = resolve_config_text(embedded, config_json)?;
        let cfgs = StackConfigs::from_config_str(&content)?;
        Self::from_source(&mut source, cfgs, audiovae_path.as_ref(), device, dtype)
    }
}

/// Pick the `config.json` body: the GGUF's embedded copy first, the path
/// argument second, an error naming both third.
///
/// Takes the embedded string rather than the [`Gguf`] so the precedence rule
/// is testable without a real 1.2 GB checkpoint.
fn resolve_config_text(embedded: Option<&str>, path: Option<&Path>) -> Result<String> {
    if let Some(content) = embedded {
        return Ok(content.to_string());
    }
    let path = path.ok_or_else(|| Error::ModelError {
        reason: format!(
            "GGUF carries no `{GGUF_CONFIG_JSON_KEY}` metadata key and no config.json \
             path was given; pass the checkpoint's config.json, or convert with a \
             compressr that embeds it"
        ),
    })?;
    std::fs::read_to_string(path).map_err(|e| Error::ModelError {
        reason: format!("failed to read {}: {e}", path.display()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuRuntime;

    /// Write `body` to a temp file and hand back its path.
    fn write_temp(name: &str, body: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(name);
        std::fs::write(&path, body).expect("write temp config");
        path
    }

    #[test]
    fn embedded_config_wins_over_the_path_argument() {
        let path = write_temp("boostr_voxcpm2_gguf_precedence.json", "from-file");
        let text = resolve_config_text(Some("from-metadata"), Some(&path)).expect("resolved");
        let _ = std::fs::remove_file(&path);
        assert_eq!(text, "from-metadata");
    }

    #[test]
    fn path_argument_is_used_when_no_key_is_embedded() {
        let path = write_temp("boostr_voxcpm2_gguf_fallback.json", "from-file");
        let text = resolve_config_text(None, Some(&path)).expect("resolved");
        let _ = std::fs::remove_file(&path);
        assert_eq!(text, "from-file");
    }

    /// Neither source: the error must name both, so the operator knows the
    /// path argument exists.
    #[test]
    fn errors_naming_both_options_when_neither_is_present() {
        let err = resolve_config_text(None, None).unwrap_err();
        let message = err.to_string();
        assert!(message.contains(GGUF_CONFIG_JSON_KEY), "got {message}");
        assert!(message.contains("config.json"), "got {message}");
    }

    #[test]
    fn missing_config_file_is_an_error() {
        assert!(resolve_config_text(None, Some(Path::new("/nonexistent/config.json"))).is_err());
    }

    #[test]
    fn rejects_missing_gguf() {
        let device = <CpuRuntime as Runtime>::default_device();
        assert!(
            VoxCpm2Model::<CpuRuntime>::from_gguf(
                "/nonexistent/voxcpm2.gguf",
                None,
                "/nonexistent/audiovae.safetensors",
                &device,
                Some(DType::F32),
            )
            .is_err()
        );
    }
}
