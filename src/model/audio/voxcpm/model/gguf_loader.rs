//! [`VoxCpm2Model::from_gguf`]: load the whole model from a single GGUF file.
//!
//! # Two naming conventions, both loadable
//!
//! A VoxCPM2 GGUF holds 577 tensors — the transformer stack — but under one
//! of two mutually exclusive name schemes:
//!
//! - **Verbatim HuggingFace**, what `compressr convert <ckpt-dir> --format
//!   gguf` writes: the checkpoint's own key strings, unchanged
//!   (`base_lm.layers.0.self_attn.q_proj.weight`). The generic
//!   [`gguf_to_hf_name`](crate::format::gguf::gguf_to_hf_name) is deliberately
//!   NOT applied to these — the sub-loaders already ask for exactly these
//!   keys, and renaming would break every one of them.
//! - **ggml-conventional**, what third-party llama.cpp-style ports such as
//!   `cstr/voxcpm2-GGUF` write (`tslm.blk.0.attn_q.weight`). These are
//!   translated on the way in by `loader::cstr::GgmlNamedGguf`.
//!
//! Which one a file uses is decided by probing for a sentinel tensor, NOT by
//! `general.architecture` — both files set that key to `voxcpm2`. See
//! `loader::cstr::probe_naming`.
//!
//! # The AudioVAE: embedded when the converter wrote it
//!
//! A verbatim GGUF written by compressr from a directory holding
//! `audiovae.pth` carries the VAE too, folded and dense, as
//! `vae.decoder.*`/`vae.encoder.*` with the `voxcpm2.config_json` key beside
//! it. `from_gguf` probes for
//! [`VAE_GGUF_PROBE_TENSOR`](crate::model::audio::voxcpm::vae::VAE_GGUF_PROBE_TENSOR)
//! and reads the VAE from the same file when it is there; the
//! `audiovae_path` argument is then optional, and ignored when given. A
//! file without it — an older conversion, or cstr's, whose `vae.*` tensors
//! use a third naming scheme with `weight_norm` still unfolded — still needs
//! the path. A compressr GGUF also carries `tokenizer.json` under
//! [`GGUF_TOKENIZER_JSON_KEY`](crate::model::audio::voxcpm::gguf_keys::GGUF_TOKENIZER_JSON_KEY);
//! `VoxCpm2Weights::tokenizer_source` reads it. cstr's file holds none.
//!
//! An embedded VAE must be stored DENSE. `load_named` dequantizes silently,
//! and the decoder is verified against F32 fixtures, so a block-quantized
//! `vae.*` tensor is refused by name before anything is read: it is a
//! converter defect, not something to decode around.
//!
//! # What stays quantized in memory, and what does not
//!
//! EVERY matmul weight in the transformer stack stays PACKED: the MiniCPM4
//! stack (`base_lm` and `residual_lm` — 327 of the 577 tensors), the shared
//! `bidirectional` block stack behind BOTH `feat_encoder` and `local_dit`,
//! those two sub-models' own projections and time MLPs, and the `fsq`
//! in/out projections plus the six auxiliary ones. Each is read through
//! [`WeightSource::load_named_weight`](crate::model::audio::voxcpm::loader::support::WeightSource::load_named_weight),
//! which hands back a `QuantTensor`, and multiplied with `quant_matmul`. A
//! Q4_K weight costs Q4_K-sized memory.
//!
//! What still arrives DENSE is what has no packed kernel to run against:
//! every norm weight, every bias, the embedding tables (a row gather, not a
//! matmul) and `feat_encoder.special_token` (concatenated, never
//! multiplied). Three matmul weights land dense too, and not by choice:
//! `feat_decoder.estimator.in_proj`/`cond_proj` and
//! `feat_encoder.in_proj` have 64-element rows, which no K-quant block size
//! divides, so a GGUF writer stores them unquantized. Nothing special-cases
//! them — `load_named_weight` returns `Weight::Standard` for whatever the
//! file did not quantize, and `MaybeLoraLinear` runs the dense path there.
//!
//! The AudioVAE decoder is untouched by all of this: embedded or separate,
//! it is never packed, and it is cast by its own `vae_decoder_dtype`
//! argument independently of the GGUF's quantization tier (`None` keeps its
//! F32, verified against PyTorch fixtures at that dtype). The encoder has no
//! dtype option at all — always F32, see [`super::loader`]'s module docs.

use crate::error::{Error, Result};
use crate::format::gguf::Gguf;
pub use crate::model::audio::voxcpm::gguf_keys::GGUF_CONFIG_JSON_KEY;
use crate::model::audio::voxcpm::loader::cstr::{GgmlNamedGguf, GgufNaming, probe_naming};
use crate::model::audio::voxcpm::loader::support::DenseWeightSource;
use crate::model::audio::voxcpm::model::loader::{StackConfigs, VoxCpm2Model, packed_vae_tensor};
use crate::model::audio::voxcpm::vae::{VAE_GGUF_PROBE_TENSOR, VAE_GGUF_ROOT};
use crate::quant::traits::DequantOps;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ReduceOps, TensorOps, TypeConversionOps, UnaryOps};
use numr::runtime::Runtime;
use std::path::Path;

impl<R: Runtime<DType = DType>> VoxCpm2Model<R>
where
    R::Client: TypeConversionOps<R> + ReduceOps<R> + UnaryOps<R> + BinaryOps<R> + TensorOps<R>,
{
    /// Load the whole model from a GGUF, and the separate AudioVAE file
    /// when the GGUF does not embed one.
    ///
    /// The architecture config is resolved in this order:
    ///
    /// 1. the GGUF's own [`GGUF_CONFIG_JSON_KEY`] metadata string, when present;
    /// 2. the `config_json` path argument;
    /// 3. neither — an error naming both options.
    ///
    /// The AudioVAE follows the same shape: the GGUF's own `vae.*` tensors
    /// when present, else `audiovae_path`, else an error naming both — see
    /// the module docs. A block-quantized `vae.*` tensor is refused by name.
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
    ///
    /// `vae_decoder_dtype` casts every AudioVAE DECODER tensor independently
    /// of `dtype` — see [`super::loader`]'s module docs. The encoder always
    /// loads at F32.
    pub fn from_gguf<P: AsRef<Path>>(
        gguf_path: P,
        config_json: Option<&Path>,
        audiovae_path: Option<&Path>,
        device: &R::Device,
        dtype: Option<DType>,
        vae_decoder_dtype: Option<DType>,
    ) -> Result<Self> {
        // Opened ONCE for all five transformer-stack sub-models, like
        // `from_checkpoint`'s safetensors source.
        let mut source = Gguf::open(gguf_path.as_ref())?;
        let embedded = source.metadata().get_string(GGUF_CONFIG_JSON_KEY);
        let content = resolve_config_text(embedded, config_json)?;
        let cfgs = StackConfigs::from_config_str(&content)?;
        check_embedded_vae_dense(&source)?;
        // Two conventions, one walk: `from_source` is generic over the
        // source, so the only difference is whether the names are rewritten
        // on the way in.
        // Bound before the match so the probe's borrow of `source` ends
        // here: a match scrutinee's temporaries live for the whole match,
        // and the arms below need `source` by `&mut` and by value.
        let naming = probe_naming(&source)?;
        match naming {
            GgufNaming::Verbatim => Self::from_source(
                &mut source,
                cfgs,
                audiovae_path,
                device,
                dtype,
                vae_decoder_dtype,
            ),
            GgufNaming::Ggml => Self::from_source(
                &mut GgmlNamedGguf::new(source),
                cfgs,
                audiovae_path,
                device,
                dtype,
                vae_decoder_dtype,
            ),
        }
    }

    /// Load the whole model from a GGUF, materializing EVERY packed weight
    /// to dense F32 instead of keeping it packed.
    ///
    /// This is the weight-encoding-only measurement mode, and it exists for
    /// one reason: a cross-format quality comparison is valid only when both
    /// artifacts run the SAME activation path. A packed block kernel
    /// quantizes activations the way `ggml-quants.c` does for its block type
    /// and backend, so two block formats pay different activation error —
    /// read
    /// [`DenseWeightSource`](crate::model::audio::voxcpm::loader::support::DenseWeightSource)
    /// for the whole argument. Loaded through here, both artifacts run dense
    /// F32 end to end and differ only in weight VALUES.
    ///
    /// There is no `dtype` argument: the mode fixes F32, the dtype
    /// `quant_matmul` would have run the packed weight at.
    /// `vae_decoder_dtype` still casts the AudioVAE decoder independently —
    /// that codec is not part of the encoding-only measurement this mode
    /// exists for. The encoder always loads at F32. The AudioVAE itself is
    /// resolved exactly as [`from_gguf`](Self::from_gguf) resolves it.
    ///
    /// A dense stack costs what an unquantized checkpoint costs. Use
    /// [`from_gguf`](Self::from_gguf) for anything but a measurement.
    pub fn from_gguf_dense<P: AsRef<Path>, C: DequantOps<R>>(
        gguf_path: P,
        config_json: Option<&Path>,
        audiovae_path: Option<&Path>,
        device: &R::Device,
        client: &C,
        vae_decoder_dtype: Option<DType>,
    ) -> Result<Self> {
        let mut source = Gguf::open(gguf_path.as_ref())?;
        let embedded = source.metadata().get_string(GGUF_CONFIG_JSON_KEY);
        let content = resolve_config_text(embedded, config_json)?;
        let cfgs = StackConfigs::from_config_str(&content)?;
        check_embedded_vae_dense(&source)?;
        // Same two-convention dispatch [`from_gguf`] makes, on the same
        // probe: the decorator changes what a weight arrives AS, never which
        // name it is read under.
        let naming = probe_naming(&source)?;
        match naming {
            GgufNaming::Verbatim => Self::from_source(
                &mut DenseWeightSource::new(&mut source, client),
                cfgs,
                audiovae_path,
                device,
                Some(DType::F32),
                vae_decoder_dtype,
            ),
            GgufNaming::Ggml => {
                let mut named = GgmlNamedGguf::new(source);
                Self::from_source(
                    &mut DenseWeightSource::new(&mut named, client),
                    cfgs,
                    audiovae_path,
                    device,
                    Some(DType::F32),
                    vae_decoder_dtype,
                )
            }
        }
    }
}

/// Refuse a GGUF whose embedded AudioVAE holds a block-quantized tensor.
///
/// Runs over the directory, before any read: `Gguf::load_named` would
/// dequantize such a tensor without complaint, and the VAE loaders are
/// verified against dense fixtures only. Integer tensors pass — the
/// `sr_bin_boundaries` I32 vector is stored verbatim and never read.
///
/// Only a file that carries OUR embedded VAE (the probe tensor is present)
/// is checked: cstr's `vae.*` tensors are never read through this path, so
/// how that file stores them is not this loader's concern.
fn check_embedded_vae_dense(gguf: &Gguf) -> Result<()> {
    if gguf.tensor_info(VAE_GGUF_PROBE_TENSOR).is_err() {
        return Ok(());
    }
    for name in gguf.tensor_names().filter(|n| n.starts_with(VAE_GGUF_ROOT)) {
        let ggml_type = gguf.tensor_info(name)?.ggml_type;
        if ggml_type.to_quant_format().is_some() {
            return Err(packed_vae_tensor(name, &format!("{ggml_type:?}")));
        }
    }
    Ok(())
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
                Some(Path::new("/nonexistent/audiovae.safetensors")),
                &device,
                Some(DType::F32),
                None,
            )
            .is_err()
        );
    }

    /// A minimal GGUF v3 image: no metadata, the named tensors each with
    /// the given GGML type, one 32-element row of zero bytes behind each.
    /// Enough for the directory-level checks under test, which never read
    /// element data.
    fn gguf_with_typed_tensors(tensors: &[(&str, GgmlType)]) -> Gguf {
        use crate::format::gguf::types::GgmlType as T;
        fn put_str(out: &mut Vec<u8>, s: &str) {
            out.extend_from_slice(&(s.len() as u64).to_le_bytes());
            out.extend_from_slice(s.as_bytes());
        }
        let row_bytes = |t: T| match t {
            T::F32 => 128u64,
            T::F16 | T::BF16 => 64,
            T::I32 => 128,
            T::Q4K | T::Q8_0 => 256,
            other => panic!("fixture has no row size for {other:?}"),
        };
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        let mut offset = 0u64;
        for (name, ty) in tensors {
            put_str(&mut out, name);
            out.extend_from_slice(&1u32.to_le_bytes());
            out.extend_from_slice(&256u64.to_le_bytes());
            out.extend_from_slice(&(*ty as u32).to_le_bytes());
            out.extend_from_slice(&offset.to_le_bytes());
            offset += row_bytes(*ty);
        }
        let aligned = out.len().div_ceil(32) * 32;
        out.resize(aligned + offset as usize, 0);
        Gguf::from_bytes(out).expect("parse synthetic GGUF")
    }

    use crate::format::gguf::types::GgmlType;
    use crate::format::weight_source::WeightSource;

    #[test]
    fn a_dense_embedded_vae_passes_the_guard() {
        let gguf = gguf_with_typed_tensors(&[
            ("vae.decoder.model.0.weight", GgmlType::F32),
            ("vae.decoder.model.1.weight", GgmlType::F16),
            ("vae.encoder.block.0.weight", GgmlType::BF16),
            ("vae.decoder.sr_bin_boundaries", GgmlType::I32),
            ("base_lm.layers.0.self_attn.q_proj.weight", GgmlType::Q4K),
        ]);
        check_embedded_vae_dense(&gguf).expect("dense VAE beside a packed stack");
        assert!(WeightSource::<CpuRuntime>::has_named(
            &gguf,
            VAE_GGUF_PROBE_TENSOR
        ));
    }

    /// A packed `vae.*` tensor is refused by name and type, whatever else
    /// the file holds.
    #[test]
    fn a_quantized_embedded_vae_tensor_is_refused_by_name() {
        let gguf = gguf_with_typed_tensors(&[
            ("vae.decoder.model.0.weight", GgmlType::F32),
            (
                "vae.decoder.sr_cond_model.2.scale_embed.weight",
                GgmlType::Q4K,
            ),
        ]);
        let err = check_embedded_vae_dense(&gguf)
            .expect_err("packed VAE tensor")
            .to_string();
        assert!(
            err.contains("vae.decoder.sr_cond_model.2.scale_embed.weight"),
            "{err}"
        );
        assert!(err.contains("Q4K"), "{err}");
    }

    /// No `vae.*` at all: nothing to guard, and the probe says "not
    /// embedded", which is what sends `from_source` to the path argument.
    #[test]
    fn a_gguf_without_a_vae_passes_the_guard_and_fails_the_probe() {
        let gguf =
            gguf_with_typed_tensors(&[("base_lm.layers.0.self_attn.q_proj.weight", GgmlType::Q4K)]);
        check_embedded_vae_dense(&gguf).expect("no VAE to check");
        assert!(!WeightSource::<CpuRuntime>::has_named(
            &gguf,
            VAE_GGUF_PROBE_TENSOR
        ));
    }

    /// A file that is not ours (no probe tensor) is not checked, whatever
    /// its own `vae.*` tensors look like: this loader never reads them.
    #[test]
    fn a_foreign_vae_scheme_is_not_checked() {
        let gguf = gguf_with_typed_tensors(&[
            ("vae.enc.conv0.weight", GgmlType::Q8_0),
            ("tslm.blk.0.attn_q.weight", GgmlType::Q4K),
        ]);
        check_embedded_vae_dense(&gguf).expect("not our VAE");
    }
}
