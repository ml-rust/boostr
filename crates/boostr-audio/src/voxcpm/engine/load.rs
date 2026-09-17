//! Loading a [`VoxCpm2Engine`]: the model, the tokenizer, the optional LoRA
//! adapter and every reference recording in the voices directory.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, RandomOps, ReduceOps,
    ScalarOps, ShapeOps, TensorOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;

use boostr::model::audio::voxcpm::model::VoxCpm2Model;
use boostr::model::audio::voxcpm::{VoxCpm2Weights, VoxCpmClient};
use boostr::quant::traits::DequantOps;

use crate::decode::{decode_audio, extension_hint};
use crate::error::{Error, Result};
use crate::resample::to_mono_at_rate;
use crate::voxcpm::engine::types::{
    EncodedVoice, REF_RATE, VOICE_EXTENSIONS, VoxCpm2Engine, ZERO_SHOT_VOICE_ID,
};
use crate::voxcpm::options::VoxCpm2LoadOptions;
use crate::voxcpm::tokenizer::load_tokenizer;

impl<R> VoxCpm2Engine<R>
where
    R: Runtime<DType = DType>,
    R::Client: VoxCpmClient<R>
        + TensorOps<R>
        + ScalarOps<R>
        + ReduceOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + ActivationOps<R>
        + BinaryOps<R>
        + UnaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + TypeConversionOps<R>
        + RandomOps<R>
        + DequantOps<R>
        + 'static,
{
    /// Load the model, the tokenizer and every voice under `voices_dir`,
    /// optionally folding a LoRA adapter into the weights first.
    ///
    /// A voice is any wav, flac, mp3 or ogg file in `voices_dir`; its id is
    /// the file stem. `voices_dir` is optional: `None`, a missing directory,
    /// or an empty directory all load fine with zero encoded voices —
    /// rendering still serves [`ZERO_SHOT_VOICE_ID`], which needs none. A
    /// voice file stemmed `zero-shot` is refused at load: it would silently
    /// shadow the reserved id.
    ///
    /// `audiovae` is the separate `audiovae.pth`/`audiovae.safetensors`. A
    /// GGUF or TCF that embeds the VAE (compressr writes it under `vae.`)
    /// needs none, and ignores one when given; a checkpoint directory, or a
    /// single-file model written without it, needs `Some` — the loader's
    /// error names both ways to supply it. The tokenizer follows
    /// [`VoxCpm2Weights::tokenizer_source`]: embedded in a compressr GGUF,
    /// else `tokenizer.json` inside the checkpoint or beside the file.
    ///
    /// `options.adapter` is applied to the model BEFORE it is wrapped as an
    /// engine — see [`VoxCpm2LoadOptions::adapter`] for the one-adapter-per-
    /// bundle model. `None` loads the base model, unchanged. Either way, the
    /// applied adapter's report (if any) is readable back via
    /// [`Self::adapter`].
    pub fn load(
        weights: &VoxCpm2Weights,
        audiovae: Option<&Path>,
        voices_dir: Option<&Path>,
        device: &R::Device,
        client: Arc<R::Client>,
        options: VoxCpm2LoadOptions,
    ) -> Result<Self> {
        let VoxCpm2LoadOptions {
            dtype,
            vae_decoder_dtype,
            synth,
            adapter,
        } = options;

        let mut model = match weights {
            VoxCpm2Weights::Checkpoint(dir) => {
                VoxCpm2Model::<R>::from_checkpoint(dir, audiovae, device, dtype, vae_decoder_dtype)?
            }
            VoxCpm2Weights::Gguf { path, config } => VoxCpm2Model::<R>::from_gguf(
                path,
                config.as_deref(),
                audiovae,
                device,
                dtype,
                vae_decoder_dtype,
            )?,
            VoxCpm2Weights::Tcf { path, config } => VoxCpm2Model::<R>::from_tcf(
                path,
                config,
                audiovae,
                device,
                dtype,
                vae_decoder_dtype,
            )?,
        };
        let adapter_report = adapter
            .as_deref()
            .map(|path| model.load_lora_adapter(path, device))
            .transpose()?;
        let tokenizer = load_tokenizer(&weights.tokenizer_source()?)?;

        let mut voices = BTreeMap::new();
        if let Some(dir) = voices_dir {
            for (id, path) in list_voice_files(dir)? {
                reject_reserved_voice_id(&id, &path)?;
                let ref_wav = load_reference_16k(&path)?;
                let ref_feat = model.encode_reference(client.as_ref(), &ref_wav)?;
                voices.insert(id, EncodedVoice { ref_feat });
            }
        }

        Ok(Self {
            model,
            client,
            tokenizer,
            voices,
            options: synth,
            adapter: adapter_report,
            render: Mutex::new(()),
        })
    }
}

/// Refuse a voice file whose stem is the reserved zero-shot id. Without this
/// check, a file named `zero-shot.wav` would silently shadow
/// [`ZERO_SHOT_VOICE_ID`] and the render path could never reach the
/// no-reference branch for that id again.
fn reject_reserved_voice_id(id: &str, path: &Path) -> Result<()> {
    if id == ZERO_SHOT_VOICE_ID {
        return Err(Error::ModelError {
            reason: format!(
                "voice file {} has stem {id:?}, which is reserved for zero-shot rendering \
                 ({ZERO_SHOT_VOICE_ID:?}); rename the file",
                path.display()
            ),
        });
    }
    Ok(())
}

/// `(voice id, path)` for every regular file in `dir` with a recognised
/// audio extension, sorted by id. A missing directory is not an error: it
/// loads as zero voices, same as an empty one — a caller relying on
/// zero-shot rendering need not create `voices/` at all.
fn list_voice_files(dir: &Path) -> Result<Vec<(String, PathBuf)>> {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => {
            return Err(Error::ModelError {
                reason: format!("reading voices directory {}: {e}", dir.display()),
            });
        }
    };
    let mut voices = Vec::new();
    for entry in entries {
        let path = entry
            .map_err(|e| Error::ModelError {
                reason: format!("reading voices directory {}: {e}", dir.display()),
            })?
            .path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let is_audio = extension_hint(name)
            .is_some_and(|ext| VOICE_EXTENSIONS.iter().any(|a| ext.eq_ignore_ascii_case(a)));
        if !is_audio {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        voices.push((stem.to_string(), path.clone()));
    }
    voices.sort();
    Ok(voices)
}

/// Decode one reference recording to mono at the encoder's rate.
fn load_reference_16k(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).map_err(|e| Error::ModelError {
        reason: format!("reading voice {}: {e}", path.display()),
    })?;
    let hint = path
        .file_name()
        .and_then(|n| n.to_str())
        .and_then(extension_hint);
    let data = decode_audio(&bytes, hint)?;
    to_mono_at_rate(&data, REF_RATE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voice_listing_takes_audio_files_only_sorted_by_id() {
        let dir = std::env::temp_dir().join("boostr_voxcpm2_engine_voices");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("zara.wav"), b"").unwrap();
        std::fs::write(dir.join("amir.flac"), b"").unwrap();
        std::fs::write(dir.join("notes.txt"), b"").unwrap();
        std::fs::create_dir_all(dir.join("sub.wav")).unwrap();
        let listed = list_voice_files(&dir).unwrap();
        let ids: Vec<&str> = listed.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(ids, ["amir", "zara"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn list_voice_files_on_missing_dir_is_empty_ok() {
        let dir = std::env::temp_dir().join("boostr_voxcpm2_engine_voices_missing");
        let _ = std::fs::remove_dir_all(&dir);
        let listed = list_voice_files(&dir).unwrap();
        assert!(listed.is_empty());
    }

    #[test]
    fn list_voice_files_on_empty_dir_is_empty_ok() {
        let dir = std::env::temp_dir().join("boostr_voxcpm2_engine_voices_empty");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let listed = list_voice_files(&dir).unwrap();
        assert!(listed.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reserved_voice_id_is_refused_at_load() {
        let path = PathBuf::from("/voices/zero-shot.wav");
        let err = reject_reserved_voice_id(ZERO_SHOT_VOICE_ID, &path).unwrap_err();
        assert!(err.to_string().contains(ZERO_SHOT_VOICE_ID));
        assert!(err.to_string().contains("zero-shot.wav"));
    }

    #[test]
    fn non_reserved_voice_id_is_accepted() {
        let path = PathBuf::from("/voices/zara.wav");
        assert!(reject_reserved_voice_id("zara", &path).is_ok());
    }
}
