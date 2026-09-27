//! [`NeuralG2pFallback`]: misaki's [`OovFallback`] hook backed by g2p_lm, a
//! char-level prefix-LM oxidizr trains (`oxidizr/models/g2p-char/g2p-char.yaml`).
//!
//! Sequence: `<bos> <us|gb> graphemes... <sep>`, greedy-decoded until
//! `<eos>`. Runs on boostr's CPU runtime; a fallback call happens for a
//! handful of out-of-vocabulary words per document, not in a hot loop.

use std::path::Path;
use std::sync::Arc;

use boostr::format::SafeTensors;
use boostr::model::{LoadedModel, UniversalConfig};
use boostr::{VarBuilder, VarMap};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

use super::checkpoint;
use super::config::parse_checkpoint_config;
use super::normalize::fold_to_model_chars;
use super::vocab::G2pVocab;
use crate::error::{Error, Result};
use crate::g2p::misaki::{MToken, OovFallback};

/// Generation never runs past this many phoneme tokens: `g2p-char.yaml`'s
/// longest packed sequence is 102 tokens total, so no real word's phoneme
/// span comes close. A safety bound against a model that never emits `eos`.
const MAX_NEW_TOKENS: usize = 64;

/// The loaded checkpoint, shared between a US and a GB [`NeuralG2pFallback`]
/// so the weights and vocab load from disk only once (see
/// [`NeuralG2pFallback::with_british`]).
struct Shared {
    client: CpuClient,
    model: LoadedModel<CpuRuntime>,
    vocab: G2pVocab,
}

/// misaki [`OovFallback`] backed by the g2p_lm char-level model.
pub struct NeuralG2pFallback {
    shared: Arc<Shared>,
    british: bool,
}

impl NeuralG2pFallback {
    /// Loads the model from a directory holding `config.json` (oxidizr's
    /// `BlazrConfig`), one `*.safetensors` checkpoint, and `vocab.json`
    /// (`assets/g2p_lm/build_dataset.py`'s output, copied or symlinked next
    /// to the checkpoint — training writes it to a separate `data/`
    /// directory, not the checkpoint directory).
    ///
    /// `british`: `true` sends every word through the `gb` dialect token,
    /// `false` through `us` — matching [`crate::g2p::misaki::MisakiG2p`]'s
    /// own per-instance dialect. Use [`Self::with_british`] to answer the
    /// other dialect without reloading the checkpoint.
    pub fn from_dir(dir: impl AsRef<Path>, british: bool) -> Result<Self> {
        let dir = dir.as_ref();
        let config_text =
            std::fs::read_to_string(dir.join("config.json")).map_err(|e| Error::ModelError {
                reason: format!("reading {}: {e}", dir.join("config.json").display()),
            })?;
        let vocab_text =
            std::fs::read_to_string(dir.join("vocab.json")).map_err(|e| Error::ModelError {
                reason: format!("reading {}: {e}", dir.join("vocab.json").display()),
            })?;
        let checkpoint_path = checkpoint::locate(dir)?;
        let checkpoint_bytes = std::fs::read(&checkpoint_path).map_err(|e| Error::ModelError {
            reason: format!("reading {}: {e}", checkpoint_path.display()),
        })?;

        let universal: UniversalConfig = parse_checkpoint_config(&config_text)?;
        let vocab = G2pVocab::from_json(&vocab_text)?;
        if universal.vocab_size as u32 != vocab.vocab_size {
            return Err(Error::ModelError {
                reason: format!(
                    "g2p_lm config.json vocab_size {} does not match vocab.json vocab_size {}",
                    universal.vocab_size, vocab.vocab_size
                ),
            });
        }

        let device = CpuDevice::default();
        let client = CpuClient::new(device.clone());
        let mut varmap = varmap_from_bytes(checkpoint_bytes, &device)?;
        let mut vb = VarBuilder::new(&mut varmap, &device);
        let model = LoadedModel::<CpuRuntime>::load(&universal, &mut vb)?;

        Ok(Self {
            shared: Arc::new(Shared {
                client,
                model,
                vocab,
            }),
            british,
        })
    }

    /// A cheap clone (the loaded checkpoint is shared, not reloaded)
    /// answering the other dialect. Used to load one checkpoint and wire
    /// both a US and a GB [`OovFallback`] from it.
    pub fn with_british(&self, british: bool) -> Self {
        Self {
            shared: self.shared.clone(),
            british,
        }
    }

    /// Predicts phonemes for an already-normalized `word` under `dialect`
    /// (`"us"` or `"gb"`), independent of the dialect this instance answers
    /// [`OovFallback`] calls in. `None` when generation never reaches a
    /// phoneme id before `eos` (an empty pronunciation) or emits an id
    /// outside the trained phoneme range (a decoding error). Exposed for
    /// evaluation tooling, which needs both dialects from one loaded model.
    pub fn predict(&self, dialect: &str, word: &str) -> Result<Option<String>> {
        let vocab = &self.shared.vocab;
        let graphemes = match vocab.encode_word(word) {
            Some(g) if !g.is_empty() => g,
            _ => return Ok(None),
        };

        let mut prompt = Vec::with_capacity(graphemes.len() + 3);
        prompt.push(vocab.bos()?);
        prompt.push(vocab.dialect(dialect)?);
        prompt.extend(graphemes);
        prompt.push(vocab.sep()?);

        let opts = boostr::inference::GenerateOptions {
            max_new_tokens: MAX_NEW_TOKENS,
            stop_tokens: vec![vocab.eos],
        };
        let generated = boostr::inference::generate_greedy(
            &self.shared.client,
            &self.shared.model,
            &prompt,
            &opts,
        )?;

        let mut phonemes = String::new();
        for id in generated {
            if id == vocab.eos {
                break;
            }
            if id < vocab.target_base {
                // Below the first phoneme id: a special or grapheme id
                // leaked into generation, a malformed output.
                return Ok(None);
            }
            match vocab.phoneme_symbol(id) {
                Some(sym) => phonemes.push(sym),
                None => return Ok(None),
            }
        }
        if phonemes.is_empty() {
            Ok(None)
        } else {
            Ok(Some(phonemes))
        }
    }
}

/// Builds a `VarMap` from already-decompressed SafeTensors bytes, without a
/// round trip through a temporary file (`VarMap::from_safetensors` only
/// takes a path).
fn varmap_from_bytes(bytes: Vec<u8>, device: &CpuDevice) -> Result<VarMap<CpuRuntime>> {
    let mut st = SafeTensors::from_bytes(bytes)?;
    let mut map = VarMap::new();
    for (name, tensor) in st.load_all::<CpuRuntime>(device)? {
        map.insert(name, tensor);
    }
    Ok(map)
}

impl OovFallback for NeuralG2pFallback {
    /// Mirrors misaki's `EspeakFallback.__call__` (`espeak.py`): phonemizes
    /// `token.text` as written (no lowercasing — the model's grapheme vocab
    /// keeps case for the same reason misaki's own lexicon does), rating 2
    /// like misaki's espeak fallback, `(None, None)` when there is nothing
    /// to say. The `ɾ`/`ʔ` cleanup `EspeakFallback` applies at the end is
    /// not repeated here: `MisakiG2p::phonemize` already runs it over every
    /// resolved token, fallback output included.
    fn phonemize(&self, token: &MToken) -> (Option<String>, Option<u8>) {
        let Some(folded) = fold_to_model_chars(&token.text) else {
            return (None, None);
        };
        let dialect = if self.british { "gb" } else { "us" };
        match self.predict(dialect, &folded) {
            Ok(Some(ps)) => (Some(ps), Some(2)),
            Ok(None) | Err(_) => (None, None),
        }
    }
}
