//! Checkpoint file parsing for [`super::tokenizer::VitsTokenizer::from_dir`].
//!
//! Reads `vocab.json`, `tokenizer_config.json`, and an optional
//! `special_tokens_map.json`, and refuses a checkpoint that needs
//! phonemization or uroman romanization.

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;
use serde::de::DeserializeOwned;

use crate::error::{Error, Result};

/// Reads `path` and parses it as JSON into `T`, wrapping either failure in
/// [`Error::ModelError`] with the path attached.
fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let bytes = std::fs::read(path).map_err(|e| Error::ModelError {
        reason: format!("reading {}: {e}", path.display()),
    })?;
    serde_json::from_slice(&bytes).map_err(|e| Error::ModelError {
        reason: format!("parsing {}: {e}", path.display()),
    })
}

fn default_true() -> bool {
    true
}

/// `tokenizer_config.json` fields this tokenizer reads. Unlisted fields are
/// ignored by serde's default behavior for structs read via `Deserialize`
/// with `#[serde(default)]` on every field (an unknown key is simply left
/// out of the struct, not rejected).
#[derive(Debug, Deserialize)]
struct TokenizerConfigFile {
    #[serde(default = "default_true")]
    add_blank: bool,
    #[serde(default = "default_true")]
    normalize: bool,
    #[serde(default = "default_true")]
    phonemize: bool,
    #[serde(default)]
    is_uroman: bool,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    pad_token: Option<String>,
    #[serde(default)]
    unk_token: Option<String>,
}

/// `special_tokens_map.json`, when the checkpoint carries one. A value can be
/// a plain string or an object with a `content` field; both forms appear
/// across HF checkpoints.
#[derive(Debug, Default, Deserialize)]
struct SpecialTokensMapFile {
    #[serde(default)]
    pad_token: Option<serde_json::Value>,
    #[serde(default)]
    unk_token: Option<serde_json::Value>,
}

fn special_token_string(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Object(map) => map.get("content")?.as_str().map(str::to_string),
        _ => None,
    }
}

/// Parsed checkpoint contents needed to build a `VitsTokenizer`.
pub(super) struct Loaded {
    pub encoder: HashMap<String, u32>,
    pub pad_token: String,
    pub unk_token: String,
    pub language: Option<String>,
    pub add_blank: bool,
    pub normalize: bool,
}

/// Read and validate the checkpoint directory. Errors when the checkpoint
/// needs phonemization (`phonemize: true`) or uroman romanization
/// (`is_uroman: true`): this crate has no G2P for VITS, and uroman
/// romanization is a queued unit.
pub(super) fn load(dir: &Path) -> Result<Loaded> {
    let vocab_path = dir.join("vocab.json");
    let raw_vocab: HashMap<String, serde_json::Value> = read_json(&vocab_path)?;
    let mut encoder = HashMap::with_capacity(raw_vocab.len());
    for (token, id) in raw_vocab {
        let id = id.as_u64().ok_or_else(|| Error::ModelError {
            reason: format!(
                "{}: token {token:?} has a non-integer id",
                vocab_path.display()
            ),
        })?;
        let id = u32::try_from(id).map_err(|_| Error::ModelError {
            reason: format!(
                "{}: token {token:?} id {id} overflows u32",
                vocab_path.display()
            ),
        })?;
        encoder.insert(token, id);
    }

    let config_path = dir.join("tokenizer_config.json");
    let config: TokenizerConfigFile = read_json(&config_path)?;

    if config.phonemize {
        return Err(Error::ModelError {
            reason: format!(
                "{}: tokenizer_config.json sets phonemize: true, which needs a \
                 grapheme-to-phoneme backend this crate does not provide for VITS",
                dir.display()
            ),
        });
    }
    if config.is_uroman {
        return Err(Error::ModelError {
            reason: format!(
                "{}: tokenizer_config.json sets is_uroman: true; uroman romanization is a \
                 queued unit, not yet implemented",
                dir.display()
            ),
        });
    }

    let special_path = dir.join("special_tokens_map.json");
    let special = if special_path.exists() {
        read_json::<SpecialTokensMapFile>(&special_path)?
    } else {
        SpecialTokensMapFile::default()
    };

    let pad_token = special
        .pad_token
        .as_ref()
        .and_then(special_token_string)
        .or(config.pad_token)
        .unwrap_or_else(|| "<pad>".to_string());
    let unk_token = special
        .unk_token
        .as_ref()
        .and_then(special_token_string)
        .or(config.unk_token)
        .unwrap_or_else(|| "<unk>".to_string());

    Ok(Loaded {
        encoder,
        pad_token,
        unk_token,
        language: config.language,
        add_blank: config.add_blank,
        normalize: config.normalize,
    })
}
