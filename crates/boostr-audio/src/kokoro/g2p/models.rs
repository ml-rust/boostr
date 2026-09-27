//! Resolves the on-disk models [`KokoroG2p`] needs — the spaCy tagger
//! misaki's pipeline requires, and an optional g2p_lm out-of-vocabulary
//! fallback — from a Kokoro model directory, and loads them.
//!
//! Directory layout, alongside a Kokoro checkpoint's own files:
//!
//! ```text
//! <kokoro dir>/g2p/tagger/   tagger.safetensors, tagger.json
//!                            (assets/spacy_en/convert_tagger.py)
//! <kokoro dir>/g2p/oov/      config.json, one *.safetensors, vocab.json
//!                            (NeuralG2pFallback::from_dir's layout)
//! ```
//!
//! `g2p/oov` is optional: without it, out-of-vocabulary words phonemize to
//! nothing, exactly as [`KokoroG2p::new`] behaves with no fallback attached.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::text_front::KokoroG2p;
use crate::error::{Error, Result};
use crate::g2p::misaki::TextFrontEnd;
use crate::g2p::neural::NeuralG2pFallback;
use crate::g2p::spacy_tagger::EnglishTagger;

/// Directories [`Self::load`] reads: the spaCy tagger (required) and an
/// optional g2p_lm out-of-vocabulary fallback.
#[derive(Debug, Clone)]
pub struct KokoroG2pModels {
    /// Holds `tagger.safetensors` + `tagger.json`.
    pub tagger_dir: PathBuf,
    /// Holds `config.json`, one `*.safetensors` checkpoint and `vocab.json`;
    /// `None` when no fallback is wired.
    pub oov_dir: Option<PathBuf>,
}

impl KokoroG2pModels {
    /// Resolves both directories from a Kokoro model directory's
    /// `g2p/tagger` and `g2p/oov`, with explicit overrides taking
    /// precedence over either. Errs naming the expected tagger path and the
    /// `convert_tagger.py` command when no tagger is found either way; a
    /// missing `oov_dir` (no override, no `g2p/oov` on disk) is not an
    /// error — [`Self::load`] then builds a `KokoroG2p` with no fallback.
    pub fn resolve(
        kokoro_dir: impl AsRef<Path>,
        tagger_dir: Option<PathBuf>,
        oov_dir: Option<PathBuf>,
    ) -> Result<Self> {
        let kokoro_dir = kokoro_dir.as_ref();
        let tagger_dir = tagger_dir.unwrap_or_else(|| kokoro_dir.join("g2p/tagger"));
        if !tagger_dir.join("tagger.safetensors").is_file()
            || !tagger_dir.join("tagger.json").is_file()
        {
            return Err(Error::ModelError {
                reason: format!(
                    "no spaCy tagger at {} (expected tagger.safetensors + tagger.json); run:\n\
                     \n    python3 crates/boostr-audio/assets/spacy_en/convert_tagger.py \\\n        \
                     <en_core_web_sm model dir> {}\n",
                    tagger_dir.display(),
                    tagger_dir.display(),
                ),
            });
        }

        let oov_dir = oov_dir.or_else(|| {
            let dir = kokoro_dir.join("g2p/oov");
            dir.is_dir().then_some(dir)
        });

        Ok(Self {
            tagger_dir,
            oov_dir,
        })
    }

    /// Builds the tagger (and, when `oov_dir` is set, the OOV fallback) and
    /// wires a ready [`KokoroG2p`] for both dialects. The tagger and the
    /// fallback checkpoint each load from disk once and are reused —
    /// [`NeuralG2pFallback::with_british`] answers the second dialect from
    /// the already-loaded checkpoint rather than reloading it.
    pub fn load(&self) -> Result<KokoroG2p> {
        let tagger = EnglishTagger::from_dir(&self.tagger_dir)?;
        let front = Arc::new(TextFrontEnd::new(tagger)?);
        let g2p = KokoroG2p::new(front)?;

        let Some(oov_dir) = &self.oov_dir else {
            return Ok(g2p);
        };
        let us_fallback = NeuralG2pFallback::from_dir(oov_dir, false)?;
        let gb_fallback = us_fallback.with_british(true);
        Ok(g2p.with_fallback(Box::new(us_fallback), Box::new(gb_fallback)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_errs_naming_the_missing_tagger_and_the_convert_command() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let err = KokoroG2pModels::resolve(dir.path(), None, None).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("tagger.safetensors"), "{msg}");
        assert!(msg.contains("convert_tagger.py"), "{msg}");
    }

    #[test]
    fn resolve_defaults_oov_dir_to_none_when_absent() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        std::fs::create_dir_all(dir.path().join("g2p/tagger")).unwrap_or_else(|e| panic!("{e}"));
        std::fs::write(dir.path().join("g2p/tagger/tagger.safetensors"), b"")
            .unwrap_or_else(|e| panic!("{e}"));
        std::fs::write(dir.path().join("g2p/tagger/tagger.json"), b"{}")
            .unwrap_or_else(|e| panic!("{e}"));
        let models = KokoroG2pModels::resolve(dir.path(), None, None)
            .unwrap_or_else(|e| panic!("resolve: {e}"));
        assert!(models.oov_dir.is_none());
    }

    #[test]
    fn resolve_honors_explicit_overrides() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let tagger_dir = dir.path().join("elsewhere/tagger");
        let oov_dir = dir.path().join("elsewhere/oov");
        std::fs::create_dir_all(&tagger_dir).unwrap_or_else(|e| panic!("{e}"));
        std::fs::create_dir_all(&oov_dir).unwrap_or_else(|e| panic!("{e}"));
        std::fs::write(tagger_dir.join("tagger.safetensors"), b"")
            .unwrap_or_else(|e| panic!("{e}"));
        std::fs::write(tagger_dir.join("tagger.json"), b"{}").unwrap_or_else(|e| panic!("{e}"));
        let models =
            KokoroG2pModels::resolve(dir.path(), Some(tagger_dir.clone()), Some(oov_dir.clone()))
                .unwrap_or_else(|e| panic!("resolve: {e}"));
        assert_eq!(models.tagger_dir, tagger_dir);
        assert_eq!(models.oov_dir, Some(oov_dir));
    }
}
