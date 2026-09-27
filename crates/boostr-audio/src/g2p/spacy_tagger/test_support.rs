//! Test-only fixture resolution shared by every inline unit test that needs
//! the packed spaCy tagger directory (`tagger.rs`, `misaki/g2p.rs`,
//! `kokoro/g2p/text_front.rs`) — one copy of the env-var fallback instead of
//! one per test module.

use std::path::PathBuf;

/// Resolves the packed tagger directory (`SPACY_EN_TAGGER_DIR`, else
/// `$BOOSTR_MODELS_DIR/spacy-en-tagger`), `None` when unavailable — callers
/// then skip, since no tagger is checked into this repo.
pub(crate) fn tagger_dir() -> Option<PathBuf> {
    let dir = std::env::var("SPACY_EN_TAGGER_DIR")
        .ok()
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var("BOOSTR_MODELS_DIR")
                .ok()
                .filter(|v| !v.is_empty())
                .map(|root| PathBuf::from(root).join("spacy-en-tagger"))
        })?;
    dir.is_dir().then_some(dir)
}
