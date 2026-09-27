//! [`KokoroG2p`]: G2P selection by dialect plus reference Kokoro's chunking,
//! ready for [`crate::kokoro::engine::KokoroEngine::synthesize`] to map each
//! chunk's phonemes to input ids and hand them to the model.

use super::chunk::{en_tokenize, split_segments};
use super::dialect::Dialect;
use crate::error::{Error, Result};
use crate::g2p::misaki::{KOKORO_UNK, MisakiG2p, OovFallback};

pub use super::chunk::TextChunk;

/// Kokoro's G2P + chunking front end.
///
/// Builds both dialects' [`MisakiG2p`] up front — a voice's prefix decides
/// which one a given [`Self::chunks`] call reads, so there is no per-call
/// reconstruction cost. No OOV fallback is attached unless
/// [`Self::with_fallback`] supplies one; an unresolved word then contributes
/// no phonemes to its chunk (misaki's `unk` for Kokoro is the empty string).
pub struct KokoroG2p {
    us: MisakiG2p,
    gb: MisakiG2p,
}

impl KokoroG2p {
    /// Builds both dialects' G2P, unk = the empty string, no fallback.
    pub fn new() -> Result<Self> {
        Ok(Self {
            us: MisakiG2p::new(false)?.with_unk(KOKORO_UNK),
            gb: MisakiG2p::new(true)?.with_unk(KOKORO_UNK),
        })
    }

    /// Attaches the out-of-vocabulary fallback for each dialect. The hook
    /// the neural G2P fallback (a separate unit) plugs in through; this
    /// crate ships no default.
    pub fn with_fallback(
        mut self,
        us_fallback: Box<dyn OovFallback + Send + Sync>,
        gb_fallback: Box<dyn OovFallback + Send + Sync>,
    ) -> Self {
        self.us = self.us.with_fallback(us_fallback);
        self.gb = self.gb.with_fallback(gb_fallback);
        self
    }

    /// Phonemizes `text` and cuts it into chunks the model can consume in
    /// one call each: split on blank lines, G2P per segment, then reference
    /// Kokoro's phoneme-budget chunking. Chunks with no phonemes (an
    /// all-blank segment, or a lone unresolved word) are dropped, mirroring
    /// `KPipeline.__call__`'s `if not ps: continue`.
    ///
    /// Errs when every segment produces no phonemes at all.
    pub fn chunks(&self, text: &str, dialect: Dialect) -> Result<Vec<TextChunk>> {
        let g2p = match dialect {
            Dialect::Us => &self.us,
            Dialect::Gb => &self.gb,
        };
        let mut out = Vec::new();
        for segment in split_segments(text) {
            if segment.trim().is_empty() {
                continue;
            }
            let phonemized = g2p.phonemize_text(segment)?;
            out.extend(
                en_tokenize(phonemized.tokens)
                    .into_iter()
                    .filter(|c| !c.phonemes.is_empty()),
            );
        }
        if out.is_empty() {
            return Err(Error::InvalidArgument {
                arg: "text",
                reason: "no phonemizable text — every segment produced empty phonemes".into(),
            });
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_splits_on_blank_lines_and_phonemizes() {
        let g2p = KokoroG2p::new().unwrap_or_else(|e| panic!("KokoroG2p::new: {e}"));
        let chunks = g2p
            .chunks("Hello world.\n\nGoodbye.", Dialect::Us)
            .unwrap_or_else(|e| panic!("chunks: {e}"));
        assert_eq!(chunks.len(), 2);
        assert!(!chunks[0].phonemes.is_empty());
        assert!(!chunks[1].phonemes.is_empty());
    }

    #[test]
    fn chunks_uses_gb_dialect_lexicon() {
        // "schedule" resolves differently between US and GB lexicons; the
        // point here is only that the GB path runs the GB lexicon, not the
        // exact phoneme string.
        let g2p = KokoroG2p::new().unwrap_or_else(|e| panic!("KokoroG2p::new: {e}"));
        let us = g2p
            .chunks("schedule", Dialect::Us)
            .unwrap_or_else(|e| panic!("chunks: {e}"));
        let gb = g2p
            .chunks("schedule", Dialect::Gb)
            .unwrap_or_else(|e| panic!("chunks: {e}"));
        assert!(!us[0].phonemes.is_empty());
        assert!(!gb[0].phonemes.is_empty());
    }

    #[test]
    fn chunks_errs_on_all_blank_text() {
        let g2p = KokoroG2p::new().unwrap_or_else(|e| panic!("KokoroG2p::new: {e}"));
        assert!(g2p.chunks("   \n \n  ", Dialect::Us).is_err());
    }
}
