//! Where each id falls: the region accessors, `is_text` / `is_special` /
//! `is_audio`, the sampling suppression list, and audio id arithmetic.

use crate::error::{Error, Result};

use super::super::codec::CodecVocab;
use super::super::special::SpecialToken;
use super::build::SpeechVocab;

impl SpeechVocab {
    /// The codec this layout was built for.
    pub fn codec(&self) -> &CodecVocab {
        &self.codec
    }

    /// Size of the text region. Text ids are `[0, text_vocab_size)`.
    ///
    /// Text ids keep their ORIGINAL values from the base tokenizer. The pretrained
    /// embedding matrix is indexed by id, so shifting text ids would point every
    /// token at another token's learned row and destroy the pretrained model.
    /// New regions are therefore only ever appended after it.
    pub fn text_vocab_size(&self) -> usize {
        self.text_vocab_size
    }

    /// Ids reserved for control tokens, defined or not. See the field docs.
    pub fn control_region_size(&self) -> usize {
        self.control_region_size
    }

    /// Number of control tokens actually defined, at most
    /// [`control_region_size`](Self::control_region_size).
    pub fn num_specials(&self) -> usize {
        self.specials.len()
    }

    /// First audio id. Audio occupies `[audio_base, total_size)`.
    ///
    /// Derived from the RESERVED region, not from the defined tokens, so adding a
    /// control token leaves every audio id where it is.
    pub fn audio_base(&self) -> usize {
        self.text_vocab_size + self.control_region_size
    }

    /// Total ids, i.e. the embedding/output-projection row count.
    pub fn total_size(&self) -> usize {
        self.audio_base() + self.codec.total_audio_tokens()
    }

    /// True if a checkpoint with `rows` embedding rows matches this layout.
    ///
    /// A loader MUST check this before using a layout. A mismatch means the
    /// checkpoint and the layout disagree about the id space, so every id past the
    /// first divergence points at the wrong embedding row; loading must abort
    /// rather than proceed with silently wrong tokens.
    pub fn matches_embedding_rows(&self, rows: usize) -> bool {
        rows == self.total_size()
    }

    /// Id of a control token, or `None` if it is not defined in this layout.
    pub fn special_id(&self, tok: SpecialToken) -> Option<u32> {
        self.specials.get(&tok).copied()
    }

    /// True if `id` sits in the reserved control region but no token claims it.
    ///
    /// These rows exist in the embedding matrix and are trained on nothing.
    pub fn is_reserved_unused(&self, id: u32) -> bool {
        self.is_special(id) && !self.specials.values().any(|claimed| *claimed == id)
    }

    /// Ids that MUST be suppressed in the logits before sampling.
    ///
    /// This is a correctness requirement, not hygiene. Under
    /// `tie_word_embeddings: true` the embedding matrix IS the output projection,
    /// so every reserved-but-undefined row is also an output logit — and those
    /// rows are never a training target, so nothing ever pushes their logit down.
    /// Untrained rows of a tied matrix are exactly the "glitch token" class
    /// described in *Fishing for Magikarp* (Land & Bartolo, EMNLP 2024): they are
    /// occasionally sampled with high probability, and the id they emit decodes to
    /// nothing at all. Suppressing them is the only reliable fix at inference time.
    ///
    /// Returned ids, ascending:
    /// - every id in the control region with no defined [`SpecialToken`], and
    /// - [`SpecialToken::SpeechPad`], which is padding for packed batches and is
    ///   defined but must never be generated.
    ///
    /// Text ids and audio ids are NOT here: both are trained targets, and
    /// restricting sampling to one of those regions is the caller's decision.
    pub fn sampling_forbidden_ids(&self) -> Vec<u32> {
        let mut out = Vec::new();
        for id in self.text_vocab_size..self.audio_base() {
            let Ok(id) = u32::try_from(id) else {
                break;
            };
            if self.is_reserved_unused(id) {
                out.push(id);
            }
        }
        if let Some(pad) = self.special_id(SpecialToken::SpeechPad) {
            match out.binary_search(&pad) {
                Ok(_) => {}
                Err(at) => out.insert(at, pad),
            }
        }
        out
    }

    /// Flat id for `code` in `codebook`, codebook-major.
    ///
    /// `audio_base + codebook * codebook_size + code`. Codebook-major keeps each
    /// codebook's ids contiguous, so a residual codec can mask logits to the one
    /// codebook legal at the current frame position with a single range.
    pub fn audio_token(&self, codebook: usize, code: usize) -> Result<u32> {
        if codebook >= self.codec.num_codebooks() {
            return Err(Error::InvalidArgument {
                arg: "codebook",
                reason: format!(
                    "codebook {codebook} out of range for {} codebooks",
                    self.codec.num_codebooks()
                ),
            });
        }
        if code >= self.codec.codebook_size() {
            return Err(Error::InvalidArgument {
                arg: "code",
                reason: format!(
                    "code {code} out of range for codebook size {}",
                    self.codec.codebook_size()
                ),
            });
        }
        let id = self.audio_base() + codebook * self.codec.codebook_size() + code;
        u32::try_from(id).map_err(|_| Error::ModelError {
            reason: format!("audio id {id} exceeds the u32 id space"),
        })
    }

    /// Inverse of [`audio_token`](Self::audio_token); `None` for non-audio ids.
    pub fn decode_audio_token(&self, id: u32) -> Option<(usize, usize)> {
        let id = id as usize;
        let offset = id.checked_sub(self.audio_base())?;
        if offset >= self.codec.total_audio_tokens() {
            return None;
        }
        Some((
            offset / self.codec.codebook_size(),
            offset % self.codec.codebook_size(),
        ))
    }

    /// True if `id` is a text token.
    pub fn is_text(&self, id: u32) -> bool {
        (id as usize) < self.text_vocab_size
    }

    /// True if `id` falls in the reserved control region.
    ///
    /// The whole reserved region answers true, including ids no token claims yet.
    /// Those rows exist in the embedding matrix and are neither text nor audio.
    pub fn is_special(&self, id: u32) -> bool {
        let id = id as usize;
        id >= self.text_vocab_size && id < self.audio_base()
    }

    /// True if `id` is an audio token.
    pub fn is_audio(&self, id: u32) -> bool {
        let id = id as usize;
        id >= self.audio_base() && id < self.total_size()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::super::build::tests::{
        QWEN3_TEXT_VOCAB, SMALL_REGION, all_specials, core_specials, neucodec_vocab,
    };
    use super::*;
    use crate::model::speech_lm::special::ALL_SPECIAL_TOKENS;

    /// Flat residual codec on a toy text vocabulary: 3 codebooks x 4096.
    fn snac_flat_vocab() -> SpeechVocab {
        let codec = CodecVocab::new(3, 4096).expect("valid codec");
        SpeechVocab::with_sequential_specials(1000, 16, &ALL_SPECIAL_TOKENS, codec)
            .expect("valid vocab")
    }

    #[test]
    fn all_thirteen_ids_are_distinct_and_inside_the_control_region() {
        let v = neucodec_vocab();
        let mut seen = BTreeMap::new();
        for (i, tok) in ALL_SPECIAL_TOKENS.iter().enumerate() {
            let id = v.special_id(*tok).expect("defined control token");
            // Canonical order is assigned sequentially from text_vocab_size.
            assert_eq!(id as usize, QWEN3_TEXT_VOCAB + i, "{tok:?}");
            assert!(v.is_special(id), "{tok:?} id {id} outside control region");
            assert!(!v.is_text(id));
            assert!(!v.is_audio(id));
            assert!(
                seen.insert(id, *tok).is_none(),
                "{tok:?} duplicates id {id}"
            );
        }
        assert_eq!(seen.len(), 13);
    }

    /// Untrained rows of a TIED embedding matrix are also output logits, and are the
    /// "glitch token" class from *Fishing for Magikarp*. They must be suppressed.
    #[test]
    fn sampling_forbidden_ids_covers_unused_rows_and_the_pad() {
        let v = neucodec_vocab();
        let forbidden = v.sampling_forbidden_ids();

        // 627 unused reserved ids, plus the defined-but-unsampleable pad.
        assert_eq!(forbidden.len(), 628);
        assert!(
            forbidden.windows(2).all(|w| w[0] < w[1]),
            "must be ascending"
        );

        let pad = v.special_id(SpecialToken::SpeechPad).expect("pad defined");
        assert!(forbidden.contains(&pad));

        // The 12 sampleable control tokens must NOT be suppressed.
        for tok in ALL_SPECIAL_TOKENS {
            let id = v.special_id(tok).expect("defined");
            assert_eq!(
                forbidden.contains(&id),
                tok == SpecialToken::SpeechPad,
                "{tok:?}"
            );
        }

        // Every reserved id past the 13 defined ones is suppressed, and only those.
        for id in v.text_vocab_size() as u32..v.audio_base() as u32 {
            assert_eq!(
                forbidden.contains(&id),
                v.is_reserved_unused(id) || id == pad
            );
        }
        // Text and audio are trained targets and are never in the list.
        assert!(!forbidden.contains(&0));
        assert!(!forbidden.contains(&(v.audio_base() as u32)));
        assert!(!forbidden.contains(&(v.total_size() as u32 - 1)));
    }

    #[test]
    fn is_reserved_unused_is_false_for_claimed_and_non_control_ids() {
        let codec = CodecVocab::new(1, 8).expect("valid codec");
        let v = SpeechVocab::with_sequential_specials(10, 6, &core_specials(), codec)
            .expect("valid vocab");
        for id in 10..14u32 {
            assert!(!v.is_reserved_unused(id), "id {id} is claimed");
        }
        assert!(v.is_reserved_unused(14));
        assert!(v.is_reserved_unused(15));
        assert!(!v.is_reserved_unused(9));
        assert!(!v.is_reserved_unused(16));
        assert!(!v.is_reserved_unused(u32::MAX));
        assert_eq!(v.sampling_forbidden_ids(), vec![14, 15]);
    }

    #[test]
    fn missing_special_has_no_id() {
        let codec = CodecVocab::new(1, 8).expect("valid codec");
        let v = SpeechVocab::with_sequential_specials(
            10,
            2,
            &[SpecialToken::Speech, SpecialToken::SpeechEnd],
            codec,
        )
        .expect("valid vocab");
        assert_eq!(v.special_id(SpecialToken::Speech), Some(10));
        assert_eq!(v.special_id(SpecialToken::SpeechEnd), Some(11));
        assert_eq!(v.special_id(SpecialToken::SpeechText), None);
        assert_eq!(v.audio_base(), 12);
        // Nothing to suppress: no unused reserved rows, and no pad defined.
        assert!(v.sampling_forbidden_ids().is_empty());
    }

    #[test]
    fn classification_is_exhaustive_and_disjoint() {
        let codec = CodecVocab::new(2, 5).expect("valid codec");
        let v = SpeechVocab::with_sequential_specials(7, SMALL_REGION, &all_specials(), codec)
            .expect("valid vocab");
        assert_eq!(v.total_size(), 7 + 16 + 10);

        for id in 0..v.total_size() as u32 {
            let flags = [v.is_text(id), v.is_special(id), v.is_audio(id)];
            let set = flags.iter().filter(|f| **f).count();
            assert_eq!(set, 1, "id {id} classified {flags:?}");
            assert_eq!(v.is_audio(id), v.decode_audio_token(id).is_some());
        }

        // Past the end nothing classifies.
        for id in [v.total_size() as u32, v.total_size() as u32 + 1, u32::MAX] {
            assert!(!v.is_text(id));
            assert!(!v.is_special(id));
            assert!(!v.is_audio(id));
        }
    }

    #[test]
    fn reserved_but_undefined_ids_classify_as_control() {
        // A region of 20 with 13 defined tokens leaves 7 spare rows: control, not audio.
        let codec = CodecVocab::new(1, 8).expect("valid codec");
        let v = SpeechVocab::with_sequential_specials(10, 20, &all_specials(), codec)
            .expect("valid vocab");
        for id in 10..30u32 {
            assert!(v.is_special(id), "id {id} must be control");
            assert!(!v.is_audio(id));
        }
        assert_eq!(v.special_id(SpecialToken::SpeechPad), Some(22));
        assert_eq!(v.audio_base(), 30);
        assert_eq!(
            v.sampling_forbidden_ids(),
            vec![22, 23, 24, 25, 26, 27, 28, 29]
        );
    }

    #[test]
    fn audio_token_round_trips_at_boundaries() {
        let v = snac_flat_vocab();
        let size = v.codec().codebook_size();
        for codebook in 0..v.codec().num_codebooks() {
            for code in [0, 1, size / 2, size - 2, size - 1] {
                let id = v.audio_token(codebook, code).expect("in-range audio token");
                assert_eq!(v.decode_audio_token(id), Some((codebook, code)));
                assert!(v.is_audio(id));
            }
        }
    }

    #[test]
    fn audio_token_boundaries_do_not_bleed_between_codebooks() {
        let v = snac_flat_vocab();
        let size = v.codec().codebook_size();
        // Last code of codebook c and first of c+1 must be adjacent but distinct.
        for c in 0..v.codec().num_codebooks() - 1 {
            let last = v.audio_token(c, size - 1).expect("last code");
            let next = v.audio_token(c + 1, 0).expect("first code of next");
            assert_eq!(next, last + 1);
            assert_eq!(v.decode_audio_token(last), Some((c, size - 1)));
            assert_eq!(v.decode_audio_token(next), Some((c + 1, 0)));
        }
    }

    #[test]
    fn ids_just_outside_the_audio_span_are_not_audio() {
        let v = snac_flat_vocab();
        let base = v.audio_base() as u32;
        let last = v.total_size() as u32 - 1;
        assert_eq!(v.decode_audio_token(base - 1), None);
        assert!(v.decode_audio_token(base).is_some());
        assert!(v.decode_audio_token(last).is_some());
        assert_eq!(v.decode_audio_token(last + 1), None);
        assert_eq!(v.decode_audio_token(u32::MAX), None);
        assert_eq!(v.decode_audio_token(0), None);
    }

    #[test]
    fn out_of_range_audio_token_errors_without_panic() {
        let v = snac_flat_vocab();
        assert!(v.audio_token(3, 0).is_err());
        assert!(v.audio_token(usize::MAX, 0).is_err());
        assert!(v.audio_token(0, 4096).is_err());
        assert!(v.audio_token(0, usize::MAX).is_err());
        assert!(v.audio_token(2, 4095).is_ok());
    }
}
