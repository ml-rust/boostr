//! The [`SpeechVocab`] type and its constructors: explicit ids, sequential
//! ids over a reserved region, and the default layout.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

use super::super::codec::CodecVocab;
use super::super::special::{ALL_SPECIAL_TOKENS, DEFAULT_CONTROL_REGION, SpecialToken};

/// The flat id layout holding text, control, and audio tokens.
///
/// **The order of the three regions is load-bearing.** Downstream code depends on
/// it: embedding resize appends rows for control+audio to the pretrained matrix,
/// loss masking selects `>= audio_base` to score audio-only, and sampling
/// constrains logits to a contiguous audio slice. Reordering these regions
/// invalidates every checkpoint trained under the old layout.
///
/// ```text
/// [0, text_vocab)                                        text tokens
/// [text_vocab, text_vocab + control_region_size)         control tokens
/// [audio_base, audio_base + num_codebooks*codebook_size) audio tokens
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeechVocab {
    pub(super) text_vocab_size: usize,
    /// Ids RESERVED for control tokens, chosen by the caller — NOT the count of
    /// tokens currently defined.
    ///
    /// This is what stops a future control token from invalidating a trained
    /// checkpoint. `audio_base` is derived from this reservation, so reserving
    /// more slots than are defined lets a fourteenth (or fiftieth) control token
    /// take an unused reserved id WITHOUT moving a single audio id. Deriving the
    /// boundary from the number of defined specials instead would shift the whole
    /// audio region every time a control token is added.
    pub(super) control_region_size: usize,
    /// Absolute id of each defined control token.
    ///
    /// Explicit and persisted: an id is never recomputed from a position, so the
    /// map read back from a layout file is the same map the checkpoint trained on.
    pub(super) specials: BTreeMap<SpecialToken, u32>,
    pub(super) codec: CodecVocab,
}

impl SpeechVocab {
    /// Build the layout from explicit control ids.
    ///
    /// This is the form to persist and reload. Every id is stated, so the layout
    /// is self-describing and checkable against a checkpoint.
    pub fn new(
        text_vocab_size: usize,
        control_region_size: usize,
        specials: BTreeMap<SpecialToken, u32>,
        codec: CodecVocab,
    ) -> Result<Self> {
        if text_vocab_size == 0 {
            return Err(Error::InvalidArgument {
                arg: "text_vocab_size",
                reason: "text vocabulary must be non-empty".to_string(),
            });
        }
        if control_region_size < specials.len() {
            return Err(Error::InvalidArgument {
                arg: "control_region_size",
                reason: format!(
                    "reserved region of {control_region_size} ids cannot hold {} control tokens",
                    specials.len()
                ),
            });
        }
        let audio_base = text_vocab_size
            .checked_add(control_region_size)
            .ok_or_else(|| Error::ModelError {
                reason: format!(
                    "{text_vocab_size} text + {control_region_size} control ids \
                     overflows the id space"
                ),
            })?;
        for (tok, id) in specials.iter() {
            let id = *id as usize;
            if id < text_vocab_size || id >= audio_base {
                return Err(Error::InvalidArgument {
                    arg: "specials",
                    reason: format!(
                        "control token {tok:?} has id {id} outside the reserved control region \
                         [{text_vocab_size}, {audio_base})"
                    ),
                });
            }
        }
        // Two tokens on one id would share an embedding row and split its gradient.
        for (tok, id) in specials.iter() {
            if specials.iter().any(|(o, oid)| o != tok && oid == id) {
                return Err(Error::InvalidArgument {
                    arg: "specials",
                    reason: format!("control token {tok:?} shares id {id} with another token"),
                });
            }
        }
        let total = audio_base
            .checked_add(codec.total_audio_tokens())
            .filter(|n| *n <= u32::MAX as usize)
            .ok_or_else(|| Error::ModelError {
                reason: format!(
                    "vocabulary of {text_vocab_size} text + {control_region_size} control + {} \
                     audio tokens overflows the u32 id space",
                    codec.total_audio_tokens()
                ),
            })?;
        debug_assert!(total > 0);
        Ok(Self {
            text_vocab_size,
            control_region_size,
            specials,
            codec,
        })
    }

    /// Build the layout by assigning control ids sequentially from `text_vocab_size`.
    ///
    /// Ergonomic path for a fresh layout. The assignment happens ONCE, here; the
    /// resulting ids are STORED and become the source of truth. Reordering the
    /// slice afterwards cannot change a persisted layout's ids.
    pub fn with_sequential_specials(
        text_vocab_size: usize,
        control_region_size: usize,
        specials: &[SpecialToken],
        codec: CodecVocab,
    ) -> Result<Self> {
        let mut map = BTreeMap::new();
        for (i, tok) in specials.iter().enumerate() {
            let id = text_vocab_size
                .checked_add(i)
                .and_then(|id| u32::try_from(id).ok())
                .ok_or_else(|| Error::ModelError {
                    reason: format!("control id for {tok:?} exceeds the u32 id space"),
                })?;
            if map.insert(*tok, id).is_some() {
                return Err(Error::InvalidArgument {
                    arg: "specials",
                    reason: format!("duplicate special token {tok:?} at index {i}"),
                });
            }
        }
        Self::new(text_vocab_size, control_region_size, map, codec)
    }

    /// The default layout: all 13 control tokens in canonical order, in a region
    /// of [`DEFAULT_CONTROL_REGION`] ids.
    ///
    /// Use this to build a fresh model. The remaining 627 reserved ids are the
    /// headroom that keeps a future control token from moving `audio_base`; see
    /// [`DEFAULT_CONTROL_REGION`] for why the region is that size.
    pub fn with_default_specials(text_vocab_size: usize, codec: CodecVocab) -> Result<Self> {
        Self::with_sequential_specials(
            text_vocab_size,
            DEFAULT_CONTROL_REGION,
            &ALL_SPECIAL_TOKENS,
            codec,
        )
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    pub(in super::super) const QWEN3_TEXT_VOCAB: usize = 151_936;
    pub(in super::super) const NEUCODEC_SIZE: usize = 65_536;

    /// Small region used by the toy layouts below. Must hold all 13 control tokens.
    pub(in super::super) const SMALL_REGION: usize = 16;

    pub(in super::super) fn all_specials() -> Vec<SpecialToken> {
        ALL_SPECIAL_TOKENS.to_vec()
    }

    /// The first four delimiters. Used where a layout needs fewer tokens than slots.
    pub(in super::super) fn core_specials() -> Vec<SpecialToken> {
        vec![
            SpecialToken::SpeechText,
            SpecialToken::SpeechTextEnd,
            SpecialToken::Speech,
            SpecialToken::SpeechEnd,
        ]
    }

    /// NeuCodec on Qwen3: 1 codebook x 65_536, one code per frame.
    pub(in super::super) fn neucodec_vocab() -> SpeechVocab {
        let codec = CodecVocab::new(1, NEUCODEC_SIZE).expect("valid codec");
        SpeechVocab::with_default_specials(QWEN3_TEXT_VOCAB, codec).expect("valid vocab")
    }

    /// Interleaved hierarchy: codebook 0 once, 1 twice, 2 four times per frame.
    fn snac_interleaved_vocab() -> SpeechVocab {
        let codec = CodecVocab::with_frame_layout(4096, vec![1, 2, 4]).expect("valid codec");
        SpeechVocab::with_sequential_specials(1000, SMALL_REGION, &all_specials(), codec)
            .expect("valid vocab")
    }

    #[test]
    fn neucodec_layout_has_concrete_ids() {
        let v = neucodec_vocab();
        assert_eq!(v.text_vocab_size(), 151_936);
        assert_eq!(v.control_region_size(), DEFAULT_CONTROL_REGION);
        assert_eq!(v.control_region_size(), 640);
        assert_eq!(v.num_specials(), 13);
        assert_eq!(v.audio_base(), 151_936 + 640);
        assert_eq!(v.audio_base(), 152_576);
        assert_eq!(v.total_size(), 152_576 + 65_536);
        assert_eq!(v.total_size(), 218_112);

        assert_eq!(v.audio_token(0, 0).expect("first audio id"), 152_576);
        assert_eq!(v.audio_token(0, 65_535).expect("last audio id"), 218_111);
        assert_eq!(v.codec().codes_per_frame(), 1);
        assert!(v.matches_embedding_rows(218_112));
        assert!(!v.matches_embedding_rows(218_111));
        assert!(!v.matches_embedding_rows(0));
    }

    /// The embedding and output-projection row count must tile at 512.
    ///
    /// This is the constraint that picks 640 over 13: 151_936 % 512 == 384 and
    /// 65_536 % 512 == 0, so only a control region congruent to 128 mod 512 aligns.
    #[test]
    fn default_region_keeps_total_size_512_aligned() {
        let v = neucodec_vocab();
        assert_eq!(v.total_size() % 512, 0);
        assert_eq!(DEFAULT_CONTROL_REGION % 512, 128);
        assert_eq!(QWEN3_TEXT_VOCAB % 512, 384);
        assert_eq!(NEUCODEC_SIZE % 512, 0);
        // 13 defined leaves 627 reserved and free for future control tokens.
        assert_eq!(DEFAULT_CONTROL_REGION - v.num_specials(), 627);
    }

    #[test]
    fn explicit_ids_are_stored_not_recomputed_from_order() {
        let codec = CodecVocab::new(1, 8).expect("valid codec");
        // Ids deliberately NOT in enum order: the map is the source of truth.
        let mut specials = BTreeMap::new();
        specials.insert(SpecialToken::SpeechText, 13u32);
        specials.insert(SpecialToken::SpeechEnd, 10u32);
        let v = SpeechVocab::new(10, 4, specials, codec).expect("valid vocab");
        assert_eq!(v.special_id(SpecialToken::SpeechText), Some(13));
        assert_eq!(v.special_id(SpecialToken::SpeechEnd), Some(10));
        assert_eq!(v.audio_base(), 14);
    }

    /// THE LANDMINE TEST.
    ///
    /// Reserving a control region LARGER than the number of defined tokens must make
    /// the audio region immovable. A control token added later takes a spare reserved
    /// id, and every audio id stays exactly where the trained checkpoint put it.
    ///
    /// The layout below defines four of the thirteen tokens; the fifth taking a
    /// previously unused reserved slot stands in for the fourteenth token a future
    /// task would add to the full default layout.
    #[test]
    fn adding_a_special_into_a_reserved_region_does_not_move_audio_base() {
        let codec = || CodecVocab::new(1, 8).expect("valid codec");
        let region = 8;

        let before = SpeechVocab::with_sequential_specials(100, region, &core_specials(), codec())
            .expect("valid vocab");
        assert_eq!(before.audio_base(), 108);
        assert_eq!(before.total_size(), 116);
        let audio_before = before.audio_token(0, 3).expect("audio id");

        // A control token now sits at reserved id 107, which nothing claimed before.
        let mut grown = BTreeMap::new();
        for (i, tok) in core_specials().iter().enumerate() {
            grown.insert(*tok, 100 + i as u32);
        }
        grown.insert(SpecialToken::Speaker, 107);
        let after = SpeechVocab::new(100, region, grown, codec()).expect("valid vocab");

        assert_eq!(after.audio_base(), before.audio_base());
        assert_eq!(after.total_size(), before.total_size());
        assert_eq!(after.audio_token(0, 3).expect("audio id"), audio_before);
        assert_eq!(after.special_id(SpecialToken::Speaker), Some(107));
        // The newly claimed row also drops out of the suppression list.
        assert!(before.sampling_forbidden_ids().contains(&107));
        assert!(!after.sampling_forbidden_ids().contains(&107));

        // Contrast: sizing the region to the token count instead ties audio_base to
        // how many controls exist, so every added token would shift all audio ids.
        let sized_to_count =
            SpeechVocab::with_sequential_specials(100, 4, &core_specials(), codec())
                .expect("valid vocab");
        assert_eq!(sized_to_count.audio_base(), 104);
        assert_ne!(sized_to_count.audio_base(), before.audio_base());
    }

    #[test]
    fn special_id_inside_the_text_region_is_rejected() {
        let codec = CodecVocab::new(1, 8).expect("valid codec");
        let mut specials = BTreeMap::new();
        specials.insert(SpecialToken::SpeechEnd, 9u32);
        let err = SpeechVocab::new(10, 4, specials, codec).expect_err("must reject text-region id");
        let msg = err.to_string();
        assert!(msg.contains("SpeechEnd"), "{msg}");
        assert!(msg.contains('9'), "{msg}");
    }

    #[test]
    fn special_id_at_or_past_audio_base_is_rejected() {
        let codec = CodecVocab::new(1, 8).expect("valid codec");
        for id in [14u32, 15, 100] {
            let mut specials = BTreeMap::new();
            specials.insert(SpecialToken::Speech, id);
            let err = SpeechVocab::new(10, 4, specials, codec.clone())
                .expect_err("must reject audio-region id");
            let msg = err.to_string();
            assert!(msg.contains("Speech"), "{msg}");
            assert!(msg.contains(&id.to_string()), "{msg}");
        }
        // The last reserved id is still legal.
        let mut ok = BTreeMap::new();
        ok.insert(SpecialToken::Speech, 13u32);
        assert!(SpeechVocab::new(10, 4, ok, codec).is_ok());
    }

    #[test]
    fn duplicate_special_ids_are_rejected() {
        let codec = CodecVocab::new(1, 8).expect("valid codec");
        let mut specials = BTreeMap::new();
        specials.insert(SpecialToken::Speech, 11u32);
        specials.insert(SpecialToken::SpeechEnd, 11u32);
        let err = SpeechVocab::new(10, 4, specials, codec).expect_err("must reject duplicate id");
        assert!(err.to_string().contains("shares id 11"), "{err}");
    }

    #[test]
    fn control_region_smaller_than_the_special_count_is_rejected() {
        let codec = CodecVocab::new(1, 8).expect("valid codec");
        let err = SpeechVocab::with_sequential_specials(10, 12, &all_specials(), codec.clone())
            .expect_err("must reject undersized region");
        assert!(err.to_string().contains("cannot hold 13"), "{err}");
        assert!(SpeechVocab::with_sequential_specials(10, 0, &[], codec).is_ok());
    }

    #[test]
    fn serde_round_trip_reproduces_an_identical_layout() {
        let v = neucodec_vocab();
        let json = serde_json::to_string(&v).expect("serialize layout");
        // Control tokens persist by NAME, never by numeric index.
        assert!(json.contains("\"SpeechPad\""), "{json}");
        assert!(json.contains("\"VoiceRefEnd\""), "{json}");
        let back: SpeechVocab = serde_json::from_str(&json).expect("deserialize layout");
        assert_eq!(back, v);
        assert_eq!(back.audio_base(), v.audio_base());
        assert_eq!(
            back.special_id(SpecialToken::SpeechEnd),
            v.special_id(SpecialToken::SpeechEnd)
        );
        assert_eq!(back.sampling_forbidden_ids(), v.sampling_forbidden_ids());

        let interleaved = snac_interleaved_vocab();
        let json = serde_json::to_string(&interleaved).expect("serialize layout");
        assert_eq!(
            serde_json::from_str::<SpeechVocab>(&json).expect("deserialize layout"),
            interleaved
        );

        // A stale or hand-edited layout carrying an unknown field fails loudly.
        let stale = serde_json::to_string(&v)
            .expect("serialize layout")
            .replacen('{', "{\"legacy_field\":1,", 1);
        assert!(serde_json::from_str::<SpeechVocab>(&stale).is_err());
    }

    #[test]
    fn speech_vocab_construction_validates_inputs() {
        let codec = CodecVocab::new(1, 8).expect("valid codec");
        assert!(SpeechVocab::with_default_specials(0, codec.clone()).is_err());
        assert!(
            SpeechVocab::with_sequential_specials(
                10,
                SMALL_REGION,
                &[SpecialToken::SpeechEnd, SpecialToken::SpeechEnd],
                codec.clone(),
            )
            .is_err()
        );
        // Empty special map is legal: some setups reuse the base tokenizer's ids.
        let v = SpeechVocab::new(10, 0, BTreeMap::new(), codec).expect("valid vocab");
        assert_eq!(v.audio_base(), 10);
        assert_eq!(v.special_id(SpecialToken::SpeechEnd), None);
        assert!(v.sampling_forbidden_ids().is_empty());

        // Total that overflows u32 is rejected rather than wrapping.
        let big = CodecVocab::new(1, u32::MAX as usize).expect("valid codec");
        assert!(SpeechVocab::new(1, 0, BTreeMap::new(), big).is_err());
    }
}
