//! Control tokens of a speech LM, and the size of the region reserved for them.
//!
//! # Why these thirteen, and why nothing else
//!
//! The set below is not a guess. It is what survives a survey of the tokenizer
//! files actually shipped by production speech LMs: Scicom Multilingual-Expressive-TTS
//! (our base), Orpheus, CosyVoice2, Higgs Audio v2, Step-Audio 2, Fish-Speech,
//! Moshi, GLM-4-Voice, Qwen2.5-Omni, Spark-TTS, Parler, XTTS-v2, Bark, Sesame CSM
//! and Zonos. Three findings decide the table:
//!
//! 1. **Segment delimiters are universal.** Every one of those systems marks where
//!    text stops and audio starts. This is the load-bearing part and the only part
//!    every system agrees on.
//! 2. **Identity, language and emotion are NOT tokenized anywhere.** No surveyed
//!    system assigns a token per speaker — speaker identity is a plain-text NAME.
//!    No system tokenizes language or code-switching either: Scicom covers 150+
//!    languages with mid-sentence switching and has zero language tokens.
//!    Categorical emotion/prosody tokens were tried only by Spark-TTS, on 100k
//!    hours, and its own paper does not claim they work.
//! 3. **Style conditioning is free text behind a delimiter**, not a closed class.
//!    Orpheus's `<laugh>` / `<sigh>` look like control tokens but are ordinary BPE
//!    text in its tokenizer.
//!
//! So: delimiters get tokens, CONTENT does not. A speaker name, a language, an
//! emotion and a style description are all written as text between delimiters.
//! Adding `<|speaker_042|>` or `<|lang_ms|>` or `<|angry|>` would spend embedding
//! rows on a closed vocabulary that no shipped system found necessary, and would
//! cap the model at the speakers, languages and moods enumerated at training time.
//! Do not add them.
//!
//! The one distinction worth a token pair is INPUT audio vs GENERATED audio: every
//! system that supports voice-prompt cloning separates them (Higgs `audio_bos` vs
//! `audio_out_bos`, Step-Audio `<audio_start>` vs `<tts_start>`). That is
//! [`SpecialToken::VoiceRef`] against [`SpecialToken::Speech`].

use serde::{Deserialize, Serialize};

/// Ids reserved for control tokens by default: 13 defined, **627 reserved and free**.
///
/// # Why 640 and not 13
///
/// Two constraints pick this number.
///
/// **512-alignment.** `total_size` is the embedding and output-projection row
/// count, and matmul tiling wants it to be a multiple of 512. With Qwen3's 151_936
/// text ids and NeuCodec's 65_536 audio ids, `151_936 % 512 == 384` and
/// `65_536 % 512 == 0`, so alignment requires `control_region_size % 512 == 128`.
/// The practical choices are therefore 128 and 640. 640 is chosen for headroom.
///
/// **Cost.** 627 unused rows x hidden 2048 x 2 bytes is about 2.6 MB. Negligible
/// against a multi-gigabyte checkpoint.
///
/// # What the reservation buys
///
/// A control token added LATER takes one of the free reserved slots and does NOT
/// move [`SpeechVocab::audio_base`](super::vocab::SpeechVocab::audio_base), so
/// every audio id keeps its embedding row and previously trained checkpoints stay
/// valid. That is the entire reason the region exists; sizing it to the number of
/// tokens defined today would shift the whole audio region on the next addition.
pub const DEFAULT_CONTROL_REGION: usize = 640;

/// Control tokens of a speech LM, in canonical id order.
///
/// See the [module docs](self) for the evidence behind this set — in particular
/// why there is no speaker-id, language or emotion token, and why there never
/// should be.
///
/// The variant ORDER is the id order used by
/// [`SpeechVocab::with_default_specials`](super::vocab::SpeechVocab::with_default_specials),
/// and `Ord` is derived from it so a `BTreeMap` keyed by this type iterates
/// deterministically and a serialized layout is byte-stable across runs.
///
/// Serialization is by VARIANT NAME (serde's default for unit variants), never by
/// numeric index: an index would silently change meaning if this enum were ever
/// reordered, turning an old layout file into a wrong-but-loadable one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum SpecialToken {
    /// Start of the text to be spoken.
    SpeechText,
    /// End of the text to be spoken.
    SpeechTextEnd,
    /// Switch from reading text to emitting audio codes.
    Speech,
    /// End of generated audio. This is the sampling stop condition.
    SpeechEnd,
    /// Start of the speaker NAME.
    ///
    /// The name is PLAIN TEXT between this and [`SpecialToken::SpeakerEnd`], never
    /// a per-speaker id. No surveyed system tokenizes speaker identity, and a
    /// per-speaker token would cap the model at the speakers seen in training.
    Speaker,
    /// End of the speaker name.
    SpeakerEnd,
    /// Start of a free-form natural-language style description.
    ///
    /// Free text, deliberately. Style conditioning is an open class everywhere it
    /// ships; a closed set of emotion tokens was tried once and not shown to work.
    Style,
    /// End of the free-form style description.
    StyleEnd,
    /// Start of a schema'd style description over a fixed field list.
    ///
    /// Distinct from [`SpecialToken::Style`] so a model can be told which
    /// convention the text between the delimiters follows: prose, or fields.
    StyleFields,
    /// End of the schema'd style description.
    StyleFieldsEnd,
    /// Start of REFERENCE audio codes supplied as a voice prompt.
    ///
    /// Input audio, not output. Every system supporting voice-prompt cloning
    /// separates the two, because the model must condition on the reference
    /// without treating it as something it produced.
    VoiceRef,
    /// End of the reference audio codes.
    VoiceRefEnd,
    /// Padding for packed or batched sequences. Never sampled.
    SpeechPad,
}

/// Every [`SpecialToken`], in canonical id order.
pub const ALL_SPECIAL_TOKENS: [SpecialToken; 13] = [
    SpecialToken::SpeechText,
    SpecialToken::SpeechTextEnd,
    SpecialToken::Speech,
    SpecialToken::SpeechEnd,
    SpecialToken::Speaker,
    SpecialToken::SpeakerEnd,
    SpecialToken::Style,
    SpecialToken::StyleEnd,
    SpecialToken::StyleFields,
    SpecialToken::StyleFieldsEnd,
    SpecialToken::VoiceRef,
    SpecialToken::VoiceRefEnd,
    SpecialToken::SpeechPad,
];

impl SpecialToken {
    /// The token's surface string.
    ///
    /// These strings are the ON-DISK CONTRACT with the tokenizer: they appear in
    /// `added_tokens` and in every training example. Changing one silently
    /// retokenizes the corpus, so a round-trip test pins them.
    pub const fn token_str(&self) -> &'static str {
        match self {
            Self::SpeechText => "<|speech_text|>",
            Self::SpeechTextEnd => "<|/speech_text|>",
            Self::Speech => "<|speech|>",
            Self::SpeechEnd => "<|/speech|>",
            Self::Speaker => "<|speaker|>",
            Self::SpeakerEnd => "<|/speaker|>",
            Self::Style => "<|style|>",
            Self::StyleEnd => "<|/style|>",
            Self::StyleFields => "<|style_fields|>",
            Self::StyleFieldsEnd => "<|/style_fields|>",
            Self::VoiceRef => "<|voice_ref|>",
            Self::VoiceRefEnd => "<|/voice_ref|>",
            Self::SpeechPad => "<|speech_pad|>",
        }
    }

    /// Inverse of [`token_str`](Self::token_str); `None` for anything else.
    pub fn from_token_str(s: &str) -> Option<Self> {
        ALL_SPECIAL_TOKENS
            .iter()
            .copied()
            .find(|t| t.token_str() == s)
    }

    /// True if sampling is allowed to emit this token.
    ///
    /// Only [`SpecialToken::SpeechPad`] is forbidden: it exists to fill packed
    /// batches and carries no meaning in generated output.
    pub const fn is_sampleable(&self) -> bool {
        !matches!(self, Self::SpeechPad)
    }
}

#[cfg(test)]
mod tests {
    //! Tests for the control-token table and its on-disk surface strings.

    use super::*;

    #[test]
    fn token_strings_are_pinned_verbatim() {
        // The on-disk contract with the tokenizer. Editing a string here without
        // retokenizing every training example silently changes what the model reads.
        let expected = [
            (SpecialToken::SpeechText, "<|speech_text|>"),
            (SpecialToken::SpeechTextEnd, "<|/speech_text|>"),
            (SpecialToken::Speech, "<|speech|>"),
            (SpecialToken::SpeechEnd, "<|/speech|>"),
            (SpecialToken::Speaker, "<|speaker|>"),
            (SpecialToken::SpeakerEnd, "<|/speaker|>"),
            (SpecialToken::Style, "<|style|>"),
            (SpecialToken::StyleEnd, "<|/style|>"),
            (SpecialToken::StyleFields, "<|style_fields|>"),
            (SpecialToken::StyleFieldsEnd, "<|/style_fields|>"),
            (SpecialToken::VoiceRef, "<|voice_ref|>"),
            (SpecialToken::VoiceRefEnd, "<|/voice_ref|>"),
            (SpecialToken::SpeechPad, "<|speech_pad|>"),
        ];
        assert_eq!(expected.len(), ALL_SPECIAL_TOKENS.len());
        for (i, (tok, s)) in expected.iter().enumerate() {
            assert_eq!(
                ALL_SPECIAL_TOKENS[i], *tok,
                "canonical order changed at {i}"
            );
            assert_eq!(tok.token_str(), *s);
        }
    }

    #[test]
    fn token_strings_round_trip_for_all_thirteen() {
        for tok in ALL_SPECIAL_TOKENS {
            assert_eq!(SpecialToken::from_token_str(tok.token_str()), Some(tok));
        }
    }

    #[test]
    fn token_strings_are_distinct() {
        let mut seen = std::collections::BTreeSet::new();
        for tok in ALL_SPECIAL_TOKENS {
            assert!(seen.insert(tok.token_str()), "duplicate string for {tok:?}");
        }
        assert_eq!(seen.len(), 13);
    }

    #[test]
    fn unknown_token_strings_are_rejected() {
        for s in [
            "",
            "<|speech_text|",
            "speech",
            "<|SPEECH|>",
            "<|lang_ms|>",
            "<|angry|>",
            "<|speaker_042|>",
            "<laugh>",
        ] {
            assert_eq!(SpecialToken::from_token_str(s), None, "accepted {s:?}");
        }
    }

    #[test]
    fn only_the_pad_token_is_unsampleable() {
        for tok in ALL_SPECIAL_TOKENS {
            assert_eq!(tok.is_sampleable(), tok != SpecialToken::SpeechPad);
        }
    }

    /// Serialization must be by VARIANT NAME, not by index.
    ///
    /// A numeric index would stay valid-looking after the enum is reordered or a
    /// variant is inserted, so an old layout file would load and mean something else.
    /// A name either matches or fails loudly.
    #[test]
    fn serde_uses_variant_names_not_indices() {
        for tok in ALL_SPECIAL_TOKENS {
            let json = serde_json::to_string(&tok).expect("serialize token");
            assert_eq!(json, format!("\"{tok:?}\""));
            let back: SpecialToken = serde_json::from_str(&json).expect("deserialize token");
            assert_eq!(back, tok);
        }
        // An index-shaped payload must not deserialize into a variant.
        assert!(serde_json::from_str::<SpecialToken>("0").is_err());
        assert!(serde_json::from_str::<SpecialToken>("12").is_err());
        // A name this enum does not know fails rather than aliasing an existing one.
        assert!(serde_json::from_str::<SpecialToken>("\"LanguageTag\"").is_err());
    }

    /// A payload written before the enum grew must still load unchanged.
    ///
    /// Serializing by name makes this hold by construction: the four names below are
    /// pinned, and inserting variants anywhere in the enum cannot move them.
    #[test]
    fn names_survive_the_enum_being_extended() {
        let json = r#"["SpeechText","Speech","SpeechEnd","SpeechPad"]"#;
        let back: Vec<SpecialToken> = serde_json::from_str(json).expect("deserialize names");
        assert_eq!(
            back,
            vec![
                SpecialToken::SpeechText,
                SpecialToken::Speech,
                SpecialToken::SpeechEnd,
                SpecialToken::SpeechPad,
            ]
        );
        assert_eq!(serde_json::to_string(&back).expect("serialize"), json);
    }
}
