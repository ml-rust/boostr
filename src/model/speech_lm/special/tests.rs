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
