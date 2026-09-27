//! End-to-end [`VitsEngine`] check against the real `facebook/mms-tts-zlm`
//! checkpoint: `TtsEngine::synthesize` returns finite audio of a plausible
//! length, and the engine's own tokenization path matches
//! [`VitsTokenizer`] exactly (already oracle-tested against HF in
//! `vits_tokenizer_oracle.rs`).
//!
//! Gated on the checkpoint: skips unless `MMS_TTS_ZLM_DIR`, else
//! `$BOOSTR_MODELS_DIR/facebook/mms-tts-zlm`, is present on disk.

#![cfg(feature = "vits")]

use boostr_audio::tts::{TtsEngine, VitsEngine, VitsTokenizer};

mod common;
use common::{model_fixture, skip_notice};

const TEXT: &str = "Selamat pagi, apa khabar?";

#[test]
fn synthesize_returns_plausible_finite_audio_and_matches_the_oracle_tokenizer() {
    let Some(checkpoint_dir) = model_fixture("MMS_TTS_ZLM_DIR", "facebook/mms-tts-zlm") else {
        skip_notice("facebook/mms-tts-zlm checkpoint", "MMS_TTS_ZLM_DIR");
        return;
    };

    let engine = VitsEngine::from_dir(&checkpoint_dir).expect("load VitsEngine");
    let voices = TtsEngine::voices(&engine);
    assert_eq!(
        voices.len(),
        1,
        "an MMS-TTS checkpoint serves exactly one voice"
    );
    let voice_id = voices[0].id.clone();

    let sample_rate = TtsEngine::sample_rate(&engine);
    assert_eq!(
        sample_rate, 16_000,
        "facebook/mms-tts-zlm renders at 16 kHz"
    );

    let samples = TtsEngine::synthesize(&engine, TEXT, &voice_id, 1.0).expect("synthesize");
    assert!(!samples.is_empty(), "synthesize produced no samples");
    assert!(
        samples.iter().all(|s| s.is_finite() && s.abs() <= 1.0),
        "every sample must be finite and within the HiFi-GAN tanh range"
    );

    // A plausible length: a few tokens of Malay speech at 16 kHz should
    // land somewhere between a fraction of a second and several seconds,
    // not near-empty or absurdly long — catches a wrong upsample product or
    // a duration-predictor unit mismatch without needing a reference dump.
    let seconds = samples.len() as f32 / sample_rate as f32;
    assert!(
        (0.3..10.0).contains(&seconds),
        "synthesized {seconds:.2}s of audio for a short sentence, expected roughly 0.3s-10s"
    );

    // The engine's own tokenization path must equal the standalone,
    // oracle-tested VitsTokenizer.
    let oracle_tokenizer = VitsTokenizer::from_dir(&checkpoint_dir).expect("load VitsTokenizer");
    let expected_ids = oracle_tokenizer
        .encode(TEXT)
        .expect("encode with the oracle tokenizer");
    let engine_ids = engine
        .tokenizer()
        .encode(TEXT)
        .expect("encode with the engine's own tokenizer");
    assert_eq!(
        engine_ids, expected_ids,
        "the engine must tokenize exactly like the standalone VitsTokenizer"
    );
}

#[test]
fn synthesize_rejects_an_unknown_voice_and_a_non_positive_speed() {
    let Some(checkpoint_dir) = model_fixture("MMS_TTS_ZLM_DIR", "facebook/mms-tts-zlm") else {
        skip_notice("facebook/mms-tts-zlm checkpoint", "MMS_TTS_ZLM_DIR");
        return;
    };
    let engine = VitsEngine::from_dir(&checkpoint_dir).expect("load VitsEngine");

    assert!(
        TtsEngine::synthesize(&engine, TEXT, "not-a-real-voice", 1.0).is_err(),
        "an unknown voice id must be refused"
    );
    let voices = TtsEngine::voices(&engine);
    assert!(
        TtsEngine::synthesize(&engine, TEXT, &voices[0].id, 0.0).is_err(),
        "a non-positive speed must be refused"
    );
}
