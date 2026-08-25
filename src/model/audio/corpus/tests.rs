//! Everything in this module that runs without a checkpoint.
//!
//! The three models (`SileroVad`, `WhisperBundle`, `NeuCodecEncoder`) can only
//! be built from real safetensors files, so [`SpeechCorpusBuilder`] itself
//! cannot be constructed here. The two behaviours that would otherwise be
//! unreachable are therefore reached through the free functions the builder
//! delegates to — [`check_max_speech_duration`] for the segmentation cap and
//! [`pack_utterances`] for the packing — which is exactly the code the methods
//! run.
//!
//! [`SpeechCorpusBuilder`]: super::SpeechCorpusBuilder

use splintr::Tokenize;

use super::options::{
    CorpusOptions, MAX_UTTERANCE_SECS, PRETRAINED_TOKENIZER_NAMES, TextTokenizer,
    check_max_speech_duration,
};
use super::utterance::{Utterance, pack_utterances, pack_utterances_with_layout};
use crate::model::audio::vad::{SpeechSegment, VadSegmentOptions};
use crate::model::speech_lm::codec::CodecVocab;
use crate::model::speech_lm::layout::SpeechLayout;
use crate::model::speech_lm::pack::{SpeechRecord, pack_records, pack_records_padded};
use crate::model::speech_lm::vocab::SpeechVocab;

/// A vocabulary shaped like the real one (NeuCodec's single 65,536-entry
/// codebook) but over a small text region, so a test can name text ids by hand.
fn test_vocab() -> SpeechVocab {
    let codec = CodecVocab::new(1, 65_536).expect("neucodec codec vocab");
    SpeechVocab::with_default_specials(256, codec).expect("speech vocab")
}

fn utterance(start: usize, end: usize, text_tokens: &[u32], frames: &[usize]) -> Utterance {
    Utterance {
        segment: SpeechSegment { start, end },
        text: "hello".to_string(),
        text_tokens: text_tokens.to_vec(),
        frames: frames.iter().map(|&code| vec![code]).collect(),
    }
}

#[test]
fn unknown_pretrained_name_lists_the_accepted_names() {
    // `AnyTokenizer` is not `Debug`, so `expect_err` is unusable here.
    let Err(err) = TextTokenizer::Pretrained("not_a_tokenizer").resolve() else {
        panic!("an unknown name must not resolve");
    };
    let msg = err.to_string();
    assert!(
        msg.contains("not_a_tokenizer"),
        "message must quote the offending name: {msg}"
    );
    for name in PRETRAINED_TOKENIZER_NAMES {
        assert!(
            msg.contains(name),
            "message must list the accepted name {name}: {msg}"
        );
    }
}

#[test]
fn cl100k_base_resolves_with_a_plausible_vocab_size() {
    let tokenizer = TextTokenizer::Pretrained("cl100k_base")
        .resolve()
        .expect("cl100k_base is bundled by this crate's splintr features");
    let size = Tokenize::vocab_size(&tokenizer);
    assert!(
        (100_000..110_000).contains(&size),
        "cl100k_base should hold roughly 100k tokens, got {size}"
    );
}

#[test]
fn infinite_max_speech_duration_is_rejected() {
    let opts = CorpusOptions {
        vad: VadSegmentOptions::default(),
        ..CorpusOptions::default()
    };
    assert!(
        !opts.vad.max_speech_duration_s.is_finite(),
        "the VAD's own default must still be uncapped for this test to mean anything"
    );
    let err = check_max_speech_duration(opts.vad.max_speech_duration_s)
        .expect_err("an uncapped segmentation length must be refused");
    let msg = err.to_string();
    assert!(
        msg.contains("max_speech_duration_s"),
        "message must name the option: {msg}"
    );
    assert!(
        msg.contains("30"),
        "message must name Whisper's 30 s window: {msg}"
    );
}

#[test]
fn over_thirty_second_max_speech_duration_is_rejected() {
    let err =
        check_max_speech_duration(45.0).expect_err("a cap over Whisper's window must be refused");
    assert!(
        err.to_string().contains("max_speech_duration_s"),
        "message must name the option: {err}"
    );
}

#[test]
fn max_speech_duration_at_the_window_is_accepted() {
    check_max_speech_duration(MAX_UTTERANCE_SECS).expect("exactly the window must be accepted");
    check_max_speech_duration(5.0).expect("a shorter cap must be accepted");
}

#[test]
fn non_positive_max_speech_duration_is_rejected() {
    assert!(check_max_speech_duration(0.0).is_err());
    assert!(check_max_speech_duration(-1.0).is_err());
    assert!(check_max_speech_duration(f32::NAN).is_err());
}

#[test]
fn corpus_options_default_is_already_capped() {
    check_max_speech_duration(CorpusOptions::default().vad.max_speech_duration_s)
        .expect("the default must be usable without further tuning");
}

#[test]
fn pack_matches_pack_records_on_equivalent_records() {
    let vocab = test_vocab();
    let utterances = vec![
        utterance(0, 320, &[7, 8, 9], &[1, 2]),
        utterance(320, 960, &[10], &[3, 4, 5]),
    ];
    let opts = CorpusOptions::default();

    let packed = pack_utterances(&vocab, &utterances, &opts).expect("pack");

    let records: Vec<SpeechRecord<'_>> = utterances
        .iter()
        .map(|u| SpeechRecord {
            speaker: None,
            style: None,
            text: &u.text_tokens,
            frames: &u.frames,
        })
        .collect();
    let expected = pack_records(&vocab, &records).expect("pack_records");

    assert_eq!(packed, expected);
}

#[test]
fn pack_honours_pad_to_multiple() {
    let vocab = test_vocab();
    let utterances = vec![utterance(0, 320, &[7, 8, 9], &[1, 2])];
    let pad_to_multiple = 64;
    let opts = CorpusOptions {
        pad_to_multiple: Some(pad_to_multiple),
        ..CorpusOptions::default()
    };

    let padded = pack_utterances(&vocab, &utterances, &opts).expect("padded pack");
    assert_eq!(
        padded.len() % pad_to_multiple,
        0,
        "padded length {} is not a multiple of {pad_to_multiple}",
        padded.len()
    );

    let records = [SpeechRecord {
        speaker: None,
        style: None,
        text: &utterances[0].text_tokens,
        frames: &utterances[0].frames,
    }];
    let expected = pack_records_padded(&vocab, &records, pad_to_multiple).expect("reference pack");
    assert_eq!(padded, expected);

    let unpadded =
        pack_utterances(&vocab, &utterances, &CorpusOptions::default()).expect("unpadded pack");
    assert!(
        unpadded.len() < padded.len(),
        "the padded stream must be longer than the unpadded one"
    );
    assert_eq!(
        &padded[..unpadded.len()],
        &unpadded[..],
        "padding must only append"
    );
}

#[test]
fn pack_of_no_utterances_is_empty() {
    let vocab = test_vocab();
    let packed = pack_utterances(&vocab, &[], &CorpusOptions::default()).expect("pack");
    assert!(packed.is_empty());
}

/// Knowing a name and being able to LOAD it are different things: splintr gates
/// each vocabulary's data behind a cargo feature. `llama3` is documented and
/// resolves to a `PretrainedVocab`, then fails unless its vocabulary is
/// bundled — which is exactly what boostr's `vocab-llama3` feature does, so
/// this only holds in a build without it.
#[cfg(not(feature = "vocab-llama3"))]
#[test]
fn a_documented_but_unbundled_name_fails_with_a_feature_error() {
    let Err(err) = TextTokenizer::Pretrained("llama3").resolve() else {
        panic!("llama3 is not bundled by this build and must not resolve");
    };
    let msg = err.to_string();
    // splintr's own error names the cargo feature that would fix it.
    assert!(
        msg.contains("vocab-llama3") || msg.contains("llama3"),
        "the error must name the vocabulary or its feature: {msg}"
    );
}

/// The unknown-name error must not imply every documented name is loadable —
/// a documented name still needs its vocabulary bundled. It must say so and
/// steer at the path that always works.
#[test]
fn the_unknown_name_error_explains_the_feature_gate() {
    let Err(err) = TextTokenizer::Pretrained("nope").resolve() else {
        panic!("an unknown name must not resolve");
    };
    let msg = err.to_string();
    assert!(
        msg.contains("vocab-"),
        "must point at the feature that bundles a vocabulary: {msg}"
    );
    assert!(
        msg.contains("tokenizer.json"),
        "must steer at the path that always works: {msg}"
    );
}

#[test]
fn the_native_layout_packs_utterances_exactly_as_before() {
    let vocab = test_vocab();
    let utterances = vec![utterance(0, 320, &[7, 8, 9], &[1, 2])];
    let opts = CorpusOptions::default();

    let by_vocab = pack_utterances(&vocab, &utterances, &opts).expect("pack by vocab");
    let by_layout = pack_utterances_with_layout(&SpeechLayout::Native(vocab), &utterances, &opts)
        .expect("pack by layout");

    // <|speech_text|> 7 8 9 <|/speech_text|> <|speech|> audio(1) audio(2) <|/speech|>
    assert_eq!(by_vocab, vec![256, 7, 8, 9, 257, 258, 897, 898, 259]);
    assert_eq!(by_layout, by_vocab);
}

#[test]
fn the_expressive_tts_layout_packs_utterances_into_the_base_sequence() {
    let utterances = vec![utterance(0, 320, &[7, 8, 9], &[1, 2])];
    let packed = pack_utterances_with_layout(
        &SpeechLayout::expressive_tts(),
        &utterances,
        &CorpusOptions::default(),
    )
    .expect("pack by layout");

    // <|im_start|> 7 8 9 <|speech_start|> 151670+1 151670+2 <|im_end|>
    assert_eq!(
        packed,
        vec![151_644, 7, 8, 9, 151_669, 151_671, 151_672, 151_645]
    );
}

#[test]
fn the_default_options_render_no_speaker_prefix() {
    assert!(
        CorpusOptions::default().speaker.is_none(),
        "adding the speaker option must not change what an existing run tokenizes"
    );
}
