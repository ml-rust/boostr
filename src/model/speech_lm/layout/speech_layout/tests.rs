//! The point of these tests is that the two layouts are DIFFERENT and both
//! stable. The native expectations are hand-computed from the region arithmetic
//! documented on [`SpeechVocab`], so introducing `SpeechLayout` cannot have
//! quietly altered what the pre-existing path emits.

use super::*;

use crate::model::speech_lm::codec::CodecVocab;
use crate::model::speech_lm::special::SpecialToken;

/// A native vocabulary over a small text region, so text ids can be named by
/// hand, with NeuCodec's real single 65,536-entry codebook.
///
/// Control ids run sequentially from 256, so `SpeechText` = 256 ... `SpeechPad`
/// = 268, and `audio_base` = 256 + 640 = 896.
fn native() -> SpeechLayout {
    let codec = CodecVocab::new(1, 65_536).expect("neucodec codec vocab");
    let vocab = SpeechVocab::with_default_specials(256, codec).expect("speech vocab");
    SpeechLayout::Native(vocab)
}

#[test]
fn the_native_layout_still_emits_exactly_what_it_did_before() {
    let layout = native();
    let frames = vec![vec![0usize], vec![5]];
    let record = SpeechRecord {
        speaker: None,
        style: None,
        text: &[1, 2, 3],
        frames: &frames,
    };
    let packed = layout.pack_record(&record).expect("native packs");

    // <|speech_text|> 1 2 3 <|/speech_text|> <|speech|> audio(0) audio(5) <|/speech|>
    assert_eq!(packed, vec![256, 1, 2, 3, 257, 258, 896, 901, 259]);

    // And the free function it delegates to agrees, id for id.
    let vocab = layout.vocab().expect("native carries a vocab");
    assert_eq!(pack_record(vocab, &record).expect("free fn packs"), packed);
}

#[test]
fn the_native_layout_still_emits_its_speaker_and_style_sections() {
    let layout = native();
    let frames = vec![vec![0usize]];
    let record = SpeechRecord {
        speaker: Some(&[10]),
        style: Some(&[20]),
        text: &[1, 2, 3],
        frames: &frames,
    };
    let packed = layout.pack_record(&record).expect("native packs");

    // <|speaker|> 10 <|/speaker|> <|style|> 20 <|/style|>
    // <|speech_text|> 1 2 3 <|/speech_text|> <|speech|> audio(0) <|/speech|>
    assert_eq!(
        packed,
        vec![260, 10, 261, 262, 20, 263, 256, 1, 2, 3, 257, 258, 896, 259]
    );
}

#[test]
fn the_two_layouts_disagree_about_the_same_record() {
    let frames = vec![vec![0usize]];
    let record = SpeechRecord {
        speaker: None,
        style: None,
        text: &[1, 2, 3],
        frames: &frames,
    };
    let native = native().pack_record(&record).expect("native packs");
    let base = SpeechLayout::expressive_tts()
        .pack_record(&record)
        .expect("expressive-tts packs");
    assert_eq!(native, vec![256, 1, 2, 3, 257, 258, 896, 259]);
    assert_eq!(base, vec![151_644, 1, 2, 3, 151_669, 151_670, 151_645]);
    assert_ne!(native, base);
}

#[test]
fn layout_metadata_is_reported_per_variant() {
    let native = native();
    assert_eq!(native.name(), "boostr-native");
    assert_eq!(native.audio_base(), 896);
    assert_eq!(native.codebook_size(), 65_536);
    assert_eq!(native.total_size(), 896 + 65_536);
    assert!(native.matches_embedding_rows(66_432));

    let base = SpeechLayout::expressive_tts();
    assert_eq!(base.name(), "expressive-tts-1.7b");
    assert_eq!(base.audio_base(), 151_670);
    assert_eq!(base.codebook_size(), 65_536);
    assert_eq!(base.total_size(), 217_208);
    assert!(base.vocab().is_none());
}

#[test]
fn packing_several_records_concatenates_them_with_no_separator() {
    let layout = SpeechLayout::expressive_tts();
    let frames = vec![vec![0usize]];
    let record = SpeechRecord {
        speaker: None,
        style: None,
        text: &[1],
        frames: &frames,
    };
    let packed = layout
        .pack_records(&[record.clone(), record])
        .expect("two records pack");
    assert_eq!(
        packed,
        vec![
            151_644, 1, 151_669, 151_670, 151_645, 151_644, 1, 151_669, 151_670, 151_645
        ]
    );
}

#[test]
fn a_bad_record_is_named_by_its_index_in_the_batch() {
    let layout = SpeechLayout::expressive_tts();
    let good_frames = vec![vec![0usize]];
    let bad_frames = vec![vec![65_536usize]];
    let good = SpeechRecord {
        speaker: None,
        style: None,
        text: &[1],
        frames: &good_frames,
    };
    let bad = SpeechRecord {
        frames: &bad_frames,
        ..good.clone()
    };
    let err = layout
        .pack_records(&[good, bad])
        .expect_err("record 1 holds an out-of-range code");
    let msg = err.to_string();
    assert!(msg.contains("record 1"), "must name the record: {msg}");
    assert!(msg.contains("65536"), "must name the code: {msg}");
}

#[test]
fn each_layout_pads_with_its_own_pad_id() {
    let frames = vec![vec![0usize]];
    let record = SpeechRecord {
        speaker: None,
        style: None,
        text: &[1],
        frames: &frames,
    };

    // Native: 8 ids packed, padded to 10 with <|speech_pad|> = 268.
    let native = native();
    let packed = native
        .pack_records_padded(std::slice::from_ref(&record), 10)
        .expect("native pads");
    assert_eq!(packed, vec![256, 1, 257, 258, 896, 259, 268, 268, 268, 268]);
    assert_eq!(
        native
            .vocab()
            .and_then(|v| v.special_id(SpecialToken::SpeechPad)),
        Some(268)
    );

    // ExpressiveTTS: 5 ids packed, padded to 8 with <|endoftext|> = 151_643.
    let base = SpeechLayout::expressive_tts();
    let packed = base
        .pack_records_padded(std::slice::from_ref(&record), 8)
        .expect("expressive-tts pads");
    assert_eq!(
        packed,
        vec![
            151_644, 1, 151_669, 151_670, 151_645, 151_643, 151_643, 151_643
        ]
    );
}

#[test]
fn a_zero_window_is_rejected_by_both_layouts() {
    let record = SpeechRecord {
        speaker: None,
        style: None,
        text: &[1],
        frames: &[],
    };
    for layout in [native(), SpeechLayout::expressive_tts()] {
        let err = layout
            .pack_records_padded(std::slice::from_ref(&record), 0)
            .expect_err("a zero window is not a window");
        assert!(
            err.to_string().contains("at least 1"),
            "message must state the bound: {err}"
        );
    }
}

#[test]
fn is_audio_classifies_both_layouts_at_their_own_boundary() {
    // The property a trainer's loss mask depends on: the id one below
    // `audio_base` is not audio, the id at it is, and `total_size` is past the
    // end. Checked on both variants because a mask built for a corpus packed
    // under either must select the same positions the packer wrote audio into.
    for layout in [SpeechLayout::expressive_tts(), native()] {
        let base = layout.audio_base() as u32;
        let total = layout.total_size() as u32;
        assert!(!layout.is_audio(0), "{}: id 0", layout.name());
        assert!(!layout.is_audio(base - 1), "{}: below base", layout.name());
        assert!(layout.is_audio(base), "{}: at base", layout.name());
        assert!(
            layout.is_audio(total - 1),
            "{}: last audio id",
            layout.name()
        );
        assert!(!layout.is_audio(total), "{}: past the end", layout.name());
    }
}

#[test]
fn is_audio_agrees_with_the_native_vocab_it_wraps() {
    // `SpeechVocab::is_audio` is the existing answer for the native layout;
    // the layout-level one must not drift from it.
    let layout = native();
    let vocab = layout
        .vocab()
        .expect("native layout carries its vocab")
        .clone();
    for id in [
        0u32,
        1,
        vocab.audio_base() as u32 - 1,
        vocab.audio_base() as u32,
        vocab.total_size() as u32 - 1,
        vocab.total_size() as u32,
    ] {
        assert_eq!(layout.is_audio(id), vocab.is_audio(id), "id {id}");
    }
}
