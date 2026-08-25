//! The sequence these tests pin IS the contract with
//! `Scicom-intl/Multilingual-Expressive-TTS-1.7B`. Every expected vector below is
//! hand-computed from the checkpoint's `added_tokens.json` ids, not from the code
//! under test, so a change to the layout fails here rather than in a fine-tune
//! that merely learns worse.

use super::*;

/// Two arbitrary text ids, well inside Qwen3's text region.
const TEXT: [u32; 3] = [9707, 11, 1879];

fn record<'a>(text: &'a [u32], frames: &'a [Vec<usize>]) -> SpeechRecord<'a> {
    SpeechRecord {
        speaker: None,
        style: None,
        text,
        frames,
    }
}

#[test]
fn packs_the_exact_sequence_from_the_model_card() {
    let layout = ExpressiveTtsLayout::new();
    let frames = vec![vec![0usize], vec![1], vec![65_535]];
    let packed = layout
        .pack_record(&record(&TEXT, &frames))
        .expect("a one-codebook record packs");

    // <|im_start|> ++ text ++ <|speech_start|> ++ 151670 + c ++ <|im_end|>
    let expected: Vec<u32> = vec![
        151_644, 9707, 11, 1879, 151_669, 151_670, 151_671, 217_205, 151_645,
    ];
    assert_eq!(packed, expected);
}

#[test]
fn code_zero_and_code_65535_pin_the_ends_of_the_audio_run() {
    let layout = ExpressiveTtsLayout::new();
    assert_eq!(layout.audio_token(0).expect("code 0 is in range"), 151_670);
    assert_eq!(
        layout.audio_token(65_535).expect("code 65535 is in range"),
        217_205
    );
    assert_eq!(layout.decode_audio_token(151_670), Some(0));
    assert_eq!(layout.decode_audio_token(217_205), Some(65_535));
    // <|description|> sits immediately after the audio run and is not audio.
    assert_eq!(layout.decode_audio_token(217_206), None);
    assert_eq!(layout.decode_audio_token(151_669), None);
}

#[test]
fn code_65536_is_rejected_and_named_with_its_frame() {
    let layout = ExpressiveTtsLayout::new();
    let frames = vec![vec![7usize], vec![65_536]];
    let err = layout
        .pack_record(&record(&TEXT, &frames))
        .expect_err("code 65536 is past the codebook");
    let msg = err.to_string();
    assert!(msg.contains("65536"), "must name the code: {msg}");
    assert!(msg.contains("frame 1"), "must name the frame: {msg}");
}

#[test]
fn a_frame_that_is_not_one_code_is_rejected_by_index_and_length() {
    let layout = ExpressiveTtsLayout::new();
    let frames = vec![vec![1usize], vec![2, 3, 4]];
    let err = layout
        .pack_record(&record(&TEXT, &frames))
        .expect_err("a 3-code frame has nowhere to go in a single-codebook layout");
    let msg = err.to_string();
    assert!(msg.contains("frame 1"), "must name the frame index: {msg}");
    assert!(msg.contains("3 codes"), "must name the length: {msg}");
}

#[test]
fn a_some_speaker_is_refused_rather_than_guessed_at() {
    let layout = ExpressiveTtsLayout::new();
    let speaker = [1234u32];
    let frames = vec![vec![0usize]];
    let rec = SpeechRecord {
        speaker: Some(&speaker),
        style: None,
        text: &TEXT,
        frames: &frames,
    };
    let err = layout
        .pack_record(&rec)
        .expect_err("this layout has no speaker delimiter");
    let msg = err.to_string();
    assert!(
        msg.contains("{speaker}: {text}"),
        "the error must tell the caller to pre-render the prefix: {msg}"
    );
}

#[test]
fn style_becomes_the_description_variant() {
    let layout = ExpressiveTtsLayout::new();
    let style = [100u32, 200];
    let frames = vec![vec![2usize]];
    let rec = SpeechRecord {
        speaker: None,
        style: Some(&style),
        text: &TEXT,
        frames: &frames,
    };
    let packed = layout
        .pack_record(&rec)
        .expect("the description variant packs");

    // <|im_start|> text <|description|> description <|speech_start|> audio <|im_end|>
    let expected: Vec<u32> = vec![
        151_644, 9707, 11, 1879, 217_206, 100, 200, 151_669, 151_672, 151_645,
    ];
    assert_eq!(packed, expected);
}

#[test]
fn empty_frames_still_emit_both_delimiters() {
    let layout = ExpressiveTtsLayout::new();
    let frames: Vec<Vec<usize>> = Vec::new();
    let packed = layout
        .pack_record(&record(&TEXT, &frames))
        .expect("a record with no audio packs");
    assert_eq!(packed, vec![151_644, 9707, 11, 1879, 151_669, 151_645]);
}

#[test]
fn empty_text_still_emits_both_delimiters() {
    let layout = ExpressiveTtsLayout::new();
    let frames = vec![vec![65_535usize]];
    let packed = layout
        .pack_record(&record(&[], &frames))
        .expect("a record with no text packs");
    assert_eq!(packed, vec![151_644, 151_669, 217_205, 151_645]);
}

#[test]
fn the_checkpoint_ids_are_the_ones_this_layout_carries() {
    let layout = ExpressiveTtsLayout::new();
    assert_eq!(layout.seq_start(), 151_644);
    assert_eq!(layout.speech_start(), 151_669);
    assert_eq!(layout.eos_id(), 151_645);
    assert_eq!(layout.pad_id(), 151_643);
    assert_eq!(layout.description_id(), 217_206);
    assert_eq!(layout.audio_base(), 151_670);
    assert_eq!(layout.codebook_size(), 65_536);
    assert_eq!(layout.vocab_size(), 217_208);
    assert_eq!(DESCRIPTION_CATEGORY, 217_207);
    // The audio run ends exactly where <|description|> begins: no gap, no overlap.
    assert_eq!(AUDIO_BASE as usize + CODEBOOK_SIZE, DESCRIPTION as usize);
    assert!(layout.matches_embedding_rows(217_208));
    assert!(!layout.matches_embedding_rows(217_207));
}
