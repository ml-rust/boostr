//! Tests for the speech-LM flat vocabulary layout.

use super::*;

const QWEN3_TEXT_VOCAB: usize = 151_936;
const NEUCODEC_SIZE: usize = 65_536;

/// Small region used by the toy layouts below. Must hold all 13 control tokens.
const SMALL_REGION: usize = 16;

fn all_specials() -> Vec<SpecialToken> {
    ALL_SPECIAL_TOKENS.to_vec()
}

/// The first four delimiters. Used where a layout needs fewer tokens than slots.
fn core_specials() -> Vec<SpecialToken> {
    vec![
        SpecialToken::SpeechText,
        SpecialToken::SpeechTextEnd,
        SpecialToken::Speech,
        SpecialToken::SpeechEnd,
    ]
}

/// NeuCodec on Qwen3: 1 codebook x 65_536, one code per frame.
fn neucodec_vocab() -> SpeechVocab {
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
    let v =
        SpeechVocab::with_sequential_specials(10, 6, &core_specials(), codec).expect("valid vocab");
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
    let sized_to_count = SpeechVocab::with_sequential_specials(100, 4, &core_specials(), codec())
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
    let v =
        SpeechVocab::with_sequential_specials(10, 20, &all_specials(), codec).expect("valid vocab");
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
