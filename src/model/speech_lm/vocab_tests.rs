//! Tests for the speech-LM flat vocabulary layout.

use super::*;

const QWEN3_TEXT_VOCAB: usize = 151_936;
const NEUCODEC_SIZE: usize = 65_536;

/// The four control tokens defined today, and the region size pinned for them.
const CONTROL_REGION: usize = 4;

fn all_specials() -> Vec<SpecialToken> {
    vec![
        SpecialToken::StartOfText,
        SpecialToken::EndOfText,
        SpecialToken::StartOfAudio,
        SpecialToken::EndOfAudio,
    ]
}

/// NeuCodec on Qwen3: 1 codebook x 65_536, one code per frame.
fn neucodec_vocab() -> SpeechVocab {
    let codec = CodecVocab::new(1, NEUCODEC_SIZE).expect("valid codec");
    SpeechVocab::with_sequential_specials(QWEN3_TEXT_VOCAB, CONTROL_REGION, &all_specials(), codec)
        .expect("valid vocab")
}

/// Flat residual codec: 3 codebooks x 4096, one code each per frame.
fn snac_flat_vocab() -> SpeechVocab {
    let codec = CodecVocab::new(3, 4096).expect("valid codec");
    SpeechVocab::with_sequential_specials(1000, CONTROL_REGION, &all_specials(), codec)
        .expect("valid vocab")
}

/// Interleaved hierarchy: codebook 0 once, 1 twice, 2 four times per frame.
fn snac_interleaved_vocab() -> SpeechVocab {
    let codec = CodecVocab::with_frame_layout(4096, vec![1, 2, 4]).expect("valid codec");
    SpeechVocab::with_sequential_specials(1000, CONTROL_REGION, &all_specials(), codec)
        .expect("valid vocab")
}

#[test]
fn neucodec_layout_has_concrete_ids() {
    let v = neucodec_vocab();
    assert_eq!(v.text_vocab_size(), 151_936);
    assert_eq!(v.control_region_size(), 4);
    assert_eq!(v.num_specials(), 4);
    assert_eq!(v.audio_base(), 151_940);
    assert_eq!(v.total_size(), 151_940 + 65_536);
    assert_eq!(v.total_size(), 217_476);

    assert_eq!(v.special_id(SpecialToken::StartOfText), Some(151_936));
    assert_eq!(v.special_id(SpecialToken::EndOfText), Some(151_937));
    assert_eq!(v.special_id(SpecialToken::StartOfAudio), Some(151_938));
    assert_eq!(v.special_id(SpecialToken::EndOfAudio), Some(151_939));

    assert_eq!(v.audio_token(0, 0).expect("first audio id"), 151_940);
    assert_eq!(v.audio_token(0, 65_535).expect("last audio id"), 217_475);
    assert_eq!(v.codec().codes_per_frame(), 1);
    assert!(v.matches_embedding_rows(217_476));
    assert!(!v.matches_embedding_rows(217_475));
    assert!(!v.matches_embedding_rows(0));
}

#[test]
fn missing_special_has_no_id() {
    let codec = CodecVocab::new(1, 8).expect("valid codec");
    let v = SpeechVocab::with_sequential_specials(
        10,
        2,
        &[SpecialToken::StartOfAudio, SpecialToken::EndOfAudio],
        codec,
    )
    .expect("valid vocab");
    assert_eq!(v.special_id(SpecialToken::StartOfAudio), Some(10));
    assert_eq!(v.special_id(SpecialToken::EndOfAudio), Some(11));
    assert_eq!(v.special_id(SpecialToken::StartOfText), None);
    assert_eq!(v.audio_base(), 12);
}

#[test]
fn explicit_ids_are_stored_not_recomputed_from_order() {
    let codec = CodecVocab::new(1, 8).expect("valid codec");
    // Ids deliberately NOT in enum order: the map is the source of truth.
    let mut specials = BTreeMap::new();
    specials.insert(SpecialToken::StartOfText, 13u32);
    specials.insert(SpecialToken::EndOfAudio, 10u32);
    let v = SpeechVocab::new(10, 4, specials, codec).expect("valid vocab");
    assert_eq!(v.special_id(SpecialToken::StartOfText), Some(13));
    assert_eq!(v.special_id(SpecialToken::EndOfAudio), Some(10));
    assert_eq!(v.audio_base(), 14);
}

/// THE LANDMINE TEST.
///
/// Reserving a control region LARGER than the number of defined tokens must make
/// the audio region immovable. A control token added later takes a spare reserved
/// id, and every audio id stays exactly where the trained checkpoint put it.
///
/// `SpecialToken` has only four variants today, so a token occupying a previously
/// unused reserved slot stands in for the fifth token a future task would add.
#[test]
fn adding_a_special_into_a_reserved_region_does_not_move_audio_base() {
    let codec = || CodecVocab::new(1, 8).expect("valid codec");
    let region = 8;

    let before = SpeechVocab::with_sequential_specials(100, region, &all_specials(), codec())
        .expect("valid vocab");
    assert_eq!(before.audio_base(), 108);
    assert_eq!(before.total_size(), 116);
    let audio_before = before.audio_token(0, 3).expect("audio id");

    // A control token now sits at reserved id 107, which nothing claimed before.
    let mut grown = BTreeMap::new();
    for (i, tok) in all_specials().iter().enumerate() {
        grown.insert(*tok, 100 + i as u32);
    }
    grown.insert(SpecialToken::StartOfText, 107);
    let after = SpeechVocab::new(100, region, grown, codec()).expect("valid vocab");

    assert_eq!(after.audio_base(), before.audio_base());
    assert_eq!(after.total_size(), before.total_size());
    assert_eq!(after.audio_token(0, 3).expect("audio id"), audio_before);
    assert_eq!(after.special_id(SpecialToken::StartOfText), Some(107));

    // Contrast: sizing the region to the token count instead ties audio_base to
    // how many controls exist, so every added token would shift all audio ids.
    let sized_to_count = SpeechVocab::with_sequential_specials(100, 4, &all_specials(), codec())
        .expect("valid vocab");
    assert_eq!(sized_to_count.audio_base(), 104);
    assert_ne!(sized_to_count.audio_base(), before.audio_base());
}

#[test]
fn special_id_inside_the_text_region_is_rejected() {
    let codec = CodecVocab::new(1, 8).expect("valid codec");
    let mut specials = BTreeMap::new();
    specials.insert(SpecialToken::EndOfAudio, 9u32);
    let err = SpeechVocab::new(10, 4, specials, codec).expect_err("must reject text-region id");
    let msg = err.to_string();
    assert!(msg.contains("EndOfAudio"), "{msg}");
    assert!(msg.contains('9'), "{msg}");
}

#[test]
fn special_id_at_or_past_audio_base_is_rejected() {
    let codec = CodecVocab::new(1, 8).expect("valid codec");
    for id in [14u32, 15, 100] {
        let mut specials = BTreeMap::new();
        specials.insert(SpecialToken::StartOfAudio, id);
        let err = SpeechVocab::new(10, 4, specials, codec.clone())
            .expect_err("must reject audio-region id");
        let msg = err.to_string();
        assert!(msg.contains("StartOfAudio"), "{msg}");
        assert!(msg.contains(&id.to_string()), "{msg}");
    }
    // The last reserved id is still legal.
    let mut ok = BTreeMap::new();
    ok.insert(SpecialToken::StartOfAudio, 13u32);
    assert!(SpeechVocab::new(10, 4, ok, codec).is_ok());
}

#[test]
fn duplicate_special_ids_are_rejected() {
    let codec = CodecVocab::new(1, 8).expect("valid codec");
    let mut specials = BTreeMap::new();
    specials.insert(SpecialToken::StartOfAudio, 11u32);
    specials.insert(SpecialToken::EndOfAudio, 11u32);
    let err = SpeechVocab::new(10, 4, specials, codec).expect_err("must reject duplicate id");
    assert!(err.to_string().contains("shares id 11"), "{err}");
}

#[test]
fn control_region_smaller_than_the_special_count_is_rejected() {
    let codec = CodecVocab::new(1, 8).expect("valid codec");
    let err = SpeechVocab::with_sequential_specials(10, 3, &all_specials(), codec.clone())
        .expect_err("must reject undersized region");
    assert!(err.to_string().contains("cannot hold 4"), "{err}");
    assert!(SpeechVocab::with_sequential_specials(10, 0, &[], codec).is_ok());
}

#[test]
fn serde_round_trip_reproduces_an_identical_layout() {
    let v = neucodec_vocab();
    let json = serde_json::to_string(&v).expect("serialize layout");
    let back: SpeechVocab = serde_json::from_str(&json).expect("deserialize layout");
    assert_eq!(back, v);
    assert_eq!(back.audio_base(), v.audio_base());
    assert_eq!(
        back.special_id(SpecialToken::EndOfAudio),
        v.special_id(SpecialToken::EndOfAudio)
    );

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

#[test]
fn frame_round_trips_for_flat_multi_codebook() {
    let v = snac_flat_vocab();
    assert_eq!(v.codec().codes_per_frame(), 3);
    assert_eq!(v.codec().frame_codebooks(), vec![0, 1, 2]);

    let codes = vec![0, 2047, 4095];
    let ids = v.encode_frame(&codes).expect("encode frame");
    assert_eq!(ids.len(), 3);
    assert_eq!(v.decode_frame(&ids).expect("decode frame"), codes);
}

#[test]
fn frame_round_trips_for_interleaved_codebooks() {
    let v = snac_interleaved_vocab();
    assert_eq!(v.codec().codes_per_frame(), 7);
    assert_eq!(v.codec().frame_codebooks(), vec![0, 1, 1, 2, 2, 2, 2]);
    assert_eq!(v.codec().codes_of_codebook(1), Some(2));
    assert_eq!(v.codec().codes_of_codebook(3), None);

    let codes = vec![1, 10, 11, 100, 101, 102, 4095];
    let ids = v.encode_frame(&codes).expect("encode frame");
    assert_eq!(v.decode_frame(&ids).expect("decode frame"), codes);
    // Position 1 and 2 both come from codebook 1, so their ids share its range.
    assert_eq!(v.decode_audio_token(ids[1]), Some((1, 10)));
    assert_eq!(v.decode_audio_token(ids[2]), Some((1, 11)));
}

#[test]
fn decode_frame_rejects_codebook_position_mismatch() {
    let v = snac_flat_vocab();
    let ids = v.encode_frame(&[5, 6, 7]).expect("encode frame");
    // Model emitted a codebook-2 token where codebook 0 belongs.
    let wrong = vec![v.audio_token(2, 5).expect("cb2 token"), ids[1], ids[2]];
    let err = v.decode_frame(&wrong).expect_err("must reject mismatch");
    assert!(err.to_string().contains("expected codebook 0"));

    // Swapping two valid ids is caught too.
    let swapped = vec![ids[1], ids[0], ids[2]];
    assert!(v.decode_frame(&swapped).is_err());
}

#[test]
fn decode_frame_rejects_non_audio_ids() {
    let v = snac_flat_vocab();
    let ids = v.encode_frame(&[5, 6, 7]).expect("encode frame");
    let text = vec![0u32, ids[1], ids[2]];
    let err = v.decode_frame(&text).expect_err("must reject text id");
    assert!(err.to_string().contains("not an audio token"));

    let special = vec![
        v.special_id(SpecialToken::EndOfAudio).expect("eoa"),
        ids[1],
        ids[2],
    ];
    assert!(v.decode_frame(&special).is_err());
}

#[test]
fn frame_helpers_reject_wrong_length() {
    let v = snac_flat_vocab();
    assert!(v.encode_frame(&[]).is_err());
    assert!(v.encode_frame(&[1, 2]).is_err());
    assert!(v.encode_frame(&[1, 2, 3, 4]).is_err());
    assert!(v.decode_frame(&[]).is_err());
    assert!(v.decode_frame(&[0, 1]).is_err());
}

#[test]
fn encode_frame_reports_out_of_range_codes() {
    let v = snac_flat_vocab();
    assert!(v.encode_frame(&[0, 4096, 0]).is_err());
    assert!(v.encode_frame(&[0, 0, usize::MAX]).is_err());
}

#[test]
fn classification_is_exhaustive_and_disjoint() {
    let codec = CodecVocab::new(2, 5).expect("valid codec");
    let v = SpeechVocab::with_sequential_specials(7, CONTROL_REGION, &all_specials(), codec)
        .expect("valid vocab");
    assert_eq!(v.total_size(), 7 + 4 + 10);

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
    // A region of 8 with 4 defined tokens leaves 4 spare rows: control, not audio.
    let codec = CodecVocab::new(1, 8).expect("valid codec");
    let v =
        SpeechVocab::with_sequential_specials(10, 8, &all_specials(), codec).expect("valid vocab");
    for id in 10..18u32 {
        assert!(v.is_special(id), "id {id} must be control");
        assert!(!v.is_audio(id));
    }
    assert_eq!(v.special_id(SpecialToken::EndOfAudio), Some(13));
    assert_eq!(v.audio_base(), 18);
}

#[test]
fn codec_construction_validates_sizes() {
    assert!(CodecVocab::new(0, 4096).is_err());
    assert!(CodecVocab::new(1, 0).is_err());
    assert!(CodecVocab::with_frame_layout(4096, vec![]).is_err());
    assert!(CodecVocab::with_frame_layout(4096, vec![1, 0, 1]).is_err());
    // A codebook span wider than the u32 id space is rejected.
    assert!(CodecVocab::new(2, usize::MAX / 2).is_err());
    assert!(CodecVocab::new(1, u32::MAX as usize + 1).is_err());
    assert!(CodecVocab::new(1, u32::MAX as usize).is_ok());
}

#[test]
fn speech_vocab_construction_validates_inputs() {
    let codec = CodecVocab::new(1, 8).expect("valid codec");
    assert!(
        SpeechVocab::with_sequential_specials(0, CONTROL_REGION, &all_specials(), codec.clone())
            .is_err()
    );
    assert!(
        SpeechVocab::with_sequential_specials(
            10,
            CONTROL_REGION,
            &[SpecialToken::EndOfAudio, SpecialToken::EndOfAudio],
            codec.clone(),
        )
        .is_err()
    );
    // Empty special map is legal: some setups reuse the base tokenizer's ids.
    let v = SpeechVocab::new(10, 0, BTreeMap::new(), codec).expect("valid vocab");
    assert_eq!(v.audio_base(), 10);
    assert_eq!(v.special_id(SpecialToken::EndOfAudio), None);

    // Total that overflows u32 is rejected rather than wrapping.
    let big = CodecVocab::new(1, u32::MAX as usize).expect("valid codec");
    assert!(SpeechVocab::new(1, 0, BTreeMap::new(), big).is_err());
}
