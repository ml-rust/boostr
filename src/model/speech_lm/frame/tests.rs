//! Tests for per-frame conversion between codec codes and flat vocabulary ids.

use crate::model::speech_lm::{CodecVocab, SpecialToken, SpeechVocab};

/// Control region large enough for the whole default table.
const SMALL_REGION: usize = 16;

fn all_specials() -> Vec<SpecialToken> {
    crate::model::speech_lm::ALL_SPECIAL_TOKENS.to_vec()
}

/// Flat residual codec: 3 codebooks x 4096, one code each per frame.
fn snac_flat_vocab() -> SpeechVocab {
    let codec = CodecVocab::new(3, 4096).expect("valid codec");
    SpeechVocab::with_sequential_specials(1000, SMALL_REGION, &all_specials(), codec)
        .expect("valid vocab")
}

/// Interleaved hierarchy: codebook 0 once, 1 twice, 2 four times per frame.
fn snac_interleaved_vocab() -> SpeechVocab {
    let codec = CodecVocab::with_frame_layout(4096, vec![1, 2, 4]).expect("valid codec");
    SpeechVocab::with_sequential_specials(1000, SMALL_REGION, &all_specials(), codec)
        .expect("valid vocab")
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
        v.special_id(SpecialToken::SpeechEnd).expect("speech end"),
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
