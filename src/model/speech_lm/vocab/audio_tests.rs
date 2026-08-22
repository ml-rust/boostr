//! Tests for the audio region of the flat layout: id arithmetic and boundaries.

use super::*;

/// Flat residual codec on a toy text vocabulary: 3 codebooks x 4096.
fn snac_flat_vocab() -> SpeechVocab {
    let codec = CodecVocab::new(3, 4096).expect("valid codec");
    SpeechVocab::with_sequential_specials(1000, 16, &ALL_SPECIAL_TOKENS, codec)
        .expect("valid vocab")
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
