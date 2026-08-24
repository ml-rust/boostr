//! Tests for flattening a speech record into a token stream and back.

use crate::model::speech_lm::{
    ALL_SPECIAL_TOKENS, CodecVocab, SpecialToken, SpeechRecord, SpeechVocab,
    pack::{
        OwnedSpeechRecord, pack_record, pack_records, pack_records_padded, unpack_record,
        unpack_records,
    },
};

const TEXT_VOCAB: usize = 1000;
const REGION: usize = 16;

/// Flat residual codec: 3 codebooks x 4096, one code each per frame.
fn vocab() -> SpeechVocab {
    let codec = CodecVocab::new(3, 4096).expect("valid codec");
    SpeechVocab::with_sequential_specials(TEXT_VOCAB, REGION, &ALL_SPECIAL_TOKENS, codec)
        .expect("valid vocab")
}

/// Interleaved hierarchy: codebook 0 once, 1 twice, 2 four times per frame.
fn interleaved_vocab() -> SpeechVocab {
    let codec = CodecVocab::with_frame_layout(4096, vec![1, 2, 4]).expect("valid codec");
    SpeechVocab::with_sequential_specials(TEXT_VOCAB, REGION, &ALL_SPECIAL_TOKENS, codec)
        .expect("valid vocab")
}

fn sid(v: &SpeechVocab, tok: SpecialToken) -> u32 {
    v.special_id(tok).expect("special defined")
}

fn frames() -> Vec<Vec<usize>> {
    vec![vec![0, 1, 2], vec![4095, 100, 7]]
}

fn owned(speaker: Option<&[u32]>, style: Option<&[u32]>, text: &[u32]) -> OwnedSpeechRecord {
    OwnedSpeechRecord {
        speaker: speaker.map(|s| s.to_vec()),
        style: style.map(|s| s.to_vec()),
        text: text.to_vec(),
        frames: frames(),
    }
}

#[test]
fn round_trips_with_speaker_and_style() {
    let v = vocab();
    let record = owned(Some(&[7, 8]), Some(&[9]), &[10, 11, 12]);
    let packed = pack_record(&v, &record.as_record()).unwrap();
    let (back, used) = unpack_record(&v, &packed).unwrap();
    assert_eq!(used, packed.len());
    assert_eq!(back, record);
}

#[test]
fn round_trips_without_speaker_or_style() {
    let v = vocab();
    let record = owned(None, None, &[10, 11, 12]);
    let packed = pack_record(&v, &record.as_record()).unwrap();
    // No speaker/style delimiters at all.
    assert_eq!(packed[0], sid(&v, SpecialToken::SpeechText));
    assert!(!packed.contains(&sid(&v, SpecialToken::Speaker)));
    assert!(!packed.contains(&sid(&v, SpecialToken::Style)));
    let (back, used) = unpack_record(&v, &packed).unwrap();
    assert_eq!(used, packed.len());
    assert_eq!(back, record);
}

#[test]
fn round_trips_speaker_only_and_style_only() {
    let v = vocab();
    for record in [
        owned(Some(&[7]), None, &[10]),
        owned(None, Some(&[9, 9]), &[10]),
    ] {
        let packed = pack_record(&v, &record.as_record()).unwrap();
        let (back, _) = unpack_record(&v, &packed).unwrap();
        assert_eq!(back, record);
    }
}

#[test]
fn round_trips_an_interleaved_codec() {
    let v = interleaved_vocab();
    let record = OwnedSpeechRecord {
        speaker: Some(vec![1]),
        style: None,
        text: vec![2, 3],
        frames: vec![vec![0, 1, 2, 3, 4, 5, 6], vec![9, 8, 7, 6, 5, 4, 3]],
    };
    let packed = pack_record(&v, &record.as_record()).unwrap();
    let (back, _) = unpack_record(&v, &packed).unwrap();
    assert_eq!(back, record);
}

#[test]
fn round_trips_an_empty_record() {
    let v = vocab();
    let record = OwnedSpeechRecord::default();
    let packed = pack_record(&v, &record.as_record()).unwrap();
    assert_eq!(packed.len(), 4);
    let (back, _) = unpack_record(&v, &packed).unwrap();
    assert_eq!(back, record);
}

#[test]
fn emitted_layout_is_exactly_the_documented_order() {
    let v = vocab();
    let record = owned(Some(&[7]), Some(&[9]), &[10]);
    let packed = pack_record(&v, &record.as_record()).unwrap();
    let f = v.encode_frame(&[0, 1, 2]).unwrap();
    let g = v.encode_frame(&[4095, 100, 7]).unwrap();
    let mut want = vec![
        sid(&v, SpecialToken::Speaker),
        7,
        sid(&v, SpecialToken::SpeakerEnd),
        sid(&v, SpecialToken::Style),
        9,
        sid(&v, SpecialToken::StyleEnd),
        sid(&v, SpecialToken::SpeechText),
        10,
        sid(&v, SpecialToken::SpeechTextEnd),
        sid(&v, SpecialToken::Speech),
    ];
    want.extend_from_slice(&f);
    want.extend_from_slice(&g);
    want.push(sid(&v, SpecialToken::SpeechEnd));
    assert_eq!(packed, want);
}

#[test]
fn audio_span_is_exactly_what_is_audio_accepts() {
    let v = vocab();
    let record = owned(Some(&[7, 8]), Some(&[9]), &[10, 11]);
    let packed = pack_record(&v, &record.as_record()).unwrap();

    let start = packed
        .iter()
        .position(|id| *id == sid(&v, SpecialToken::Speech))
        .unwrap();
    let end = packed
        .iter()
        .position(|id| *id == sid(&v, SpecialToken::SpeechEnd))
        .unwrap();

    // Everything strictly between the delimiters is audio, and nothing else is.
    assert!(packed[start + 1..end].iter().all(|id| v.is_audio(*id)));
    assert!(packed[..=start].iter().all(|id| !v.is_audio(*id)));
    assert!(packed[end..].iter().all(|id| !v.is_audio(*id)));
    assert_eq!(end - start - 1, 2 * v.codec().codes_per_frame());
}

#[test]
fn text_spans_are_exactly_what_is_text_accepts() {
    let v = vocab();
    let record = owned(Some(&[7, 8]), Some(&[9]), &[10, 11]);
    let packed = pack_record(&v, &record.as_record()).unwrap();

    // Every non-delimiter id before the audio section is text; every delimiter is
    // special; nothing is both.
    let speech = packed
        .iter()
        .position(|id| *id == sid(&v, SpecialToken::Speech))
        .unwrap();
    let text_ids: Vec<u32> = packed[..speech]
        .iter()
        .copied()
        .filter(|id| !v.is_special(*id))
        .collect();
    assert_eq!(text_ids, vec![7, 8, 9, 10, 11]);
    assert!(text_ids.iter().all(|id| v.is_text(*id)));
    assert!(packed.iter().all(|id| !(v.is_text(*id) && v.is_audio(*id))));
}

#[test]
fn rejects_an_audio_id_supplied_as_text() {
    let v = vocab();
    let audio = v.audio_token(1, 5).unwrap();
    let frames = frames();
    let record = SpeechRecord {
        speaker: None,
        style: None,
        text: &[10, audio],
        frames: &frames,
    };
    let err = pack_record(&v, &record).unwrap_err().to_string();
    assert!(err.contains(&audio.to_string()), "{err}");
    assert!(err.contains("audio id"), "{err}");
    assert!(err.contains("text"), "{err}");
}

#[test]
fn rejects_a_control_id_supplied_as_speaker() {
    let v = vocab();
    let pad = sid(&v, SpecialToken::SpeechPad);
    let frames = frames();
    let record = SpeechRecord {
        speaker: Some(&[pad]),
        style: None,
        text: &[10],
        frames: &frames,
    };
    let err = pack_record(&v, &record).unwrap_err().to_string();
    assert!(err.contains(&pad.to_string()), "{err}");
    assert!(err.contains("speaker"), "{err}");
}

#[test]
fn rejects_an_out_of_range_code() {
    let v = vocab();
    let frames = vec![vec![0, 1, 4096]];
    let record = SpeechRecord {
        speaker: None,
        style: None,
        text: &[10],
        frames: &frames,
    };
    let err = pack_record(&v, &record).unwrap_err().to_string();
    assert!(err.contains("4096"), "{err}");
    assert!(err.contains("frame 0"), "{err}");
}

#[test]
fn rejects_a_wrong_length_frame() {
    let v = vocab();
    let frames = vec![vec![0, 1]];
    let record = SpeechRecord {
        speaker: None,
        style: None,
        text: &[10],
        frames: &frames,
    };
    let err = pack_record(&v, &record).unwrap_err().to_string();
    assert!(err.contains("expected 3 codes per frame"), "{err}");
}

#[test]
fn rejects_a_vocabulary_missing_a_required_control_token() {
    let codec = CodecVocab::new(3, 4096).unwrap();
    // Every token except the audio terminator.
    let partial: Vec<SpecialToken> = ALL_SPECIAL_TOKENS
        .iter()
        .copied()
        .filter(|t| *t != SpecialToken::SpeechEnd)
        .collect();
    let v = SpeechVocab::with_sequential_specials(TEXT_VOCAB, REGION, &partial, codec).unwrap();
    let frames = frames();
    let record = SpeechRecord {
        speaker: None,
        style: None,
        text: &[10],
        frames: &frames,
    };
    let err = pack_record(&v, &record).unwrap_err().to_string();
    assert!(err.contains("SpeechEnd"), "{err}");
}

#[test]
fn multi_record_packing_concatenates_without_gaps() {
    let v = vocab();
    let a = owned(Some(&[7]), None, &[10, 11]);
    let b = owned(None, Some(&[9]), &[12]);
    let records = [a.as_record(), b.as_record()];

    let packed = pack_records(&v, &records).unwrap();
    let pa = pack_record(&v, &a.as_record()).unwrap();
    let pb = pack_record(&v, &b.as_record()).unwrap();
    let mut want = pa.clone();
    want.extend_from_slice(&pb);
    assert_eq!(packed, want);

    // Nothing between the records: the first id after record a is whatever opens
    // record b. Assert against `pb[0]` rather than a fixed delimiter — which
    // section opens a record depends on whether speaker/style are present, and
    // record b carries a style, so it opens with `Style`, not `SpeechText`.
    assert_eq!(packed[pa.len()], pb[0]);
    assert_eq!(pb[0], sid(&v, SpecialToken::Style));
    assert_eq!(unpack_records(&v, &packed).unwrap(), vec![a, b]);
}

#[test]
fn padding_pads_to_the_requested_multiple_with_speech_pad() {
    let v = vocab();
    let a = owned(Some(&[7]), None, &[10, 11]);
    let records = [a.as_record()];
    let unpadded = pack_records(&v, &records).unwrap();

    let window = 64;
    let padded = pack_records_padded(&v, &records, window).unwrap();
    assert_eq!(padded.len() % window, 0);
    assert!(padded.len() >= unpadded.len());
    assert!(padded.len() - unpadded.len() < window);
    assert_eq!(&padded[..unpadded.len()], &unpadded[..]);
    let pad = sid(&v, SpecialToken::SpeechPad);
    assert!(padded[unpadded.len()..].iter().all(|id| *id == pad));

    // Padding is never sampled back out, and unpacking ignores the tail.
    assert!(v.sampling_forbidden_ids().contains(&pad));
    assert_eq!(unpack_records(&v, &padded).unwrap(), vec![a]);
}

#[test]
fn padding_is_a_no_op_when_already_aligned() {
    let v = vocab();
    let a = owned(None, None, &[10]);
    let records = [a.as_record()];
    let unpadded = pack_records(&v, &records).unwrap();
    let padded = pack_records_padded(&v, &records, unpadded.len()).unwrap();
    assert_eq!(padded, unpadded);
}

#[test]
fn padding_rejects_a_zero_window() {
    let v = vocab();
    let err = pack_records_padded(&v, &[], 0).unwrap_err().to_string();
    assert!(err.contains("pad_to_multiple"), "{err}");
}

#[test]
fn unpacking_rejects_a_truncated_stream() {
    let v = vocab();
    let a = owned(Some(&[7]), None, &[10]);
    let packed = pack_record(&v, &a.as_record()).unwrap();
    let err = unpack_record(&v, &packed[..packed.len() - 1])
        .unwrap_err()
        .to_string();
    assert!(err.contains("audio section"), "{err}");
}

#[test]
fn unpacking_rejects_a_half_frame() {
    let v = vocab();
    let a = owned(None, None, &[10]);
    let mut packed = pack_record(&v, &a.as_record()).unwrap();
    // Drop one audio id, leaving a partial frame before the terminator.
    let end = packed.len() - 1;
    packed.remove(end - 1);
    let err = unpack_record(&v, &packed).unwrap_err().to_string();
    assert!(err.contains("whole number"), "{err}");
}

#[test]
fn unpacking_rejects_data_after_padding_starts() {
    let v = vocab();
    let a = owned(None, None, &[10]);
    let mut packed = pack_record(&v, &a.as_record()).unwrap();
    packed.push(sid(&v, SpecialToken::SpeechPad));
    packed.push(10);
    let err = unpack_records(&v, &packed).unwrap_err().to_string();
    assert!(err.contains("padding"), "{err}");
}
