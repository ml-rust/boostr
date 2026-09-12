//! Flattening records into the token stream: [`pack_record`],
//! [`pack_records`], [`pack_records_padded`].

use crate::error::{Error, Result};

use super::super::special::SpecialToken;
use super::super::vocab::SpeechVocab;
use super::record::SpeechRecord;

/// Id of a control token, or a descriptive error naming what the vocabulary lacks.
pub(super) fn special(vocab: &SpeechVocab, tok: SpecialToken) -> Result<u32> {
    vocab.special_id(tok).ok_or_else(|| Error::InvalidArgument {
        arg: "vocab",
        reason: format!(
            "control token {tok:?} (\"{}\") has no id in this vocabulary, \
             so a record cannot be packed",
            tok.token_str()
        ),
    })
}

/// Reject any id in `ids` that is not a text id.
fn check_text(vocab: &SpeechVocab, ids: &[u32], field: &'static str) -> Result<()> {
    for (i, id) in ids.iter().enumerate() {
        if !vocab.is_text(*id) {
            let kind = if vocab.is_audio(*id) {
                "an audio id"
            } else if vocab.is_special(*id) {
                "a control id"
            } else {
                "outside the vocabulary"
            };
            return Err(Error::InvalidArgument {
                arg: field,
                reason: format!(
                    "id {id} at position {i} of {field} is {kind}, expected a text id in \
                     [0, {})",
                    vocab.text_vocab_size()
                ),
            });
        }
    }
    Ok(())
}

/// Emit `open ids close`, after checking every id is text.
fn push_text_section(
    out: &mut Vec<u32>,
    vocab: &SpeechVocab,
    open: SpecialToken,
    close: SpecialToken,
    ids: &[u32],
    field: &'static str,
) -> Result<()> {
    check_text(vocab, ids, field)?;
    out.push(special(vocab, open)?);
    out.extend_from_slice(ids);
    out.push(special(vocab, close)?);
    Ok(())
}

/// Flatten one record into the layout documented at [module level](super).
pub fn pack_record(vocab: &SpeechVocab, record: &SpeechRecord) -> Result<Vec<u32>> {
    let codes_per_frame = vocab.codec().codes_per_frame();
    let audio_len = record.frames.len() * codes_per_frame;
    let mut out = Vec::with_capacity(record.text.len() + audio_len + 8);

    if let Some(speaker) = record.speaker {
        push_text_section(
            &mut out,
            vocab,
            SpecialToken::Speaker,
            SpecialToken::SpeakerEnd,
            speaker,
            "speaker",
        )?;
    }
    if let Some(style) = record.style {
        push_text_section(
            &mut out,
            vocab,
            SpecialToken::Style,
            SpecialToken::StyleEnd,
            style,
            "style",
        )?;
    }
    push_text_section(
        &mut out,
        vocab,
        SpecialToken::SpeechText,
        SpecialToken::SpeechTextEnd,
        record.text,
        "text",
    )?;

    out.push(special(vocab, SpecialToken::Speech)?);
    for (i, frame) in record.frames.iter().enumerate() {
        let ids = vocab
            .encode_frame(frame)
            .map_err(|e| Error::InvalidArgument {
                arg: "frames",
                reason: format!("frame {i}: {e}"),
            })?;
        out.extend_from_slice(&ids);
    }
    out.push(special(vocab, SpecialToken::SpeechEnd)?);

    Ok(out)
}

/// Flatten records back to back, with no separator between them.
///
/// Record boundaries are already unambiguous: every record ends with
/// [`SpecialToken::SpeechEnd`] and the next begins with a delimiter, so an extra
/// separator token would be a second, redundant encoding of the same boundary.
pub fn pack_records(vocab: &SpeechVocab, records: &[SpeechRecord]) -> Result<Vec<u32>> {
    let mut out = Vec::new();
    for (i, record) in records.iter().enumerate() {
        let packed = pack_record(vocab, record).map_err(|e| Error::InvalidArgument {
            arg: "records",
            reason: format!("record {i}: {e}"),
        })?;
        out.extend_from_slice(&packed);
    }
    Ok(out)
}

/// [`pack_records`], then pad the tail with [`SpecialToken::SpeechPad`] up to a
/// multiple of `pad_to_multiple`.
///
/// # Why a separate function rather than an option on `pack_records`
///
/// Padding is a property of the FILE being written, not of the records. Exactly
/// one caller wants it — the writer of a training file whose loader reads fixed
/// windows — while every other caller (a server building a prompt, a test, a
/// corpus checker) would pass `None` forever. A separate entry point keeps that
/// `None` out of every call site and keeps the padding decision visible at the one
/// place it is actually made.
///
/// The pad token is [`SpecialToken::SpeechPad`], which exists for exactly this and
/// is the one control token
/// [`sampling_forbidden_ids`](SpeechVocab::sampling_forbidden_ids) always
/// suppresses, so padding can never be generated back out.
pub fn pack_records_padded(
    vocab: &SpeechVocab,
    records: &[SpeechRecord],
    pad_to_multiple: usize,
) -> Result<Vec<u32>> {
    if pad_to_multiple == 0 {
        return Err(Error::InvalidArgument {
            arg: "pad_to_multiple",
            reason: "window length must be at least 1".to_string(),
        });
    }
    let mut out = pack_records(vocab, records)?;
    let remainder = out.len() % pad_to_multiple;
    if remainder != 0 {
        let pad = special(vocab, SpecialToken::SpeechPad)?;
        out.resize(out.len() + (pad_to_multiple - remainder), pad);
    }
    Ok(out)
}

#[cfg(test)]
pub(super) mod tests {
    use super::super::{OwnedSpeechRecord, unpack_records};
    use super::*;
    use crate::model::speech_lm::{ALL_SPECIAL_TOKENS, CodecVocab};

    pub(in super::super) const TEXT_VOCAB: usize = 1000;
    pub(in super::super) const REGION: usize = 16;

    /// Flat residual codec: 3 codebooks x 4096, one code each per frame.
    pub(in super::super) fn vocab() -> SpeechVocab {
        let codec = CodecVocab::new(3, 4096).expect("valid codec");
        SpeechVocab::with_sequential_specials(TEXT_VOCAB, REGION, &ALL_SPECIAL_TOKENS, codec)
            .expect("valid vocab")
    }

    /// Interleaved hierarchy: codebook 0 once, 1 twice, 2 four times per frame.
    pub(in super::super) fn interleaved_vocab() -> SpeechVocab {
        let codec = CodecVocab::with_frame_layout(4096, vec![1, 2, 4]).expect("valid codec");
        SpeechVocab::with_sequential_specials(TEXT_VOCAB, REGION, &ALL_SPECIAL_TOKENS, codec)
            .expect("valid vocab")
    }

    pub(in super::super) fn sid(v: &SpeechVocab, tok: SpecialToken) -> u32 {
        v.special_id(tok).expect("special defined")
    }

    pub(in super::super) fn frames() -> Vec<Vec<usize>> {
        vec![vec![0, 1, 2], vec![4095, 100, 7]]
    }

    pub(in super::super) fn owned(
        speaker: Option<&[u32]>,
        style: Option<&[u32]>,
        text: &[u32],
    ) -> OwnedSpeechRecord {
        OwnedSpeechRecord {
            speaker: speaker.map(|s| s.to_vec()),
            style: style.map(|s| s.to_vec()),
            text: text.to_vec(),
            frames: frames(),
        }
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
}
