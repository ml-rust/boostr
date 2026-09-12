//! Reading records back out of the token stream: [`unpack_record`],
//! [`unpack_records`].

use crate::error::{Error, Result};

use super::super::special::SpecialToken;
use super::super::vocab::SpeechVocab;
use super::encode::special;
use super::record::OwnedSpeechRecord;

/// Read ids up to `close_id`, requiring every one before it to be text.
fn take_text_body(
    vocab: &SpeechVocab,
    ids: &[u32],
    pos: &mut usize,
    close_id: u32,
    field: &'static str,
) -> Result<Vec<u32>> {
    let mut body = Vec::new();
    loop {
        let Some(id) = ids.get(*pos).copied() else {
            return Err(Error::InvalidArgument {
                arg: "ids",
                reason: format!(
                    "stream ends inside the {field} section: no closing id {close_id} found"
                ),
            });
        };
        *pos += 1;
        if id == close_id {
            return Ok(body);
        }
        if !vocab.is_text(id) {
            return Err(Error::InvalidArgument {
                arg: "ids",
                reason: format!(
                    "id {id} at position {} inside the {field} section is not a text id; \
                     expected a text id in [0, {}) or the closing id {close_id}",
                    *pos - 1,
                    vocab.text_vocab_size()
                ),
            });
        }
        body.push(id);
    }
}

/// Read an optional `open ... close` text section, if `open` is at `pos`.
fn take_optional_section(
    vocab: &SpeechVocab,
    ids: &[u32],
    pos: &mut usize,
    open: SpecialToken,
    close: SpecialToken,
    field: &'static str,
) -> Result<Option<Vec<u32>>> {
    let Some(open_id) = vocab.special_id(open) else {
        return Ok(None);
    };
    if ids.get(*pos).copied() != Some(open_id) {
        return Ok(None);
    }
    *pos += 1;
    let close_id = special(vocab, close)?;
    Ok(Some(take_text_body(vocab, ids, pos, close_id, field)?))
}

/// Read a required `open ... close` text section at `pos`.
fn take_required_section(
    vocab: &SpeechVocab,
    ids: &[u32],
    pos: &mut usize,
    open: SpecialToken,
    close: SpecialToken,
    field: &'static str,
) -> Result<Vec<u32>> {
    let open_id = special(vocab, open)?;
    match ids.get(*pos).copied() {
        Some(id) if id == open_id => *pos += 1,
        Some(id) => {
            return Err(Error::InvalidArgument {
                arg: "ids",
                reason: format!(
                    "expected the {field} opening id {open_id} at position {pos}, found id {id}"
                ),
            });
        }
        None => {
            return Err(Error::InvalidArgument {
                arg: "ids",
                reason: format!(
                    "stream ends at position {pos}, expected the {field} opening id {open_id}"
                ),
            });
        }
    }
    let close_id = special(vocab, close)?;
    take_text_body(vocab, ids, pos, close_id, field)
}

/// Read the audio section and decode it back into frames.
fn take_frames(vocab: &SpeechVocab, ids: &[u32], pos: &mut usize) -> Result<Vec<Vec<usize>>> {
    let open_id = special(vocab, SpecialToken::Speech)?;
    match ids.get(*pos).copied() {
        Some(id) if id == open_id => *pos += 1,
        Some(id) => {
            return Err(Error::InvalidArgument {
                arg: "ids",
                reason: format!(
                    "expected the audio opening id {open_id} at position {pos}, found id {id}"
                ),
            });
        }
        None => {
            return Err(Error::InvalidArgument {
                arg: "ids",
                reason: format!(
                    "stream ends at position {pos}, expected the audio opening id {open_id}"
                ),
            });
        }
    }
    let close_id = special(vocab, SpecialToken::SpeechEnd)?;

    let mut audio = Vec::new();
    loop {
        let Some(id) = ids.get(*pos).copied() else {
            return Err(Error::InvalidArgument {
                arg: "ids",
                reason: format!(
                    "stream ends inside the audio section: no closing id {close_id} found"
                ),
            });
        };
        *pos += 1;
        if id == close_id {
            break;
        }
        if !vocab.is_audio(id) {
            return Err(Error::InvalidArgument {
                arg: "ids",
                reason: format!(
                    "id {id} at position {} inside the audio section is not an audio id; \
                     expected an audio id in [{}, {}) or the closing id {close_id}",
                    *pos - 1,
                    vocab.audio_base(),
                    vocab.total_size()
                ),
            });
        }
        audio.push(id);
    }

    let codes_per_frame = vocab.codec().codes_per_frame();
    if audio.len() % codes_per_frame != 0 {
        return Err(Error::InvalidArgument {
            arg: "ids",
            reason: format!(
                "audio section holds {} ids, which is not a whole number of {codes_per_frame}-code \
                 frames",
                audio.len()
            ),
        });
    }
    let mut frames = Vec::with_capacity(audio.len() / codes_per_frame);
    for (i, chunk) in audio.chunks(codes_per_frame).enumerate() {
        let codes = vocab
            .decode_frame(chunk)
            .map_err(|e| Error::InvalidArgument {
                arg: "ids",
                reason: format!("frame {i}: {e}"),
            })?;
        frames.push(codes);
    }
    Ok(frames)
}

/// Recover one record from the start of a packed stream.
///
/// Returns the record and how many ids it consumed, so a caller walking a packed
/// file knows where the next record begins.
pub fn unpack_record(vocab: &SpeechVocab, ids: &[u32]) -> Result<(OwnedSpeechRecord, usize)> {
    let mut pos = 0usize;
    let speaker = take_optional_section(
        vocab,
        ids,
        &mut pos,
        SpecialToken::Speaker,
        SpecialToken::SpeakerEnd,
        "speaker",
    )?;
    let style = take_optional_section(
        vocab,
        ids,
        &mut pos,
        SpecialToken::Style,
        SpecialToken::StyleEnd,
        "style",
    )?;
    let text = take_required_section(
        vocab,
        ids,
        &mut pos,
        SpecialToken::SpeechText,
        SpecialToken::SpeechTextEnd,
        "text",
    )?;
    let frames = take_frames(vocab, ids, &mut pos)?;
    Ok((
        OwnedSpeechRecord {
            speaker,
            style,
            text,
            frames,
        },
        pos,
    ))
}

/// Recover every record from a packed stream, ignoring a
/// [`SpecialToken::SpeechPad`] tail.
///
/// Padding is only accepted as a TAIL: a pad id found where a record should start
/// requires every remaining id to be padding too, so a corrupted or misaligned
/// file is reported rather than silently truncated.
pub fn unpack_records(vocab: &SpeechVocab, ids: &[u32]) -> Result<Vec<OwnedSpeechRecord>> {
    let pad = vocab.special_id(SpecialToken::SpeechPad);
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < ids.len() {
        if Some(ids[pos]) == pad {
            if let Some(bad) = ids[pos..].iter().position(|id| Some(*id) != pad) {
                return Err(Error::InvalidArgument {
                    arg: "ids",
                    reason: format!(
                        "padding starts at position {pos} but id {} at position {} is not padding",
                        ids[pos + bad],
                        pos + bad
                    ),
                });
            }
            break;
        }
        let (record, used) =
            unpack_record(vocab, &ids[pos..]).map_err(|e| Error::InvalidArgument {
                arg: "ids",
                reason: format!("record {} starting at position {pos}: {e}", out.len()),
            })?;
        pos += used;
        out.push(record);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::super::encode::tests::{interleaved_vocab, owned, sid, vocab};
    use super::super::pack_record;
    use super::*;

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
}
