//! Turning one utterance into a flat token stream, and back.
//!
//! A speech LM trains and serves on ONE sequence of ids. The record it is trained
//! on — who speaks, in what style, what words, and the codec frames realising them
//! — has to be flattened into that sequence in an order every stage agrees on:
//! the trainer's loss mask, the sampler's stop condition, and the decoder that
//! turns emitted ids back into codec frames.
//!
//! This module is that flattening, and nothing else. It is pure integer logic over
//! [`SpeechVocab`] and [`SpecialToken`]: no tensors, no `Runtime`, no file IO, no
//! codec. It lives here rather than in a trainer because every trainer and every
//! server needs the same layout, and two implementations of it would silently
//! disagree.
//!
//! # Emitted layout
//!
//! ```text
//! [Speaker]     speaker ids     [SpeakerEnd]      omitted entirely when None
//! [Style]       style ids       [StyleEnd]        omitted entirely when None
//! [SpeechText]  text ids        [SpeechTextEnd]
//! [Speech]      audio ids       [SpeechEnd]
//! ```
//!
//! The order comes from the [`SpecialToken`] docs: conditioning first — identity,
//! then manner, then the words — and audio last, because generation reads
//! everything before [`SpecialToken::Speech`] and emits everything after it.
//! Speaker and style are PLAIN TEXT between delimiters, never per-speaker or
//! per-emotion ids; see [`super::special`] for why that is not negotiable.
//!
//! Audio ids come from [`SpeechVocab::encode_frame`], one call per frame,
//! concatenated in frame order. Per-frame code ordering is that method's business
//! and is never re-derived here.
//!
//! # What is validated
//!
//! Nothing about an input record is trusted. A text, speaker or style id that is
//! not actually a text id, a control token this vocabulary does not define, and a
//! frame `encode_frame` rejects all produce a descriptive `Err` naming the
//! offending value. A record that packs is a record whose every id lands in the
//! region its position claims, which is exactly the property a loss mask over
//! `is_audio` depends on.

use crate::error::{Error, Result};

use super::special::SpecialToken;
use super::vocab::SpeechVocab;

/// One utterance: who says it, optionally how, what the text is, and the codec
/// frames that realise it.
///
/// All text fields hold ids ALREADY produced by the base tokenizer. This type
/// does no tokenization; it only checks that what it is handed is text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeechRecord<'a> {
    /// Already-tokenised speaker NAME, as plain text ids. `None` omits the
    /// speaker section and both its delimiters.
    pub speaker: Option<&'a [u32]>,
    /// Already-tokenised free-form style description. `None` omits the style
    /// section and both its delimiters.
    pub style: Option<&'a [u32]>,
    /// Already-tokenised text to be spoken.
    pub text: &'a [u32],
    /// Per-frame codec codes; one inner `Vec` per frame, each of length
    /// [`CodecVocab::codes_per_frame`](super::codec::CodecVocab::codes_per_frame).
    pub frames: &'a [Vec<usize>],
}

/// A record recovered from a packed stream, owning its ids.
///
/// The borrowing [`SpeechRecord`] cannot be returned from unpacking: the text
/// sections are copied out of the stream and the frames are decoded into fresh
/// `Vec`s. Use [`OwnedSpeechRecord::as_record`] to feed one straight back into
/// [`pack_record`], which is how a packed corpus is checked rather than trusted.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct OwnedSpeechRecord {
    /// Speaker name ids, or `None` if the packed stream had no speaker section.
    pub speaker: Option<Vec<u32>>,
    /// Style description ids, or `None` if the packed stream had no style section.
    pub style: Option<Vec<u32>>,
    /// Text ids.
    pub text: Vec<u32>,
    /// Per-frame codec codes, in frame order.
    pub frames: Vec<Vec<usize>>,
}

impl OwnedSpeechRecord {
    /// Borrow this record as a [`SpeechRecord`], for re-packing.
    pub fn as_record(&self) -> SpeechRecord<'_> {
        SpeechRecord {
            speaker: self.speaker.as_deref(),
            style: self.style.as_deref(),
            text: &self.text,
            frames: &self.frames,
        }
    }
}

/// Id of a control token, or a descriptive error naming what the vocabulary lacks.
fn special(vocab: &SpeechVocab, tok: SpecialToken) -> Result<u32> {
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

/// Flatten one record into the layout documented at [module level](self).
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
mod tests;
