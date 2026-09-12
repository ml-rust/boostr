//! Turning one utterance into a flat token stream, and back.
//!
//! A speech LM trains and serves on ONE sequence of ids. The record it is trained
//! on — who speaks, in what style, what words, and the codec frames realising them
//! — has to be flattened into that sequence in an order every stage agrees on:
//! the trainer's loss mask, the sampler's stop condition, and the decoder that
//! turns emitted ids back into codec frames.
//!
//! This module is that flattening, and nothing else. It is pure integer logic over
//! [`SpeechVocab`](super::vocab::SpeechVocab) and
//! [`SpecialToken`](super::special::SpecialToken): no tensors, no `Runtime`, no
//! file IO, no codec. It lives here rather than in a trainer because every trainer and every
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
//! The order comes from the [`SpecialToken`](super::special::SpecialToken) docs:
//! conditioning first — identity, then manner, then the words — and audio last,
//! because generation reads everything before
//! [`SpecialToken::Speech`](super::special::SpecialToken::Speech) and emits
//! everything after it.
//! Speaker and style are PLAIN TEXT between delimiters, never per-speaker or
//! per-emotion ids; see [`super::special`] for why that is not negotiable.
//!
//! Audio ids come from
//! [`SpeechVocab::encode_frame`](super::vocab::SpeechVocab::encode_frame), one
//! call per frame, concatenated in frame order. Per-frame code ordering is that method's business
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
//!
//! - `record`: the [`SpeechRecord`] and [`OwnedSpeechRecord`] types
//! - `encode`: [`pack_record`], [`pack_records`], [`pack_records_padded`]
//! - `decode`: [`unpack_record`], [`unpack_records`]

mod decode;
mod encode;
mod record;

pub use decode::{unpack_record, unpack_records};
pub use encode::{pack_record, pack_records, pack_records_padded};
pub use record::{OwnedSpeechRecord, SpeechRecord};
