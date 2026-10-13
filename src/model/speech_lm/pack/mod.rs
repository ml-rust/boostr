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
//! disagree. `encode` documents the emitted layout and what is validated.
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
