//! Error-rate metrics for ASR output and for the intelligibility leg of TTS eval.
//!
//! Two callers need this and neither should reimplement it:
//!
//! - **Corpus QC.** A transcript is training ground truth. Measuring an ASR's
//!   word error rate against a read script is how we learn whether an
//!   ASR-transcribed corpus is usable at all.
//! - **TTS eval.** Intelligibility of generated speech is ASR WER over that
//!   speech against the text it was asked to say.
//!
//! **CPU-only, no tensors.** Levenshtein over token sequences: two rows of
//! `O(hypothesis)` state, one pass per utterance.
//!
//! # Two tokenizers, one metric each
//!
//! WER is meaningless without saying how text was tokenized, because that
//! choice moves the number by several points. This module offers two, and a
//! caller states which one it used:
//!
//! - [`normalize()`] counts what is written. Hyphens become spaces and digits
//!   stay digits, so `ter-delete` is two tokens and `250` never matches
//!   `two hundred and fifty`. Word counts over a prompt need this.
//! - [`canonicalize`] counts what is spoken. Number words and digits collapse
//!   to one form and a standalone Malay affix joins the token after it, so
//!   two spellings of the same utterance score as a match.
//!
//! Scoring speech against a prompt uses [`canonicalize`]. Measuring how a
//! corpus is written uses [`normalize()`].

pub mod align;
pub mod canonical;
pub mod normalize;

pub use align::{
    ErrorRate, align, by_group, character_error_rate, grand_total, total, word_error_rate,
};
pub use canonical::{canonical_character_error_rate, canonical_word_error_rate, canonicalize};
pub use normalize::normalize;
