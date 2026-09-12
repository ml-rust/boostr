//! Flat token layout shared by text and neural-audio-codec tokens.
//!
//! A text-to-speech LM is a causal decoder over ONE vocabulary: it reads text ids
//! and emits audio-codec ids. That only works if both live in a single flat id
//! space with a layout every downstream stage agrees on — embedding resize, loss
//! masking, sampling constraints, and audio decode all index into it.
//!
//! The layout is EXPLICIT and SERIALIZABLE. Control ids are stored as absolute
//! numbers, never derived from a position in a list, and the control region has a
//! size the caller reserves up front. Both properties exist so a layout can be
//! written next to a checkpoint and checked back against it: a vocabulary whose
//! layout is not recorded cannot be reloaded correctly.
//!
//! Nothing here is codec-specific. Every size is derived from
//! [`CodecVocab`](super::codec::CodecVocab), so a single-codebook codec (NeuCodec: 1 codebook x 65_536 entries, 50 frames/sec)
//! and a residual/interleaved codec (SNAC: 3-4 codebooks x 4096 entries, several
//! codes per frame) are both expressible without touching this file.
//!
//! Control tokens themselves, and the evidence for the set that exists, live in
//! [`super::special`].
//!
//! # Extension path: how each future addition lands
//!
//! The region order below is chosen so growth does not move existing ids. State
//! plainly what each kind of extension costs.
//!
//! - **A new control token.** Takes the next FREE reserved slot inside the control
//!   region. `audio_base` is unchanged, every audio id keeps its embedding row,
//!   and checkpoints trained before the addition stay valid. This is free.
//! - **A richer codec** (more codebooks, or a bigger codebook). NOT free. It is a
//!   new [`SpeechVocab`] over a different [`CodecVocab`](super::codec::CodecVocab):
//!   every id at or after
//!   `audio_base` changes meaning, so it is a new layout and a NEW CHECKPOINT.
//!   There is no in-place upgrade — do not pretend otherwise.
//! - **A second modality** (say a video codec block). Appends AFTER the audio
//!   block. Text, control and audio ids are all untouched, so an audio-only
//!   checkpoint's rows stay valid and the new rows are appended to the embedding
//!   matrix.
//!
//! That last case is why control ids are reserved BEFORE audio rather than
//! appended after it. Appending a MODALITY after audio is fine — it is one more
//! contiguous block. Appending CONTROL tokens after audio, which Scicom's
//! checkpoint does, permanently fragments the layout: control ids end up split
//! across two disjoint ranges, and every mask, resize and range check has to carry
//! both forever.
//!
//! Pure logic: no tensors, no `Runtime`, no device code. Testable without weights.
//!
//! - `build`: the [`SpeechVocab`] type and its constructors
//! - `regions`: the region accessors, id classification, and audio id arithmetic

mod build;
mod regions;

pub use build::SpeechVocab;
