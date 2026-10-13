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
//! [`super::special`]. `build` documents how each future addition lands.
//!
//! Pure logic: no tensors, no `Runtime`, no device code. Testable without weights.
//!
//! - `build`: the [`SpeechVocab`] type and its constructors
//! - `regions`: the region accessors, id classification, and audio id arithmetic

mod build;
mod regions;

pub use build::SpeechVocab;
