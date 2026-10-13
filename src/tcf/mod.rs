//! TCF: GGUF's block-quantized payloads in a strict, self-describing
//! container. See `FORMAT.md` beside this module for the binary layout.
//!
//! What the container adds over a GGUF file holding the same bytes:
//! - an activation contract every tensor names, refused at open when it dangles and checked
//!   against the kernel at dispatch;
//! - a proof vector per quantized tensor that the READER's block decoder must reproduce, plus
//!   payload and directory digests;
//! - provenance: module records, a fallback reason on every tensor not at its module's preferred
//!   encoding, calibration and per-tensor sensitivity records when the allocation was measured;
//! - a directory computable from headers alone, so a placement planner never touches a payload
//!   page.
//!
//! `encoding` documents the payload encodings; `enums::contract` the block contract.

pub mod binary16;
pub mod consts;
pub mod digest;
pub mod encoding;
pub mod enums;
pub mod error;
pub mod flags;
pub mod proof;
pub mod reader;
pub mod record;
mod reexports;
pub mod streaming;
pub mod writer;

pub use reexports::*;
