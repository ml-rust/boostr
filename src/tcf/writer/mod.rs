//! `TcfWriter`: the TCF file writer. `FORMAT.md` Section 4, Section 4.1,
//! Section 5, Section 5.3, Section 8, Section 14.4, Section 15.
//!
//! `finish` documents the three passes and the memory each entry point holds.
//! `records` documents what the writer owns. `payload` documents what it does
//! not own.
//!
//! # Where each pass lives
//!
//! `records` holds the [`TcfWriter`] struct, the record `add_*` and
//! [`TcfWriter::intern`] API, and the derived record digests; `finish`
//! holds the [`TcfWriter::finish`] entry points. The tensor entry points
//! are in `tensors`. Pass 1 and pass 3 are in `layout`; pass 2 is in
//! `payload`; [`TcfWriter::finish_streaming`] is in
//! [`crate::tcf::streaming`].

mod finish;
pub(crate) mod layout;
mod payload;
mod records;
mod tensors;

pub use payload::Payload;
pub use records::TcfWriter;

pub(crate) use records::LAYOUT_BOUNDS;
