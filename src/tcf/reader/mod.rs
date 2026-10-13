//! `TcfFile`: the TCF reader. `FORMAT.md` Section 4, Section 4.1,
//! Section 5, Section 6, Section 15, Section 16, Section 17.
//!
//! `file` documents the Section 16 directory-only guarantee `open` makes
//! true by construction.
//!
//! - `file`: the [`TcfFile`] type, [`TcfFile::open`], and the accessors
//! - `header_checks`: the header-level checks `open` runs first
//! - `sections`: section ranges, record decoding, the string table
//! - `checks`: record digests and the per-tensor cross-record invariants
//! - `verify`: the Section 15 payload checks

mod checks;
mod file;
mod header_checks;
mod sections;
mod verify;

pub use file::TcfFile;
