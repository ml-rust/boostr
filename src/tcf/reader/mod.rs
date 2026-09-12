//! `TcfFile`: the TCF reader. `FORMAT.md` Section 4, Section 4.1,
//! Section 5, Section 6, Section 15, Section 16, Section 17.
//!
//! # The directory-only guarantee, by construction
//!
//! Section 16 requires a placement planner to produce a complete plan
//! without touching a tensor payload page. [`TcfFile::open`] therefore never
//! reads a byte at or past `data_off`.
//!
//! That is structural here, not a convention: `open` splits the caller's
//! slice once, at `data_off`, and every later step of `open` reads the
//! `directory` half. The `data` half is reachable only from
//! [`TcfFile::payload`] and [`TcfFile::verify_tensor`]. A payload page is
//! mapped in when a caller asks for that tensor's bytes, never before.
//!
//! Splitting a slice reads nothing, so this holds for a memory-mapped file
//! as well as an in-memory one. Mapping is the host's job: this reader
//! borrows a slice and owns none of it.
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
