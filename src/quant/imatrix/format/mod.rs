//! The importance-matrix FILE: its layout, its writer, its reader, and the
//! check a consumer runs before trusting it.
//!
//! One definition, in the library, because `compressr` reads this file for
//! BOTH codecs (GGUF K-quants and TCF) and `compressr` depends on `boostr`.
//! A parser written a second time beside the consumer is the exact failure
//! this repo already records for block layouts: a writer checked against a
//! reader beside it proves nothing.
//!
//! # Layout
//!
//! Little-endian throughout. No padding, no alignment requirement: every
//! multi-byte field is read from a byte slice, so a file written on one
//! machine is read identically on another.
//!
//! Header, 32 bytes:
//!
//! | offset | size | field         | value                                    |
//! | ------ | ---- | ------------- | ---------------------------------------- |
//! | 0      | 8    | `magic`       | `BSTRIMTX`                               |
//! | 8      | 4    | `version`     | `1`                                      |
//! | 12     | 4    | `flags`       | `0` — reserved, a reader REJECTS nonzero |
//! | 16     | 8    | `entry_count` | entries that follow                      |
//! | 24     | 8    | `token_count` | tokens the whole run accumulated over    |
//!
//! Then `entry_count` entries, back to back, each:
//!
//! | size            | field         | meaning                                |
//! | --------------- | ------------- | -------------------------------------- |
//! | 4               | `name_len`    | length of `name` in bytes              |
//! | 8               | `in_features` | columns of the weight this describes   |
//! | 8               | `rows`        | activation rows summed into this entry |
//! | `name_len`      | `name`        | UTF-8 tensor name, no trailing NUL     |
//! | `in_features*4` | `sums`        | `f32` per column                       |
//!
//! # What a value means
//!
//! `sums[j]` is `sum over rows of x_j * x_j` — a SUM, never a mean. The
//! divisor is `rows`, carried beside it, so a consumer picks its own
//! normalization and two files collected over different corpus sizes stay
//! combinable. `token_count` is the run-level total, which lets a consumer
//! tell a 10-token collection from a 100k-token one at a glance.
//!
//! # Absent is not zero
//!
//! A tensor the run never exercised has NO entry. It is never written as a
//! zero vector: zero importance and "never measured" mean opposite things to
//! a quantizer, and only one of them is safe to act on.
//!
//! # Entry order
//!
//! Entries are written sorted by name, so two runs that collected the same
//! statistics produce byte-identical files regardless of the order the
//! collector observed the tensors in.

mod codec;
mod matrix;

pub use matrix::{
    IMATRIX_HEADER_LEN, IMATRIX_MAGIC, IMATRIX_VERSION, ImportanceCheck, ImportanceEntry,
    ImportanceMatrix, MAX_NAME_LEN,
};
