//! The importance-matrix FILE: its layout, its writer, its reader, and the
//! check a consumer runs before trusting it.
//!
//! One definition, in the library, because `compressr` reads this file for
//! BOTH codecs (GGUF K-quants and TCF) and `compressr` depends on `boostr`.
//! A parser written a second time beside the consumer is the exact failure
//! this repo already records for block layouts: a writer checked against a
//! reader beside it proves nothing.
//!
//! `codec` documents the byte layout and the entry order. `matrix` documents
//! what a value means and why an absent tensor is not a zero entry.

mod codec;
mod matrix;

pub use matrix::{
    IMATRIX_HEADER_LEN, IMATRIX_MAGIC, IMATRIX_VERSION, ImportanceCheck, ImportanceEntry,
    ImportanceMatrix, MAX_NAME_LEN,
};
