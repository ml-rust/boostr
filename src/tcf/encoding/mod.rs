//! The TCF encoding registry: raw dtypes, GGML block encodings, and their
//! unified `Encoding` identifier.
//!
//! The payload encodings are `Encoding::Raw` (literal element types) and
//! `Encoding::Block` (GGML block streams, identified by `ggml_type`). TCF's
//! own tile encodings were removed after measurement found no advantage
//! over the GGML blocks at any bit width; their identifier range stays
//! unassigned so an old file is refused by name.

pub mod block;
pub mod raw;
pub mod registry;

pub use block::BlockEncoding;
pub use raw::RawEncoding;
pub use registry::Encoding;
