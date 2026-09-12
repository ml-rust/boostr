//! The TCF encoding registry: raw dtypes, GGML block encodings, and their
//! unified `Encoding` identifier.

pub mod block;
pub mod raw;
pub mod registry;

pub use block::BlockEncoding;
pub use raw::RawEncoding;
pub use registry::Encoding;
