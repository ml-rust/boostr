//! Generic Gated DeltaNet implementation — same algorithm for every backend.

pub mod chunk;
pub mod common;
pub mod step;

pub use chunk::gdn_chunk_prefill_impl;
pub use common::{GdnDims, check_gdn_shapes};
pub use step::gdn_step_impl;
