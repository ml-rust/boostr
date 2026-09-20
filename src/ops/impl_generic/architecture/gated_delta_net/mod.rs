//! Generic Gated DeltaNet implementation — same algorithm for every backend.

pub mod chunk;
pub mod common;
pub mod from_conv;
pub mod step;

pub use chunk::gdn_chunk_prefill_impl;
pub use common::{GdnConvDims, GdnDims, check_gdn_conv_shapes, check_gdn_shapes};
pub use from_conv::gdn_step_from_conv_impl;
pub use step::gdn_step_impl;
