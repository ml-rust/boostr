//! Generic Gated DeltaNet implementation — same algorithm for every backend.

pub mod chunk;
pub(crate) mod chunk_bwd;
mod chunk_layout;
mod chunk_mask;
mod chunk_terms;
pub mod common;
pub mod from_conv;
pub(crate) mod recurrent_bwd;
mod recurrent_replay;
pub mod step;
pub mod ut_transform;

pub use chunk::gdn_chunk_prefill_impl;
pub use chunk_bwd::gdn_chunk_backward_impl;
pub use common::{GdnConvDims, GdnDims, check_gdn_conv_shapes, check_gdn_shapes};
pub use from_conv::gdn_step_from_conv_impl;
pub use recurrent_bwd::gdn_recurrent_backward;
pub use step::gdn_step_impl;
