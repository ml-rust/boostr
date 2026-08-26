//! VoxCPM2's shared bidirectional MiniCPM4 transformer block stack, reused
//! by `feat_encoder` (`local_encoder`) and the local DiT (`feat_decoder`).

pub mod attention;
pub mod layer;
pub(crate) mod loader;
pub mod mlp;

pub use attention::BidirectionalAttention;
pub use layer::BidirectionalLayer;
pub(crate) use loader::{BidirectionalLayerDims, load_bidirectional_layer};
pub use mlp::BidirectionalMlp;
