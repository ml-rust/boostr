//! VoxCPM2 `feat_encoder` local encoder ("locenc"): per-frame patch
//! transformer producing one pooled embedding per frame.

pub mod attention;
pub mod config;
pub mod encoder;
pub mod layer;
pub mod loader;
pub mod mlp;

pub use attention::LocalEncoderAttention;
pub use config::LocalEncoderConfig;
pub use encoder::LocalEncoder;
pub use layer::LocalEncoderLayer;
pub use loader::DEFAULT_LOCAL_ENCODER_PREFIX;
pub use mlp::LocalEncoderMlp;
