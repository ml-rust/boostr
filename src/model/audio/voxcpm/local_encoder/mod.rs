//! VoxCPM2 `feat_encoder` local encoder ("locenc"): per-frame patch
//! transformer producing one pooled embedding per frame. Its transformer
//! blocks are the shared bidirectional MiniCPM4 stack in
//! `crate::model::audio::voxcpm::bidirectional`.

pub mod config;
pub mod encoder;
pub mod loader;

pub use config::LocalEncoderConfig;
pub use encoder::LocalEncoder;
pub use loader::DEFAULT_LOCAL_ENCODER_PREFIX;
