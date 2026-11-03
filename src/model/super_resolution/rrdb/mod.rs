//! RRDBNet at ×2: `RealESRGAN_x2plus` and the legacy-key `2x_PureVision`.

pub mod config;
pub mod dense_block;
pub mod keys;
pub mod metadata;
pub mod net;
pub mod rrdb_block;

pub use config::{RRDB_SUPPORTED_SCALE, RrdbConfig};
pub use dense_block::ResidualDenseBlock;
pub use keys::{
    LEGACY_ESRGAN_MARKER, is_legacy_esrgan, legacy_esrgan_renames, legacy_esrgan_target,
};
pub use metadata::check_esrgan_metadata;
pub use net::RrdbNet;
pub use rrdb_block::RrdbBlock;
