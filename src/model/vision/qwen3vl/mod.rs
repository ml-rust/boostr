//! Qwen3-VL vision tower loaded from an mmproj GGUF.
//!
//! `encoder` documents the encoder graph. `preprocess` documents the input
//! size, resampling and normalization.

pub mod bicubic;
mod block;
pub mod config;
mod embed;
pub mod encoder;
pub mod loader;
pub mod preprocess;
pub mod rope2d;

pub use bicubic::{RgbImage, resize_bicubic};
pub use config::Qwen3VlVisionConfig;
pub use encoder::Qwen3VlVision;
pub use loader::{load_qwen3vl_vision_from_reader, load_qwen3vl_vision_gguf};
pub use preprocess::{
    PreprocessedImage, decode_image, preprocess, preprocess_bytes, resize_pad_ceil, smart_resize,
};
pub use rope2d::{Rope2dTables, rope2d_tables};
