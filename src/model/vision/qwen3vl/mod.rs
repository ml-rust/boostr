//! Qwen3-VL vision tower loaded from an mmproj GGUF.
//!
//! # Encoder graph
//!
//! 1. The preprocessed `[1, 3, H, W]` image passes through two stride-16
//!    convolutions over the same pixels. Their sum is the patch token grid
//!    `[hidden, H/16, W/16]`.
//! 2. Patch tokens reorder into merge-block order: for each 2x2 block in
//!    raster order `(by, bx)`, the four patches `(dy, dx)` follow one
//!    another. Token `t` holds patch `(2*by + dy, 2*bx + dx)` with
//!    `t = ((by * W/32 + bx) * 2 + dy) * 2 + dx`.
//! 3. The learned 48x48 position table resamples bilinearly with aligned
//!    corners to the `(H/16, W/16)` grid, takes the same block order, and
//!    adds to the tokens together with the patch bias.
//! 4. Each of the 27 blocks runs `LN1 -> fused qkv -> 2D rope on q and k
//!    -> full softmax attention at scale 1/sqrt(72) -> out`, adds the
//!    residual, then `LN2 -> up -> gelu -> down` and adds the residual.
//!    The 2D rope rotates the first 18 pairs of each 72-wide head by the
//!    patch row and the next 18 by the patch column.
//! 5. `post_ln` normalizes every token. The four tokens of one block
//!    concatenate into a 4608-wide row, `mm.0 -> gelu -> mm.2` maps each
//!    row to 5120, and the rows come out in raster order over the
//!    `(H/32, W/32)` token grid.
//!
//! # Preprocessing
//!
//! [`preprocess`] picks a size on the 32-pixel grid between 8 and 4096
//! output tokens, scales the image with a fixed-point bicubic filter,
//! centers it on black, and normalizes bytes to `(v/255 - 0.5) / 0.5`.

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
