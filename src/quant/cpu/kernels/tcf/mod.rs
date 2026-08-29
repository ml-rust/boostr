//! CPU kernels for TCF native quantized payloads.
//!
//! [`dequant`] rebuilds a whole tensor as f32. [`stream`] decodes one bounded
//! tile range at a time. [`decode`] turns decoded tiles into f32 with the
//! vector element loop. [`matmul`] multiplies against a packed weight without
//! ever holding a full f32 copy of it.

mod decode;
mod dequant;
mod matmul;
mod stream;

pub use decode::dequantize_tiles_into;
pub use dequant::{dequant_tcf, unpack_tiles};
pub use matmul::{FUSED_TILE_CHUNK, tcf_matmul_f32};
pub use stream::unpack_tile_range;
