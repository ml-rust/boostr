//! CPU kernels for TCF native quantized payloads.
//!
//! [`dequant`] rebuilds a whole tensor as f32. [`decode`] turns decoded tiles
//! into f32 with the vector element loop. [`matmul`] multiplies against a
//! packed weight without ever holding a full f32 copy of it.
//!
//! Both [`dequant`] and [`matmul`] walk a payload through `tcf-core`'s
//! `for_each_group`, which yields one quantization group at a time and keeps
//! every plane offset and bit position inside the codec. Neither materializes
//! an intermediate, so neither needs a bounded-range entry point of its own.

mod decode;
mod dequant;
mod matmul;

pub use decode::{dequantize_tiles_append, dequantize_tiles_into};
pub use dequant::{dequant_tcf, unpack_tiles};
pub use matmul::tcf_matmul_f32;
