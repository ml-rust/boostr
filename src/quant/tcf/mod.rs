//! TCF native quantized encodings, and the plane geometry a GPU kernel takes.
//!
//! `encoding` wraps `tcf-core`'s encoding identifier and answers every size
//! question through `tcf-core`'s own `QuantLayout`. `planes` turns that layout
//! into the byte offsets and geometry a device kernel needs, so no shader and
//! no `.cu` file holds a plane order or a plane size.

mod encoding;
#[cfg(any(feature = "cuda", feature = "wgpu", test))]
mod planes;

pub use encoding::TcfEncoding;

#[cfg(any(feature = "cuda", feature = "wgpu"))]
pub(crate) use planes::TcfPlanes;
#[cfg(feature = "wgpu")]
pub(crate) use planes::{
    SCALE_FORM_FLAT, SCALE_FORM_TWO_LEVEL_U6M6, SCALE_FORM_TWO_LEVEL_U8, TILE as TCF_TILE,
};
