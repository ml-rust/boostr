//! CUDA graph decode implementation, gated on `cfg(feature = "cuda")` by the
//! parent module.

mod argmax;
mod decode_graph;
mod device_scalars;
mod paged_decode_graph;

pub use argmax::{argmax_to_buf, batch_argmax_to_buf};
pub use decode_graph::DecodeGraph;
pub use device_scalars::DeviceScalars;
pub use paged_decode_graph::PagedDecodeGraph;
