//! CUDA graph decode implementation, gated on `cfg(feature = "cuda")` by the
//! parent module.

mod argmax;
mod attention_step;
mod decode_graph;
mod device_scalars;
mod mrope_scalars;
mod paged_decode_graph;
mod stable_copy;

pub use argmax::{argmax_to_buf, batch_argmax_to_buf};
pub use attention_step::insert_and_decode_attention;
pub use decode_graph::DecodeGraph;
pub use device_scalars::DeviceScalars;
pub use mrope_scalars::MropeScalars;
pub use paged_decode_graph::PagedDecodeGraph;
pub use stable_copy::copy_into_stable;
