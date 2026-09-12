//! SafeTensors file format parser and loader
//!
//! SafeTensors is a simple, safe format for storing tensors developed by HuggingFace.
//!
//! # Format
//!
//! ```text
//! [8 bytes] header_size (little-endian u64)
//! [header_size bytes] JSON header containing:
//!   - "__metadata__": optional dict of string key-value pairs
//!   - "<tensor_name>": { "dtype": str, "shape": [int], "data_offsets": [start, end] }
//! [remaining bytes] raw tensor data
//! ```

mod header;
mod load;
mod save;

pub use header::{SafeTensors, TensorInfo};
pub use save::save_safetensors;
