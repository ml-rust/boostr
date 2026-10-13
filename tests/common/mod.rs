// Each file in `tests/` compiles as its own crate and pulls in this module
// wholesale, so any helper a given test crate does not call reads as dead code
// there. The helpers ARE used — just not all of them by every crate.
#![allow(dead_code, unused_imports)]

mod blocks;
pub mod decision_fixture;
pub mod decision_gguf;
pub mod decision_model;
pub mod decision_pointer_fixture;
mod fixtures;
pub mod kv_start;
#[cfg(feature = "cuda")]
pub mod llama_graph;
#[cfg(feature = "cuda")]
pub mod llama_tiny;
mod params;
mod q8_0;
#[cfg(feature = "cuda")]
pub mod qwen35_cuda;
#[cfg(feature = "cuda")]
pub mod qwen35_graph;
pub mod qwen35_lanes;
pub mod qwen35_tiny;

pub use blocks::*;
pub use fixtures::*;
pub use params::*;
pub use q8_0::*;
