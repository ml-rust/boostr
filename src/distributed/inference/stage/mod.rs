//! One pipeline stage: a contiguous layer range of a model, with one KV
//! cache per request.
//!
//! - `contract`: the `Stage` trait a worker serves, `StageOutput`
//! - `book`: per-request caches, positions and the in-flight cap
//! - `config`: `StageConfig`, which models and ranges a stage accepts
//! - `runner`: `StageRunner`, the stage over a `LoadedModel`
//! - `tensors`: payload to device tensors, and device tensors to the host

mod book;
pub mod config;
pub mod contract;
pub mod runner;
mod tensors;

pub use config::StageConfig;
pub use contract::{Stage, StageOutput};
pub use runner::StageRunner;
