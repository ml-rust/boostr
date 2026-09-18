//! `qwen35`: Gated DeltaNet + gated full-attention hybrid (Bonsai).
//!
//! - `model`: the [`Qwen35Model`] type, its blocks, and the state-carrying
//!   forward

pub mod model;

pub use model::{Qwen35AttentionLayer, Qwen35Block, Qwen35GdnLayer, Qwen35Model};
