//! Shared helper utilities for LLaMA building blocks.
//!
//! Re-exports the canonical `Var` layout helpers from [`crate::nn::var_ops`].

pub use crate::nn::var_ops::{repeat_kv, var_contiguous};
