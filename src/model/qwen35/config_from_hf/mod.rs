//! `qwen35` config from a Hugging Face Qwen3.5 `config.json`.
//!
//! - `build`: [`qwen35_config_from_hf`], with the key table
//! - `keys`: key lookup, error text, and the rejection check
//! - `fields`: rotated width, IMROPE sections, layer roles

mod build;
mod fields;
#[cfg(test)]
mod fixtures;
mod keys;

pub use build::{QWEN35_HF_MODEL_TYPES, is_qwen35_hf_config, qwen35_config_from_hf};
