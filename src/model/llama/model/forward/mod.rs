//! LLaMA model: struct definition, construction, and training forward pass.

mod construction;
mod experts;
mod types;

#[cfg(test)]
mod test_support;

pub use types::Llama;
