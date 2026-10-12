//! LLaMA model: struct definition, construction, and training forward pass.

mod construction;
mod experts;
mod types;

#[cfg(test)]
mod test_support;
#[cfg(test)]
mod variant_tests;

pub use types::Llama;
