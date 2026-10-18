//! Drafters: token proposers that run without a model.

pub mod drafter;
pub mod prompt_lookup;

pub use drafter::Drafter;
pub use prompt_lookup::{DEFAULT_LOOKUP_WINDOW, PromptLookup};
