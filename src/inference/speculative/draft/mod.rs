//! Drafters: token proposers with no model, or with a draft model.

pub mod drafter;
pub mod model_drafter;
pub mod prob_drafter;
pub mod prompt_lookup;

pub use drafter::Drafter;
pub use model_drafter::{DRAFT_KEY_DOMAIN, ModelDrafter};
pub use prob_drafter::{DraftModel, ProbDraft, ProbDrafter};
pub use prompt_lookup::{DEFAULT_LOOKUP_WINDOW, PromptLookup};
