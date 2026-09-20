//! Image inputs for `qwen35`: prompt layout, IMROPE positions, and the
//! embedding splice that feeds `Qwen35Model::forward_qwen35_embeds`.
//!
//! - `layout`: [`plan_positions_only`] and the row/position rule, no tensors
//! - `splice`: [`Qwen35PromptPlan`], text lookups plus vision-tower rows

pub mod layout;
pub mod splice;

pub use layout::{ImageGrid, PromptLayout, Segment, VisionMarkers, plan_positions_only};
pub use splice::{ImageEmbeds, Qwen35PromptPlan};
