//! `qwen35`: Gated DeltaNet + gated full-attention hybrid (Bonsai).
//!
//! - `config_from_gguf`: [`UniversalConfig`](crate::model::config::UniversalConfig)
//!   from the `qwen35.*` GGUF namespace
//! - `model`: the [`Qwen35Model`] type, its blocks, the state-carrying
//!   forward, and the GGUF loader
//! - `multimodal`: image placeholders expanded to vision-tower rows with
//!   IMROPE positions, the input of `forward_qwen35_embeds`

pub mod config_from_gguf;
pub mod model;
pub mod multimodal;

pub use config_from_gguf::qwen35_config_from_gguf;
pub use model::{Qwen35AttentionLayer, Qwen35Block, Qwen35GdnLayer, Qwen35Model};
pub use multimodal::{
    ImageEmbeds, ImageGrid, PromptLayout, Qwen35PromptPlan, Segment, VisionMarkers,
    plan_positions_only,
};
