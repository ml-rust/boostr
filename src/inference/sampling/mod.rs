//! Host-side logit shaping: penalty window, DRY, typical-p, dynamic temperature, mirostat, logit bias.

pub mod bias;
pub mod dry;
pub mod dynatemp;
pub mod mirostat;
pub mod typical;
pub mod window;

pub use bias::LogitBias;
pub use dry::{DryParams, apply_dry_penalty};
pub use dynatemp::{DynaTemp, MIN_DYNAMIC_TEMPERATURE};
pub use mirostat::{MirostatV2, MirostatV2State};
pub use typical::apply_typical_filter;
pub use window::penalty_window;
