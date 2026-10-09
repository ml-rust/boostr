//! CLAP (contrastive language-audio pretraining): transformers'
//! `ClapModel` for unfused checkpoints such as `laion/larger_clap_general`.
//! The HTSAT audio tower, the RoBERTa text tower, their projections, and
//! the log-mel front end that feeds the audio tower.

pub mod audio_encoder;
pub mod client;
pub mod config;
pub mod features;
mod linear;
pub mod model;
pub mod projection;
pub mod resize;
pub mod stage;
pub mod swin_block;
pub mod text;
pub mod window_attention;

pub use audio_encoder::ClapAudioEncoder;
pub use client::ClapClient;
pub use config::{ClapAudioConfig, ClapConfig, ClapTextConfig};
pub use features::{ClapFeatureExtractor, ClapPadding};
pub use model::ClapModel;
pub use projection::ClapProjection;
pub use resize::resize_bicubic;
pub use text::ClapTextEncoder;
