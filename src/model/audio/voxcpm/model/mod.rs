//! VoxCPM2 end-to-end orchestrator: [`config`] (patch geometry and special
//! token ids), [`loader`] ([`VoxCpm2Model`], every sub-model plus its
//! checkpoint loader), [`sequence`] (host-side prefix layout and mask
//! complementarity), [`patches`] (wav padding and the VAE patch fold),
//! [`prefill`] (reference encode and the two-LM prefill).
//! The per-patch sampling loop is a separate unit and does not live here.

pub mod config;
pub mod loader;
pub mod patches;
pub mod prefill;
pub mod sequence;

pub use config::{
    AUDIO_START_ID, REF_AUDIO_END_ID, REF_AUDIO_FILLER_ID, REF_AUDIO_START_ID, VoxCpm2Config,
};
pub use loader::{DEFAULT_CONFIG_FILE, DEFAULT_WEIGHTS_FILE, VoxCpm2Model};
pub use patches::{fold_patches, pad_to_multiple};
pub use prefill::{PrefillIntermediates, PrefillState};
pub use sequence::{SequenceLayout, check_mask_complementarity};
