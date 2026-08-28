//! VoxCPM2 end-to-end orchestrator: [`config`] (patch geometry and special token ids),
//! [`loader`] ([`VoxCpm2Model`], every sub-model plus its checkpoint loader), [`gguf_loader`]
//! (the single-file GGUF entry point), [`sequence`] (prefix layout and mask complementarity),
//! [`patches`] (wav padding and the VAE patch fold), [`prefill`] (reference encode and the
//! two-LM prefill), [`generate`] (the per-patch sampling loop and its stop logic), [`decode`]
//! (unfolding patches back to a latent and VAE-decoding to a waveform), `chunked_decode`
//! (windowed VAE decode so peak memory does not scale with utterance length), [`train`] (the
//! CFM training loss, wiring teacher-forced conditioning into a differentiable `Var`).
//! The wav file wrapper is a separate, later unit and does not live here.

pub(crate) mod chunked_decode;
pub mod config;
pub mod decode;
pub mod generate;
pub mod gguf_loader;
pub mod loader;
pub mod patches;
pub mod prefill;
pub mod sequence;
pub mod train;

pub use config::{
    AUDIO_START_ID, REF_AUDIO_END_ID, REF_AUDIO_FILLER_ID, REF_AUDIO_START_ID, VoxCpm2Config,
};
pub use decode::unfold_patches;
pub use generate::{
    GenerateOptions, GenerateOutcome, GenerateState, PatchGenerator, StepIntermediates,
    StepOutcome, TeacherForcedConditioning,
};
pub use gguf_loader::GGUF_CONFIG_JSON_KEY;
pub use loader::{DEFAULT_CONFIG_FILE, DEFAULT_WEIGHTS_FILE, VoxCpm2Model};
pub use patches::{fold_patches, pad_to_multiple};
pub use prefill::{PrefillIntermediates, PrefillState};
pub use sequence::{SequenceLayout, check_mask_complementarity};
