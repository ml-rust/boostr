//! The flat `voxcpm::model::*` names, re-exported from each stage module.

pub use super::config::{
    AUDIO_START_ID, REF_AUDIO_END_ID, REF_AUDIO_FILLER_ID, REF_AUDIO_START_ID,
    STREAMING_PREFIX_LEN, VoxCpm2Config,
};
pub use super::decode::{unfold_patches, unfold_patches_row};
pub use super::generate::{
    GenerateOptions, GenerateOutcome, GenerateState, PatchGenerator, RowOptions, StepIntermediates,
    StepOutcome, TeacherForcedConditioning,
};
pub use super::gguf_loader::GGUF_CONFIG_JSON_KEY;
pub use super::loader::{
    DEFAULT_CONFIG_FILE, DEFAULT_WEIGHTS_FILE, LoraAdapterReport, VoxCpm2Model,
};
pub use super::patches::{PadSide, fold_patches, pad_to_multiple};
pub use super::prefill::{PaddedBatch, PrefillIntermediates, PrefillRow, PrefillState};
pub use super::sequence::{SequenceLayout, check_mask_complementarity};
pub use super::widths::{BATCH_WIDTHS, batch_width, clamp_max_batch};
