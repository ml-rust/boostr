//! [`VoxCpm2Model`]: the end-to-end VoxCPM2 orchestrator, owning every ported
//! sub-model, plus the loader that builds it from a checkpoint directory.
//!
//! `model` documents where the AudioVAE comes from and the dtype rules.
//!
//! - `configs`: the checkpoint file names and `StackConfigs`
//! - `model`: [`VoxCpm2Model`], its constructors, and the stack dtype/device
//! - `module`: the `Module` impl (parameter enumeration)
//! - `lora`: whole-model adapter attachment, projection names, write-back
//! - `lora_named`: `load_lora_named`
//! - `lora_adapter`: `load_lora_adapter` and [`LoraAdapterReport`]
//! - `checkpointing`: `set_activation_checkpointing`

mod checkpointing;
mod configs;
mod lora;
mod lora_adapter;
mod lora_named;
mod model;
mod module;

pub(crate) use configs::StackConfigs;
pub use configs::{DEFAULT_CONFIG_FILE, DEFAULT_WEIGHTS_FILE};
pub use lora_adapter::LoraAdapterReport;
pub use model::VoxCpm2Model;
pub(crate) use model::packed_vae_tensor;
