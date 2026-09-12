//! [`VoxCpm2Model`]: the end-to-end VoxCPM2 orchestrator, owning every ported
//! sub-model, plus the loader that builds it from a checkpoint directory.
//!
//! # Two checkpoints, not one
//!
//! The transformer stack (`base_lm`, `residual_lm`, `feat_encoder`,
//! `feat_decoder`, `fsq_layer` and the six auxiliary projections) lives in the
//! checkpoint directory's `model.safetensors`. The AudioVAE ships SEPARATELY
//! as `audiovae.pth`, so its path is a second argument — see
//! [`VoxCpm2Model::from_checkpoint`]. That `.pth` is read as published,
//! `weight_norm` folded at load time
//! ([`VaeCheckpoint`](crate::model::audio::voxcpm::vae::VaeCheckpoint)); an
//! `audiovae.safetensors` converted by the reference repo's
//! `convert_audiovae.py` is still accepted, so a tree that already holds one
//! keeps loading.
//!
//! The same split holds for the GGUF entry point
//! ([`from_gguf`](crate::model::audio::voxcpm::model::gguf_loader)): a
//! VoxCPM2 GGUF written by `compressr convert --format gguf` carries the
//! TRANSFORMER STACK ONLY. The AudioVAE is not in it, because it is not part
//! of the checkpoint compressr converts — it arrives as its own
//! `audiovae.pth` — so `from_gguf` takes the VAE path as its own argument
//! exactly like `from_checkpoint`.
//!
//! # Dtype
//!
//! The VoxCPM2 checkpoint is BF16, the AudioVAE is F32-native. The transformer
//! stack takes a `dtype` argument that casts every tensor it reads (pass
//! `Some(DType::F32)` to run the whole stack in F32); the AudioVAE loaders
//! take none and are always left at their checkpoint dtype, because that model
//! is verified against PyTorch fixtures in F32 and must not be cast.
//!
//! - `configs`: the checkpoint file names and [`StackConfigs`]
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
