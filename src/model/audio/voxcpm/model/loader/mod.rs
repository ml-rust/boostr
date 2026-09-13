//! [`VoxCpm2Model`]: the end-to-end VoxCPM2 orchestrator, owning every ported
//! sub-model, plus the loader that builds it from a checkpoint directory.
//!
//! # Where the AudioVAE comes from
//!
//! The transformer stack (`base_lm`, `residual_lm`, `feat_encoder`,
//! `feat_decoder`, `fsq_layer` and the six auxiliary projections) lives in the
//! checkpoint directory's `model.safetensors`. The AudioVAE ships SEPARATELY
//! as `audiovae.pth`, so [`VoxCpm2Model::from_checkpoint`] takes its path as
//! a second argument. That `.pth` is read as published, `weight_norm` folded
//! at load time
//! ([`VaeCheckpoint`](crate::model::audio::voxcpm::vae::VaeCheckpoint)); an
//! `audiovae.safetensors` converted by the reference repo's
//! `convert_audiovae.py` is still accepted, so a tree that already holds one
//! keeps loading.
//!
//! A GGUF or TCF written by `compressr convert` from a directory that holds
//! `audiovae.pth` beside `model.safetensors` EMBEDS the VAE: folded, dense,
//! under `vae.decoder.*`/`vae.encoder.*`. The single-file entry points
//! ([`from_gguf`](crate::model::audio::voxcpm::model::gguf_loader),
//! [`from_tcf`](crate::model::audio::voxcpm::model::tcf_loader)) probe for
//! it and read it from the same file when present; the separate path is
//! then optional and, when given, ignored — the embedded copy wins. A file
//! written without it (an older conversion, or a third-party GGUF such as
//! `cstr/voxcpm2-GGUF`, whose `vae.*` tensors use another scheme) still
//! needs the path. The rule is one function, `model::vae_origin`.
//!
//! # Dtype
//!
//! The VoxCPM2 checkpoint is BF16, the AudioVAE is F32-native. The transformer
//! stack takes a `dtype` argument that casts every tensor it reads (pass
//! `Some(DType::F32)` to run the whole stack in F32). The AudioVAE DECODER
//! takes its own, independent `vae_decoder_dtype` argument: `None` keeps the
//! checkpoint's F32 (verified against PyTorch fixtures at that dtype),
//! `Some(F16)`/`Some(BF16)` casts every decoder weight and runs its forward
//! pass in that dtype — every conv, `Snake`'s `sin`/`recip`, and the final
//! `tanh` all have CUDA F16/BF16 kernels, so nothing in the decoder path
//! falls back to F32. The boundary cast (decoder output back to F32) lives in
//! [`super::decode::VoxCpm2Model::decode_patches`].
//!
//! The AudioVAE ENCODER has NO dtype option: it always loads and runs at F32.
//! It runs once per render, over a short reference clip (and during
//! training), so its cost is negligible — but casting it changes the LATENT
//! handed to the transformer stack, which shifts the stack's own
//! conditioning and therefore the whole generation: a lower-precision
//! encoder changes the patch sequence itself, including where it stops, so a
//! decoder-only comparison would measure the wrong thing. See
//! [`AudioVaeEncoder::from_checkpoint`](crate::model::audio::voxcpm::vae::AudioVaeEncoder::from_checkpoint)'s
//! docs.
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
pub(crate) use model::packed_vae_tensor;
