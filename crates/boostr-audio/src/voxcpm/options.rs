//! Load-time and per-request settings for [`VoxCpm2Engine`](super::engine::VoxCpm2Engine).

use numr::dtype::DType;
use std::path::PathBuf;

/// Generation settings applied to every request. Defaults are the clone
/// pipeline's verified values.
#[derive(Debug, Clone)]
pub struct VoxCpm2SynthOptions {
    /// Flow-matching solver steps per patch.
    pub n_timesteps: usize,
    /// Classifier-free guidance scale.
    pub cfg_value: f32,
    /// Patches during which the stop token is ignored.
    pub min_len: usize,
    /// Base seed; every request draws from it, so equal requests render
    /// equal audio on one backend.
    pub seed: u64,
}

impl Default for VoxCpm2SynthOptions {
    fn default() -> Self {
        Self {
            n_timesteps: 10,
            cfg_value: 2.0,
            min_len: 2,
            seed: 0,
        }
    }
}

/// Everything [`VoxCpm2Engine::load`] needs beyond the checkpoint location:
/// the dtype to cast to, per-request generation defaults, and an optional
/// LoRA adapter to fold into the weights. Bundled into one struct because
/// `load` already takes five positional arguments (weights, audiovae,
/// voices_dir, device, client) — three more flat parameters would trip
/// clippy's too-many-arguments limit.
#[derive(Debug, Clone, Default)]
pub struct VoxCpm2LoadOptions {
    /// Casts every transformer-stack tensor; `None` keeps the checkpoint's
    /// own dtype (BF16).
    pub dtype: Option<DType>,
    /// Casts every AudioVAE DECODER tensor (conv weights/biases, `Snake`
    /// alphas, sr-cond embeds), independently of `dtype`. `None` keeps the
    /// checkpoint's own F32, verified against PyTorch fixtures at that
    /// dtype.
    ///
    /// The encoder is NOT affected: it always loads and runs at F32. It
    /// runs once per render, on a short reference clip, so its cost is
    /// negligible — but casting it changes the latent handed to the
    /// transformer stack, shifting the stack's own conditioning and
    /// therefore the whole generation, including where it stops.
    pub vae_decoder_dtype: Option<DType>,
    /// Per-request generation settings — see [`VoxCpm2SynthOptions`].
    pub synth: VoxCpm2SynthOptions,
    /// A LoRA adapter safetensors file, folded into the model's weights
    /// once, at load time. `None` loads the base model, unchanged.
    ///
    /// The adapter is folded in ONCE: an engine serves ONE adapted model for
    /// its lifetime. Per-request adapter switching is not offered — serving
    /// several adapters means loading several engines (a `--tts-model
    /// NAME=DIR` bundle per adapter, in blazr's terms), one adapter each.
    pub adapter: Option<PathBuf>,
}
