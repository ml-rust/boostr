//! [`VoxCpm2Model`]: the end-to-end VoxCPM2 orchestrator, owning every ported
//! sub-model, plus the loader that builds it from a checkpoint directory.
//!
//! # Two checkpoints, not one
//!
//! The transformer stack (`base_lm`, `residual_lm`, `feat_encoder`,
//! `feat_decoder`, `fsq_layer` and the six auxiliary projections) lives in the
//! checkpoint directory's `model.safetensors`. The AudioVAE ships SEPARATELY
//! as `audiovae.safetensors` (produced by the reference repo's
//! `convert_audiovae.py`), so its path is a second argument — see
//! [`VoxCpm2Model::from_checkpoint`].
//!
//! The same split holds for the GGUF entry point
//! ([`from_gguf`](crate::model::audio::voxcpm::model::gguf_loader)): a
//! VoxCPM2 GGUF written by `compressr convert --format gguf` carries the
//! TRANSFORMER STACK ONLY. The AudioVAE is not in it, because it is not part
//! of the checkpoint compressr converts — it arrives as its own
//! `audiovae.safetensors` from a separate conversion script — so `from_gguf`
//! takes the VAE path as its own argument exactly like `from_checkpoint`.
//!
//! # Dtype
//!
//! The VoxCPM2 checkpoint is BF16, the AudioVAE is F32-native. The transformer
//! stack takes a `dtype` argument that casts every tensor it reads (pass
//! `Some(DType::F32)` to run the whole stack in F32); the AudioVAE loaders
//! take none and are always left at their checkpoint dtype, because that model
//! is verified against PyTorch fixtures in F32 and must not be cast.

use crate::error::{Error, Result};
use crate::format::safetensors_loader::SafeTensorsLoader;
use crate::model::audio::voxcpm::fsq::{AuxProjections, FsqConfig, ScalarQuantization};
use crate::model::audio::voxcpm::loader::support::WeightSource;
use crate::model::audio::voxcpm::local_dit::{DEFAULT_LOCAL_DIT_PREFIX, LocalDit, LocalDitConfig};
use crate::model::audio::voxcpm::local_encoder::{
    DEFAULT_LOCAL_ENCODER_PREFIX, LocalEncoder, LocalEncoderConfig,
};
use crate::model::audio::voxcpm::minicpm4::{
    DEFAULT_MINICPM4_PREFIX, DEFAULT_RESIDUAL_LM_PREFIX, MiniCpm4Config, MiniCpm4Model,
};
use crate::model::audio::voxcpm::model::config::VoxCpm2Config;
use crate::model::audio::voxcpm::vae::{AudioVaeDecoder, AudioVaeEncoder};
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::Runtime;
use std::path::Path;

/// Checkpoint file name holding the transformer stack.
pub const DEFAULT_WEIGHTS_FILE: &str = "model.safetensors";
/// Checkpoint file name holding the architecture config.
pub const DEFAULT_CONFIG_FILE: &str = "config.json";

/// The six configs the transformer stack needs, all resolved from ONE
/// `config.json`.
///
/// Grouped so the file entry point and the GGUF entry point share both the
/// parse and the sub-model walk below: a GGUF holds the same architecture
/// under the same tensor names, and only the byte container differs.
pub(crate) struct StackConfigs {
    pub(crate) model: VoxCpm2Config,
    pub(crate) base_lm: MiniCpm4Config,
    pub(crate) residual_lm: MiniCpm4Config,
    pub(crate) encoder: LocalEncoderConfig,
    pub(crate) dit: LocalDitConfig,
    pub(crate) fsq: FsqConfig,
}

impl StackConfigs {
    /// Resolve all six out of the verbatim contents of a `config.json`.
    pub(crate) fn from_config_str(content: &str) -> Result<Self> {
        Ok(Self {
            model: VoxCpm2Config::from_config_str(content)?,
            base_lm: MiniCpm4Config::from_config_str(content)?,
            residual_lm: MiniCpm4Config::residual_lm_from_config_str(content)?,
            encoder: LocalEncoderConfig::from_config_str(content)?,
            dit: LocalDitConfig::from_config_str(content)?,
            fsq: FsqConfig::from_config_str(content)?,
        })
    }

    /// Read a `config.json` once and resolve all six out of it. The six
    /// per-type `from_config_json` constructors each read the file
    /// themselves, which would be six reads of the same bytes here.
    pub(crate) fn from_config_json(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| Error::ModelError {
            reason: format!("failed to read {}: {e}", path.display()),
        })?;
        Self::from_config_str(&content)
    }
}

/// Every VoxCPM2 sub-model, loaded and ready, plus the patch geometry the
/// orchestrator needs.
///
/// The fields are public so a caller can drive a sub-model directly (the
/// gate examples do). `feat_decoder` is loaded here but is NOT used by the
/// prefill path — it belongs to the per-patch sampling loop.
pub struct VoxCpm2Model<R: Runtime> {
    /// Waveform -> latent `[1, feat_dim, frames]`.
    pub vae_encoder: AudioVaeEncoder<R>,
    /// Latent -> waveform. Unused by prefill; the decode path is a later unit.
    pub vae_decoder: AudioVaeDecoder<R>,
    /// `feat_encoder`: `[B, T, patch_size, feat_dim]` -> `[B, T, 1024]`.
    pub feat_encoder: LocalEncoder<R>,
    /// `base_lm`: the 28-layer rotary decoder.
    pub base_lm: MiniCpm4Model<R>,
    /// `residual_lm`: the 8-layer NoPE decoder, fed pre-computed embeddings.
    pub residual_lm: MiniCpm4Model<R>,
    /// `feat_decoder`: the CFM estimator. Unused by prefill.
    pub feat_decoder: LocalDit<R>,
    /// `fsq_layer`: the bottleneck applied to AUDIO positions only.
    pub fsq: ScalarQuantization<R>,
    /// The six auxiliary projections (`enc_to_lm_proj`, `fusion_concat_proj`,
    /// the DiT bridges, and the stop chain).
    pub aux: AuxProjections<R>,
    /// Patch geometry (`patch_size`, `feat_dim`).
    pub config: VoxCpm2Config,
}

impl<R: Runtime<DType = DType>> VoxCpm2Model<R>
where
    R::Client: TypeConversionOps<R>,
{
    /// Load the whole model.
    ///
    /// `checkpoint_dir` must contain `config.json` and `model.safetensors`.
    /// `audiovae_path` is the separately converted `audiovae.safetensors`
    /// (file or its containing directory).
    ///
    /// `dtype` casts every transformer-stack tensor (`None` keeps the
    /// checkpoint's BF16). The AudioVAE is never cast — see the module docs.
    pub fn from_checkpoint<P: AsRef<Path>, Q: AsRef<Path>>(
        checkpoint_dir: P,
        audiovae_path: Q,
        device: &R::Device,
        dtype: Option<DType>,
    ) -> Result<Self> {
        let dir = checkpoint_dir.as_ref();
        let cfgs = StackConfigs::from_config_json(&dir.join(DEFAULT_CONFIG_FILE))?;
        // Opened ONCE for all five transformer-stack sub-models. Each
        // sub-loader's own `from_safetensors*` would reopen and re-parse this
        // 4.3 GB file's header, five times over.
        let mut source = SafeTensorsLoader::open(dir.join(DEFAULT_WEIGHTS_FILE))?;
        Self::from_source(&mut source, cfgs, audiovae_path.as_ref(), device, dtype)
    }

    /// Assemble every sub-model from one already-open weight source.
    ///
    /// Shared by [`from_checkpoint`](Self::from_checkpoint) and
    /// [`from_gguf`](Self::from_gguf) — the tensor names and shapes are the
    /// same in both containers, so the walk is written once.
    ///
    /// The two AudioVAE loaders deliberately do NOT go through `source`: the
    /// VAE lives in its own separate file (see the module docs), and it is
    /// never cast, so it takes neither `source` nor `dtype`.
    pub(crate) fn from_source<S: WeightSource<R>>(
        source: &mut S,
        cfgs: StackConfigs,
        audiovae_path: &Path,
        device: &R::Device,
        dtype: Option<DType>,
    ) -> Result<Self> {
        Ok(Self {
            vae_encoder: AudioVaeEncoder::from_safetensors(audiovae_path, device)?,
            vae_decoder: AudioVaeDecoder::from_safetensors(audiovae_path, device)?,
            feat_encoder: LocalEncoder::from_source(
                source,
                DEFAULT_LOCAL_ENCODER_PREFIX,
                cfgs.encoder,
                device,
                dtype,
            )?,
            base_lm: MiniCpm4Model::from_source(
                source,
                DEFAULT_MINICPM4_PREFIX,
                cfgs.base_lm,
                device,
                dtype,
            )?,
            residual_lm: MiniCpm4Model::from_source(
                source,
                DEFAULT_RESIDUAL_LM_PREFIX,
                cfgs.residual_lm,
                device,
                dtype,
            )?,
            feat_decoder: LocalDit::from_source(
                source,
                DEFAULT_LOCAL_DIT_PREFIX,
                cfgs.dit,
                device,
                dtype,
            )?,
            fsq: ScalarQuantization::from_source(source, cfgs.fsq, device, dtype)?,
            aux: AuxProjections::from_source(source, cfgs.fsq, device, dtype)?,
            config: cfgs.model,
        })
    }
}

impl<R: Runtime<DType = DType>> VoxCpm2Model<R> {
    /// Dtype every transformer-stack tensor was loaded at.
    ///
    /// Read off `enc_to_lm_proj`, an always-present weight the prefill path
    /// itself multiplies against — so the masks and the zero patches this
    /// module builds are guaranteed to match the tensors they meet. Reading
    /// it beats threading the loader's `Option<DType>` through, which is
    /// `None` for "whatever the checkpoint had" and so answers nothing.
    pub fn lm_dtype(&self) -> DType {
        self.aux.enc_to_lm_proj.weight().tensor().dtype()
    }

    /// Device every transformer-stack tensor lives on.
    pub fn device(&self) -> &R::Device {
        self.aux.enc_to_lm_proj.weight().tensor().device()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn rejects_missing_checkpoint() {
        let device = <CpuRuntime as Runtime>::default_device();
        assert!(
            VoxCpm2Model::<CpuRuntime>::from_checkpoint(
                "/nonexistent/voxcpm2",
                "/nonexistent/audiovae.safetensors",
                &device,
                Some(DType::F32),
            )
            .is_err()
        );
    }
}
