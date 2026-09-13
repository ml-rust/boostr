//! [`VoxCpm2Model`]: the sub-model bundle, its two constructors, and the
//! dtype/device the transformer stack runs at.

use super::configs::{DEFAULT_CONFIG_FILE, DEFAULT_WEIGHTS_FILE, StackConfigs};
use crate::error::{Error, Result};
use crate::format::safetensors_loader::SafeTensorsLoader;
use crate::model::audio::voxcpm::fsq::{AuxProjections, ScalarQuantization};
use crate::model::audio::voxcpm::loader::support::WeightSource;
use crate::model::audio::voxcpm::local_dit::{DEFAULT_LOCAL_DIT_PREFIX, LocalDit};
use crate::model::audio::voxcpm::local_encoder::{DEFAULT_LOCAL_ENCODER_PREFIX, LocalEncoder};
use crate::model::audio::voxcpm::minicpm4::{
    DEFAULT_MINICPM4_PREFIX, DEFAULT_RESIDUAL_LM_PREFIX, MiniCpm4Model,
};
use crate::model::audio::voxcpm::model::config::VoxCpm2Config;
use crate::model::audio::voxcpm::vae::{
    AudioVaeDecoder, AudioVaeEncoder, VAE_GGUF_DECODER_PREFIX, VAE_GGUF_ENCODER_PREFIX,
    VAE_GGUF_PROBE_TENSOR,
};
use crate::nn::MaybeQuantLinear;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ReduceOps, TensorOps, TypeConversionOps, UnaryOps};
use numr::runtime::Runtime;
use std::path::Path;

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
    R::Client: TypeConversionOps<R> + ReduceOps<R> + UnaryOps<R> + BinaryOps<R> + TensorOps<R>,
{
    /// Load the whole model.
    ///
    /// `checkpoint_dir` must contain `config.json` and `model.safetensors`.
    /// `audiovae_path` is the separately shipped `audiovae.pth`, or an
    /// `audiovae.safetensors` converted from it (file or containing
    /// directory) — see the module docs. A safetensors checkpoint never
    /// embeds the AudioVAE, so `None` here is an error naming the fix; the
    /// parameter is an `Option` only so every entry point shares one rule
    /// ([`vae_origin`]).
    ///
    /// `dtype` casts every transformer-stack tensor (`None` keeps the
    /// checkpoint's BF16). `vae_decoder_dtype` casts every AudioVAE DECODER
    /// tensor independently — see the module docs. The encoder always loads
    /// at F32; there is no encoder dtype option (see
    /// [`AudioVaeEncoder::from_checkpoint`]'s docs for why).
    pub fn from_checkpoint<P: AsRef<Path>>(
        checkpoint_dir: P,
        audiovae_path: Option<&Path>,
        device: &R::Device,
        dtype: Option<DType>,
        vae_decoder_dtype: Option<DType>,
    ) -> Result<Self> {
        let dir = checkpoint_dir.as_ref();
        let cfgs = StackConfigs::from_config_json(&dir.join(DEFAULT_CONFIG_FILE))?;
        // Opened ONCE for all five transformer-stack sub-models. Each
        // sub-loader's own `from_safetensors*` would reopen and re-parse this
        // 4.3 GB file's header, five times over.
        let mut source = SafeTensorsLoader::open(dir.join(DEFAULT_WEIGHTS_FILE))?;
        Self::from_source(
            &mut source,
            cfgs,
            audiovae_path,
            device,
            dtype,
            vae_decoder_dtype,
        )
    }

    /// Assemble every sub-model from one already-open weight source.
    ///
    /// Shared by [`from_checkpoint`](Self::from_checkpoint),
    /// [`from_gguf`](Self::from_gguf) and [`from_tcf`](Self::from_tcf) —
    /// the tensor names and shapes are the same in every container, so the
    /// walk is written once.
    ///
    /// The AudioVAE comes from `source` when `source` carries it (compressr
    /// embeds it under `vae.` — probed by [`VAE_GGUF_PROBE_TENSOR`]) and from
    /// `audiovae_path` otherwise; [`vae_origin`] is the rule. The decoder
    /// takes its own `vae_decoder_dtype` rather than `source`'s `dtype`; the
    /// encoder takes no dtype at all (always F32). A source that stores a
    /// `vae.*` tensor block-quantized must be refused by the caller before
    /// reaching here — `load_named` would dequantize it silently.
    pub(crate) fn from_source<S: WeightSource<R>>(
        source: &mut S,
        cfgs: StackConfigs,
        audiovae_path: Option<&Path>,
        device: &R::Device,
        dtype: Option<DType>,
        vae_decoder_dtype: Option<DType>,
    ) -> Result<Self> {
        let (vae_encoder, vae_decoder) =
            match vae_origin(source.has_named(VAE_GGUF_PROBE_TENSOR), audiovae_path)? {
                VaeOrigin::Embedded => (
                    AudioVaeEncoder::from_source(source, VAE_GGUF_ENCODER_PREFIX, device)?,
                    AudioVaeDecoder::from_source(
                        source,
                        VAE_GGUF_DECODER_PREFIX,
                        device,
                        vae_decoder_dtype,
                    )?,
                ),
                VaeOrigin::Separate(path) => (
                    AudioVaeEncoder::from_checkpoint(path, device)?,
                    AudioVaeDecoder::from_checkpoint(path, device, vae_decoder_dtype)?,
                ),
            };
        Ok(Self {
            vae_encoder,
            vae_decoder,
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

/// Where the AudioVAE is read from.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum VaeOrigin<'a> {
    /// The `vae.*` tensors of the weight source itself.
    Embedded,
    /// A separate `audiovae.pth` / `audiovae.safetensors`.
    Separate(&'a Path),
}

/// Decide where the AudioVAE comes from.
///
/// An embedded VAE WINS over a path: the converter wrote it from the same
/// checkpoint, and loading a different file beside it would silently pair
/// the stack with a VAE it was not converted with. No VAE anywhere is an
/// error that names both ways to supply one.
pub(crate) fn vae_origin(embedded: bool, audiovae_path: Option<&Path>) -> Result<VaeOrigin<'_>> {
    if embedded {
        return Ok(VaeOrigin::Embedded);
    }
    audiovae_path
        .map(VaeOrigin::Separate)
        .ok_or_else(|| Error::ModelError {
            reason: format!(
                "the weight source carries no AudioVAE (no `{VAE_GGUF_PROBE_TENSOR}` \
                 tensor) and no audiovae path was given; pass the checkpoint's \
                 audiovae.pth, or convert with a compressr that embeds the VAE"
            ),
        })
}

/// The error for an embedded `vae.*` tensor stored block-quantized.
///
/// Shared by the GGUF and TCF entry points, which each run the check over
/// their own directory before handing the source to [`VoxCpm2Model::from_source`].
pub(crate) fn packed_vae_tensor(name: &str, stored_as: &str) -> Error {
    Error::ModelError {
        reason: format!(
            "{name}: an embedded AudioVAE tensor is stored as {stored_as}, a block-quantized \
             type; the AudioVAE must be stored dense (F32/F16/BF16), so this file was \
             written by a converter that quantized it — reconvert"
        ),
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
    pub fn lm_dtype(&self) -> Result<DType> {
        Ok(self.lm_dtype_device()?.0)
    }

    /// Device every transformer-stack tensor lives on.
    pub fn device(&self) -> Result<&R::Device> {
        Ok(self.lm_dtype_device()?.1)
    }

    /// The dtype and device the stack's ARITHMETIC actually runs at.
    ///
    /// A packed `enc_to_lm_proj` cannot answer this from its weight:
    /// `quant_matmul` consumes F32 and emits F32 whatever block format the
    /// weight holds, so the answer is F32 there — the packed weight has no
    /// element dtype to copy. Same shape as
    /// `MiniCpm4Attention::kv_dtype_device`, and both return `Result` rather
    /// than guessing for the decomposed-quant arm no VoxCPM2 checkpoint
    /// loads.
    pub(crate) fn lm_dtype_device(&self) -> Result<(DType, &R::Device)> {
        match self.aux.enc_to_lm_proj.base() {
            MaybeQuantLinear::Standard(linear) => {
                let w = linear.weight().tensor();
                Ok((w.dtype(), w.device()))
            }
            MaybeQuantLinear::Quantized(qlinear) => Ok((DType::F32, qlinear.weight().device())),
            MaybeQuantLinear::DecomposedQuant(_) => Err(Error::ModelError {
                reason: "VoxCPM2 enc_to_lm_proj: no VoxCPM2 checkpoint loads \
                         decomposed-quantized (AWQ/GPTQ) weights, so the stack \
                         dtype is undefined here"
                    .to_string(),
            }),
        }
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
                Some(Path::new("/nonexistent/audiovae.safetensors")),
                &device,
                Some(DType::F32),
                None,
            )
            .is_err()
        );
    }

    #[test]
    fn embedded_vae_wins_over_a_path() {
        let path = Path::new("/somewhere/audiovae.pth");
        assert_eq!(vae_origin(true, Some(path)).unwrap(), VaeOrigin::Embedded);
        assert_eq!(vae_origin(true, None).unwrap(), VaeOrigin::Embedded);
    }

    #[test]
    fn a_path_is_used_when_nothing_is_embedded() {
        let path = Path::new("/somewhere/audiovae.pth");
        assert_eq!(
            vae_origin(false, Some(path)).unwrap(),
            VaeOrigin::Separate(path)
        );
    }

    /// Neither: the error names the probe tensor and the path option, so
    /// the operator knows both ways to supply a VAE.
    #[test]
    fn no_vae_anywhere_names_both_options() {
        let err = vae_origin(false, None).unwrap_err().to_string();
        assert!(err.contains(VAE_GGUF_PROBE_TENSOR), "{err}");
        assert!(err.contains("audiovae"), "{err}");
    }

    #[test]
    fn packed_vae_error_names_tensor_and_type() {
        let err = packed_vae_tensor("vae.decoder.model.1.weight", "Q4K").to_string();
        assert!(err.contains("vae.decoder.model.1.weight"), "{err}");
        assert!(err.contains("Q4K"), "{err}");
    }
}
