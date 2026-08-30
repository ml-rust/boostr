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

use crate::error::{Error, Result};
use crate::format::safetensors_loader::SafeTensorsLoader;
use crate::model::audio::voxcpm::fsq::loader::FSQ_LAYER_PREFIX;
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
use crate::nn::{LoraTargets, MaybeQuantLinear, Module, extend_named};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ReduceOps, TensorOps, TypeConversionOps, UnaryOps};
use numr::runtime::Runtime;
use numr::tensor::{Tensor, TensorId};
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
    R::Client: TypeConversionOps<R> + ReduceOps<R> + UnaryOps<R> + BinaryOps<R> + TensorOps<R>,
{
    /// Load the whole model.
    ///
    /// `checkpoint_dir` must contain `config.json` and `model.safetensors`.
    /// `audiovae_path` is the separately shipped `audiovae.pth`, or an
    /// `audiovae.safetensors` converted from it (file or containing
    /// directory) — see the module docs.
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
            vae_encoder: AudioVaeEncoder::from_checkpoint(audiovae_path, device)?,
            vae_decoder: AudioVaeDecoder::from_checkpoint(audiovae_path, device)?,
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

    /// THE entry point: wrap every `targets`-named projection across the
    /// whole model — `feat_encoder`, `base_lm`, `residual_lm`,
    /// `feat_decoder`, `fsq_layer`, and `aux`'s six root-level projections —
    /// with a fresh LoRA adapter, so a fine-tune can train adapters over the
    /// frozen base (every VoxCPM2 weight loads `requires_grad = false`; see
    /// `local_encoder/encoder.rs:19`, `minicpm4/model.rs:30`). Returns the
    /// total number of projections adapted.
    ///
    /// `vae_encoder`/`vae_decoder` are never touched: they are a separately
    /// checkpointed, frozen audio codec (see the module docs) and neither
    /// implements `Module<R>` — same exclusion [`Self::named_parameters`]
    /// documents.
    ///
    /// Delegates to each sub-model's own `apply_lora`, joining ITS prefix
    /// with the same constant [`Self::named_parameters`] uses below
    /// (`DEFAULT_LOCAL_ENCODER_PREFIX`, `DEFAULT_MINICPM4_PREFIX`, ...) so a
    /// target's full path here always matches the same-named path
    /// `named_parameters()` would produce — the two are never built by
    /// separately hand-written logic.
    ///
    /// As the actual top of the call graph, this is the ONE place that
    /// validates every target up front with [`LoraTargets::ensure_all_match`]
    /// against the WHOLE model's candidate set — [`Self::lora_projection_names`],
    /// NOT `self.named_parameters()` — before adapting anything. Each
    /// sub-model child is then walked via its `apply_lora_unchecked`, not
    /// its validating `apply_lora`: re-validating a cross-subtree target
    /// list against only one child's own candidate set would reject a
    /// target that lives in a sibling (e.g. `stop_proj`, which lives under
    /// `aux`, is not a candidate inside `feat_encoder`'s own subtree).
    ///
    /// # Why the candidate set must be STRUCTURAL, not parameter-derived
    ///
    /// `named_parameters()` answers "which projections currently carry a
    /// dense `Var<R>`", which on a QUANTIZED (GGUF) checkpoint is nearly
    /// EMPTY: `MaybeQuantLinear::named_parameters()` returns nothing for a
    /// block-quantized projection (the weight has no `Var<R>`, only packed
    /// bytes `quant_matmul` reads directly). Measured on a real VoxCPM2
    /// GGUF, that shrank the candidate set from 577 (dense) to 131,
    /// rejecting a perfectly valid `["q_proj", "v_proj"]` target with
    /// "matched no projection by dot-segment name" — QLoRA's entire
    /// reason to exist is adapting a quantized base, so that checkpoint is
    /// exactly the one this validation must not reject.
    /// [`Self::lora_projection_names`] instead answers "which projections
    /// this tree's `MaybeLoraLinear` FIELDS structurally are", which is the
    /// same 577-projection set on every checkpoint dtype: dense,
    /// block-quantized, or decomposed-quantized. The wrapping itself was
    /// never the bug — [`LoraLinear::new`](crate::nn::LoraLinear::new) sizes
    /// its adapter from [`MaybeQuantLinear::shape`](crate::nn::MaybeQuantLinear::shape),
    /// which works on every variant — only this validation's candidate
    /// source was.
    pub fn apply_lora(
        &mut self,
        targets: &LoraTargets,
        rank: usize,
        alpha: f32,
        device: &R::Device,
    ) -> Result<usize> {
        let candidates = self.lora_projection_names();
        targets.ensure_all_match(&candidates)?;

        let mut adapted = self.feat_encoder.apply_lora_unchecked(
            targets,
            rank,
            alpha,
            device,
            DEFAULT_LOCAL_ENCODER_PREFIX,
        )?;
        adapted += self.base_lm.apply_lora_unchecked(
            targets,
            rank,
            alpha,
            device,
            DEFAULT_MINICPM4_PREFIX,
        )?;
        adapted += self.residual_lm.apply_lora_unchecked(
            targets,
            rank,
            alpha,
            device,
            DEFAULT_RESIDUAL_LM_PREFIX,
        )?;
        adapted += self.feat_decoder.apply_lora_unchecked(
            targets,
            rank,
            alpha,
            device,
            DEFAULT_LOCAL_DIT_PREFIX,
        )?;
        adapted += self
            .fsq
            .apply_lora(targets, rank, alpha, device, FSQ_LAYER_PREFIX)?;
        // Root-level, no prefix — see `Module::named_parameters` above.
        adapted += self.aux.apply_lora(targets, rank, alpha, device, "")?;
        Ok(adapted)
    }

    /// Every dotted projection path [`Self::apply_lora`] would adapt across
    /// the WHOLE model — `feat_encoder`, `base_lm`, `residual_lm`,
    /// `feat_decoder`, `fsq_layer`, and `aux`'s six root-level projections —
    /// INDEPENDENT of whether any of them is dense, block-quantized, or
    /// decomposed-quantized. See [`Self::apply_lora`]'s doc comment for WHY
    /// this must be structural rather than parameter-derived (the GGUF
    /// case).
    ///
    /// Delegates to each sub-model's own `lora_projection_names`, joined at
    /// the SAME prefix constants [`Self::apply_lora`] passes to that same
    /// sub-model's `apply_lora_unchecked` — `DEFAULT_LOCAL_ENCODER_PREFIX`,
    /// `DEFAULT_MINICPM4_PREFIX`, `DEFAULT_RESIDUAL_LM_PREFIX`,
    /// `DEFAULT_LOCAL_DIT_PREFIX`, `FSQ_LAYER_PREFIX`, and `aux`'s bare `""`
    /// — so a path here is never built by separately hand-written logic:
    /// [`Self::apply_lora`] and this walk read the SAME constants in the
    /// SAME order, the only difference being `_unchecked`'s mutable adapt
    /// vs. this method's read-only name collection.
    pub fn lora_projection_names(&self) -> Vec<String> {
        let mut names = self
            .feat_encoder
            .lora_projection_names(DEFAULT_LOCAL_ENCODER_PREFIX);
        names.extend(self.base_lm.lora_projection_names(DEFAULT_MINICPM4_PREFIX));
        names.extend(
            self.residual_lm
                .lora_projection_names(DEFAULT_RESIDUAL_LM_PREFIX),
        );
        names.extend(
            self.feat_decoder
                .lora_projection_names(DEFAULT_LOCAL_DIT_PREFIX),
        );
        names.extend(self.fsq.lora_projection_names(FSQ_LAYER_PREFIX));
        // Root-level, no prefix — see `Module::named_parameters` above.
        names.extend(self.aux.lora_projection_names(""));
        names
    }

    /// THE write-back entry point: apply an optimizer's updated adapter
    /// tensors — keyed by [`TensorId`], e.g.
    /// [`SimpleTrainer::step`](crate::trainer::simple::SimpleTrainer::step)'s
    /// output — back onto every `MaybeLoraLinear` adapter across the whole
    /// model, in place, preserving each adapter's `TensorId`.
    ///
    /// Without this, a training loop that calls `backward` then
    /// `SimpleTrainer::step` never actually updates the model: `step` writes
    /// into the `HashMap<TensorId, Tensor<R>>` it returns, but the `Var`s
    /// this model's `MaybeLoraLinear`s hold are untouched by that write, so
    /// every subsequent forward pass would keep recomputing from the SAME
    /// pre-update weights.
    ///
    /// Unlike [`Self::apply_lora`], this needs no `targets`/`prefix`
    /// threading and no [`LoraTargets::ensure_all_match`] validation:
    /// [`crate::nn::MaybeLoraLinear::load_lora_parameters`] looks each
    /// adapter up by its own stable `TensorId`, which is already unique —
    /// there is no dotted path to match against and no zero-match trap to
    /// guard. Returns the total number of adapter TENSORS written (2 per
    /// adapted projection whose ids are both present in `params`).
    pub fn load_lora_parameters(
        &mut self,
        params: &std::collections::HashMap<TensorId, Tensor<R>>,
    ) -> Result<usize> {
        let mut written = self.feat_encoder.load_lora_parameters(params)?;
        written += self.base_lm.load_lora_parameters(params)?;
        written += self.residual_lm.load_lora_parameters(params)?;
        written += self.feat_decoder.load_lora_parameters(params)?;
        written += self.fsq.load_lora_parameters(params)?;
        written += self.aux.load_lora_parameters(params)?;
        Ok(written)
    }
}

// `load_lora_named` lives in `loader/lora_named.rs`, in its own
// `impl<R: Runtime<DType = DType>> VoxCpm2Model<R>` block — split out to
// keep this file under the crate's 500-line hard limit for
// model-architecture files.
mod lora_named;

// `set_activation_checkpointing` lives in `loader/checkpointing.rs`, in its
// own `impl<R: Runtime<DType = DType>> VoxCpm2Model<R>` block — split out for
// the same 500-line reason as `lora_named` above.
mod checkpointing;

/// Whole-model parameter enumeration for fine-tuning (e.g. LoRA target
/// matching, [`SimpleTrainer`](crate::trainer::simple::SimpleTrainer)'s
/// `HashMap<TensorId, Tensor<R>>` build). Names are checkpoint keys
/// verbatim, reusing each sub-model's own default checkpoint prefix
/// constant rather than a hand-copied literal:
///
/// - `feat_encoder.*`, `base_lm.*`, `residual_lm.*`, `feat_decoder.*` —
///   [`DEFAULT_LOCAL_ENCODER_PREFIX`], [`DEFAULT_MINICPM4_PREFIX`],
///   [`DEFAULT_RESIDUAL_LM_PREFIX`], [`DEFAULT_LOCAL_DIT_PREFIX`].
/// - `fsq_layer.*` — [`FSQ_LAYER_PREFIX`], for `fsq` (`ScalarQuantization`)
///   only.
/// - `aux`'s six projections (`enc_to_lm_proj`, `lm_to_dit_proj`,
///   `res_to_dit_proj`, `fusion_concat_proj`, `stop_proj`, `stop_head`) live
///   at the checkpoint ROOT with no prefix at all — see
///   [`AuxProjections`]'s own `Module` impl — so they are appended
///   unprefixed here, unlike every other sub-model.
///
/// `vae_encoder`/`vae_decoder` are DELIBERATELY EXCLUDED: the AudioVAE is a
/// frozen audio codec loaded from a SEPARATE checkpoint
/// (`audiovae.pth`, see the module docs) that is never a
/// fine-tuning target, and neither `AudioVaeEncoder` nor `AudioVaeDecoder`
/// implements `Module<R>` — their `CausalConv1d`/`EncoderBlock` internals
/// were not audited for this unit. A caller needing to enumerate them
/// would need that follow-up unit first.
impl<R: Runtime<DType = DType>> Module<R> for VoxCpm2Model<R> {
    fn parameters(&self) -> Vec<&Var<R>> {
        let mut params = self.feat_encoder.parameters();
        params.extend(self.base_lm.parameters());
        params.extend(self.residual_lm.parameters());
        params.extend(self.feat_decoder.parameters());
        params.extend(self.fsq.parameters());
        params.extend(self.aux.parameters());
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        let mut params = Vec::new();
        extend_named(
            &mut params,
            DEFAULT_LOCAL_ENCODER_PREFIX,
            self.feat_encoder.named_parameters(),
        );
        extend_named(
            &mut params,
            DEFAULT_MINICPM4_PREFIX,
            self.base_lm.named_parameters(),
        );
        extend_named(
            &mut params,
            DEFAULT_RESIDUAL_LM_PREFIX,
            self.residual_lm.named_parameters(),
        );
        extend_named(
            &mut params,
            DEFAULT_LOCAL_DIT_PREFIX,
            self.feat_decoder.named_parameters(),
        );
        extend_named(&mut params, FSQ_LAYER_PREFIX, self.fsq.named_parameters());
        // Root-level, no prefix — see the impl doc above.
        params.extend(self.aux.named_parameters());
        params
    }
}

#[cfg(test)]
mod tests;
