//! Unit A of the VoxCPM2 end-to-end orchestrator: reference-audio encoding
//! and the deterministic two-LM prefill.
//!
//! ```text
//! ref_wav_16k -> pad to patch_size*640 -> AudioVAE -> fold -> [T_ref, 4, 64]
//!             -> reference prefix + prompt (SequenceLayout)
//!             -> base_lm.prefill    -> fsq blend -> lm_hidden
//!             -> residual_lm.prefill              -> residual_hidden
//! ```
//!
//! The per-patch sampling loop, the stop logic and the VAE decode path are a
//! LATER unit and are deliberately absent.
//!
//! # Traps this file exists to get right
//!
//! - `lm_hidden` is the LAST row of the post-blend `enc_outputs`. In
//!   reference mode that row is a TEXT position, so the blend left it
//!   UN-fsq'd. Do NOT apply `fsq` to it again.
//! - `feat_encoder` runs over ALL `S` rows, text rows included, whose patches
//!   are zeros. Skipping them is not an optimization; it changes the result.
//! - `fusion_concat_proj`'s argument is `cat(enc_outputs, audio_mask *
//!   feat_embed)` in THAT order. Swapping the halves of a 4096-wide concat is
//!   shape-valid and silently computes a different model.
//! - `enc_outputs` is NOT masked again inside the fusion. Only `feat_embed`
//!   is.
//! - Both KV caches come back primed to `current_length == S`. The later
//!   sampling loop advances ONE shared position counter, starting at
//!   [`PrefillState::position`] (`== S`), across both caches.

use crate::error::{Error, Result};
use crate::inference::LayeredKvCache;
use crate::model::audio::voxcpm::client::VoxCpmClient;
use crate::model::audio::voxcpm::model::loader::VoxCpm2Model;
use crate::model::audio::voxcpm::model::patches::{fold_patches, pad_to_multiple};
use crate::model::audio::voxcpm::model::sequence::SequenceLayout;
use crate::model::traits::ModelClient;
use crate::nn::var_contiguous;
use crate::quant::traits::DequantOps;
use numr::autograd::{Var, var_add, var_cat, var_mul, var_narrow, var_reshape};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Everything the per-patch sampling loop needs from the prefill.
pub struct PrefillState<R: Runtime> {
    /// `[1, lm_hidden]` — the LAST row of the post-blend `enc_outputs`,
    /// un-fsq'd because it is a text position.
    pub lm_hidden: Var<R>,
    /// `[1, lm_hidden]` — the LAST row of `residual_lm`'s output.
    pub residual_hidden: Var<R>,
    /// `base_lm`'s cache, primed to `current_length == position`.
    pub base_cache: LayeredKvCache<R>,
    /// `residual_lm`'s cache, primed to the SAME `current_length`.
    pub residual_cache: LayeredKvCache<R>,
    /// `S`. The single shared position counter for the later decode loop:
    /// both caches advance together from here.
    pub position: usize,
    /// Present only when the prefill ran via
    /// [`VoxCpm2Model::prefill_capturing`]; always `None` on the plain
    /// [`VoxCpm2Model::prefill`] path, which allocates nothing extra.
    pub intermediates: Option<PrefillIntermediates<R>>,
}

/// Full-sequence intermediates, for the gate example to compare against the
/// reference. Each is `[1, S, _]`.
pub struct PrefillIntermediates<R: Runtime> {
    /// `text_mask * text_embed + audio_mask * feat_embed`, `[1, S, 2048]`.
    pub combined_embed: Var<R>,
    /// `base_lm`'s output AFTER the fsq blend, `[1, S, 2048]`.
    pub enc_outputs: Var<R>,
    /// `enc_to_lm_proj(feat_encoder(audio_feat))`, UNMASKED, `[1, S, 2048]`.
    pub feat_embed: Var<R>,
    /// `fusion_concat_proj(...)`, i.e. what `residual_lm` was prefilled with,
    /// `[1, S, 2048]`.
    pub residual_enc_inputs: Var<R>,
}

impl<R: Runtime<DType = DType>> VoxCpm2Model<R> {
    /// Encode the reference waveform into per-patch features `[T_ref,
    /// patch_size, feat_dim]`.
    ///
    /// `ref_wav_16k` is mono 16 kHz PCM. It is right-padded with zeros to a
    /// multiple of `patch_size * 640` BEFORE the VAE encode — see
    /// [`VoxCpm2Config::ref_pad_multiple`](super::config::VoxCpm2Config::ref_pad_multiple)
    /// for why the VAE's own 640 modulus is not enough.
    ///
    /// Voice-clone mode uses the reference AUDIO only. There is no reference
    /// transcript on this path.
    pub fn encode_reference<C>(&self, client: &C, ref_wav_16k: &[f32]) -> Result<Tensor<R>>
    where
        C: VoxCpmClient<R>,
    {
        if ref_wav_16k.is_empty() {
            return Err(Error::InvalidArgument {
                arg: "ref_wav_16k",
                reason: "expected at least 1 sample, got 0".to_string(),
            });
        }
        let padded = pad_to_multiple(ref_wav_16k, self.config.ref_pad_multiple())?;
        let wave = Tensor::<R>::from_slice(padded.as_ref(), &[1, 1, padded.len()], self.device()?)?;
        let latent = self.vae_encoder.forward(client, &wave)?;
        fold_patches(&latent, self.config.patch_size, self.config.feat_dim)
    }

    /// Prefill both LMs over the reference prefix and the prompt.
    ///
    /// `ref_feat` is [`encode_reference`](Self::encode_reference)'s output,
    /// `[T_ref, patch_size, feat_dim]`. `text_token_ids` is the already
    /// tokenized prompt and must end with
    /// [`AUDIO_START_ID`](super::config::AUDIO_START_ID) — boostr does not
    /// tokenize here. `max_length` sizes both KV caches; it must be at least
    /// `S = T_ref + 2 + text_token_ids.len()` and should leave room for
    /// however many patches the later sampling loop will generate.
    ///
    /// [`PrefillState::intermediates`] is `None`; use
    /// [`prefill_capturing`](Self::prefill_capturing) to get them.
    pub fn prefill<C>(
        &self,
        client: &C,
        ref_feat: &Tensor<R>,
        text_token_ids: &[u32],
        max_length: usize,
    ) -> Result<PrefillState<R>>
    where
        C: ModelClient<R> + TypeConversionOps<R> + DequantOps<R> + 'static,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + DequantOps<R>,
    {
        self.prefill_inner(client, ref_feat, text_token_ids, max_length, false)
    }

    /// [`prefill`](Self::prefill), additionally returning the full-sequence
    /// intermediates in [`PrefillState::intermediates`].
    ///
    /// The values are the SAME tensors the plain path computes and drops —
    /// capturing them keeps them alive, it does not recompute anything.
    pub fn prefill_capturing<C>(
        &self,
        client: &C,
        ref_feat: &Tensor<R>,
        text_token_ids: &[u32],
        max_length: usize,
    ) -> Result<PrefillState<R>>
    where
        C: ModelClient<R> + TypeConversionOps<R> + DequantOps<R> + 'static,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + DequantOps<R>,
    {
        self.prefill_inner(client, ref_feat, text_token_ids, max_length, true)
    }

    fn prefill_inner<C>(
        &self,
        client: &C,
        ref_feat: &Tensor<R>,
        text_token_ids: &[u32],
        max_length: usize,
        capture: bool,
    ) -> Result<PrefillState<R>>
    where
        C: ModelClient<R> + TypeConversionOps<R> + DequantOps<R> + 'static,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + DequantOps<R>,
    {
        let (patch_size, feat_dim) = (self.config.patch_size, self.config.feat_dim);
        let ref_shape = ref_feat.shape().to_vec();
        if ref_shape.len() != 3 || ref_shape[1] != patch_size || ref_shape[2] != feat_dim {
            return Err(Error::InvalidArgument {
                arg: "ref_feat",
                reason: format!("expected [T_ref, {patch_size}, {feat_dim}], got {ref_shape:?}"),
            });
        }
        let layout = SequenceLayout::build(ref_shape[0], text_token_ids)?;
        let seq_len = layout.seq_len();
        if max_length < seq_len {
            return Err(Error::InvalidArgument {
                arg: "max_length",
                reason: format!(
                    "expected at least the prefill length S ({seq_len}), got {max_length}"
                ),
            });
        }

        let (dtype, device) = self.lm_dtype_device()?;

        // audio_feat: z1 ++ ref_feat ++ z1 ++ zeros(text_length), i.e. one
        // leading zero patch and `1 + text_length` trailing ones.
        let ref_patches = Var::new(
            ref_feat
                .to_dtype(dtype)?
                .reshape(&[1, layout.t_ref, patch_size, feat_dim])?,
            false,
        );
        let head = Var::new(
            Tensor::<R>::zeros(&[1, 1, patch_size, feat_dim], dtype, device)?,
            false,
        );
        let tail = Var::new(
            Tensor::<R>::zeros(
                &[1, 1 + layout.text_length, patch_size, feat_dim],
                dtype,
                device,
            )?,
            false,
        );
        let audio_feat = var_cat(&[&head, &ref_patches, &tail], 1, client)?;

        // feat_encoder runs over ALL S rows, text rows included — their
        // patches are zeros, not absent.
        let encoded = self.feat_encoder.forward(client, &audio_feat)?;
        let feat_embed = self.aux.enc_to_lm_proj.forward(client, &encoded)?;

        // `scale_emb` is 1.0 on this checkpoint (muP off), so the lookup is
        // UNSCALED — `MiniCpm4Model::embed` already leaves it alone.
        let ids = Tensor::<R>::from_slice(&layout.token_ids, &[1, seq_len], device)?;
        let text_embed = self.base_lm.embed(client, &ids)?;

        let text_mask = mask_var::<R>(&layout.text_mask, dtype, device)?;
        let audio_mask = mask_var::<R>(&layout.audio_mask, dtype, device)?;

        // The two masks are complementary (checked in `SequenceLayout`), so
        // this SUM picks exactly one term per position.
        let masked_feat = var_mul(&feat_embed, &audio_mask, client)?;
        let combined_embed = var_add(
            &var_mul(&text_embed, &text_mask, client)?,
            &masked_feat,
            client,
        )?;

        let mut base_cache = self.base_lm.new_kv_cache(1, max_length)?;
        let enc = self
            .base_lm
            .prefill(client, &combined_embed, &mut base_cache)?;

        // fsq on AUDIO positions, identity on TEXT positions.
        let enc_outputs = var_add(
            &var_mul(&self.fsq.forward(client, &enc)?, &audio_mask, client)?,
            &var_mul(&enc, &text_mask, client)?,
            client,
        )?;

        // LAST row. In reference mode this is a text position, hence
        // un-fsq'd by the blend above; do NOT fsq it again.
        let lm_hidden = last_row(&enc_outputs, seq_len)?;

        // Argument order is (enc_outputs, masked feat_embed) — and
        // `enc_outputs` is NOT masked again here.
        let fused = var_cat(&[&enc_outputs, &masked_feat], 2, client)?;
        let residual_enc_inputs = self.aux.fusion_concat_proj.forward(client, &fused)?;

        let mut residual_cache = self.residual_lm.new_kv_cache(1, max_length)?;
        let residual_out =
            self.residual_lm
                .prefill(client, &residual_enc_inputs, &mut residual_cache)?;
        let residual_hidden = last_row(&residual_out, seq_len)?;

        let intermediates = capture.then(|| PrefillIntermediates {
            combined_embed,
            enc_outputs,
            feat_embed,
            residual_enc_inputs,
        });

        Ok(PrefillState {
            lm_hidden,
            residual_hidden,
            base_cache,
            residual_cache,
            position: seq_len,
            intermediates,
        })
    }
}

/// `[1, S, hidden]` -> `[1, hidden]`, taking row `seq_len - 1`.
///
/// `narrow` yields a strided view; `reshape` needs it materialized first.
fn last_row<R: Runtime<DType = DType>>(x: &Var<R>, seq_len: usize) -> Result<Var<R>>
where
    R::Client: TensorOps<R>,
{
    let shape = x.shape().to_vec();
    if shape.len() != 3 || shape[1] != seq_len {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: format!("expected [batch, {seq_len}, hidden], got {shape:?}"),
        });
    }
    let row = var_contiguous(&var_narrow(x, 1, seq_len - 1, 1)?)?;
    Ok(var_reshape(&row, &[shape[0], shape[2]])?)
}

/// Upload a per-position mask as `[1, S, 1]` in `dtype`, ready to broadcast
/// against `[1, S, hidden]`.
fn mask_var<R: Runtime<DType = DType>>(
    mask: &[f32],
    dtype: DType,
    device: &R::Device,
) -> Result<Var<R>>
where
    R::Client: TypeConversionOps<R>,
{
    let tensor = Tensor::<R>::from_slice(mask, &[1, mask.len(), 1], device)?;
    Ok(Var::new(tensor.to_dtype(dtype)?, false))
}
