//! Teacher-forced batched counterpart of the per-patch [`super`] loop: given
//! GROUND-TRUTH patches, produce every patch's DiT conditioning (`mu`,
//! `cond`) in ONE batched pass instead of stepping `base_lm`/`residual_lm`
//! one position at a time. The forward half of a training step — loss,
//! backward and the optimizer are a LATER unit and are deliberately absent.
//!
//! # Why this reproduces the autoregressive loop
//!
//! Under teacher forcing every `pred_feat_i` is KNOWN up front (the ground
//! truth, not a CFM draw), so step 3's `curr_embed_i` depends only on that
//! known input — every patch's `curr_embed` is computed by ONE
//! `feat_encoder` call over all `T` patches at once, instead of one call per
//! loop iteration.
//!
//! [`MiniCpm4Model::decode_step`] is the KV-cached, one-position-at-a-time
//! twin of [`MiniCpm4Model::forward`] (the full-sequence causal forward).
//! With every `curr_embed` known ahead of time, ONE `forward` call over the
//! whole sequence yields every position's hidden state — the standard
//! teacher-forced-training trick, and the reason this file exists instead of
//! looping [`super::PatchGenerator::step_with_noise`] with the ground truth
//! spliced in.
//!
//! # The prefix: re-run, don't reuse the cache
//!
//! [`MiniCpm4Model::forward`] takes no KV cache — it recomputes every
//! position from scratch. For correct causal attention at the first patch
//! position (which must attend back through the whole prefill prefix), the
//! prefix embeddings have to be part of THIS call's input, not read out of
//! [`PrefillState`]'s already-built cache. So this re-runs the prefix
//! through `forward` alongside the patch positions in the SAME call (the
//! unit brief's option (a)) rather than splicing new positions into the
//! prefill's KV cache (option (b), faster in principle, but
//! `decode_step`'s cache write-order coupling makes it substantially more
//! invasive for a training forward that is not on any latency-critical
//! path — not attempted here).
//!
//! The prefix embeddings are [`PrefillIntermediates::combined_embed`] /
//! [`PrefillIntermediates::residual_enc_inputs`] — the SAME tensors
//! `base_lm`/`residual_lm` were prefilled with — so [`PrefillState`] must
//! carry [`PrefillState::intermediates`] (i.e. come from
//! [`VoxCpm2Model::prefill_capturing`]) whenever the prefix is non-empty
//! (`prefill.position > 0`). A `position == 0` prefill has no prefix to
//! re-run and accepts `intermediates: None`.
//!
//! # The shift
//!
//! `mu_i` conditions on the hidden state PRODUCED BEFORE patch `i` was
//! emitted — the loop's step 1 reads `state.prefill.lm_hidden`/
//! `residual_hidden` BEFORE steps 6-7 overwrite them for the NEXT
//! iteration. So `mu_0` uses the prefill's own last-row hidden states
//! verbatim, and `mu_i` (`i >= 1`) uses the hidden state this call computed
//! from `curr_embed_{i-1}` — a plain shift-by-one over the batched output.
//! Getting this shift wrong trains the model to predict the wrong patch,
//! and nothing downstream catches it.

use super::*;
use crate::nn::var_contiguous;
use crate::quant::traits::DequantOps;
use numr::autograd::{var_cat, var_narrow, var_reshape};

/// Every patch's DiT conditioning, teacher-forced from ground-truth patches
/// in ONE batched pass. Mirrors [`StepIntermediates`]'s `mu`/`curr_embed`,
/// stacked over `T` patches instead of one per loop iteration.
pub struct TeacherForcedConditioning<R: Runtime> {
    /// `[T, 2 * dit_hidden]` — per-patch DiT conditioning (step 1's `mu`,
    /// batched, shifted — see the module docs).
    pub mu: Var<R>,
    /// `[T, patch_size, feat_dim]` — per-patch DiT prefix condition (step
    /// 2's `cond`, batched): zeros for patch 0, `target_patches[i - 1]`
    /// otherwise.
    pub cond: Var<R>,
    /// `[T, lm_hidden]` — per-patch LM input embeddings (step 3's
    /// `curr_embed`, batched). Kept for tests/debug; nothing past this call
    /// requires it.
    pub curr_embed: Var<R>,
    /// `[1, T, lm_hidden]` — per-patch stop-head input: the SAME shifted
    /// hidden state `mu`'s `lm_to_dit_proj` half reads (step 1's row,
    /// pre-`lm_to_dit_proj`), because step 5 in `generate.rs`'s per-patch
    /// loop (`aux.stop(client, &state.prefill.lm_hidden)`) reads that
    /// identical CURRENT `lm_hidden` before steps 6-7 overwrite it for the
    /// next iteration. This is `lm_shifted` below, exposed rather than
    /// recomputed, so `Self::stop_loss` never re-runs `base_lm`.
    pub lm_hidden: Var<R>,
}

impl<R: Runtime<DType = DType>> PatchGenerator<'_, R> {
    /// Teacher-forced counterpart of the per-patch loop's steps 1-3 and 6-7,
    /// batched over every patch in ONE pass. See the module docs for why
    /// this reproduces [`Self::step_with_noise`] (up to float
    /// reassociation) when `target_patches` is fed the loop's own output.
    ///
    /// `feat_decoder` (the CFM estimator) is UNUSED here: teacher forcing
    /// means `pred_feat` is already known, so nothing samples it. This is
    /// why `teacher_forced_conditioning` lives on [`PatchGenerator`] rather
    /// than [`VoxCpm2Model`] — it needs exactly the sub-models
    /// [`VoxCpm2Model::patch_generator`] borrows, and none of the AudioVAE
    /// fields `VoxCpm2Model` also carries.
    ///
    /// `target_patches` is `[T, patch_size, feat_dim]`, `T >= 1`, ground
    /// truth (no grad needed — it is a target, never trained). `prefill` is
    /// borrowed, not mutated: this recomputes everything the loop would
    /// have produced from it without advancing it. When `prefill.position >
    /// 0`, `prefill.intermediates` must be `Some` (built via
    /// [`VoxCpm2Model::prefill_capturing`]) — the prefix embeddings live
    /// there, not in the KV caches `forward` cannot read. `position == 0`
    /// needs no prefix and accepts `intermediates: None`.
    ///
    /// Errors on a shape-mismatched `target_patches`, on a malformed
    /// `prefill` row, or on a non-empty prefix with no captured
    /// intermediates. Never panics.
    pub fn teacher_forced_conditioning<C>(
        &self,
        client: &C,
        prefill: &PrefillState<R>,
        target_patches: &Tensor<R>,
    ) -> Result<TeacherForcedConditioning<R>>
    where
        C: ModelClient<R> + TypeConversionOps<R> + 'static,
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
            + TypeConversionOps<R>
            + DequantOps<R>,
    {
        let (patch_size, feat_dim) = (self.config.patch_size, self.config.feat_dim);
        let shape = target_patches.shape().to_vec();
        if shape.len() != 3 || shape[0] == 0 || shape[1] != patch_size || shape[2] != feat_dim {
            return Err(Error::InvalidArgument {
                arg: "target_patches",
                reason: format!("expected [T >= 1, {patch_size}, {feat_dim}], got {shape:?}"),
            });
        }
        let t = shape[0];
        let lm_width = check_row("prefill.lm_hidden", &prefill.lm_hidden)?;
        check_row("prefill.residual_hidden", &prefill.residual_hidden)?;

        let hidden = prefill.lm_hidden.tensor();
        let (dtype, device) = (hidden.dtype(), hidden.device());
        let patches = Var::new(target_patches.to_dtype(dtype)?, false);

        // Step 3, batched: every curr_embed in one `feat_encoder` call.
        let patches_4d = var_reshape(&patches, &[1, t, patch_size, feat_dim])?;
        let encoded = self.feat_encoder.forward(client, &patches_4d)?;
        let curr_embed_full = self.aux.enc_to_lm_proj.forward(client, &encoded)?;

        // The prefix `base_lm`/`residual_lm` saw before patch 0 — required
        // only when there IS one. See the module docs.
        let s = prefill.position;
        let prefix = if s == 0 {
            None
        } else {
            Some(
                prefill
                    .intermediates
                    .as_ref()
                    .ok_or_else(|| Error::InvalidArgument {
                        arg: "prefill.intermediates",
                        reason: format!(
                            "prefill.position is {s} (> 0) but prefill.intermediates is None; \
                             teacher_forced_conditioning needs the prefix embeddings \
                             VoxCpm2Model::prefill_capturing captures — build the \
                             PrefillState with prefill_capturing, not prefill"
                        ),
                    })?,
            )
        };

        // Step 6, batched: run `base_lm` over prefix ++ every curr_embed in
        // ONE full-sequence forward (no KV cache — see the module docs for
        // why this re-runs the prefix instead of splicing the prefill's
        // cache), then slice out the `T` patch positions and fsq them
        // exactly as step 6 does per-iteration.
        let lm_full_in = match prefix {
            Some(inter) => var_cat(&[&inter.combined_embed, &curr_embed_full], 1, client)?,
            None => curr_embed_full.clone(),
        };
        let lm_full_out = self.base_lm.forward(client, &lm_full_in)?;
        let lm_pre_fsq = var_contiguous(&var_narrow(&lm_full_out, 1, s, t)?)?;
        let lm_post_fsq = self.fsq.forward(client, &lm_pre_fsq)?;

        // Step 7, batched: `fusion_concat_proj(cat(lm_hidden, curr_embed))`
        // — CONCAT ORDER LOAD-BEARING, `(lm_hidden, curr_embed)`, and
        // `lm_hidden` is the POST-fsq value, exactly like the loop.
        let fused = var_cat(&[&lm_post_fsq, &curr_embed_full], 2, client)?;
        let residual_in = self.aux.fusion_concat_proj.forward(client, &fused)?;
        let res_full_in = match prefix {
            Some(inter) => var_cat(&[&inter.residual_enc_inputs, &residual_in], 1, client)?,
            None => residual_in,
        };
        let res_full_out = self.residual_lm.forward(client, &res_full_in)?;
        let res_out = var_contiguous(&var_narrow(&res_full_out, 1, s, t)?)?;

        // Step 1, batched, WITH THE SHIFT: mu_i reads the hidden state from
        // BEFORE patch i — the prefill's own row for i = 0, this call's row
        // (i - 1) for i >= 1. Getting this off by one trains the model on
        // the wrong patch.
        let prev_lm = var_reshape(&prefill.lm_hidden, &[1, 1, lm_width])?;
        let prev_res = var_reshape(&prefill.residual_hidden, &[1, 1, lm_width])?;
        let lm_shifted = if t > 1 {
            let head = var_contiguous(&var_narrow(&lm_post_fsq, 1, 0, t - 1)?)?;
            var_cat(&[&prev_lm, &head], 1, client)?
        } else {
            prev_lm
        };
        let res_shifted = if t > 1 {
            let head = var_contiguous(&var_narrow(&res_out, 1, 0, t - 1)?)?;
            var_cat(&[&prev_res, &head], 1, client)?
        } else {
            prev_res
        };

        let mu_lm = self.aux.lm_to_dit_proj.forward(client, &lm_shifted)?;
        let mu_res = self.aux.res_to_dit_proj.forward(client, &res_shifted)?;
        let mu_full = var_contiguous(&var_cat(&[&mu_lm, &mu_res], 2, client)?)?;
        let mu_width = mu_full.shape()[2];
        let mu = var_reshape(&mu_full, &[t, mu_width])?;

        let curr_embed = var_reshape(&var_contiguous(&curr_embed_full)?, &[t, lm_width])?;

        // Step 4, batched, WITH THE SAME SHIFT AS `mu`: cond_i is zeros for
        // i = 0 (the text-pad patch, never the reference audio's tail — see
        // `generate.rs`'s module docs) and `target_patches[i - 1]` for
        // i >= 1.
        let zeros = Var::new(
            Tensor::<R>::zeros(&[1, patch_size, feat_dim], dtype, device)?,
            false,
        );
        let cond = if t > 1 {
            let head = var_contiguous(&var_narrow(&patches, 0, 0, t - 1)?)?;
            var_cat(&[&zeros, &head], 0, client)?
        } else {
            zeros
        };

        Ok(TeacherForcedConditioning {
            mu,
            cond,
            curr_embed,
            lm_hidden: lm_shifted,
        })
    }
}
