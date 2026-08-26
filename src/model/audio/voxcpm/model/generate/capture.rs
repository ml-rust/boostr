//! [`StepIntermediates`] and the capturing variant of
//! [`PatchGenerator::step_with_noise`](super::PatchGenerator::step_with_noise),
//! split out of `generate.rs` to stay inside the architecture-file size
//! limit — the same split `prefill.rs` uses for
//! `prefill`/`prefill_capturing` over a shared `prefill_inner`.
//!
//! `step_with_noise_inner` is `generate`-module-private (default visibility),
//! reachable here because this file is a descendant module of `generate`.
//! Tests live in `generate/tests.rs`, which already owns the loop's shared
//! fixtures — not here.

use super::*;
use crate::model::audio::voxcpm::local_dit::cfm_time_span;
use crate::nn::var_contiguous;
use numr::autograd::{var_cat, var_reshape, var_transpose};

/// Per-step intermediates for a gate to compare against the reference,
/// mirroring [`PrefillIntermediates`](super::super::prefill::PrefillIntermediates)
/// but for ONE iteration of the per-patch loop rather than the whole prefill.
pub struct StepIntermediates<R: Runtime> {
    /// Step 1's output, `[1, 2 * hidden]` — two DiT tokens wide.
    pub mu: Var<R>,
    /// Step 3's output, `[1, hidden]`.
    pub curr_embed: Var<R>,
    /// `base_lm.decode_step`'s output BEFORE `fsq`, `[1, hidden]`. `None`
    /// when the stop guard fired first (steps 6-8 did not run — see
    /// [`StepOutcome::Stopped`]).
    pub lm_hidden_pre_fsq: Option<Var<R>>,
    /// The raw `aux.stop(...)` output, `[1, 2]`, ALWAYS present — captured
    /// unconditionally even on iterations the guard would otherwise skip.
    /// See `step_with_noise_inner`'s doc comment for why.
    pub stop_logits: Var<R>,
}

impl<R: Runtime<DType = DType>> PatchGenerator<'_, R> {
    /// [`step_with_noise`](Self::step_with_noise), additionally returning the
    /// per-step [`StepIntermediates`] — `mu`, `curr_embed`,
    /// `lm_hidden_pre_fsq` and `stop_logits` — so a gate can localize a
    /// discontinuity to a specific sub-step instead of only comparing the
    /// loop's end state. See `step_with_noise_inner`'s doc comment for the
    /// capture-only `stop_logits` asymmetry.
    pub fn step_with_noise_capturing<C>(
        &self,
        client: &C,
        state: &mut GenerateState<R>,
        z: &Var<R>,
        options: &GenerateOptions,
    ) -> Result<(StepOutcome, StepIntermediates<R>)>
    where
        C: ModelClient<R> + TypeConversionOps<R>,
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
            + TypeConversionOps<R>,
    {
        let (outcome, intermediates) =
            self.step_with_noise_inner(client, state, z, options, true)?;
        let intermediates = intermediates.ok_or_else(|| Error::InvalidArgument {
            arg: "capture",
            reason: "step_with_noise_inner(capture = true) returned no intermediates".to_string(),
        })?;
        Ok((outcome, intermediates))
    }

    /// Shared body of [`step_with_noise`](Self::step_with_noise) and
    /// [`step_with_noise_capturing`](Self::step_with_noise_capturing),
    /// mirroring `prefill.rs`'s `prefill`/`prefill_capturing` split over
    /// `prefill_inner`. When `capture` is `false` the returned `Option` is
    /// always `None` and nothing extra is allocated or recomputed relative
    /// to the pre-capturing code.
    ///
    /// **Deliberate asymmetry**: step 5's stop check is normally SKIPPED
    /// entirely on iterations where `i <= options.min_len` (`aux.stop` is
    /// not even called, matching the reference and the non-capturing path
    /// exactly). When `capture` is `true`, `aux.stop` is called
    /// UNCONDITIONALLY so [`StepIntermediates::stop_logits`] is populated on
    /// every iteration, including ones the guard would otherwise skip — so a
    /// gate can compare every step. Do NOT "fix" this into a shared
    /// unconditional call: that would make the non-capturing path do work
    /// (and allocate) it does today, on every real generation call.
    pub(super) fn step_with_noise_inner<C>(
        &self,
        client: &C,
        state: &mut GenerateState<R>,
        z: &Var<R>,
        options: &GenerateOptions,
        capture: bool,
    ) -> Result<(StepOutcome, Option<StepIntermediates<R>>)>
    where
        C: ModelClient<R> + TypeConversionOps<R>,
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
            + TypeConversionOps<R>,
    {
        let (patch_size, feat_dim) = (self.config.patch_size, self.config.feat_dim);
        check_patch("z", z, &[1, feat_dim, patch_size])?;
        check_patch(
            "state.prefix_feat_cond",
            &state.prefix_feat_cond,
            &[1, patch_size, feat_dim],
        )?;
        let lm_width = check_row("state.prefill.lm_hidden", &state.prefill.lm_hidden)?;
        check_row(
            "state.prefill.residual_hidden",
            &state.prefill.residual_hidden,
        )?;

        // 1. mu = cat(lm_to_dit_proj(lm_hidden), res_to_dit_proj(residual_hidden)),
        // two DiT tokens wide.
        let from_lm = self
            .aux
            .lm_to_dit_proj
            .forward(client, &state.prefill.lm_hidden)?;
        let from_res = self
            .aux
            .res_to_dit_proj
            .forward(client, &state.prefill.residual_hidden)?;
        let mu = var_cat(&[&from_lm, &from_res], 1, client).map_err(Error::Numr)?;

        // 2. The DiT takes its condition as [1, feat_dim, patch_size] and
        // returns the same layout; `prefix_feat_cond` and the emitted patches
        // are stored [1, patch_size, feat_dim], so transpose on the way in
        // and back on the way out. `var_transpose` yields a strided view and
        // every consumer here reshapes, so materialize both.
        let cond = var_contiguous(&var_transpose(&state.prefix_feat_cond).map_err(Error::Numr)?)?;
        let t_span = cfm_time_span(options.cfm.n_timesteps, options.cfm.sway_sampling_coef)?;
        let solved = self.feat_decoder.solve_euler(
            client,
            z,
            &t_span,
            &mu,
            &cond,
            options.cfm.cfg_value,
            options.cfm.use_cfg_zero_star,
            None,
        )?;
        let pred_feat = var_contiguous(&var_transpose(&solved).map_err(Error::Numr)?)?;

        // 3. The encoder runs on ONE patch: [1, 1, patch_size, feat_dim].
        let single = var_reshape(&pred_feat, &[1, 1, patch_size, feat_dim]).map_err(Error::Numr)?;
        let encoded = self.feat_encoder.forward(client, &single)?;
        let projected = self.aux.enc_to_lm_proj.forward(client, &encoded)?;
        let curr_embed = var_reshape(&projected, &[1, lm_width]).map_err(Error::Numr)?;

        // 4. Emit, and condition the NEXT patch on this one.
        state.patches.push(pred_feat.clone());
        state.prefix_feat_cond = pred_feat;

        // 5. Stop check on the CURRENT `lm_hidden` — the hidden state that
        // produced the patch just emitted, BEFORE step 6 replaces it. The
        // guard is strictly greater than `min_len`, and `i` is the index of
        // the patch just pushed.
        //
        // `capture` asymmetry (deliberate, see this fn's doc comment): the
        // non-capturing path only calls `aux.stop` when `i > options.min_len`
        // and otherwise skips the work entirely, matching the reference and
        // every generation call today. The capturing path calls it on EVERY
        // iteration so `StepIntermediates::stop_logits` is always populated,
        // even on iterations the guard would skip. The guard decision itself
        // (`i > options.min_len`) is unchanged either way.
        let i = state.patches.len() - 1;
        let guard_open = i > options.min_len;
        let stop_logits = if guard_open || capture {
            Some(self.aux.stop(client, &state.prefill.lm_hidden)?)
        } else {
            None
        };
        if guard_open {
            // `stop_logits` is always `Some` here: `guard_open` is one of the
            // two disjuncts above.
            let should_stop = match &stop_logits {
                Some(logits) => stop_predicted(client, logits)?,
                None => false,
            };
            if should_stop {
                // `capture` implies `stop_logits` is `Some` (the disjunction
                // above), so the `(true, None)` arm is unreachable, not a
                // silent data loss.
                let intermediates = match (capture, stop_logits) {
                    (true, Some(logits)) => Some(StepIntermediates {
                        mu: mu.clone(),
                        curr_embed: curr_embed.clone(),
                        lm_hidden_pre_fsq: None,
                        stop_logits: logits,
                    }),
                    _ => None,
                };
                return Ok((StepOutcome::Stopped, intermediates));
            }
        }

        // 6. Step `base_lm`, then fsq. Unlike the prefill's last row, every
        // hidden state from here on IS fsq'd.
        let position = state.prefill.position;
        let stepped = self.base_lm.decode_step(
            client,
            &curr_embed,
            &mut state.prefill.base_cache,
            position,
        )?;
        let captured_lm_hidden_pre_fsq = capture.then(|| stepped.clone());
        state.prefill.lm_hidden = self.fsq.forward(client, &stepped)?;

        // 7. `residual_lm` consumes the POST-fsq hidden state, concatenated
        // with `curr_embed` in THAT order.
        let fused =
            var_cat(&[&state.prefill.lm_hidden, &curr_embed], 1, client).map_err(Error::Numr)?;
        let residual_in = self.aux.fusion_concat_proj.forward(client, &fused)?;
        let residual = self.residual_lm.decode_step(
            client,
            &residual_in,
            &mut state.prefill.residual_cache,
            position,
        )?;
        state.prefill.residual_hidden = residual;

        // 8. One counter, both caches, in lockstep.
        state.prefill.position = position + 1;

        // `capture` implies `stop_logits` is `Some` (the disjunction at step
        // 5), so the `(true, None)` arm is unreachable, not a silent data
        // loss.
        let intermediates = match (capture, stop_logits) {
            (true, Some(logits)) => Some(StepIntermediates {
                mu: mu.clone(),
                curr_embed: curr_embed.clone(),
                lm_hidden_pre_fsq: captured_lm_hidden_pre_fsq,
                stop_logits: logits,
            }),
            _ => None,
        };
        Ok((StepOutcome::Continued, intermediates))
    }
}
