//! [`StepIntermediates`] and the capturing variant of
//! [`PatchGenerator::step_with_noise`](super::PatchGenerator::step_with_noise).
//!
//! `step_with_noise_inner` is `generate`-module-private (default visibility),
//! reachable here because this file is a descendant module of `generate`.

use super::*;
use crate::model::audio::voxcpm::local_dit::cfm_time_span;
use crate::ops::FlashAttentionOps;
use crate::quant::traits::DequantOps;
use numr::autograd::{var_cat, var_reshape};

/// Per-step intermediates for a gate to compare against the reference,
/// mirroring [`PrefillIntermediates`](super::super::prefill::PrefillIntermediates)
/// but for ONE iteration of the per-patch loop rather than the whole prefill.
pub struct StepIntermediates<R: Runtime> {
    /// Step 1's output, `[batch, 2 * hidden]` — two DiT tokens wide.
    pub mu: Var<R>,
    /// Step 3's output, `[batch, hidden]`.
    pub curr_embed: Var<R>,
    /// `base_lm.decode_step`'s output BEFORE `fsq`, `[batch, hidden]`. `None`
    /// when the stop guard fired first (steps 6-8 did not run — see
    /// [`StepOutcome::Stopped`]).
    pub lm_hidden_pre_fsq: Option<Var<R>>,
    /// The raw `aux.stop(...)` output, `[batch, 2]`, ALWAYS present —
    /// captured unconditionally even on iterations the guard would
    /// otherwise skip. See `step_with_noise_inner`'s doc comment for why.
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
            + DequantOps<R>
            + FlashAttentionOps<R>,
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
    /// Every row of the batch runs every sub-step; a finished row's patch is
    /// computed like any other and only its bookkeeping (`patch_len`,
    /// `outcomes`) stands still. Steps 6-8 are skipped exactly when a stop
    /// token fires on this step and no row is left unfinished, which for
    /// one row is the reference's `break`.
    ///
    /// **Deliberate asymmetry**: step 5's stop check is normally SKIPPED
    /// entirely on iterations where `i <= options.min_len` (`aux.stop` is
    /// not even called, matching the reference and the non-capturing path
    /// exactly). When `capture` is `true`, `aux.stop` is called
    /// UNCONDITIONALLY so [`StepIntermediates::stop_logits`] is populated on
    /// every iteration, including ones the guard would otherwise skip — so
    /// a gate can compare every step. Do NOT "fix" this into a shared
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
            + DequantOps<R>
            + FlashAttentionOps<R>,
    {
        let (patch_size, feat_dim) = (self.config.patch_size, self.config.feat_dim);
        let batch = state.batch;
        options.check(batch)?;
        if state.all_finished() {
            return Err(Error::InvalidArgument {
                arg: "state",
                reason: "every row is finished; nothing left to step".to_string(),
            });
        }
        check_patch("z", z, &[batch, patch_size, feat_dim])?;
        check_patch(
            "state.prefix_feat_cond",
            &state.prefix_feat_cond,
            &[batch, patch_size, feat_dim],
        )?;
        let lm_width = check_row("state.prefill.lm_hidden", &state.prefill.lm_hidden, batch)?;
        check_row(
            "state.prefill.residual_hidden",
            &state.prefill.residual_hidden,
            batch,
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

        // 2. The DiT takes `z` and its condition as [batch, patch_size,
        // feat_dim] and returns the same layout — the layout
        // `prefix_feat_cond` and the emitted patches are stored in, so
        // nothing is transposed here.
        let t_span = cfm_time_span(options.cfm.n_timesteps, options.cfm.sway_sampling_coef)?;
        // Inference-only entry: one CUDA graph launch per patch on CUDA, the
        // eager loop elsewhere. Fine-tuning never comes through here (see
        // `train/cfm.rs`), so no autograd tape is lost.
        let pred_feat = self.feat_decoder.solve_euler_graphed(
            client,
            z,
            &t_span,
            &mu,
            &state.prefix_feat_cond,
            options.cfm.cfg_value,
            options.cfm.use_cfg_zero_star,
            None,
        )?;

        // 3. The encoder runs on ONE patch per row: [batch, 1, patch_size,
        // feat_dim].
        let single =
            var_reshape(&pred_feat, &[batch, 1, patch_size, feat_dim]).map_err(Error::Numr)?;
        let encoded = self.feat_encoder.forward(client, &single)?;
        let projected = self.aux.enc_to_lm_proj.forward(client, &encoded)?;
        let curr_embed = var_reshape(&projected, &[batch, lm_width]).map_err(Error::Numr)?;

        // 4. Emit, and condition the NEXT patch on this one. Only a row still
        // generating counts the patch as its own.
        state.patches.push(pred_feat.clone());
        state.prefix_feat_cond = pred_feat;
        for b in 0..batch {
            if !state.finished(b) {
                state.patch_len[b] += 1;
            }
        }

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
        let mut stopped_now = false;
        if guard_open {
            // `stop_logits` is always `Some` here: `guard_open` is one of the
            // two disjuncts above.
            let stops = match &stop_logits {
                Some(logits) => stop_predicted(client, logits, batch)?,
                None => vec![false; batch],
            };
            for (b, &stop) in stops.iter().enumerate() {
                if stop && !state.finished(b) {
                    state.outcomes[b] = Some(GenerateOutcome::StopToken);
                    stopped_now = true;
                }
            }
        }
        // A row still open after the stop check finishes on its cap; a stop
        // on the same step wins, as it does in the reference's loop order.
        for b in 0..batch {
            if !state.finished(b) && state.patch_len[b] >= options.row(b).max_len {
                state.outcomes[b] = Some(GenerateOutcome::MaxLen);
            }
        }
        if stopped_now && state.all_finished() {
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

        // 6. Step `base_lm`, then fsq. Unlike the prefill's last row, every
        // hidden state from here on IS fsq'd.
        let position = state.prefill.position;
        let kv_start = state.prefill.kv_start.as_ref();
        let stepped = self.base_lm.decode_step(
            client,
            &curr_embed,
            &mut state.prefill.base_cache,
            position,
            kv_start,
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
            kv_start,
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

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use super::*;
    use crate::test_utils::cpu_setup;

    /// Capturing must not perturb step 1-8 arithmetic. Guard OPEN (`i >
    /// min_len`), so `aux.stop` runs on both paths here — the closed-guard
    /// asymmetry is covered separately below.
    #[test]
    fn capturing_and_non_capturing_paths_agree() {
        let (client, device) = cpu_setup();
        let fx = fixture(false, &device);
        let generator = fx.generator();
        let opts = options(0, 8);

        let mut plain = state(&fx, &device);
        let outcome = generator
            .step_with_noise(&client, &mut plain, &noise(0.2, &device), &opts)
            .expect("plain step");

        let mut captured = state(&fx, &device);
        let (captured_outcome, intermediates) = generator
            .step_with_noise_capturing(&client, &mut captured, &noise(0.2, &device), &opts)
            .expect("capturing step");

        assert_eq!(outcome, captured_outcome);
        assert_eq!(values(&plain.patches[0]), values(&captured.patches[0]));
        assert_eq!(
            values(&plain.prefill.lm_hidden),
            values(&captured.prefill.lm_hidden)
        );
        assert_eq!(
            values(&plain.prefill.residual_hidden),
            values(&captured.prefill.residual_hidden)
        );
        assert_eq!(plain.prefill.position, captured.prefill.position);

        // Non-degenerate: this must not pass by both sides silently zeroing out.
        assert!(values(&intermediates.mu).iter().any(|v| v.abs() > 1e-6));
        assert!(
            values(&intermediates.curr_embed)
                .iter()
                .any(|v| v.abs() > 1e-6)
        );
        assert!(
            intermediates.lm_hidden_pre_fsq.is_some(),
            "the guard was open, so steps 6-8 ran and lm_hidden_pre_fsq must be captured"
        );
        assert_eq!(intermediates.stop_logits.shape(), &[1, 2]);
    }

    /// Guard CLOSED (`i <= min_len`): the non-capturing path skips `aux.stop`
    /// entirely, but capturing must still populate `stop_logits`, without
    /// changing any other output.
    #[test]
    fn capturing_computes_stop_logits_even_when_the_guard_is_closed() {
        let (client, device) = cpu_setup();
        let fx = fixture(false, &device);
        let generator = fx.generator();
        let opts = options(2, 8);

        let mut plain = state(&fx, &device);
        generator
            .step_with_noise(&client, &mut plain, &noise(0.2, &device), &opts)
            .expect("plain step");

        let mut captured = state(&fx, &device);
        let (outcome, intermediates) = generator
            .step_with_noise_capturing(&client, &mut captured, &noise(0.2, &device), &opts)
            .expect("capturing step");

        assert_eq!(
            outcome,
            StepOutcome::Continued,
            "i = 0 is below min_len = 2"
        );
        assert_eq!(intermediates.stop_logits.shape(), &[1, 2]);
        assert_eq!(values(&plain.patches[0]), values(&captured.patches[0]));
        assert_eq!(
            values(&plain.prefill.lm_hidden),
            values(&captured.prefill.lm_hidden)
        );
    }
}
