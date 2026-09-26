//! [`StepIntermediates`] and the capturing variant of
//! [`PatchGenerator::step_with_noise`](super::PatchGenerator::step_with_noise).
//!
//! `step_with_noise_inner` is `generate`-module-private (default visibility),
//! reachable here because this file is a descendant module of `generate`.

use super::*;
use crate::model::audio::voxcpm::local_dit::cfm_time_span;
use crate::ops::FlashAttentionOps;
use crate::quant::traits::DequantOps;
use numr::autograd::{var_cat, var_narrow, var_reshape};

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
    /// captured unconditionally, including on iterations the guard otherwise
    /// skips. See `step_with_noise_inner`'s doc comment for why.
    pub stop_logits: Var<R>,
}

impl<R: Runtime<DType = DType>> PatchGenerator<'_, R> {
    /// [`step_with_noise`](Self::step_with_noise), additionally returning the
    /// per-step [`StepIntermediates`] — `mu`, `curr_embed`,
    /// `lm_hidden_pre_fsq` and `stop_logits` — so a gate localizes a
    /// discontinuity to one sub-step. See `step_with_noise_inner` for the
    /// capture-only `stop_logits` asymmetry.
    pub fn step_with_noise_capturing<C>(
        &self,
        client: &C,
        state: &mut GenerateState<R>,
        z: &Var<R>,
        options: &GenerateOptions,
    ) -> Result<(StepOutcome, StepIntermediates<R>)>
    where
        C: ModelClient<R> + TypeConversionOps<R> + HostCopyOps<R> + 'static,
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
    /// `prefill_inner`. With `capture` `false` the returned `Option` is
    /// always `None` and nothing extra is allocated.
    ///
    /// Every row runs every sub-step. A finished row's patch is computed like
    /// any other and only its bookkeeping (`patch_len`, `outcomes`) stands
    /// still. Steps 6-8 are skipped exactly when a stop token fires on this
    /// step and no row is left unfinished.
    ///
    /// **Deliberate asymmetry**: step 5's stop check is SKIPPED on iterations
    /// where `i <= options.min_len`, so `aux.stop` is not called. With
    /// `capture` `true`, `aux.stop` runs UNCONDITIONALLY so
    /// [`StepIntermediates::stop_logits`] is populated on every iteration. Do
    /// NOT merge the two into one unconditional call: that adds work and an
    /// allocation to every real generation call.
    pub(super) fn step_with_noise_inner<C>(
        &self,
        client: &C,
        state: &mut GenerateState<R>,
        z: &Var<R>,
        options: &GenerateOptions,
        capture: bool,
    ) -> Result<(StepOutcome, Option<StepIntermediates<R>>)>
    where
        C: ModelClient<R> + TypeConversionOps<R> + HostCopyOps<R> + 'static,
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
        // The CEILING: what the per-row host vectors and the options' `rows`
        // list are sized at.
        let ceiling = state.batch;
        // The ALLOCATION width: the rows every stored device tensor holds.
        // The shape checks below read this, never the ceiling.
        let batch = state.alloc_width;
        // The STEPPED width: the prefix of slots this iteration runs. The
        // slots at or above it hold no request and are untouched below.
        let rows = state.active_width;
        options.check(ceiling)?;
        if state.all_finished() {
            return Err(Error::InvalidArgument {
                arg: "state",
                reason: "every row is finished; nothing left to step".to_string(),
            });
        }
        check_patch("z", z, &[rows, patch_size, feat_dim])?;
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

        // The three per-step inputs, narrowed to the stepped rows. Each is
        // stored at the allocation width and the stepped rows are its prefix,
        // so a dim-0 narrow moves no bytes.
        let lm_hidden = var_narrow(&state.prefill.lm_hidden, 0, 0, rows).map_err(Error::Numr)?;
        let residual_hidden =
            var_narrow(&state.prefill.residual_hidden, 0, 0, rows).map_err(Error::Numr)?;
        let prefix_cond = var_narrow(&state.prefix_feat_cond, 0, 0, rows).map_err(Error::Numr)?;

        // 1. mu = cat(lm_to_dit_proj(lm_hidden), res_to_dit_proj(residual_hidden)),
        // two DiT tokens wide.
        let from_lm = self.aux.lm_to_dit_proj.forward(client, &lm_hidden)?;
        let from_res = self.aux.res_to_dit_proj.forward(client, &residual_hidden)?;
        let mu = var_cat(&[&from_lm, &from_res], 1, client).map_err(Error::Numr)?;

        // 2. The DiT takes `z` and its condition as [rows, patch_size,
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
            &prefix_cond,
            options.cfm.cfg_value,
            options.cfm.use_cfg_zero_star,
            None,
        )?;

        // 3. The encoder runs on ONE patch per stepped row: [rows, 1,
        // patch_size, feat_dim].
        let single =
            var_reshape(&pred_feat, &[rows, 1, patch_size, feat_dim]).map_err(Error::Numr)?;
        let encoded = self.feat_encoder.forward(client, &single)?;
        let projected = self.aux.enc_to_lm_proj.forward(client, &encoded)?;
        let curr_embed = var_reshape(&projected, &[rows, lm_width]).map_err(Error::Numr)?;

        // 4. Emit, and condition the NEXT patch on this one. Each row stores
        // its OWN `[1, patch_size, feat_dim]` slice of `pred_feat`, a dim-0
        // narrow that copies no bytes. Only a row still generating counts the
        // patch as its own.
        // A FREE slot is skipped: its patch is computed and dropped. Storing
        // it grows without bound for as long as the engine runs.
        for b in 0..rows {
            if !state.occupied(b) {
                continue;
            }
            let own = var_narrow(&pred_feat, 0, b, 1).map_err(Error::Numr)?;
            state.patches[b].push(own);
            if !state.finished(b) {
                state.patch_len[b] += 1;
            }
        }
        state.prefix_feat_cond = widen(client, &state.prefix_feat_cond, pred_feat, rows)?;

        // 5. Stop check on the CURRENT `lm_hidden` — the hidden state that
        // produced the patch just emitted, BEFORE step 6 replaces it. The
        // guard is strictly greater than `min_len`, and row `b`'s `i_b` is
        // the index of the patch that row just counted —
        // `state.patch_len[b] - 1`. Every row still generating counted the
        // patch pushed above, so their `i_b` agree. A finished row's count is
        // frozen, and the stop loop below skips finished rows.
        //
        // `capture` asymmetry (see this fn's doc comment): the non-capturing
        // path calls `aux.stop` only when some row's guard is open. The
        // capturing path calls it every iteration. The guard decision
        // (`i_b > options.min_len`) is the same either way.
        let guard_open: Vec<bool> = (0..rows)
            .map(|b| {
                debug_assert!(
                    state.finished(b) || state.patch_len[b] == state.patches[b].len(),
                    "a live row's patch count must equal the loop's iteration index"
                );
                state.patch_len[b].saturating_sub(1) > options.min_len
            })
            .collect();
        // At least one row was open on entry and none finishes before the
        // loop below. A frozen count never exceeds a live one, so a finished
        // row cannot be the only open entry.
        let any_guard_open = guard_open.contains(&true);
        let stop_logits = if any_guard_open || capture {
            Some(self.aux.stop(client, &lm_hidden)?)
        } else {
            None
        };
        let mut stopped_now = false;
        if any_guard_open {
            // `stop_logits` is always `Some` here: `any_guard_open` is one of
            // the two disjuncts above.
            let stops = match &stop_logits {
                Some(logits) => stop_predicted(client, logits, rows)?,
                None => vec![false; rows],
            };
            for (b, &stop) in stops.iter().enumerate() {
                if stop && guard_open[b] && !state.finished(b) {
                    state.outcomes[b] = Some(GenerateOutcome::StopToken);
                    stopped_now = true;
                }
            }
        }
        // A row still open after the stop check finishes on its cap; a stop
        // on the same step wins, as it does in the reference's loop order.
        for b in 0..rows {
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
        //
        // Both stacks read the SAME step off the lane table, so both write
        // row `b` at `lanes.position(b)` and attend over the same range. One
        // advance follows step 7. An advance between them files `residual_lm`
        // a position ahead of `base_lm`.
        let step = state.prefill.lanes.step();
        let stepped =
            self.base_lm
                .decode_step(client, &curr_embed, &mut state.prefill.base_cache, &step)?;
        let captured_lm_hidden_pre_fsq = capture.then(|| stepped.clone());
        let lm_next = self.fsq.forward(client, &stepped)?;

        // 7. `residual_lm` consumes the POST-fsq hidden state, concatenated
        // with `curr_embed` in THAT order. Both are the stepped rows, so the
        // concat runs at the stepped width like everything above it.
        let fused = var_cat(&[&lm_next, &curr_embed], 1, client).map_err(Error::Numr)?;
        let residual_in = self.aux.fusion_concat_proj.forward(client, &fused)?;
        let residual = self.residual_lm.decode_step(
            client,
            &residual_in,
            &mut state.prefill.residual_cache,
            &step,
        )?;

        // 8. One table, both caches, in lockstep: ONE advance, here, after
        // both stacks have stepped. `step` borrows the table, so the borrow
        // checker refuses an advance placed between them.
        state.prefill.lanes.advance(client)?;

        // Both handoff rows go back into the allocation-width tensors the
        // state holds, so a slot seated above the stepped prefix still has a
        // row of its own for `admit` to write into.
        state.prefill.lm_hidden = widen(client, &state.prefill.lm_hidden, lm_next, rows)?;
        state.prefill.residual_hidden =
            widen(client, &state.prefill.residual_hidden, residual, rows)?;

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

/// `full` with its first `rows` rows replaced by `part`, or `part` itself
/// when the two are already the same width.
///
/// The per-step tensors a state carries between iterations — both handoff
/// rows and the prefix condition — stay at the state's allocation width,
/// while a step computes only the stepped rows. A slot seated above the
/// stepped prefix needs its own row to be written into before the prefix
/// widens to cover it, so the width cannot follow the step. The stepped rows
/// are the tensor's prefix, so this replaces exactly them.
fn widen<R, C>(client: &C, full: &Var<R>, part: Var<R>, rows: usize) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: IndexingOps<R>,
{
    if full.shape()[0] == rows {
        return Ok(part);
    }
    Ok(Var::new(
        client
            .slice_assign(full.tensor(), part.tensor(), 0, 0)
            .map_err(Error::Numr)?,
        false,
    ))
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use super::*;
    use crate::test_utils::cpu_setup;

    /// Capturing must not perturb step 1-8 arithmetic. Guard OPEN (`i >
    /// min_len`), so `aux.stop` runs on both paths.
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
        assert_eq!(
            values(&plain.patches[0][0]),
            values(&captured.patches[0][0])
        );
        assert_eq!(
            values(&plain.prefill.lm_hidden),
            values(&captured.prefill.lm_hidden)
        );
        assert_eq!(
            values(&plain.prefill.residual_hidden),
            values(&captured.prefill.residual_hidden)
        );
        assert_eq!(
            plain.prefill.lanes.positions(),
            captured.prefill.lanes.positions()
        );

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
        assert_eq!(
            values(&plain.patches[0][0]),
            values(&captured.patches[0][0])
        );
        assert_eq!(
            values(&plain.prefill.lm_hidden),
            values(&captured.prefill.lm_hidden)
        );
    }
}
