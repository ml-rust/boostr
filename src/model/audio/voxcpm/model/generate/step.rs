//! One iteration of the per-patch loop: [`PatchGenerator::step_with_noise`]
//! and the drawing [`PatchGenerator::step`]. The whole-run driver is
//! `super::run`, the capturing variant and the shared inner body
//! `super::capture`.
//!
//! ```text
//! per iteration i, over every STEPPED row of the batch at once:
//!   1. mu         = cat(lm_to_dit_proj(lm_hidden), res_to_dit_proj(residual_hidden))
//!   2. pred_feat  = feat_decoder(mu, cond = prefix_feat_cond^T)
//!   3. curr_embed = enc_to_lm_proj(feat_encoder(pred_feat.unsqueeze(1)))
//!   4. patches[b].push(pred_feat[b]); prefix_feat_cond = pred_feat
//!   5. stop?      = argmax(stop(lm_hidden)) == 1   <- the CURRENT lm_hidden
//!   6. lm_hidden  = fsq(base_lm.decode_step(curr_embed, base_cache, position))
//!   7. residual_hidden = residual_lm.decode_step(
//!          fusion_concat_proj(cat(lm_hidden, curr_embed)), residual_cache, position)
//!   8. position  += 1
//! ```
//!
//! # Traps this module exists to get right
//!
//! - **`prefix_feat_cond` starts as each row's last raw prefill patch.**
//!   Without a continuation tail that is the ZERO text-pad patch — NOT the
//!   last patch of the reference audio. With a tail it is the prompt's last
//!   patch. [`PrefillState::prefix_feat_cond`] carries it and
//!   [`GenerateState::start`] takes it.
//! - **A continuation row's decode context is not generated output.**
//!   [`GenerateState::context`] holds it apart from `patches`, so
//!   `patch_len`, the stop guard, the noise index and `max_len` all count
//!   generated patches only. The decode prepends the context and trims it.
//! - **The initial `lm_hidden` is the prefill's blended last row.** At a
//!   TEXT position the blend left it UN-fsq'd, at a continuation tail's
//!   AUDIO position it fsq'd it. From step 6 onward every `lm_hidden` IS
//!   fsq'd. Do not normalise them. Re-fsq'ing the prefill's row, and
//!   dropping the fsq in step 6, both stay shape-valid.
//! - **Step 7 consumes the POST-fsq `lm_hidden`**, and the concat order is
//!   `(lm_hidden, curr_embed)`. That concat is `2 * lm_hidden` wide, so
//!   swapping the halves is shape-valid and computes a different model.
//! - **A state carries THREE widths, none interchangeable.**
//!
//!   | Width | Meaning |
//!   | --- | --- |
//!   | [`GenerateState::batch`] | CEILING: slots the state can ever hold. A slot index, a per-row HOST vector and [`GenerateOptions::rows`] are checked against it. |
//!   | `GenerateState::alloc_width` | ALLOCATION: rows every DEVICE tensor and both caches hold. Opens at ONE row whatever the ceiling, widens before a slot above it is seated, never shrinks. A shape check on a stored tensor reads this. |
//!   | `GenerateState::active_width` | STEPPED: the slot prefix one iteration runs. Rows at or above it hold no request and no step touches them. |
//!
//!   The stepped width is the occupied prefix rounded UP to an entry of
//!   [`widths::BATCH_WIDTHS`](crate::model::audio::voxcpm::model::widths::BATCH_WIDTHS).
//!   The DiT's Euler solver keys captured graphs by batch size, and the
//!   allocation rounds to the same list.
//! - **ONE [`LanePositions`](crate::model::audio::voxcpm::minicpm4::LanePositions)
//!   table drives BOTH caches and EVERY STEPPED row.** The prefill primes
//!   every row to `S_max` and the table advances every row by one per
//!   iteration, so the rows hold one shared position throughout this loop. A
//!   finished row keeps stepping — its patches are computed and discarded —
//!   because the caches have no per-row compaction. Only its `patch_len` and
//!   `outcomes` entries stand still.
//!   [`MiniCpm4Model::decode_step`] rejects a write position the cache will
//!   not fill, so a drift errors rather than corrupts. That check is why the
//!   table is not duplicated per cache.
//! - **The stop guard is STRICTLY greater**: `i > min_len`, so a stop token
//!   at `i <= min_len` is ignored and at least `min_len + 2` patches are
//!   always emitted (`i = 0..=min_len` push unconditionally, and `i = min_len
//!   + 1` pushes before its check can fire).
//! - **The `max_len` exit is silent in the reference.** Here it is
//!   [`GenerateOutcome::MaxLen`], distinct from
//!   [`GenerateOutcome::StopToken`], so a caller tells a finished utterance
//!   from a truncated one per row, in [`GenerateState::outcomes`].
//! - **A stop that finishes the LAST open row skips steps 6-8**, exactly as
//!   the reference's `break` does for one row. A stop that leaves other
//!   rows open does not: the step runs through for everyone.
//! - **`step_with_noise` ignores `temperature`.** Scaling the noise draw is
//!   the drawing wrapper's job, the same split `solve_euler`/`sample` uses.
//!   A caller injecting `z` owns its scale.

use super::prelude::*;
use super::*;
use crate::nn::var_contiguous;
use crate::ops::FlashAttentionOps;
use numr::autograd::var_transpose;

impl<R: Runtime<DType = DType>> PatchGenerator<'_, R> {
    /// One iteration, with the CFM noise supplied by the caller.
    ///
    /// The primitive. It draws nothing, so a caller pins `z` per step and
    /// reproduces a run exactly. [`step`](Self::step) is the drawing wrapper.
    ///
    /// `z` is `[state.active_width(), patch_size, feat_dim]`, the patch layout
    /// over the STEPPED rows, and is used AS GIVEN. `options.cfm.temperature`
    /// is not applied here; see the module docs. A caller holding the
    /// reference's `[batch, feat_dim, patch_size]` noise transposes it first.
    ///
    /// Runs steps 1-8 in order. Returns [`StepOutcome::Stopped`] when a stop
    /// token fires and every row is then finished. Steps 6-8 then did not
    /// run, so the caches and `position` are unchanged. Errors when every row
    /// was already finished on entry.
    pub fn step_with_noise<C>(
        &self,
        client: &C,
        state: &mut GenerateState<R>,
        z: &Var<R>,
        options: &GenerateOptions,
    ) -> Result<StepOutcome>
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
        self.step_with_noise_inner(client, state, z, options, false)
            .map(|(outcome, _)| outcome)
    }

    /// One iteration, drawing the CFM noise itself.
    ///
    /// Row `b`'s `z` is `randn_seeded(options.row(b).seed + i_b) *
    /// options.cfm.temperature`, drawn over `[1, feat_dim, patch_size]` (the
    /// reference's element order, so a seed places the same draw on the same
    /// feature and position), stacked over the STEPPED rows, then transposed
    /// to the patch layout.
    ///
    /// `i_b` is `state.patch_len[b]`, THAT ROW's own emitted-patch count, so
    /// consecutive patches never share noise and each row is reproducible
    /// from its own seed whatever else is in the batch. Everything after the
    /// draw is [`step_with_noise`](Self::step_with_noise). `randn_seeded` is
    /// reproducible per backend, so one seed draws differently on CPU and
    /// CUDA.
    pub fn step<C>(
        &self,
        client: &C,
        state: &mut GenerateState<R>,
        options: &GenerateOptions,
    ) -> Result<StepOutcome>
    where
        C: ModelClient<R> + TypeConversionOps<R> + RandomOps<R> + HostCopyOps<R> + 'static,
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
        options.check(state.batch)?;
        let hidden = state.prefill.lm_hidden.tensor();
        let shape = [1, self.config.feat_dim, self.config.patch_size];
        // One draw per STEPPED row. Slots above the stepped prefix take no
        // part in this iteration, so a draw for them allocates unread noise.
        let stepped = state.active_width;
        let mut rows = Vec::with_capacity(stepped);
        for b in 0..stepped {
            // Row `b` advances its noise by its OWN emitted-patch count, not
            // the loop's iteration index. A live row has emitted one patch
            // per iteration, so the two agree. A finished row's count is
            // frozen and its further patches are discarded.
            debug_assert!(
                state.finished(b) || state.patch_len[b] == state.patches[b].len(),
                "a live row's patch count must equal the loop's iteration index"
            );
            let i = state.patch_len[b] as u64;
            let seed = options.row(b).seed.wrapping_add(i);
            rows.push(
                client
                    .randn_seeded(&shape, hidden.dtype(), seed)
                    .map_err(Error::Numr)?,
            );
        }
        let noise = if stepped == 1 {
            rows.swap_remove(0)
        } else {
            let refs: Vec<&Tensor<R>> = rows.iter().collect();
            Tensor::cat(&refs, 0).map_err(Error::Numr)?
        };
        let z = var_mul_scalar(
            &Var::new(noise, false),
            options.cfm.temperature as f64,
            client,
        )
        .map_err(Error::Numr)?;
        let z = var_contiguous(&var_transpose(&z).map_err(Error::Numr)?)?;
        self.step_with_noise(client, state, &z, options)
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use super::*;
    use crate::model::audio::voxcpm::local_dit::tests::{FEAT_DIM, HIDDEN_DIM, PATCH_SIZE, t};
    use crate::model::audio::voxcpm::minicpm4::model::tests::HIDDEN;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    /// Trap 1: iteration 0 is conditioned on the ZERO text-pad patch, and
    /// iteration 1 on the patch iteration 0 emitted — never on anything from
    /// the reference audio.
    #[test]
    fn prefix_feat_cond_is_zero_then_the_previous_patch() {
        let (client, device) = cpu_setup();
        let fx = fixture(false, &device);
        let generator = fx.generator();
        let mut st = state(&fx, &device);
        let opts = options(2, 8);

        assert_eq!(st.prefix_feat_cond.shape(), &[1, PATCH_SIZE, FEAT_DIM]);
        assert!(
            values(&st.prefix_feat_cond).iter().all(|v| *v == 0.0),
            "iteration 0 must be conditioned on zeros"
        );

        generator
            .step_with_noise(&client, &mut st, &noise(0.2, &device), &opts)
            .expect("step 0");
        let patch0 = values(&st.patches[0][0]);
        assert!(
            patch0.iter().any(|v| v.abs() > 1e-6),
            "degenerate patch would make the comparison below vacuous"
        );
        assert_eq!(
            values(&st.prefix_feat_cond),
            patch0,
            "iteration 1 must be conditioned on iteration 0's patch"
        );
    }

    /// Trap 2, the strictly-greater half that fires: with `min_len = 2` and a
    /// stop token on every iteration, the break lands at `i = 3` — so exactly
    /// `min_len + 2` patches come out.
    #[test]
    fn stop_guard_fires_one_past_min_len() {
        let (client, device) = cpu_setup();
        let fx = fixture(true, &device);
        let mut st = state(&fx, &device);
        let opts = options(2, 12);

        let outcome = fx
            .generator()
            .generate(&client, &mut st, &opts)
            .expect("generate");
        assert_eq!(outcome, GenerateOutcome::StopToken);
        assert_eq!(
            st.emitted(),
            opts.min_len + 2,
            "the guard is `i > min_len`, so i = min_len + 1 is the first break"
        );
    }

    /// Trap 2, the half that must NOT fire: capped at `min_len + 1` patches, a
    /// stop token on every iteration cannot end the run, because `i` never
    /// exceeds `min_len`. A `>=` guard returns `StopToken` here.
    #[test]
    fn stop_guard_never_fires_at_min_len() {
        let (client, device) = cpu_setup();
        let fx = fixture(true, &device);
        let mut st = state(&fx, &device);
        let opts = options(2, 3);

        let outcome = fx
            .generator()
            .generate(&client, &mut st, &opts)
            .expect("generate");
        assert_eq!(
            outcome,
            GenerateOutcome::MaxLen,
            "a stop token at i <= min_len must be ignored"
        );
        assert_eq!(st.emitted(), 3);
    }

    /// Trap 3: the cap exit is distinguishable from a stop-token exit.
    #[test]
    fn max_len_exit_reports_max_len() {
        let (client, device) = cpu_setup();
        let fx = fixture(false, &device);
        let mut st = state(&fx, &device);
        let opts = options(2, 5);

        let outcome = fx
            .generator()
            .generate(&client, &mut st, &opts)
            .expect("generate");
        assert_eq!(outcome, GenerateOutcome::MaxLen);
        assert_eq!(st.emitted(), 5);
        assert_eq!(st.prefill.lanes.position(0), 5);
    }

    /// Trap 4: ONE table, both caches. Each iteration advances every row of
    /// the lane table and BOTH cache lengths by exactly one, and the table's
    /// device `kv_end` follows.
    #[test]
    fn position_advances_both_caches_in_lockstep() {
        let (client, device) = cpu_setup();
        let fx = fixture(false, &device);
        let generator = fx.generator();
        let mut st = state(&fx, &device);
        let opts = options(2, 8);

        for expected in 1..=4 {
            let outcome = generator.step(&client, &mut st, &opts).expect("step");
            assert_eq!(outcome, StepOutcome::Continued);
            assert_eq!(st.prefill.lanes.positions(), &[expected]);
            assert_eq!(st.prefill.base_cache.seq_len(), expected);
            assert_eq!(st.prefill.residual_cache.seq_len(), expected);
            assert_eq!(
                st.prefill.lanes.kv_end().to_vec::<i32>(),
                vec![expected as i32 + 1]
            );
        }
    }

    /// Trap 4's guard rail: the lane table advances ONCE per iteration, AFTER
    /// both LMs have stepped.
    ///
    /// A successful step proves the ordering, since an advance between the
    /// two files `residual_lm` ahead of `base_lm` and `decode_step`'s
    /// write-order guard rejects it. The second half injects that extra
    /// advance and pins that the guard fires.
    #[test]
    fn the_lane_table_advances_once_after_both_lms() {
        let (client, device) = cpu_setup();
        let fx = fixture(false, &device);
        let generator = fx.generator();
        let mut st = state(&fx, &device);
        let opts = options(2, 8);

        generator.step(&client, &mut st, &opts).expect("step 0");
        assert_eq!(st.prefill.lanes.positions(), &[1]);
        assert_eq!(st.prefill.base_cache.seq_len(), 1);
        assert_eq!(st.prefill.residual_cache.seq_len(), 1);

        // Deliberate drift: the caches hold 1 position, the table claims 2.
        st.prefill.lanes.advance(&client).expect("advance");
        assert!(
            generator.step(&client, &mut st, &opts).is_err(),
            "a table/cache drift must error, not corrupt the cache"
        );
    }

    /// Trap 5: the stop guard fires BEFORE step 6, so a stopped iteration
    /// leaves the caches and `position` where the previous iteration left
    /// them, exactly as the reference's `break` does.
    #[test]
    fn a_stopped_step_does_not_advance_the_caches() {
        let (client, device) = cpu_setup();
        let fx = fixture(true, &device);
        let generator = fx.generator();
        let mut st = state(&fx, &device);
        let opts = options(2, 12);

        assert_eq!(
            generator
                .generate(&client, &mut st, &opts)
                .expect("generate"),
            GenerateOutcome::StopToken
        );
        // 4 patches emitted, but the last one broke before stepping the LMs.
        assert_eq!(st.emitted(), 4);
        assert_eq!(st.prefill.lanes.position(0), 3);
        assert_eq!(st.prefill.base_cache.seq_len(), 3);
        assert_eq!(st.prefill.residual_cache.seq_len(), 3);
    }

    /// The same `z` against the same state reproduces the same patch bit for
    /// bit, and a different `z` changes it.
    #[test]
    fn injected_noise_is_reproducible_and_load_bearing() {
        let (client, device) = cpu_setup();
        let fx = fixture(false, &device);
        let generator = fx.generator();
        let opts = options(2, 8);

        let run = |z: &Var<CpuRuntime>| {
            let mut st = state(&fx, &device);
            generator
                .step_with_noise(&client, &mut st, z, &opts)
                .expect("step");
            values(&st.patches[0][0])
        };

        let a = run(&noise(0.2, &device));
        assert_eq!(a, run(&noise(0.2, &device)), "same z must give same patch");
        assert_ne!(
            a,
            run(&noise(4.6, &device)),
            "a different z must change the patch, or the noise is being ignored"
        );
    }

    /// Shape validation, at the two inputs a caller drives directly.
    #[test]
    fn rejects_wrong_shapes_and_a_zero_cap() {
        let (client, device) = cpu_setup();
        let fx = fixture(false, &device);
        let generator = fx.generator();
        let mut st = state(&fx, &device);
        let opts = options(2, 8);

        // `z` is the [1, patch_size, feat_dim] patch layout, NOT the
        // reference's noise layout.
        let transposed = Var::new(t(&[1, FEAT_DIM, PATCH_SIZE], 0.5, &device), false);
        assert!(
            generator
                .step_with_noise(&client, &mut st, &transposed, &opts)
                .is_err()
        );
        assert!(
            generator
                .generate(&client, &mut st, &options(2, 0))
                .is_err()
        );
    }

    /// The loop's mu is two DiT tokens wide, which is what `check_mu`
    /// derives.
    #[test]
    fn mu_is_two_dit_tokens_wide() {
        let (client, device) = cpu_setup();
        let fx = fixture(false, &device);
        let hidden = Var::new(t(&[1, HIDDEN], 0.4, &device), false);
        let lm = fx
            .aux
            .lm_to_dit_proj
            .forward(&client, &hidden)
            .expect("lm_to_dit");
        let res = fx
            .aux
            .res_to_dit_proj
            .forward(&client, &hidden)
            .expect("res_to_dit");
        assert_eq!(lm.shape()[1] + res.shape()[1], 2 * HIDDEN_DIM);
    }
}
