//! The per-patch loop itself: [`PatchGenerator::step_with_noise`],
//! [`PatchGenerator::step`] and [`PatchGenerator::generate`]. The capturing
//! variant and its shared inner body live in `super::capture`.

use super::*;
use crate::nn::var_contiguous;
use numr::autograd::var_transpose;

impl<R: Runtime<DType = DType>> PatchGenerator<'_, R> {
    /// One iteration, with the CFM noise supplied by the caller.
    ///
    /// This is the primitive: it draws nothing, so a caller (the CFM gate)
    /// can pin `z` per step and reproduce a run exactly.
    /// [`step`](Self::step) is the thin drawing wrapper over it. `z` is `[1,
    /// patch_size, feat_dim]` — the patch layout, same as
    /// `state.prefix_feat_cond` — and is used AS GIVEN;
    /// `options.cfm.temperature` is not applied here; see the module docs.
    /// A caller holding the reference's `[1, feat_dim, patch_size]` noise
    /// transposes it first, as [`step`](Self::step) does.
    /// Runs steps 1-8 in order. Returns [`StepOutcome::Stopped`] when the
    /// stop guard fires, in which case steps 6-8 did not run and the caches
    /// and `position` are unchanged.
    pub fn step_with_noise<C>(
        &self,
        client: &C,
        state: &mut GenerateState<R>,
        z: &Var<R>,
        options: &GenerateOptions,
    ) -> Result<StepOutcome>
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
        self.step_with_noise_inner(client, state, z, options, false)
            .map(|(outcome, _)| outcome)
    }

    /// One iteration, drawing the CFM noise itself.
    ///
    /// `z` is `randn_seeded(options.seed + i) * options.cfm.temperature`
    /// drawn over `[1, feat_dim, patch_size]` — the reference's element order,
    /// so a seed keeps placing the same draw on the same (feature, position)
    /// — then transposed to the `[1, patch_size, feat_dim]` patch layout.
    /// `i` is the index of the patch about to be emitted, so consecutive
    /// patches never share noise and the run is reproducible from
    /// `options.seed`. Everything after the draw is
    /// [`step_with_noise`](Self::step_with_noise). `randn_seeded` is
    /// reproducible per backend, so one seed draws differently on CPU and
    /// CUDA.
    pub fn step<C>(
        &self,
        client: &C,
        state: &mut GenerateState<R>,
        options: &GenerateOptions,
    ) -> Result<StepOutcome>
    where
        C: ModelClient<R> + TypeConversionOps<R> + RandomOps<R> + 'static,
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
        let hidden = state.prefill.lm_hidden.tensor();
        let noise = client
            .randn_seeded(
                &[1, self.config.feat_dim, self.config.patch_size],
                hidden.dtype(),
                options.seed.wrapping_add(state.patches.len() as u64),
            )
            .map_err(Error::Numr)?;
        let z = var_mul_scalar(
            &Var::new(noise, false),
            options.cfm.temperature as f64,
            client,
        )
        .map_err(Error::Numr)?;
        let z = var_contiguous(&var_transpose(&z).map_err(Error::Numr)?)?;
        self.step_with_noise(client, state, &z, options)
    }

    /// Run the loop to a stop token or `max_len`.
    ///
    /// Steps with [`step`](Self::step), so the noise comes from
    /// `options.seed`. The emitted patches stay in `state.patches`, each `[1,
    /// patch_size, feat_dim]`; this returns only WHY the loop ended, so the
    /// caller can tell a finished utterance ([`GenerateOutcome::StopToken`])
    /// from a truncated one ([`GenerateOutcome::MaxLen`]). Does NOT
    /// VAE-decode and does NOT write audio — that is a later unit. Errors
    /// when `max_len` is 0, and propagates the first step error (a
    /// `position`/cache drift included) rather than continuing.
    pub fn generate<C>(
        &self,
        client: &C,
        state: &mut GenerateState<R>,
        options: &GenerateOptions,
    ) -> Result<GenerateOutcome>
    where
        C: ModelClient<R> + TypeConversionOps<R> + RandomOps<R> + 'static,
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
        if options.max_len == 0 {
            return Err(Error::InvalidArgument {
                arg: "options.max_len",
                reason: "expected at least 1, got 0".to_string(),
            });
        }
        while state.patches.len() < options.max_len {
            if self.step(client, state, options)? == StepOutcome::Stopped {
                return Ok(GenerateOutcome::StopToken);
            }
        }
        Ok(GenerateOutcome::MaxLen)
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
        let patch0 = values(&st.patches[0]);
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
            st.patches.len(),
            opts.min_len + 2,
            "the guard is `i > min_len`, so i = min_len + 1 is the first break"
        );
    }

    /// Trap 2, the half that must NOT fire: capped at `min_len + 1` patches, a
    /// stop token on every iteration still cannot end the run, because `i`
    /// never exceeds `min_len`. A `>=` guard would return `StopToken` here.
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
        assert_eq!(st.patches.len(), 3);
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
        assert_eq!(st.patches.len(), 5);
        assert_eq!(st.prefill.position, 5);
    }

    /// Trap 4: ONE counter, both caches. Each iteration advances `position`
    /// and BOTH cache lengths by exactly one.
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
            assert_eq!(st.prefill.position, expected);
            assert_eq!(st.prefill.base_cache.seq_len(), expected);
            assert_eq!(st.prefill.residual_cache.seq_len(), expected);
        }
    }

    /// Trap 4's guard rail: a `position` that no longer matches the caches is
    /// rejected by `decode_step`, so a drift errors instead of rotating a
    /// query at one position while filing its key at another.
    #[test]
    fn desynced_position_errors() {
        let (client, device) = cpu_setup();
        let fx = fixture(false, &device);
        let generator = fx.generator();
        let mut st = state(&fx, &device);
        let opts = options(2, 8);

        generator.step(&client, &mut st, &opts).expect("step 0");
        assert_eq!(st.prefill.position, 1);

        // Deliberate drift: the caches hold 1 position, the counter claims 2.
        st.prefill.position += 1;
        assert!(
            generator.step(&client, &mut st, &opts).is_err(),
            "a position/cache drift must error, not corrupt the cache"
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
        assert_eq!(st.patches.len(), 4);
        assert_eq!(st.prefill.position, 3);
        assert_eq!(st.prefill.base_cache.seq_len(), 3);
        assert_eq!(st.prefill.residual_cache.seq_len(), 3);
    }

    /// The injected-noise path is the primitive: the same `z` against the
    /// same state reproduces the same patch bit for bit, which is what makes
    /// the CFM gate possible.
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
            values(&st.patches[0])
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

    /// A rank-2 client sanity check on the fixture wiring: the loop's mu is
    /// two DiT tokens wide, which is what `check_mu` derives.
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
