//! The Euler integration loop and the noise-drawing `sample` wrapper.
//!
//! ```text
//! t_span = linspace(1, 0, n_timesteps + 1)
//! t_span = t_span + coef * (cos(pi/2 * t_span) - 1 + t_span)   # "sway"
//! dt = t_span[0] - t_span[1]
//! for step in 1..=n_timesteps:
//!     v  = 0                       if step <= zero_init_steps
//!     v  = cfg(estimator(...))     otherwise
//!     x  = x - dt * v
//!     t  = t - dt
//!     dt = t - t_span[step + 1]    if step < n_timesteps
//! ```
//!
//! Traps this implementation is pinned against:
//!
//! - **The doubled batch is asymmetric in exactly one input.** Each estimator
//!   call runs batch `2b`: `x`, `t` and `cond` are written IDENTICALLY to both
//!   halves, and only `mu` differs — the first half gets the real `mu`, the
//!   second half stays ZERO. So the FIRST half is the CONDITIONAL velocity and
//!   the SECOND is the UNCONDITIONAL one.
//! - **The reference's naming is inverted.** `unified_cfm.py:118` names the
//!   second (zero-`mu`, i.e. unconditional) half `cfg_dphi_dt`, which reads
//!   like it is the guided output. It is not. The variables here are named
//!   `v_cond` / `v_uncond` after what they actually hold.
//! - **`dt` is fed to the estimator as ZERO, not as the Euler step.** The
//!   estimator's `dt` input is the mean-velocity delta, live only when
//!   `mean_mode` is set. It is `false` on this checkpoint — see
//!   [`LocalDitConfig::mean_mode`] for why the checkpoint's
//!   `dit_config.mean_mode` key is dead — so `dt_in` is all zeros. That is
//!   NOT the loop's `dt`, and the estimator's zero-`dt` branch still
//!   contributes a real bias.
//! - **The warmup step calls nothing.** While
//!   `use_cfg_zero_star && step <= zero_init_steps`, the velocity is zero, the
//!   estimator is never evaluated, and `x` is returned unchanged — but `t` and
//!   `dt` still advance.
//! - **The `1e-8` in `optimized_scale` sits INSIDE the denominator sum**,
//!   added to `sum(v_uncond^2)` before the divide. Adding it after the divide,
//!   or to the numerator, changes the answer whenever the velocities are near
//!   orthogonal.
//! - **`dt` is a recurrence, not a table lookup.** After the first step it is
//!   recomputed from the RUNNING `t` (`dt = t - t_span[step + 1]`), never read
//!   back as `t_span[step] - t_span[step + 1]`. The two agree only in exact
//!   arithmetic.
//! - The reference's `inference_cfg_rate` config key is NOT wired in here —
//!   [`LocalDitConfig`](crate::model::audio::voxcpm::local_dit::LocalDitConfig)
//!   has no such field, `solve_euler` never reads one, and guidance comes
//!   from the `cfg_value` argument alone.
//!
//! [`LocalDitConfig::mean_mode`]:
//!     crate::model::audio::voxcpm::local_dit::LocalDitConfig::mean_mode

use super::guidance::{cfg_combine, optimized_scale};
use super::schedule::{CfmOptions, EulerStep, cfm_time_span, euler_steps};
use crate::error::{Error, Result};
use crate::model::audio::voxcpm::local_dit::loader::LocalDit;
use crate::model::traits::ModelClient;
use crate::nn::var_contiguous;
use crate::ops::FlashAttentionOps;
use crate::quant::traits::DequantOps;
use numr::autograd::{Var, var_cat, var_mul_scalar, var_narrow, var_reshape, var_sub};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, RandomOps, ReduceOps,
    ScalarOps, ShapeOps, TensorOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> LocalDit<R> {
    /// Integrate the estimator from `t_span[0]` to `t_span[len - 1]`, starting
    /// from `z`.
    ///
    /// Draws NOTHING: `z` and the schedule are both inputs, so a caller can
    /// pin the noise and reproduce a run bit for bit. Use
    /// [`sample`](Self::sample) for the noise-drawing wrapper.
    ///
    /// - `z`: `[batch, patch_size, feat_dim]` — the starting sample.
    /// - `t_span`: the schedule from [`cfm_time_span`], at least 2 entries.
    /// - `mu`: `[batch, mu_tokens * hidden_dim]` — the global condition. It is
    ///   the ONLY input zeroed on the unconditional half of the doubled batch.
    /// - `cond`: `[batch, patch_size, feat_dim]` — the prefix condition.
    /// - `cfg_value`: guidance weight; `1.0` means no guidance.
    /// - `use_cfg_zero_star`: enables the zero-velocity warmup steps.
    /// - `trajectory`: when `Some`, receives `x` AFTER every step, including
    ///   the warmup steps that leave it untouched — `trajectory[k]` is the
    ///   state after step `k + 1`, and its length is `t_span.len() - 1`. When
    ///   `None` nothing is recorded and the hot path allocates nothing extra.
    ///
    /// Returns `[batch, patch_size, feat_dim]`.
    #[allow(clippy::too_many_arguments)]
    pub fn solve_euler<C>(
        &self,
        client: &C,
        z: &Var<R>,
        t_span: &[f32],
        mu: &Var<R>,
        cond: &Var<R>,
        cfg_value: f32,
        use_cfg_zero_star: bool,
        mut trajectory: Option<&mut Vec<Var<R>>>,
    ) -> Result<Var<R>>
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
        let batch = self.check_patch_input("z", z, None)?;
        self.check_patch_input("cond", cond, Some(batch))?;
        self.check_mu(mu, batch)?;
        if t_span.len() < 2 {
            return Err(Error::InvalidArgument {
                arg: "t_span",
                reason: format!("expected at least 2 entries, got {}", t_span.len()),
            });
        }

        let dtype = z.tensor().dtype();
        let device = z.tensor().device();

        // The doubled batch differs in `mu` ALONE: real on the first half,
        // zero on the second. `cond` is written identically to both halves.
        let mu_zero = Var::new(
            Tensor::<R>::zeros(mu.shape(), mu.tensor().dtype(), mu.tensor().device())
                .map_err(Error::Numr)?,
            false,
        );
        let mu_in = var_cat(&[mu, &mu_zero], 0, client).map_err(Error::Numr)?;
        let cond_in = var_cat(&[cond, cond], 0, client).map_err(Error::Numr)?;
        // `mu_in` does not change across steps: tokenize it ONCE here rather
        // than paying `LocalDit::forward`'s reshape + contiguous on every
        // one of `t_span.len() - 1` estimator calls below.
        let mu_tokens = self.check_mu(&mu_in, 2 * batch)?;
        let mu_tok = var_reshape(
            &var_contiguous(&mu_in)?,
            &[2 * batch, mu_tokens, self.hidden_dim()],
        )
        .map_err(Error::Numr)?;
        // `cond_in` is constant across the steps too: project it ONCE.
        let cond_h = self.project_cond(client, &cond_in)?;
        // The estimator's `dt` is the mean-velocity delta, not the Euler step:
        // `mean_mode` is false on this checkpoint, so it is zero throughout.
        let dt_in = Var::new(
            Tensor::<R>::zeros(&[2 * batch], dtype, device).map_err(Error::Numr)?,
            false,
        );

        let mut x = z.clone();

        // `t`/`dt` per step come from the shared host plan, which the
        // graph-captured integrator bakes in verbatim (see `graph::capture`).
        for slot in euler_steps(t_span, use_cfg_zero_star) {
            if let Some(EulerStep { t, dt }) = slot {
                let x_in = var_cat(&[&x, &x], 0, client).map_err(Error::Numr)?;
                let t_in = Var::new(
                    Tensor::<R>::full_scalar(&[2 * batch], dtype, t as f64, device)
                        .map_err(Error::Numr)?,
                    false,
                );
                let out = self.forward_prepared(client, &x_in, &mu_tok, &t_in, &cond_h, &dt_in)?;

                // First half = real `mu` = conditional. Second half = zero
                // `mu` = unconditional. The reference calls the second one
                // `cfg_dphi_dt`, which is the opposite of what it holds.
                // Leading-axis halves of a dense tensor are dense views, so
                // no copy happens here.
                let v_cond = var_narrow(&out, 0, 0, batch).map_err(Error::Numr)?;
                let v_uncond = var_narrow(&out, 0, batch, batch).map_err(Error::Numr)?;

                let st_star = optimized_scale(client, &v_cond, &v_uncond)?;
                let velocity = cfg_combine(client, &v_cond, &v_uncond, &st_star, cfg_value)?;
                let move_by = var_mul_scalar(&velocity, dt as f64, client).map_err(Error::Numr)?;
                x = var_sub(&x, &move_by, client).map_err(Error::Numr)?;
            }

            // Recorded on a warmup step too: `x` is unchanged there.
            if let Some(trace) = trajectory.as_deref_mut() {
                trace.push(x.clone());
            }
        }

        Ok(x)
    }

    /// Draw noise and integrate: the full CFM sample.
    ///
    /// `z` is `randn_seeded(seed) * temperature` over
    /// `[batch, patch_size, feat_dim]`, taking `batch`, dtype and device from
    /// `cond`. Everything after the draw is
    /// [`solve_euler_graphed`](Self::solve_euler_graphed): CUDA graph replay
    /// where available, the eager [`solve_euler`](Self::solve_euler) loop
    /// elsewhere. Callers that need the per-step trajectory call
    /// `solve_euler` directly.
    ///
    /// `randn_seeded` is reproducible per backend, so a CPU run and a CUDA run
    /// of one seed start from different noise.
    ///
    /// Returns `[batch, patch_size, feat_dim]`.
    pub fn sample<C>(
        &self,
        client: &C,
        mu: &Var<R>,
        cond: &Var<R>,
        options: &CfmOptions,
        seed: u64,
    ) -> Result<Var<R>>
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
            + DequantOps<R>
            + FlashAttentionOps<R>,
    {
        let batch = self.check_patch_input("cond", cond, None)?;
        self.check_mu(mu, batch)?;
        let t_span = cfm_time_span(options.n_timesteps, options.sway_sampling_coef)?;

        let noise = client
            .randn_seeded(
                &[batch, self.patch_size, self.feat_dim],
                cond.tensor().dtype(),
                seed,
            )
            .map_err(Error::Numr)?;
        let z = var_mul_scalar(&Var::new(noise, false), options.temperature as f64, client)
            .map_err(Error::Numr)?;

        self.solve_euler_graphed(
            client,
            &z,
            &t_span,
            mu,
            cond,
            options.cfg_value,
            options.use_cfg_zero_star,
            None,
        )
    }
}

#[cfg(test)]
mod tests {
    //! Pins the untouched warmup step and the seeded `sample` wrapper. The
    //! estimator itself is the tiny synthetic one from
    //! [`crate::model::audio::voxcpm::local_dit::tests`].

    use super::super::guidance::tests::values;
    use super::*;
    use crate::model::audio::voxcpm::local_dit::tests as fixture;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    struct Setup {
        z: Var<CpuRuntime>,
        mu: Var<CpuRuntime>,
        cond: Var<CpuRuntime>,
    }

    fn setup(batch: usize, device: &CpuDevice) -> Setup {
        Setup {
            z: Var::new(
                fixture::t(
                    &[batch, fixture::PATCH_SIZE, fixture::FEAT_DIM],
                    0.9,
                    device,
                ),
                false,
            ),
            mu: Var::new(
                fixture::t(
                    &[batch, fixture::MU_TOKENS * fixture::HIDDEN_DIM],
                    1.3,
                    device,
                ),
                false,
            ),
            cond: Var::new(
                fixture::t(
                    &[batch, fixture::PATCH_SIZE, fixture::FEAT_DIM],
                    1.7,
                    device,
                ),
                false,
            ),
        }
    }

    /// `zero_init_steps` is 1 for an 11-entry schedule, so step 1 has zero
    /// velocity and NO estimator call: `x` after it is bitwise `z`. Step 2 must
    /// then move `x`, otherwise the loop is inert and the test proves nothing.
    #[test]
    fn warmup_step_leaves_x_exactly_equal_to_z() {
        let (client, device) = cpu_setup();
        let m = fixture::model(1, &device);
        let s = setup(2, &device);
        let span = cfm_time_span(10, 1.0).unwrap();

        let mut trace = Vec::new();
        let out = m
            .solve_euler(
                &client,
                &s.z,
                &span,
                &s.mu,
                &s.cond,
                2.0,
                true,
                Some(&mut trace),
            )
            .unwrap();

        assert_eq!(trace.len(), 10);
        let z = values(&s.z);
        let after_warmup = values(&trace[0]);
        for (i, (got, want)) in after_warmup.iter().zip(z.iter()).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "warmup step moved element {i}: {got:?} vs {want:?}"
            );
        }

        let after_second = values(&trace[1]);
        assert!(
            after_second
                .iter()
                .zip(z.iter())
                .any(|(a, b)| (a - b).abs() > 1e-6),
            "step 2 must move x: {after_second:?} vs {z:?}"
        );
        assert_eq!(out.shape(), s.z.shape());
    }

    /// The warmup is gated on `use_cfg_zero_star`. With it off, step 1 integrates
    /// like any other step and `x` moves immediately.
    #[test]
    fn disabling_cfg_zero_star_integrates_the_first_step() {
        let (client, device) = cpu_setup();
        let m = fixture::model(1, &device);
        let s = setup(2, &device);
        let span = cfm_time_span(10, 1.0).unwrap();

        let mut trace = Vec::new();
        m.solve_euler(
            &client,
            &s.z,
            &span,
            &s.mu,
            &s.cond,
            2.0,
            false,
            Some(&mut trace),
        )
        .unwrap();

        let z = values(&s.z);
        let first = values(&trace[0]);
        assert!(
            first
                .iter()
                .zip(z.iter())
                .any(|(a, b)| (a - b).abs() > 1e-6),
            "step 1 must move x when the warmup is off: {first:?} vs {z:?}"
        );
    }

    #[test]
    fn solve_euler_rejects_a_one_entry_schedule() {
        let (client, device) = cpu_setup();
        let m = fixture::model(1, &device);
        let s = setup(1, &device);
        assert!(
            m.solve_euler(&client, &s.z, &[1.0], &s.mu, &s.cond, 2.0, true, None)
                .is_err()
        );
    }

    /// `sample` is `solve_euler` after a seeded draw: one seed reproduces a run,
    /// and a different seed does not.
    #[test]
    fn sample_is_reproducible_for_a_seed() {
        let (client, device) = cpu_setup();
        let m = fixture::model(1, &device);
        let s = setup(2, &device);
        let options = CfmOptions {
            n_timesteps: 3,
            ..CfmOptions::default()
        };

        let a = values(&m.sample(&client, &s.mu, &s.cond, &options, 7).unwrap());
        let b = values(&m.sample(&client, &s.mu, &s.cond, &options, 7).unwrap());
        let c = values(&m.sample(&client, &s.mu, &s.cond, &options, 8).unwrap());

        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(a.len(), 2 * fixture::FEAT_DIM * fixture::PATCH_SIZE);
    }
}
