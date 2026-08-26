//! Conditional flow-matching (CFM) sampler around VoxCPM2's local DiT
//! estimator ([`LocalDit::forward`]).
//!
//! Reference: `voxcpm/modules/locdit/unified_cfm.py`. The estimator is a
//! velocity field; this file integrates it backwards from `t = 1` to `t = 0`
//! with an explicit Euler scheme, classifier-free guidance (CFG), and the
//! "CFG-zero-star" rescale.
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
//! - **The `1e-8` in [`optimized_scale`] sits INSIDE the denominator sum**,
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
//! The schedule is computed ONCE on the host as a `Vec<f32>`
//! ([`cfm_time_span`]) and indexed per step, so the loop never reads a scalar
//! back off a device tensor.
//!
//! [`LocalDitConfig::mean_mode`]:
//!     crate::model::audio::voxcpm::local_dit::LocalDitConfig::mean_mode

use crate::error::{Error, Result};
use crate::model::audio::voxcpm::local_dit::loader::LocalDit;
use crate::model::traits::ModelClient;
use crate::nn::var_contiguous;
use numr::autograd::{
    Var, var_add, var_add_scalar, var_cat, var_div, var_mul, var_mul_scalar, var_narrow,
    var_reshape, var_square, var_sub, var_sum,
};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, RandomOps, ReduceOps,
    ScalarOps, ShapeOps, TensorOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Sampling knobs for [`LocalDit::sample`].
///
/// [`Default`] is the real VoxCPM2 inference path: 10 steps, CFG 2.0,
/// temperature 1.0, full sway, CFG-zero-star on.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CfmOptions {
    /// Euler steps. The schedule has `n_timesteps + 1` entries.
    pub n_timesteps: usize,
    /// Classifier-free guidance weight. `1.0` disables guidance: the combine
    /// then returns the conditional velocity unchanged.
    pub cfg_value: f32,
    /// Scales the initial noise draw.
    pub temperature: f32,
    /// Sway strength. `0.0` leaves the schedule a plain `linspace(1, 0)`.
    pub sway_sampling_coef: f32,
    /// Enables the leading zero-velocity warmup steps.
    pub use_cfg_zero_star: bool,
}

impl Default for CfmOptions {
    fn default() -> Self {
        Self {
            n_timesteps: 10,
            cfg_value: 2.0,
            temperature: 1.0,
            sway_sampling_coef: 1.0,
            use_cfg_zero_star: true,
        }
    }
}

/// Build the sway-corrected time schedule: `n_timesteps + 1` values running
/// from exactly `1.0` down to exactly `0.0`.
///
/// ```text
/// s = linspace(1, 0, n_timesteps + 1)
/// t = s + coef * (cos(pi/2 * s) - 1 + s)
/// ```
///
/// The `linspace` base is accumulated in `f64` and rounded to `f32` once (the
/// endpoints are pinned rather than accumulated), matching the reference's
/// `torch.linspace`; the sway itself is evaluated in `f32`. Accumulating the
/// base in `f32` instead reproduces 8 of the 11 values at `n_timesteps = 10`
/// and misses the other 3.
///
/// The sway fixes both endpoints — `cos(pi/2) - 1 + 1 = 0` and
/// `cos(0) - 1 + 0 = 0` — so `t[0]` is `1.0` and `t[n]` is `0.0` for any
/// `coef`.
///
/// Errors when `n_timesteps` is 0.
pub fn cfm_time_span(n_timesteps: usize, sway_sampling_coef: f32) -> Result<Vec<f32>> {
    if n_timesteps == 0 {
        return Err(Error::InvalidArgument {
            arg: "n_timesteps",
            reason: "expected at least 1, got 0".to_string(),
        });
    }
    let step = -1.0f64 / n_timesteps as f64;
    Ok((0..=n_timesteps)
        .map(|i| {
            let s = match i {
                0 => 1.0f32,
                i if i == n_timesteps => 0.0f32,
                i => (1.0f64 + step * i as f64) as f32,
            };
            s + sway_sampling_coef * ((std::f32::consts::FRAC_PI_2 * s).cos() - 1.0 + s)
        })
        .collect())
}

/// Leading steps whose velocity is forced to zero:
/// `max(1, int(len(t_span) * 0.04))` (`unified_cfm.py:96`). This is 1 for
/// every schedule shorter than 50 entries, including the 11-entry default.
fn zero_init_steps(span_len: usize) -> usize {
    ((span_len as f64 * 0.04) as usize).max(1)
}

/// CFG-zero-star rescale (`unified_cfm.py:79-82`).
///
/// ```text
/// st_star = sum(pos * neg) / (sum(neg^2) + 1e-8)
/// ```
///
/// `pos` is the CONDITIONAL velocity and `neg` the UNCONDITIONAL one, both
/// `[batch, feat_dim, patch_size]`. The reduction is per batch row over the
/// flattened `feat_dim * patch_size`, and `1e-8` is added INSIDE the
/// denominator sum, before the divide — not to the quotient. The result is
/// `[batch, 1, 1]`, shaped to broadcast back over the velocities.
fn optimized_scale<R, C>(client: &C, pos: &Var<R>, neg: &Var<R>) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: ModelClient<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    let shape = pos.shape().to_vec();
    if shape.len() != 3 || neg.shape() != shape.as_slice() {
        return Err(Error::InvalidArgument {
            arg: "pos",
            reason: format!(
                "expected two matching 3D velocities, got {shape:?} and {:?}",
                neg.shape()
            ),
        });
    }
    let batch = shape[0];
    let flat: usize = shape[1..].iter().product();
    let pos_flat = var_reshape(pos, &[batch, flat]).map_err(Error::Numr)?;
    let neg_flat = var_reshape(neg, &[batch, flat]).map_err(Error::Numr)?;

    let dot = var_sum(
        &var_mul(&pos_flat, &neg_flat, client).map_err(Error::Numr)?,
        &[1],
        true,
        client,
    )
    .map_err(Error::Numr)?;
    let sq = var_sum(
        &var_square(&neg_flat, client).map_err(Error::Numr)?,
        &[1],
        true,
        client,
    )
    .map_err(Error::Numr)?;
    let sq = var_add_scalar(&sq, 1e-8, client).map_err(Error::Numr)?;

    let scale = var_div(&dot, &sq, client).map_err(Error::Numr)?;
    var_reshape(&scale, &[batch, 1, 1]).map_err(Error::Numr)
}

/// Classifier-free guidance combine (`unified_cfm.py:128`).
///
/// ```text
/// v = v_uncond * st_star + cfg_value * (v_cond - v_uncond * st_star)
/// ```
///
/// `st_star` is `[batch, 1, 1]` and broadcasts over the velocities. At
/// `cfg_value == 1.0` the two terms telescope and the result is `v_cond`, so
/// guidance is off — `st_star` drops out entirely.
fn cfg_combine<R, C>(
    client: &C,
    v_cond: &Var<R>,
    v_uncond: &Var<R>,
    st_star: &Var<R>,
    cfg_value: f32,
) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: ModelClient<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    let scaled = var_mul(v_uncond, st_star, client).map_err(Error::Numr)?;
    let delta = var_sub(v_cond, &scaled, client).map_err(Error::Numr)?;
    let guided = var_mul_scalar(&delta, cfg_value as f64, client).map_err(Error::Numr)?;
    var_add(&scaled, &guided, client).map_err(Error::Numr)
}

impl<R: Runtime<DType = DType>> LocalDit<R> {
    /// Integrate the estimator from `t_span[0]` to `t_span[len - 1]`, starting
    /// from `z`.
    ///
    /// Draws NOTHING: `z` and the schedule are both inputs, so a caller can
    /// pin the noise and reproduce a run bit for bit. Use
    /// [`sample`](Self::sample) for the noise-drawing wrapper.
    ///
    /// - `z`: `[batch, feat_dim, patch_size]` — the starting sample.
    /// - `t_span`: the schedule from [`cfm_time_span`], at least 2 entries.
    /// - `mu`: `[batch, mu_tokens * hidden_dim]` — the global condition. It is
    ///   the ONLY input zeroed on the unconditional half of the doubled batch.
    /// - `cond`: `[batch, feat_dim, patch_size]` — the prefix condition.
    /// - `cfg_value`: guidance weight; `1.0` means no guidance.
    /// - `use_cfg_zero_star`: enables the zero-velocity warmup steps.
    /// - `trajectory`: when `Some`, receives `x` AFTER every step, including
    ///   the warmup steps that leave it untouched — `trajectory[k]` is the
    ///   state after step `k + 1`, and its length is `t_span.len() - 1`. When
    ///   `None` nothing is recorded and the hot path allocates nothing extra.
    ///
    /// Returns `[batch, feat_dim, patch_size]`.
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
        // The estimator's `dt` is the mean-velocity delta, not the Euler step:
        // `mean_mode` is false on this checkpoint, so it is zero throughout.
        let dt_in = Var::new(
            Tensor::<R>::zeros(&[2 * batch], dtype, device).map_err(Error::Numr)?,
            false,
        );

        let warmup = zero_init_steps(t_span.len());
        let mut x = z.clone();
        let mut t = t_span[0];
        // Seeded from the schedule once; every later value comes from the
        // running `t` instead.
        let mut dt = t_span[0] - t_span[1];

        for step in 1..t_span.len() {
            if !(use_cfg_zero_star && step <= warmup) {
                let x_in = var_cat(&[&x, &x], 0, client).map_err(Error::Numr)?;
                let t_in = Var::new(
                    Tensor::<R>::full_scalar(&[2 * batch], dtype, t as f64, device)
                        .map_err(Error::Numr)?,
                    false,
                );
                let out = self.forward(client, &x_in, &mu_in, &t_in, &cond_in, &dt_in)?;

                // First half = real `mu` = conditional. Second half = zero
                // `mu` = unconditional. The reference calls the second one
                // `cfg_dphi_dt`, which is the opposite of what it holds.
                let v_cond = var_contiguous(&var_narrow(&out, 0, 0, batch).map_err(Error::Numr)?)?;
                let v_uncond =
                    var_contiguous(&var_narrow(&out, 0, batch, batch).map_err(Error::Numr)?)?;

                let st_star = optimized_scale(client, &v_cond, &v_uncond)?;
                let velocity = cfg_combine(client, &v_cond, &v_uncond, &st_star, cfg_value)?;
                let move_by = var_mul_scalar(&velocity, dt as f64, client).map_err(Error::Numr)?;
                x = var_sub(&x, &move_by, client).map_err(Error::Numr)?;
            }

            // Bookkeeping advances even on a warmup step.
            t -= dt;
            if step < t_span.len() - 1 {
                dt = t - t_span[step + 1];
            }
            if let Some(trace) = trajectory.as_deref_mut() {
                trace.push(x.clone());
            }
        }

        Ok(x)
    }

    /// Draw noise and integrate: the full CFM sample.
    ///
    /// `z` is `randn_seeded(seed) * temperature` over
    /// `[batch, feat_dim, patch_size]`, taking `batch`, dtype and device from
    /// `cond`. Everything after the draw is [`solve_euler`](Self::solve_euler),
    /// which is where the per-step trajectory can be captured.
    ///
    /// `randn_seeded` is reproducible per backend, so a CPU run and a CUDA run
    /// of one seed start from different noise.
    ///
    /// Returns `[batch, feat_dim, patch_size]`.
    pub fn sample<C>(
        &self,
        client: &C,
        mu: &Var<R>,
        cond: &Var<R>,
        options: &CfmOptions,
        seed: u64,
    ) -> Result<Var<R>>
    where
        C: ModelClient<R> + TypeConversionOps<R> + RandomOps<R>,
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
        let batch = self.check_patch_input("cond", cond, None)?;
        self.check_mu(mu, batch)?;
        let t_span = cfm_time_span(options.n_timesteps, options.sway_sampling_coef)?;

        let noise = client
            .randn_seeded(
                &[batch, self.feat_dim, self.patch_size],
                cond.tensor().dtype(),
                seed,
            )
            .map_err(Error::Numr)?;
        let z = var_mul_scalar(&Var::new(noise, false), options.temperature as f64, client)
            .map_err(Error::Numr)?;

        self.solve_euler(
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
mod tests;
