//! Conditional flow-matching (CFM) sampler around VoxCPM2's local DiT
//! estimator ([`LocalDit::forward`]).
//!
//! Reference: `voxcpm/modules/locdit/unified_cfm.py`. The estimator is a
//! velocity field; this module integrates it backwards from `t = 1` to `t = 0`
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
//! - `schedule`: [`CfmOptions`] and the sway-corrected time schedule
//! - `guidance`: the CFG-zero-star rescale and the guidance combine
//! - `euler`: `LocalDit::solve_euler` and `LocalDit::sample`
//!
//! [`LocalDit::forward`]: crate::model::audio::voxcpm::local_dit::LocalDit::forward
//! [`LocalDitConfig::mean_mode`]:
//!     crate::model::audio::voxcpm::local_dit::LocalDitConfig::mean_mode
//! [`optimized_scale`]: guidance::optimized_scale

mod euler;
mod guidance;
mod schedule;

pub use schedule::{CfmOptions, cfm_time_span};
