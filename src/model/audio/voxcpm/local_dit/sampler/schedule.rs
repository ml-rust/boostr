//! Sampling knobs and the sway-corrected time schedule.

use crate::error::{Error, Result};

/// Sampling knobs for [`LocalDit::sample`](crate::model::audio::voxcpm::local_dit::LocalDit::sample).
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
pub(super) fn zero_init_steps(span_len: usize) -> usize {
    ((span_len as f64 * 0.04) as usize).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 11 values `torch.linspace(1, 0, 11)` plus the `coef = 1.0` sway
    /// actually realizes. The schedule must reproduce them BIT for bit, because
    /// every later `t` and `dt` is derived from them.
    ///
    /// Written as the shortest round-tripping `f32` literals; the reference
    /// fixture prints them at full width as:
    ///
    /// ```text
    /// 1.0, 0.956434428691864, 0.9090169668197632, 0.8539904952049255,
    /// 0.7877852320671082, 0.7071067690849304, 0.609017014503479,
    /// 0.4910065531730652, 0.3510565459728241, 0.18768836557865143, 0.0
    /// ```
    const SCHEDULE_10: [f32; 11] = [
        1.0, 0.9564344, 0.90901697, 0.8539905, 0.78778523, 0.70710677, 0.609017, 0.49100655,
        0.35105655, 0.18768837, 0.0,
    ];

    #[test]
    fn time_span_reproduces_the_reference_schedule() {
        let span = cfm_time_span(10, 1.0).unwrap();
        assert_eq!(span.len(), 11);
        for (i, (got, want)) in span.iter().zip(SCHEDULE_10.iter()).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "t_span[{i}]: got {got:?}, want {want:?}"
            );
        }
    }

    /// `coef = 0` leaves the bare `linspace`, so the sway term is genuinely the
    /// thing the previous test is measuring.
    #[test]
    fn zero_sway_coefficient_leaves_a_plain_linspace() {
        let span = cfm_time_span(10, 0.0).unwrap();
        for (i, got) in span.iter().enumerate() {
            let want = 1.0f32 - i as f32 / 10.0;
            assert!(
                (got - want).abs() < 1e-7,
                "t_span[{i}]: got {got:?}, want {want:?}"
            );
        }
        assert_ne!(span[1].to_bits(), SCHEDULE_10[1].to_bits());
    }

    #[test]
    fn time_span_rejects_zero_timesteps() {
        assert!(cfm_time_span(0, 1.0).is_err());
    }
}
