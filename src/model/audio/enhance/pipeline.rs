//! The full raw-recording-to-reference chain.

use super::biquad::Biquad;
use super::denoise::{DenoiseOptions, denoise, denoise_with_profile, noise_floor_dbfs};
use super::limiter::{LimiterOptions, LimiterReport, limit};
use super::loudness::{integrated_lufs, peak_dbfs};
use crate::error::{Error, Result};

/// Controls for [`enhance`].
#[derive(Debug, Clone, Copy)]
pub struct EnhanceOptions {
    /// Corner of the rumble filter, in Hz. Below roughly 60 Hz a speech
    /// recording holds only desk thumps, HVAC and handling noise — no voice.
    /// Removing it before anything else stops that energy from dominating the
    /// loudness measurement and eating headroom.
    pub highpass_hz: f64,
    /// Spectral gate settings. `None` skips denoising.
    pub denoise: Option<DenoiseOptions>,
    /// Low-shelf gain in dB, and its corner in Hz — the "more bass" control.
    /// Applied AFTER denoising: boosting first would amplify the rumble the
    /// gate then has to fight.
    pub bass_boost_db: f64,
    pub bass_corner_hz: f64,
    /// High-shelf gain in dB and corner in Hz. A small lift here restores the
    /// air a spectral gate always takes off the top.
    pub presence_db: f64,
    pub presence_corner_hz: f64,
    /// Target integrated loudness.
    pub target_lufs: f64,
    /// Sample peak that must not be exceeded.
    pub peak_ceiling_dbfs: f64,
    /// Limiter window. `None` skips limiting, in which case the loudness
    /// target yields to the ceiling and a normal speech take lands roughly
    /// 6 dB short of it.
    pub limiter_window_ms: Option<f64>,
    /// Most *sustained* gain reduction the limiter is allowed to apply, in dB
    /// — [`LimiterReport::sustained_reduction_db`], not the deepest single
    /// sample. Past roughly 6 dB a voice audibly pumps, and a pumping
    /// reference teaches the pumping to the model.
    ///
    /// Capping the deepest sample instead would let one plosive hold the whole
    /// take quiet, since a lone transient can need 15 dB while nothing else
    /// needs any.
    ///
    /// When the target needs more than this, the chain stops at whatever
    /// loudness that much limiting reached and reports `reached_target: false`.
    pub max_limiting_db: f64,
}

impl Default for EnhanceOptions {
    fn default() -> Self {
        Self {
            highpass_hz: 70.0,
            denoise: Some(DenoiseOptions::default()),
            bass_boost_db: 3.0,
            bass_corner_hz: 140.0,
            presence_db: 1.5,
            presence_corner_hz: 7000.0,
            target_lufs: -18.0,
            peak_ceiling_dbfs: -1.0,
            limiter_window_ms: Some(5.0),
            max_limiting_db: 6.0,
        }
    }
}

/// What the chain measured, before and after.
///
/// Kept because "it sounds better" is not a check. A take whose noise floor
/// barely moved was never noisy, and one whose floor dropped 20 dB while
/// loudness stayed put had its voice gated along with the hiss.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnhanceReport {
    pub input_lufs: f64,
    pub output_lufs: f64,
    pub input_peak_dbfs: f64,
    pub output_peak_dbfs: f64,
    pub input_noise_floor_dbfs: f64,
    pub output_noise_floor_dbfs: f64,
    /// Gain the loudness stage applied, in dB. `-inf` when the input was
    /// silent and nothing was applied.
    pub applied_gain_db: f64,
    /// Sustained gain reduction the limiter applied, in dB — the 99th
    /// percentile, which is what the cap is measured against. `0.0` when the
    /// limiter never engaged or was disabled.
    pub limiter_reduction_db: f64,
    /// Deepest single-sample gain reduction the limiter applied, in dB.
    pub limiter_peak_reduction_db: f64,
    /// Whether the chain reached [`EnhanceOptions::target_lufs`], within
    /// 0.5 LU. False means the ceiling and the limiting cap stopped it short;
    /// [`Self::output_lufs`] is then what it actually reached.
    pub reached_target: bool,
}

/// Bring a raw recording to reference quality.
///
/// Stage order is fixed:
///
/// | Stage | Why here |
/// |-------|----------|
/// | High-pass | Rumble skews every later measurement |
/// | Denoise | Operates on a signal whose floor is already rumble-free |
/// | Bass shelf | Restores weight the high-pass and gate removed |
/// | Presence shelf | Restores air the gate removed |
/// | Loudness + ceiling | Last: every earlier stage changes the level |
pub fn enhance(
    samples: &[f32],
    rate: u32,
    opts: EnhanceOptions,
) -> Result<(Vec<f32>, EnhanceReport)> {
    enhance_with_noise_profile(samples, None, rate, opts)
}

/// [`enhance`], with the noise profile taken from a supplied silent clip.
///
/// Prefer this whenever a silent stretch of the same chain is available. The
/// automatic path in [`denoise`] has to find the pauses inside the take, and a
/// take recorded without any pause gets no denoising at all.
///
/// `noise_clip` is high-passed alongside the signal, so the profile it yields
/// describes the same filtered floor the mask is compared against.
pub fn enhance_with_noise_profile(
    samples: &[f32],
    noise_clip: Option<&[f32]>,
    rate: u32,
    opts: EnhanceOptions,
) -> Result<(Vec<f32>, EnhanceReport)> {
    if rate == 0 {
        return Err(Error::InvalidArgument {
            arg: "rate",
            reason: "sample rate is 0".to_string(),
        });
    }
    if samples.is_empty() {
        return Err(Error::InvalidArgument {
            arg: "samples",
            reason: "no samples".to_string(),
        });
    }

    let input_lufs = integrated_lufs(samples, rate)?;
    let input_peak_dbfs = peak_dbfs(samples);
    let input_noise_floor_dbfs = noise_floor_dbfs(samples, rate);

    let rate_f = rate as f64;
    let mut work = samples.to_vec();

    let mut noise = noise_clip.map(|c| c.to_vec());
    if opts.highpass_hz > 0.0 {
        let hp = Biquad::highpass(rate_f, opts.highpass_hz, std::f64::consts::FRAC_1_SQRT_2);
        hp.clone().process_buffer(&mut work);
        // A fresh copy of the filter, not the one above: carrying its state
        // across two independent buffers puts a transient at the start of the
        // second.
        if let Some(n) = noise.as_mut() {
            hp.clone().process_buffer(n);
        }
    }

    if let Some(d) = opts.denoise {
        work = match noise.as_deref() {
            Some(n) => denoise_with_profile(&work, n, d)?,
            None => denoise(&work, d)?,
        };
    }

    if opts.bass_boost_db != 0.0 {
        Biquad::low_shelf(
            rate_f,
            opts.bass_corner_hz,
            std::f64::consts::FRAC_1_SQRT_2,
            opts.bass_boost_db,
        )
        .process_buffer(&mut work);
    }

    if opts.presence_db != 0.0 {
        Biquad::high_shelf(
            rate_f,
            opts.presence_corner_hz,
            std::f64::consts::FRAC_1_SQRT_2,
            opts.presence_db,
        )
        .process_buffer(&mut work);
    }

    let before_gain = integrated_lufs(&work, rate)?;
    let (levelled, limiting) = reach_target(&work, rate, before_gain, &opts)?;
    work = levelled;

    let output_lufs = integrated_lufs(&work, rate)?;
    let applied_gain_db = if before_gain.is_finite() && output_lufs.is_finite() {
        output_lufs - before_gain
    } else {
        f64::NEG_INFINITY
    };

    let report = EnhanceReport {
        input_lufs,
        output_lufs,
        input_peak_dbfs,
        output_peak_dbfs: peak_dbfs(&work),
        input_noise_floor_dbfs,
        output_noise_floor_dbfs: noise_floor_dbfs(&work, rate),
        applied_gain_db,
        limiter_reduction_db: limiting.sustained_reduction_db,
        limiter_peak_reduction_db: limiting.max_reduction_db,
        reached_target: (output_lufs - opts.target_lufs).abs() < 0.5,
    };
    Ok((work, report))
}

/// Gain to the loudness target, holding the ceiling with the limiter.
///
/// Limiting lowers loudness, so one pass undershoots. Each further pass closes
/// the remaining gap; three is enough to converge to well inside 0.1 LU, and a
/// fixed count keeps the stage from looping on a signal it cannot lift.
///
/// Returns the samples and what the limiter did in total.
fn reach_target(
    samples: &[f32],
    rate: u32,
    current_lufs: f64,
    opts: &EnhanceOptions,
) -> Result<(Vec<f32>, LimiterReport)> {
    let idle = LimiterReport {
        max_reduction_db: 0.0,
        sustained_reduction_db: 0.0,
        fraction_reduced: 0.0,
    };
    if !current_lufs.is_finite() {
        // Silence has no loudness to move. Applying a gain to it would only
        // raise whatever numerical dust is in the buffer.
        return Ok((samples.to_vec(), idle));
    }

    let Some(window_ms) = opts.limiter_window_ms else {
        // No limiter: the ceiling caps the gain, and the target is missed by
        // however far the peaks stick out.
        let gain = gain_capped_by_peak(samples, opts.target_lufs - current_lufs, opts);
        return Ok((apply_gain(samples, gain), idle));
    };

    let lim = LimiterOptions {
        ceiling_dbfs: opts.peak_ceiling_dbfs,
        window_ms,
    };

    let mut out = samples.to_vec();
    let mut total = idle;
    for _ in 0..6 {
        let lufs = integrated_lufs(&out, rate)?;
        if !lufs.is_finite() || (lufs - opts.target_lufs).abs() < 0.1 {
            break;
        }
        let wanted_db = opts.target_lufs - lufs;
        let (stepped, report) = largest_step_within_cap(&out, rate, wanted_db, lim, opts);
        // A pass that moves nothing means the cap is already binding; another
        // would compute the same thing again.
        if stepped == out {
            break;
        }
        out = stepped;
        total = LimiterReport {
            max_reduction_db: total.max_reduction_db.max(report.max_reduction_db),
            sustained_reduction_db: total
                .sustained_reduction_db
                .max(report.sustained_reduction_db),
            fraction_reduced: total.fraction_reduced.max(report.fraction_reduced),
        };
    }
    Ok((out, total))
}

/// Apply as much of `wanted_db` as the limiting cap allows.
///
/// The limiter's gain reduction rises monotonically with the gain fed to it,
/// so the largest acceptable gain is found by bisection. Ten halvings settle a
/// 30 dB span to under 0.03 dB, far finer than the 0.1 LU the caller stops at.
///
/// A reduction of gain — `wanted_db` below zero — never engages the limiter,
/// so it is applied whole.
fn largest_step_within_cap(
    samples: &[f32],
    rate: u32,
    wanted_db: f64,
    lim: LimiterOptions,
    opts: &EnhanceOptions,
) -> (Vec<f32>, LimiterReport) {
    let attempt = |db: f64| limit(&apply_gain(samples, 10f64.powf(db / 20.0)), rate, lim);

    if wanted_db <= 0.0 {
        return attempt(wanted_db);
    }

    let (full, full_report) = attempt(wanted_db);
    if full_report.sustained_reduction_db <= opts.max_limiting_db {
        return (full, full_report);
    }

    let (mut lo, mut hi) = (0.0f64, wanted_db);
    let mut best = (
        samples.to_vec(),
        LimiterReport {
            max_reduction_db: 0.0,
            sustained_reduction_db: 0.0,
            fraction_reduced: 0.0,
        },
    );
    for _ in 0..10 {
        let mid = 0.5 * (lo + hi);
        let (out, r) = attempt(mid);
        if r.sustained_reduction_db <= opts.max_limiting_db {
            best = (out, r);
            lo = mid;
        } else {
            hi = mid;
        }
    }
    best
}

/// The requested gain, reduced so the sample peak stays under the ceiling.
fn gain_capped_by_peak(samples: &[f32], wanted_db: f64, opts: &EnhanceOptions) -> f64 {
    let mut gain = 10f64.powf(wanted_db / 20.0);
    let peak = peak_dbfs(samples);
    if peak.is_finite() {
        let ceiling_gain = 10f64.powf((opts.peak_ceiling_dbfs - peak) / 20.0);
        if ceiling_gain < gain {
            gain = ceiling_gain;
        }
    }
    gain
}

fn apply_gain(samples: &[f32], gain: f64) -> Vec<f32> {
    samples.iter().map(|&s| (s as f64 * gain) as f32).collect()
}

#[cfg(test)]
mod tests;
