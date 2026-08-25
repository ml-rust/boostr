//! Fundamental-frequency (F0) estimation with YIN.
//!
//! Implements de Cheveigné & Kawahara (2002), "YIN, a fundamental frequency
//! estimator for speech and music", steps 1-5: squared difference function,
//! cumulative mean normalised difference, absolute threshold, parabolic
//! interpolation, voicing decision. The cumulative mean normalisation is what
//! separates YIN from a plain autocorrelation peak-pick: on a harmonic-rich
//! voice, autocorrelation dips just as deeply at half the period, so a
//! peak-pick reports the first or second harmonic instead of F0.
//!
//! Feeds `describe`'s `pitch_mean` / `pitch_std` speaker statistics. No model
//! weights are involved — this is pure DSP over the decoded waveform.
//!
//! **CPU-only.** The difference function is a serial reduction over one
//! analysis frame per lag, with lag-dependent trip counts and an early exit as
//! soon as the first sub-threshold dip is found. There is no batched matrix
//! structure for a GPU kernel to exploit, so this stays plain scalar loops
//! like `super::stft` and `super::quality`.

use crate::error::{Error, Result};

/// Per-frame F0 track plus the aggregate statistics `describe` needs.
#[derive(Debug, Clone, PartialEq)]
pub struct PitchTrack {
    /// Hz per analysis frame; `None` where the frame is unvoiced.
    pub f0: Vec<Option<f64>>,
    /// Seconds between frame centres.
    pub hop_s: f64,
    /// Mean F0 over VOICED frames only. `None` when nothing is voiced.
    pub mean_hz: Option<f64>,
    /// Population standard deviation of F0 over voiced frames. `None` when
    /// fewer than two frames are voiced.
    pub std_hz: Option<f64>,
    /// Fraction of frames judged voiced, in `[0, 1]`.
    pub voiced_fraction: f64,
    /// YIN aperiodicity `d'(tau)` at the chosen lag, per frame; `None` where
    /// the frame is unvoiced.
    ///
    /// 0 is perfectly periodic, 1 is no periodicity found. YIN computes this to
    /// make the voicing decision and it is the continuous quantity behind that
    /// binary — a breathy or noisy voiced frame sits near the threshold, which
    /// `voiced_fraction` alone cannot show.
    pub aperiodicity: Vec<Option<f64>>,
    /// Mean harmonic-to-noise ratio over VOICED frames, dB. `None` when nothing
    /// is voiced.
    ///
    /// `10 * log10((1 - a) / a)` from the aperiodicity `a`. This measures noise
    /// DURING speech, which a noise-floor measurement over an entire signal
    /// cannot: the floor is dominated by the pauses, and a recording whose
    /// pauses are digitally silent can still be hissy under the voice.
    pub mean_hnr_db: Option<f64>,
}

/// Harmonic-to-noise ratio in dB from a YIN aperiodicity.
///
/// Clamped away from 0 and 1 because both ends are singular: a perfectly
/// periodic frame would give `+inf` and a fully aperiodic one `-inf`, and
/// neither survives averaging. The bounds put the reportable range at
/// +/- 40 dB, well outside anything real speech produces.
fn hnr_db(aperiodicity: f64) -> f64 {
    const MIN_A: f64 = 1e-4;
    let a = aperiodicity.clamp(MIN_A, 1.0 - MIN_A);
    10.0 * ((1.0 - a) / a).log10()
}

/// Search range and framing for [`estimate_pitch`].
#[derive(Debug, Clone, Copy)]
pub struct PitchOptions {
    /// Lowest F0 to search for, Hz. Default 60.
    pub min_hz: f64,
    /// Highest F0 to search for, Hz. Default 400.
    pub max_hz: f64,
    /// Analysis window, seconds. Default 0.045.
    ///
    /// The window must span at least two periods of `min_hz`, i.e.
    /// `window_s >= 2 / min_hz`: the difference function compares an
    /// integration window of `window_s - 1/min_hz` seconds against a copy of
    /// itself shifted by up to one `min_hz` period, so a shorter window leaves
    /// nothing to integrate over at the longest lag.
    pub window_s: f64,
    /// Hop between frames, seconds. Default 0.010.
    pub hop_s: f64,
    /// YIN absolute threshold. Default 0.15.
    pub threshold: f64,
}

impl Default for PitchOptions {
    /// 60-400 Hz covers adult speech of any gender: adult male speech centres
    /// near 100-120 Hz and adult female speech near 180-220 Hz, and the range
    /// leaves headroom for creak at the bottom and emphatic pitch at the top.
    fn default() -> Self {
        Self {
            min_hz: 60.0,
            max_hz: 400.0,
            window_s: 0.045,
            hop_s: 0.010,
            threshold: 0.15,
        }
    }
}

/// Vertex offset, in lags, of the parabola through `(-1, a)`, `(0, b)`,
/// `(1, c)`. Returns 0 when the three points are collinear or form a maximum.
fn parabolic_offset(a: f64, b: f64, c: f64) -> f64 {
    let denom = 2.0 * (a - 2.0 * b + c);
    if denom <= 0.0 || !denom.is_finite() {
        return 0.0;
    }
    ((a - c) / denom).clamp(-1.0, 1.0)
}

/// YIN over one frame.
///
/// Returns `(f0_hz, aperiodicity)`, or `None` when the frame is unvoiced. The
/// aperiodicity is `d'(tau)` at the chosen lag — the same value the voicing
/// decision is made from, returned rather than discarded.
///
/// `scratch` is `tau_max + 1` long and is overwritten; it is owned by the
/// caller so no allocation happens per frame.
fn frame_f0(
    frame: &[f32],
    scratch: &mut [f64],
    sample_rate: f64,
    integration: usize,
    tau_min: usize,
    tau_max: usize,
    threshold: f64,
) -> Option<(f64, f64)> {
    // Step 1 + 2: squared difference d(tau), normalised in place into the
    // cumulative mean normalised difference d'(tau). `running` accumulates
    // d(1..=tau) before the slot is overwritten, so one buffer serves both.
    if let Some(slot) = scratch.get_mut(0) {
        *slot = 1.0;
    }
    let mut running = 0.0f64;
    for tau in 1..=tau_max {
        let mut d = 0.0f64;
        if let (Some(a), Some(b)) = (frame.get(..integration), frame.get(tau..tau + integration)) {
            d = a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| {
                    let delta = x as f64 - y as f64;
                    delta * delta
                })
                .sum();
        }
        running += d;
        let normalised = if running > 0.0 {
            d * tau as f64 / running
        } else {
            // Digital silence: d(tau) == 0 for every lag, so d'(tau) is 0/0.
            // 1.0 is the "no periodicity found" value, keeping silence unvoiced.
            1.0
        };
        if let Some(slot) = scratch.get_mut(tau) {
            *slot = normalised;
        }
    }

    let at = |i: usize| scratch.get(i).copied().unwrap_or(1.0);

    // Step 3: absolute threshold. Take the FIRST lag whose d' dips below the
    // threshold, then descend to the bottom of that dip — the global minimum
    // would bias towards lower octaves.
    let mut tau = tau_min;
    let mut chosen = None;
    while tau <= tau_max {
        if at(tau) < threshold {
            while tau < tau_max && at(tau + 1) < at(tau) {
                tau += 1;
            }
            chosen = Some(tau);
            break;
        }
        tau += 1;
    }
    // Step 5: voicing decision. No lag reached the threshold, so the frame has
    // no periodic component worth reporting.
    let tau = chosen?;

    // Step 4: parabolic interpolation for sub-sample lag precision. Skipped at
    // the range edges, where one neighbour is missing.
    let refined = if tau > tau_min && tau < tau_max {
        tau as f64 + parabolic_offset(at(tau - 1), at(tau), at(tau + 1))
    } else {
        tau as f64
    };
    if refined > 0.0 {
        // Report d' at the integer lag actually chosen: the parabolic step
        // refines the LAG, and interpolating the depth as well would report a
        // periodicity the frame did not measure.
        Some((sample_rate / refined, at(tau)))
    } else {
        None
    }
}

/// Reject a non-finite or non-positive option value, naming it and its value.
fn require_positive(arg: &'static str, value: f64) -> Result<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(Error::InvalidArgument {
            arg,
            reason: format!("must be > 0, got {value}"),
        })
    }
}

/// Estimate the F0 track of mono `samples` in `[-1, 1]` at `sample_rate`.
///
/// Returns [`Error::InvalidArgument`] when `sample_rate` is 0, `samples` is
/// empty, `min_hz` is not positive, `min_hz >= max_hz`, `window_s` or `hop_s`
/// is not positive, `window_s` spans fewer than two periods of `min_hz`, or
/// `max_hz` is at or above the Nyquist frequency.
///
/// Input shorter than one analysis window is NOT an error: it yields a
/// [`PitchTrack`] with an empty `f0`, `voiced_fraction == 0.0`, and `None`
/// for both statistics.
pub fn estimate_pitch(samples: &[f32], sample_rate: u32, opts: PitchOptions) -> Result<PitchTrack> {
    if sample_rate == 0 {
        return Err(Error::InvalidArgument {
            arg: "sample_rate",
            reason: "must be > 0, got 0".into(),
        });
    }
    if samples.is_empty() {
        return Err(Error::InvalidArgument {
            arg: "samples",
            reason: "must be non-empty, got 0 samples".into(),
        });
    }
    require_positive("opts.min_hz", opts.min_hz)?;
    require_positive("opts.window_s", opts.window_s)?;
    require_positive("opts.hop_s", opts.hop_s)?;
    if !opts.max_hz.is_finite() || opts.min_hz >= opts.max_hz {
        return Err(Error::InvalidArgument {
            arg: "opts.max_hz",
            reason: format!("must be > min_hz {}, got {}", opts.min_hz, opts.max_hz),
        });
    }
    let rate = sample_rate as f64;
    if opts.max_hz >= rate / 2.0 {
        return Err(Error::InvalidArgument {
            arg: "opts.max_hz",
            reason: format!("must be below Nyquist {}, got {}", rate / 2.0, opts.max_hz),
        });
    }
    let min_window_s = 2.0 / opts.min_hz;
    if opts.window_s < min_window_s {
        return Err(Error::InvalidArgument {
            arg: "opts.window_s",
            reason: format!(
                "must span two periods of min_hz {} (>= {min_window_s} s), got {}",
                opts.min_hz, opts.window_s
            ),
        });
    }

    let tau_max = (rate / opts.min_hz).floor() as usize;
    let tau_min = ((rate / opts.max_hz).floor() as usize).max(2);
    let window = (opts.window_s * rate).round() as usize;
    let hop = ((opts.hop_s * rate).round() as usize).max(1);
    if tau_min + 1 >= tau_max || window <= tau_max {
        return Err(Error::InvalidArgument {
            arg: "opts.min_hz",
            reason: format!(
                "search range {}-{} Hz at {sample_rate} Hz gives lags {tau_min}-{tau_max} in a \
                 {window}-sample window, which leaves no lag to search",
                opts.min_hz, opts.max_hz
            ),
        });
    }
    // Lags run to `tau_max`, so the integration window is what remains of the
    // analysis window after the longest shift.
    let integration = window - tau_max;

    let mut scratch = vec![0.0f64; tau_max + 1];
    let mut f0 = Vec::new();
    let mut aperiodicity = Vec::new();
    let mut start = 0usize;
    while let Some(frame) = samples.get(start..start + window) {
        let measured = frame_f0(
            frame,
            &mut scratch,
            rate,
            integration,
            tau_min,
            tau_max,
            opts.threshold,
        );
        f0.push(measured.map(|(hz, _)| hz));
        aperiodicity.push(measured.map(|(_, a)| a));
        start += hop;
    }

    let voiced: Vec<f64> = f0.iter().filter_map(|&v| v).collect();
    let voiced_aperiodicity: Vec<f64> = aperiodicity.iter().filter_map(|&v| v).collect();
    let mean_hnr_db = if voiced_aperiodicity.is_empty() {
        None
    } else {
        Some(
            voiced_aperiodicity.iter().copied().map(hnr_db).sum::<f64>()
                / voiced_aperiodicity.len() as f64,
        )
    };
    let voiced_fraction = if f0.is_empty() {
        0.0
    } else {
        voiced.len() as f64 / f0.len() as f64
    };
    let mean_hz = if voiced.is_empty() {
        None
    } else {
        Some(voiced.iter().sum::<f64>() / voiced.len() as f64)
    };
    let std_hz = match mean_hz {
        Some(mean) if voiced.len() >= 2 => {
            let var = voiced
                .iter()
                .map(|v| {
                    let d = v - mean;
                    d * d
                })
                .sum::<f64>()
                / voiced.len() as f64;
            Some(var.sqrt())
        }
        _ => None,
    };

    Ok(PitchTrack {
        f0,
        hop_s: hop as f64 / rate,
        mean_hz,
        std_hz,
        voiced_fraction,
        aperiodicity,
        mean_hnr_db,
    })
}

#[cfg(test)]
mod tests;
