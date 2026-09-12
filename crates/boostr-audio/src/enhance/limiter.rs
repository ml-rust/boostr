//! Look-ahead peak limiter.
//!
//! Gain alone cannot bring a speech take to a loudness target. Real speech has
//! a peak-to-loudness ratio around 20 dB — a measured 30 s reference take came
//! in at -38 LUFS with a -15 dBFS peak, 23 dB apart — so raising it to -18
//! LUFS under a -1 dBFS ceiling needs 20 dB of gain and would clip by 19 dB.
//! Clamping the gain instead, which is what a normalizer without a limiter
//! does, simply misses the target by that margin.
//!
//! The gain envelope is built in two passes over `|x|`, and the pair is what
//! makes it both safe and smooth:
//!
//! 1. A running **minimum** of the required gain over a window of half-width
//!    `L`, so the envelope is already down before a peak arrives.
//! 2. A running **mean** of that minimum over the same half-width, which
//!    removes the corners that would otherwise be heard as clicks.
//!
//! The output never exceeds the ceiling, and that is a property rather than a
//! tuning: every term averaged at sample `n` is a minimum taken over a window
//! that contains `n`, so each term is at most the gain `n` itself requires,
//! and so is their mean.

/// Controls for [`limit`].
#[derive(Debug, Clone, Copy)]
pub struct LimiterOptions {
    /// Sample peak the output must not exceed, in dBFS.
    pub ceiling_dbfs: f64,
    /// Half-width of both envelope windows, in milliseconds.
    ///
    /// Sets attack and release together. 5 ms is short enough to catch a
    /// plosive and long enough that the gain movement is inaudible on speech.
    /// Longer ducks whole syllables; shorter distorts low frequencies, because
    /// a window under one period of the fundamental modulates the waveform
    /// rather than its envelope.
    pub window_ms: f64,
}

impl Default for LimiterOptions {
    fn default() -> Self {
        Self {
            ceiling_dbfs: -1.0,
            window_ms: 5.0,
        }
    }
}

/// What the limiter did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LimiterReport {
    /// Deepest gain reduction applied anywhere, in dB. `0.0` when nothing
    /// reached the ceiling.
    ///
    /// Reported, but a poor control to cap on: one plosive sets it, and
    /// refusing gain because a single transient needed 15 dB would hold the
    /// whole take 15 dB quiet. Cap on [`Self::sustained_reduction_db`].
    pub max_reduction_db: f64,
    /// Reduction exceeded by 1% of samples, in dB — the 99th percentile.
    ///
    /// This is what pumping is. A lone transient leaves it at zero, because a
    /// lone transient is far less than 1% of a take; a limiter working
    /// continuously drives it up.
    pub sustained_reduction_db: f64,
    /// Fraction of samples that were reduced at all, in `[0, 1]`. High values
    /// mean the limiter is doing compression rather than catching peaks.
    pub fraction_reduced: f64,
}

/// Hold `samples` under `ceiling_dbfs`, returning the result and a report.
pub fn limit(samples: &[f32], rate: u32, opts: LimiterOptions) -> (Vec<f32>, LimiterReport) {
    let quiet = LimiterReport {
        max_reduction_db: 0.0,
        sustained_reduction_db: 0.0,
        fraction_reduced: 0.0,
    };
    if samples.is_empty() || rate == 0 {
        return (samples.to_vec(), quiet);
    }

    let ceiling = 10f64.powf(opts.ceiling_dbfs / 20.0);
    let half = ((opts.window_ms / 1000.0) * rate as f64).round().max(1.0) as usize;

    // Gain each sample needs on its own.
    let required: Vec<f64> = samples
        .iter()
        .map(|&x| {
            let a = (x as f64).abs();
            if a > ceiling { ceiling / a } else { 1.0 }
        })
        .collect();
    if required.iter().all(|&g| g >= 1.0) {
        return (samples.to_vec(), quiet);
    }

    let envelope = running_mean(&running_min(&required, half), half);

    let out: Vec<f32> = samples
        .iter()
        .zip(envelope.iter())
        .map(|(&x, &g)| (x as f64 * g) as f32)
        .collect();

    let mut reductions: Vec<f64> = envelope
        .iter()
        .map(|&g| if g < 1.0 { -20.0 * g.log10() } else { 0.0 })
        .collect();
    let reduced = reductions.iter().filter(|&&d| d > 1e-9).count();
    reductions.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p99 = reductions[((reductions.len() as f64 * 0.99) as usize).min(reductions.len() - 1)];

    (
        out,
        LimiterReport {
            max_reduction_db: reductions[reductions.len() - 1],
            sustained_reduction_db: p99,
            fraction_reduced: reduced as f64 / samples.len() as f64,
        },
    )
}

/// Minimum over `[n - half, n + half]`, clamped at the ends.
///
/// A monotonic deque keeps this linear in the input rather than quadratic in
/// the window, which matters because the window is thousands of samples wide
/// at 48 kHz and the input is minutes long.
fn running_min(src: &[f64], half: usize) -> Vec<f64> {
    let n = src.len();
    let mut out = vec![0.0; n];
    // Holds indices whose values increase from front to back; the front is
    // always the minimum of the current window.
    let mut deque: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    let mut next = 0usize;

    for i in 0..n {
        let hi = (i + half + 1).min(n);
        while next < hi {
            while deque.back().is_some_and(|&b| src[b] >= src[next]) {
                deque.pop_back();
            }
            deque.push_back(next);
            next += 1;
        }
        let lo = i.saturating_sub(half);
        while deque.front().is_some_and(|&f| f < lo) {
            deque.pop_front();
        }
        out[i] = deque.front().map_or(src[i], |&f| src[f]);
    }
    out
}

/// Mean over `[n - half, n + half]`, clamped at the ends.
fn running_mean(src: &[f64], half: usize) -> Vec<f64> {
    let n = src.len();
    let mut prefix = vec![0.0f64; n + 1];
    for i in 0..n {
        prefix[i + 1] = prefix[i] + src[i];
    }
    (0..n)
        .map(|i| {
            let lo = i.saturating_sub(half);
            let hi = (i + half + 1).min(n);
            (prefix[hi] - prefix[lo]) / (hi - lo) as f64
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::loudness::peak_dbfs;
    use super::*;

    const RATE: u32 = 48_000;

    fn tone(amp: f64, secs: f64, freq: f64) -> Vec<f32> {
        let n = (RATE as f64 * secs) as usize;
        (0..n)
            .map(|i| (amp * (std::f64::consts::TAU * freq * i as f64 / RATE as f64).sin()) as f32)
            .collect()
    }

    #[test]
    fn nothing_under_the_ceiling_is_touched() {
        let quiet = tone(0.2, 1.0, 200.0);
        let (out, r) = limit(&quiet, RATE, LimiterOptions::default());
        assert_eq!(out, quiet);
        assert_eq!(r.max_reduction_db, 0.0);
        assert_eq!(r.sustained_reduction_db, 0.0);
        assert_eq!(r.fraction_reduced, 0.0);
    }

    #[test]
    fn the_ceiling_holds_on_a_loud_tone() {
        let loud = tone(0.95, 1.0, 200.0);
        let (out, r) = limit(&loud, RATE, LimiterOptions::default());
        let peak = peak_dbfs(&out);
        assert!(
            peak <= -1.0 + 1e-4,
            "peak {peak:.3} dBFS breached the ceiling"
        );
        assert!(r.max_reduction_db > 0.0);
    }

    #[test]
    fn the_ceiling_holds_on_an_isolated_transient() {
        // The case gain-clamping handles badly and a limiter exists for: one spike
        // in an otherwise quiet signal. The rest of the signal must survive it.
        // Three seconds, so the ~20 ms the limiter spends on one spike is under
        // the 1% that `sustained_reduction_db` measures. That is the point of the
        // metric: brief is not sustained.
        let mut s = tone(0.1, 3.0, 200.0);
        let spike = RATE as usize * 3 / 2;
        s[spike] = 0.99;
        s[spike + 1] = -0.99;

        let (out, r) = limit(&s, RATE, LimiterOptions::default());
        let peak = peak_dbfs(&out);
        assert!(peak <= -1.0 + 1e-4, "peak {peak:.3} dBFS");

        // Far from the spike the signal is untouched.
        let far = RATE as usize / 4;
        assert!(
            (out[far] - s[far]).abs() < 1e-6,
            "a spike at 0.5 s changed the signal at {far}"
        );
        assert!(
            r.fraction_reduced < 0.05,
            "one spike ducked {:.1}% of the file",
            r.fraction_reduced * 100.0
        );
        // A lone transient is not pumping, however deeply it was reduced.
        assert!(r.max_reduction_db > 0.5, "{:.2} dB", r.max_reduction_db);
        assert_eq!(
            r.sustained_reduction_db, 0.0,
            "one spike registered as {:.2} dB of sustained reduction",
            r.sustained_reduction_db
        );
    }

    #[test]
    fn continuous_limiting_registers_as_sustained() {
        // The other half of the contract above: a signal over the ceiling
        // throughout must drive the sustained figure, since that is pumping.
        let loud = tone(0.95, 1.0, 200.0);
        let (_, r) = limit(&loud, RATE, LimiterOptions::default());
        assert!(
            r.sustained_reduction_db > 0.3,
            "a tone 0.6 dB over the ceiling gave {:.2} dB sustained",
            r.sustained_reduction_db
        );
        assert!(r.fraction_reduced > 0.5, "{:.2}", r.fraction_reduced);
    }

    #[test]
    fn the_envelope_never_overshoots_anywhere() {
        // The safety property stated in the module doc, checked sample by sample
        // on a signal whose amplitude sweeps through the ceiling repeatedly.
        let n = RATE as usize * 2;
        let s: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f64 / RATE as f64;
                let env = 0.5 + 0.5 * (std::f64::consts::TAU * 3.0 * t).sin();
                (env * 1.2 * (std::f64::consts::TAU * 220.0 * t).sin()) as f32
            })
            .collect();
        let (out, _) = limit(&s, RATE, LimiterOptions::default());
        let ceiling = 10f64.powf(-1.0 / 20.0) as f32;
        for (i, &x) in out.iter().enumerate() {
            assert!(
                x.abs() <= ceiling + 1e-5,
                "sample {i} is {x} against a ceiling of {ceiling}"
            );
        }
    }

    #[test]
    fn the_gain_envelope_is_smooth_enough_not_to_click() {
        // A limiter that switches gain abruptly is audible as a click. Bounding
        // the per-sample change is what the running mean buys.
        let mut s = tone(0.1, 1.0, 200.0);
        s[RATE as usize / 2] = 0.99;
        let (out, _) = limit(&s, RATE, LimiterOptions::default());

        let ratio = |i: usize| {
            if s[i].abs() < 1e-6 {
                1.0
            } else {
                (out[i] / s[i]) as f64
            }
        };
        let mut worst = 0.0f64;
        for i in 1..s.len() {
            if s[i].abs() > 1e-3 && s[i - 1].abs() > 1e-3 {
                worst = worst.max((ratio(i) - ratio(i - 1)).abs());
            }
        }
        assert!(worst < 0.01, "gain jumped {worst:.4} in one sample");
    }

    #[test]
    fn a_wider_window_ducks_more_of_the_signal() {
        let mut s = tone(0.1, 1.0, 200.0);
        s[RATE as usize / 2] = 0.99;
        let narrow = limit(
            &s,
            RATE,
            LimiterOptions {
                window_ms: 1.0,
                ..Default::default()
            },
        )
        .1;
        let wide = limit(
            &s,
            RATE,
            LimiterOptions {
                window_ms: 20.0,
                ..Default::default()
            },
        )
        .1;
        assert!(
            wide.fraction_reduced > narrow.fraction_reduced * 4.0,
            "20 ms ducked {:.4}, 1 ms ducked {:.4}",
            wide.fraction_reduced,
            narrow.fraction_reduced
        );
    }

    #[test]
    fn empty_input_is_handled() {
        let (out, r) = limit(&[], RATE, LimiterOptions::default());
        assert!(out.is_empty());
        assert_eq!(r.max_reduction_db, 0.0);
        assert_eq!(r.sustained_reduction_db, 0.0);
    }

    #[test]
    fn the_running_min_matches_a_direct_scan() {
        // The deque is an optimization; this pins it against the obvious version.
        let src: Vec<f64> = (0..500).map(|i| ((i * 37) % 101) as f64 / 101.0).collect();
        for half in [0usize, 1, 7, 64, 600] {
            for (i, &got) in running_min(&src, half).iter().enumerate() {
                let lo = i.saturating_sub(half);
                let hi = (i + half + 1).min(src.len());
                let slow = src[lo..hi].iter().copied().fold(f64::INFINITY, f64::min);
                assert!(
                    (got - slow).abs() < 1e-12,
                    "half={half} i={i}: {got} vs {slow}"
                );
            }
        }
    }
}
