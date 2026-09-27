//! Glitches at a generated take's edges: a dark thump set apart from the
//! speech is silenced, a take cut off mid-sound is reported, and both edges
//! fade so no take starts or stops on a click.
//!
//! Every sum runs in `f64` in sample order, and every stored sample is
//! rounded to `f32`, so the result matches the JavaScript tidy bit for bit.

use crate::narration::js::{js_round, js_to_fixed_text};

/// A frame is loud within this many dB of the take's loudest frame.
const LOUD_DB: f64 = -40.0;
/// A thump is shorter than this many 10 ms frames.
const THUMP_FRAMES: usize = 7;
/// A thump's mean brightness is under this many dB.
const THUMP_BRIGHTNESS_DB: f64 = -16.0;
/// A thump is set apart from the speech by this many frames or more.
const THUMP_GAP_FRAMES: i64 = 4;
/// Frames past the front thump's end that are silenced with it.
const FRONT_CLEAR_FRAMES: usize = 3;
/// Frames before the back thump's start that are silenced with it.
const BACK_CLEAR_FRAMES: i64 = 2;
/// The last frames of a cut-off take all sound within this many dB of the
/// peak.
const CUT_OFF_DB: f64 = -30.0;
/// Frames at the end that the cut-off test reads.
const CUT_OFF_FRAMES: usize = 3;
/// A cut-off take's last frame keeps more than this share of the level three
/// frames back.
const CUT_OFF_HOLD: f64 = 0.5;
/// Fade-in length, in seconds.
const FADE_IN: f64 = 0.01;
/// Fade-out length, in seconds.
const FADE_OUT: f64 = 0.025;
/// Decimals of the times in the repair messages.
const TIME_DECIMALS: usize = 2;

/// The fault a cut-off take reports.
pub const CUT_OFF: &str = "it stops mid-sound, cut off";

/// A take after [`tidy`]: its samples, and what it found, in words.
#[derive(Debug, Clone, PartialEq)]
pub struct TidyReport {
    /// The take with thumps silenced and both edges faded.
    pub samples: Vec<f32>,
    /// What cannot be repaired: [`CUT_OFF`].
    pub faults: Vec<String>,
    /// What was repaired: `a thump before the speech at 0.04s`, `a thump
    /// after the speech at 0.85s`.
    pub repairs: Vec<String>,
}

/// Repairs a take's edges and reports a cut-off end.
///
/// The take is read in 10 ms frames: each frame's RMS level and its
/// brightness, the energy of the first difference against the frame's
/// energy in dB. Runs of frames within 40 dB of the loudest frame are the
/// sounds. A thump is a first or last run of fewer than 7 frames, darker
/// than −16 dB on average, and 4 or more frames from the run beside it.
/// The back thump's distance runs from the previous run's end to the
/// thump's end. A front thump is zeroed from the start to 3 frames past
/// its end, a back thump from 2 frames before its start to the end.
///
/// A take whose last 3 frames all sound within 30 dB of the peak, and whose
/// last frame keeps more than half the level of the frame 3 back, stops
/// mid-sound: that is a fault, and nothing is repaired for it.
///
/// Last, the first 10 ms fade in and the last 25 ms fade out, linearly.
/// A `rate` under 50 Hz leaves no whole frame, so nothing is found.
pub fn tidy(x: &[f32], rate: u32) -> TidyReport {
    let rate = f64::from(rate);
    let hop = js_round(rate / 100.0) as usize;
    let mut env: Vec<f64> = Vec::new();
    let mut bright: Vec<f64> = Vec::new();
    if hop > 0 {
        let mut i = 0;
        while i + hop <= x.len() {
            let mut e = 0.0f64;
            let mut d = 0.0f64;
            for (k, &s) in x.iter().enumerate().skip(i).take(hop) {
                let v = f64::from(s);
                e += v * v;
                if k > 0 {
                    let step = v - f64::from(x[k - 1]);
                    d += step * step;
                }
            }
            env.push((e / hop as f64).sqrt());
            bright.push(10.0 * (d / (e + 1e-12) + 1e-12).log10());
            i += hop;
        }
    }
    // `Math.max`: a NaN level makes the peak NaN.
    let peak = env.iter().fold(1e-9f64, |p, &v| {
        if p.is_nan() || v.is_nan() {
            f64::NAN
        } else {
            p.max(v)
        }
    });
    let loud_floor = peak * 10f64.powf(LOUD_DB / 20.0);
    let runs = loud_runs(&env, loud_floor);

    let mut faults = Vec::new();
    let mut repairs = Vec::new();
    let mut out = x.to_vec();
    // Summed from +0 in frame order, as `reduce((t, v) => t + v, 0)`.
    let mean = |a: usize, b: usize| {
        let mut total = 0.0f64;
        for &v in &bright[a..=b] {
            total += v;
        }
        total / (b - a + 1) as f64
    };
    let is_thump = |run: (usize, usize), next_at: usize| {
        run.1 - run.0 + 1 < THUMP_FRAMES
            && mean(run.0, run.1) < THUMP_BRIGHTNESS_DB
            && (next_at as i64 - run.1 as i64).abs() >= THUMP_GAP_FRAMES
    };
    let at = |frame: usize| js_to_fixed_text((frame * hop) as f64 / rate, TIME_DECIMALS);
    if runs.len() > 1 {
        let first = runs[0];
        if is_thump(first, runs[1].0) {
            let end = out.len().min((first.1 + FRONT_CLEAR_FRAMES) * hop);
            out[..end].fill(0.0);
            repairs.push(format!("a thump before the speech at {}s", at(first.0)));
        }
        let last = runs[runs.len() - 1];
        if is_thump(last, runs[runs.len() - 2].1) {
            let start = ((last.0 as i64 - BACK_CLEAR_FRAMES) * hop as i64).max(0) as usize;
            if start < out.len() {
                out[start..].fill(0.0);
            }
            repairs.push(format!("a thump after the speech at {}s", at(last.0)));
        }
    }

    let n = env.len();
    let cut_floor = peak * 10f64.powf(CUT_OFF_DB / 20.0);
    if n > CUT_OFF_FRAMES
        && env[n - CUT_OFF_FRAMES..].iter().all(|&v| v > cut_floor)
        && env[n - 1] > CUT_OFF_HOLD * env[n - CUT_OFF_FRAMES]
    {
        faults.push(CUT_OFF.to_string());
    }

    let len = out.len();
    let fade_in = len.min(js_round(rate * FADE_IN) as usize);
    let fade_out = len.min(js_round(rate * FADE_OUT) as usize);
    for (k, s) in out.iter_mut().enumerate().take(fade_in) {
        *s = (f64::from(*s) * (k as f64 / fade_in as f64)) as f32;
    }
    for (k, s) in out.iter_mut().rev().enumerate().take(fade_out) {
        *s = (f64::from(*s) * (k as f64 / fade_out as f64)) as f32;
    }
    TidyReport {
        samples: out,
        faults,
        repairs,
    }
}

/// Runs of consecutive frames over `floor`, as first and last frame.
fn loud_runs(env: &[f64], floor: f64) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut f = 0;
    while f < env.len() {
        if env[f] > floor {
            let mut g = f;
            while g + 1 < env.len() && env[g + 1] > floor {
                g += 1;
            }
            runs.push((f, g));
            f = g;
        }
        f += 1;
    }
    runs
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 48000;
    const HOP: usize = 480;

    /// Samples alternating ±0.3 over frames `a..b`: loud and bright.
    fn bright(x: &mut [f32], a: usize, b: usize) {
        for (k, s) in x[a * HOP..b * HOP].iter_mut().enumerate() {
            *s = if k % 2 == 0 { 0.3 } else { -0.3 };
        }
    }

    /// A constant 0.3 over frames `a..b`: loud and dark.
    fn dark(x: &mut [f32], a: usize, b: usize) {
        x[a * HOP..b * HOP].fill(0.3);
    }

    #[test]
    fn db_floors_match_javascript() {
        // `10 ** (-40 / 20)` and `10 ** (-30 / 20)` in JavaScript.
        assert_eq!(10f64.powf(LOUD_DB / 20.0), 0.01);
        assert_eq!(10f64.powf(CUT_OFF_DB / 20.0), 0.03162277660168379);
    }

    #[test]
    fn silence_is_left_as_it_is() {
        let r = tidy(&[0.0; 4800], RATE);
        assert!(r.faults.is_empty() && r.repairs.is_empty());
        assert!(r.samples.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn edges_fade_and_a_held_end_is_cut_off() {
        let r = tidy(&[1.0; 2000], RATE);
        assert_eq!(r.samples[0], 0.0);
        assert_eq!(r.samples[240], 0.5);
        assert_eq!(r.samples[600], 1.0);
        assert_eq!(r.samples[1999 - 600], 0.5);
        assert_eq!(r.samples[1999], 0.0);
        assert_eq!(r.faults, [CUT_OFF]);
        assert!(r.repairs.is_empty());
    }

    #[test]
    fn a_buffer_shorter_than_its_fades_fades_whole() {
        let r = tidy(&[1.0; 100], RATE);
        assert_eq!(r.samples[0], 0.0);
        assert_eq!(r.samples[99], 0.0);
        assert_eq!(r.samples[50], (0.5f64 * (49.0 / 100.0)) as f32);
        assert!(r.faults.is_empty());
    }

    #[test]
    fn a_thump_before_the_speech_is_silenced() {
        let mut x = vec![0.0f32; 30 * HOP];
        dark(&mut x, 1, 3);
        bright(&mut x, 10, 20);
        let r = tidy(&x, RATE);
        assert_eq!(r.repairs, ["a thump before the speech at 0.01s"]);
        assert!(r.faults.is_empty());
        assert!(r.samples[..5 * HOP].iter().all(|&s| s == 0.0));
        assert_eq!(r.samples[10 * HOP + 300], x[10 * HOP + 300]);
    }

    #[test]
    fn a_thump_after_the_speech_is_silenced() {
        let mut x = vec![0.0f32; 30 * HOP];
        bright(&mut x, 10, 20);
        dark(&mut x, 25, 27);
        let r = tidy(&x, RATE);
        assert_eq!(r.repairs, ["a thump after the speech at 0.25s"]);
        assert!(r.samples[23 * HOP..].iter().all(|&s| s == 0.0));
        assert_eq!(r.samples[20 * HOP - 1], x[20 * HOP - 1]);
    }

    #[test]
    fn a_dark_burst_near_the_speech_is_kept() {
        let mut x = vec![0.0f32; 30 * HOP];
        dark(&mut x, 1, 3);
        bright(&mut x, 5, 20);
        let r = tidy(&x, RATE);
        assert!(r.repairs.is_empty());
        assert_eq!(r.samples[2 * HOP], 0.3);
    }
}
