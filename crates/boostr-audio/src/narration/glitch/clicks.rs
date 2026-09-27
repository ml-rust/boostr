//! Clicks in a take: impulses a few milliseconds wide in a pause, found in
//! the second difference and smoothed away.
//!
//! Every sum runs in `f64` in sample order, and every stored sample is
//! rounded to `f32`, so the results match the JavaScript click check and
//! repair bit for bit.

use crate::narration::js::{js_round, js_to_fixed};

/// Hops per second of the second-difference energy: 2 ms hops.
const HOPS_PER_SECOND: f64 = 500.0;
/// Hops in one frame of the take's loud level.
const LEVEL_FRAME_HOPS: usize = 5;
/// The loud level is this percentile of the frame levels.
const LOUD_SHARE: f64 = 0.95;
/// A pause sounds this many dB or more under the loud level.
const PAUSE_DB: f64 = -30.0;
/// The quiet test reads from this many seconds beside the click.
const PAUSE_NEAR: f64 = 0.003;
/// ... out to this many seconds beside it.
const PAUSE_FAR: f64 = 0.02;
/// Hops at each end that are never a click.
const EDGE_HOPS: usize = 2;
/// A click's energy is at least this.
const MIN_ENERGY: f64 = 0.01;
/// Hops on each side that make up the local level.
const LOCAL_HOPS: usize = 75;
/// Hops on each side of the spike that the local level leaves out.
const LOCAL_SKIP: usize = 3;
/// The local level is this percentile of the hops around the spike.
const LOCAL_SHARE: f64 = 0.9;
/// A click rises more than this many times over the local level.
const SHARPNESS: f64 = 6.0;
/// A click is at most this many hops wide at half its height.
const MAX_WIDTH: usize = 3;
/// Seconds past a hop's start where its discontinuity sits.
const HOP_CENTER: f64 = 0.001;
/// A click this many seconds or less after the last one is left out.
const MIN_SPACING: f64 = 0.05;
/// Decimals of the click times.
const TIME_DECIMALS: usize = 2;
/// Half the length of the moving average, in samples: 17 taps.
const TAPS_HALF: usize = 8;
/// Seconds over which a repair blends in at each edge.
const BLEND: f64 = 0.001;
/// Half widths of the smoothed span, in milliseconds, tried in turn.
pub const REPAIR_HALF_MS: [f64; 4] = [2.0, 4.0, 8.0, 12.0];
/// A click found before and after a repair within this many seconds is the
/// same click.
const SAME_CLICK: f64 = 0.05;

/// A take after [`repair_clicks`].
#[derive(Debug, Clone, PartialEq)]
pub struct ClickRepair {
    /// The take with its clicks smoothed.
    pub samples: Vec<f32>,
    /// Clicks found and smoothed away, in seconds.
    pub fixed: Vec<f64>,
    /// Clicks left after every repair, in seconds.
    pub left: Vec<f64>,
}

/// The median magnitude of `x[a..b]`, the upper middle of an even count; 0
/// for an empty span. Ends outside the take are clamped to it.
fn level(x: &[f32], a: i64, b: i64) -> f64 {
    let len = x.len() as i64;
    let (a, b) = (a.clamp(0, len) as usize, b.clamp(0, len) as usize);
    if a >= b {
        return 0.0;
    }
    let mut v: Vec<f32> = x[a..b].iter().map(|s| s.abs()).collect();
    let mid = v.len() >> 1;
    let (_, m, _) = v.select_nth_unstable_by(mid, f32::total_cmp);
    f64::from(*m)
}

/// The value at `floor(len * share)` of `v` sorted ascending, or `fallback`
/// when there is none or it is 0 or NaN.
fn floor_percentile(v: &mut [f64], share: f64, fallback: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    let at = (v.len() as f64 * share).floor() as usize;
    // `|| fallback`: 0 and NaN are falsy.
    v.get(at)
        .copied()
        .filter(|&p| p != 0.0 && !p.is_nan())
        .unwrap_or(fallback)
}

/// The times of the clicks in a take, in seconds, rounded to 10 ms.
///
/// The take is read in 2 ms hops, each hop's energy the RMS of the second
/// difference, which an impulse dominates and a voice does not. A click
/// is a hop of energy 0.01 or more that no neighbor exceeds, at most 3 hops
/// wide at half its height, and more than 6 times the 90th percentile of
/// the hops within 75 of it (those within 3 left out). Only a click in a
/// pause counts: the median magnitude from 3 to 20 ms on each side of the
/// hop's middle is 30 dB or more under the take's loud level, the 95th
/// percentile of its 10 ms median magnitudes. Inside speech a consonant
/// burst has the same shape, so speech is never flagged. A click 0.05 s or
/// less after the last one found is left out.
///
/// A percentile is the value at `floor(len * share)` of the sorted list; an
/// empty list or a 0 there reads as 1e-9. A `rate` under 250 Hz leaves no
/// whole hop, so nothing is found.
pub fn clicks(x: &[f32], rate: u32) -> Vec<f64> {
    let rate = f64::from(rate);
    let hop = js_round(rate / HOPS_PER_SECOND) as usize;
    let mut out = Vec::new();
    if hop == 0 {
        return out;
    }
    let frame = hop * LEVEL_FRAME_HOPS;
    let mut frames = Vec::new();
    let mut k = 0;
    while k + frame <= x.len() {
        frames.push(level(x, k as i64, (k + frame) as i64));
        k += frame;
    }
    let loud = floor_percentile(&mut frames, LOUD_SHARE, 1e-9);
    let quiet = loud * 10f64.powf(PAUSE_DB / 20.0);
    let near = js_round(PAUSE_NEAR * rate) as i64;
    let far = js_round(PAUSE_FAR * rate) as i64;
    let in_pause = |c: i64| level(x, c - far, c - near).max(level(x, c + near, c + far)) < quiet;

    let mut e = Vec::new();
    let mut i = 0;
    while i + hop < x.len() {
        let mut s = 0.0f64;
        // Each window is x[k - 2..=k] for k from max(2, i) to the hop's end.
        for w in x[i.max(2) - 2..i + hop].windows(3) {
            let d = f64::from(w[2]) - 2.0 * f64::from(w[1]) + f64::from(w[0]);
            s += d * d;
        }
        e.push((s / hop as f64).sqrt());
        i += hop;
    }

    let center = js_round(HOP_CENTER * rate) as i64;
    let mut local = Vec::with_capacity(2 * LOCAL_HOPS + 1);
    for f in EDGE_HOPS..e.len().saturating_sub(EDGE_HOPS) {
        if e[f] < MIN_ENERGY || e[f] < e[f - 1] || e[f] < e[f + 1] {
            continue;
        }
        local.clear();
        let lo = f.saturating_sub(LOCAL_HOPS);
        let hi = (f + LOCAL_HOPS).min(e.len() - 1);
        local.extend(
            (lo..=hi)
                .filter(|&k| k.abs_diff(f) > LOCAL_SKIP)
                .map(|k| e[k]),
        );
        let p90 = floor_percentile(&mut local, LOCAL_SHARE, 1e-9);
        let half = e[f] / 2.0;
        let width = 1
            + e[..f].iter().rev().take_while(|&&v| v > half).count()
            + e[f + 1..].iter().take_while(|&&v| v > half).count();
        if width > MAX_WIDTH {
            continue;
        }
        let t = (f * hop) as f64 / rate;
        if e[f] > SHARPNESS * p90
            && in_pause((f * hop) as i64 + center)
            && out.last().is_none_or(|&last| t - last > MIN_SPACING)
        {
            out.push(js_to_fixed(t, TIME_DECIMALS));
        }
    }
    out
}

/// Smooths each click at `times` with a 17-sample moving average.
///
/// A click time is the start of a 2 ms hop, so each span centers 1 ms
/// after it. The average fully replaces the take within `half_ms` of the
/// center and blends in linearly over 1 ms beyond that. Every average reads
/// the take as given, so overlapping spans do not compound: a later one
/// overwrites. The first and last 8 samples are never changed.
pub fn declick(x: &[f32], times: &[f64], rate: u32, half_ms: f64) -> Vec<f32> {
    let rate = f64::from(rate);
    let mut y = x.to_vec();
    let half = js_round(half_ms / 1000.0 * rate) as i64;
    let blend = js_round(BLEND * rate) as i64;
    let taps = TAPS_HALF as i64;
    let len = x.len() as i64;
    for &t in times {
        let c = js_round((t + HOP_CENTER) * rate) as i64;
        let from = taps.max(c.saturating_sub(half + blend));
        let to = (len - taps).min(c.saturating_add(half + blend));
        for k in from..to {
            let ku = k as usize;
            let mut m = 0.0f64;
            for s in &x[ku - TAPS_HALF..=ku + TAPS_HALF] {
                m += f64::from(*s);
            }
            m /= (2 * TAPS_HALF + 1) as f64;
            let d = (k - c).abs();
            let w = if d <= half {
                1.0
            } else {
                1.0 - (d - half) as f64 / blend as f64
            };
            y[ku] = (f64::from(x[ku]) * (1.0 - w) + m * w) as f32;
        }
    }
    y
}

/// Repairs every click it can.
///
/// The clicks found are smoothed with [`declick`], then the take is checked
/// again, and any click left is smoothed over a wider span: half widths of
/// 2, 4, 8, then 12 ms. A click only counts in a pause, so the smoothing
/// never reaches the voice. `fixed` holds the clicks first found that no
/// click left sits within 0.05 s of.
pub fn repair_clicks(x: &[f32], rate: u32) -> ClickRepair {
    let fixed = clicks(x, rate);
    let mut y = x.to_vec();
    let mut left = fixed.clone();
    for half_ms in REPAIR_HALF_MS {
        if left.is_empty() {
            break;
        }
        y = declick(&y, &left, rate, half_ms);
        left = clicks(&y, rate);
    }
    let fixed = fixed
        .into_iter()
        .filter(|t| !left.iter().any(|v| (v - t).abs() < SAME_CLICK))
        .collect();
    ClickRepair {
        samples: y,
        fixed,
        left,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 48000;

    #[test]
    fn pause_floor_matches_javascript() {
        // `10 ** (-30 / 20)` in JavaScript.
        assert_eq!(10f64.powf(PAUSE_DB / 20.0), 0.03162277660168379);
    }

    #[test]
    fn a_spike_in_silence_is_a_click_and_is_repaired() {
        let mut x = vec![0.0f32; 48000];
        x[24010] = 0.5;
        assert_eq!(clicks(&x, RATE), [0.5]);
        let r = repair_clicks(&x, RATE);
        assert_eq!(r.fixed, [0.5]);
        assert!(r.left.is_empty());
        assert!(r.samples[24010] < 0.1);
    }

    #[test]
    fn a_click_right_after_another_is_left_out() {
        let mut x = vec![0.0f32; 48000];
        x[24010] = 0.5;
        x[24970] = 0.5;
        assert_eq!(clicks(&x, RATE), [0.5]);
    }

    #[test]
    fn a_spike_inside_speech_is_not_a_click() {
        let mut x: Vec<f32> = (0..48000)
            .map(|k| (0.3 * (k as f64 * 0.026).sin()) as f32)
            .collect();
        x[24010] += 0.3;
        assert!(clicks(&x, RATE).is_empty());
    }

    #[test]
    fn a_wide_burst_is_not_a_click() {
        let mut x = vec![0.0f32; 48000];
        for (k, s) in x[24000..24480].iter_mut().enumerate() {
            *s = if k % 2 == 0 { 0.2 } else { -0.2 };
        }
        assert!(clicks(&x, RATE).is_empty());
    }

    #[test]
    fn declick_averages_and_keeps_the_first_samples() {
        let mut x = vec![1.0f32; 40];
        x[3] = 5.0;
        let y = declick(&x, &[0.0], RATE, 2.0);
        assert_eq!(y[3], 5.0);
        assert_eq!(y[8], (21.0f64 / 17.0) as f32);
        assert_eq!(y[31], 1.0);
        assert_eq!(y[32], 1.0);
    }

    #[test]
    fn nothing_is_found_in_short_or_empty_takes() {
        assert!(clicks(&[], RATE).is_empty());
        assert!(clicks(&[0.5; 100], RATE).is_empty());
        let r = repair_clicks(&[], RATE);
        assert!(r.samples.is_empty() && r.fixed.is_empty() && r.left.is_empty());
    }
}
