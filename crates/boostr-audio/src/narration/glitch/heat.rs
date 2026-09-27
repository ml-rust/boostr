//! How far a part's pitch rises above the voice's usual pitch: a part read
//! calm stays under about 1.5, a part over [`HEATED`] tends to sound
//! shouted or excited.
//!
//! Every sum runs in `f64` in sample order, and the decimated signal is
//! stored as `f32`, so the pitches match the JavaScript pitch track bit for
//! bit.

/// A part whose heat is over this tends to sound shouted or excited.
pub const HEATED: f64 = 1.6;

/// The input's sample rate, 48 kHz, is decimated by this factor.
const DECIMATE: usize = 3;
/// The pitch track's sample rate after decimation.
const TRACK_RATE: f64 = 16000.0;
/// Samples in one pitch window at the track rate: 40 ms.
const WINDOW: usize = 640;
/// Samples between pitch windows at the track rate: 20 ms.
const STEP: usize = 320;
/// The highest pitch read, in Hz.
const MAX_PITCH: f64 = 400.0;
/// The lowest pitch read, in Hz.
const MIN_PITCH: f64 = 60.0;
/// A window with an RMS level under this is silent and has no pitch.
const MIN_RMS: f64 = 0.01;
/// A window is voiced when its best autocorrelation is over this share of
/// its energy.
const VOICED: f64 = 0.4;
/// A reading more than this factor off the part's own median is a tracking
/// error: an octave.
const OCTAVE: f64 = 1.9;
/// A part with fewer readings than this, about 0.8 s of voicing, is too
/// short to judge.
const MIN_READINGS: usize = 40;
/// The median.
const MEDIAN: f64 = 0.5;
/// A part's pitch peaks are this percentile of its readings.
const PEAK: f64 = 0.95;

/// The pitch, in Hz, of each voiced 40 ms window of 48 kHz audio.
///
/// The audio is decimated to 16 kHz by averaging each 3 samples. Windows of
/// 640 samples step by 320. A window with an RMS level under 0.01 is
/// skipped. The pitch is 16000 over the lag from 40 to 266 samples (400 to
/// 60 Hz) with the largest autocorrelation of the mean-removed window,
/// the first on a tie. It counts only when that correlation is positive and
/// over 0.4 of the window's energy.
pub fn pitches(x: &[f32]) -> Vec<f64> {
    let y: Vec<f32> = x
        .as_chunks::<DECIMATE>()
        .0
        .iter()
        .map(|c| ((f64::from(c[0]) + f64::from(c[1]) + f64::from(c[2])) / 3.0) as f32)
        .collect();
    let lo = (TRACK_RATE / MAX_PITCH).floor() as usize;
    let hi = (TRACK_RATE / MIN_PITCH).floor() as usize;
    let mut out = Vec::new();
    let mut centered = vec![0.0f64; WINDOW];
    let mut i = 0;
    while i + WINDOW <= y.len() {
        let window = &y[i..i + WINDOW];
        let mut mean = 0.0f64;
        for &v in window {
            mean += f64::from(v);
        }
        mean /= WINDOW as f64;
        for (c, &v) in centered.iter_mut().zip(window) {
            *c = f64::from(v) - mean;
        }
        let mut e = 0.0f64;
        for &c in &centered {
            e += c * c;
        }
        // A NaN level is not under the floor, so the window is read on.
        let rms = (e / WINDOW as f64).sqrt();
        if rms.is_nan() || rms >= MIN_RMS {
            let mut best = 0.0f64;
            let mut lag = 0;
            for l in lo..=hi {
                let mut c = 0.0f64;
                for (a, b) in centered.iter().zip(&centered[l..]) {
                    c += a * b;
                }
                if c > best {
                    best = c;
                    lag = l;
                }
            }
            if lag > 0 && best > VOICED * e {
                out.push(TRACK_RATE / lag as f64);
            }
        }
        i += STEP;
    }
    out
}

/// The value at `floor(len * p)` of `list` sorted ascending, capped at the
/// last; 0 for an empty list.
pub fn percentile(list: &[f64], p: f64) -> f64 {
    let mut s = list.to_vec();
    s.sort_by(f64::total_cmp);
    if s.is_empty() {
        return 0.0;
    }
    let at = ((s.len() as f64 * p).floor() as usize).min(s.len() - 1);
    s[at]
}

/// The part's pitch peaks, its 95th-percentile pitch, over `voice_median`,
/// the voice's usual pitch (the median pitch of its reference).
///
/// Readings more than [`OCTAVE`] times off the part's own median pitch are
/// dropped as tracking errors. Fewer than 40 readings left, or a
/// `voice_median` of 0 or NaN (no reference pitch), gives 0.
///
/// A question rises on purpose: a phrase whose trimmed text ends in `?`,
/// optionally followed by closing quotes (`"`, `'`, `”`, `’`) or `)`, is not
/// judged by heat.
pub fn heat(part: &[f32], voice_median: f64) -> f64 {
    let p = pitches(part);
    let own = percentile(&p, MEDIAN);
    let kept: Vec<f64> = p
        .into_iter()
        .filter(|&f| f < own * OCTAVE && f > own / OCTAVE)
        .collect();
    if kept.len() >= MIN_READINGS && voice_median != 0.0 && !voice_median.is_nan() {
        percentile(&kept, PEAK) / voice_median
    } else {
        0.0
    }
}

/// The median pitch of `x`, 0 when no window is voiced.
pub fn median_pitch(x: &[f32]) -> f64 {
    percentile(&pitches(x), MEDIAN)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 200 Hz sine at 48 kHz: 80 samples a period at the track rate.
    fn tone(seconds: f64) -> Vec<f32> {
        let n = (48000.0 * seconds) as usize;
        (0..n)
            .map(|k| (0.3 * (std::f64::consts::TAU * 200.0 * k as f64 / 48000.0).sin()) as f32)
            .collect()
    }

    #[test]
    fn percentile_takes_the_floor_index() {
        assert_eq!(percentile(&[3.0, 1.0, 2.0], 0.5), 2.0);
        assert_eq!(percentile(&[3.0, 1.0, 2.0], 0.95), 3.0);
        assert_eq!(percentile(&[3.0, 1.0, 2.0], 1.0), 3.0);
        assert_eq!(percentile(&[4.0, 1.0, 3.0, 2.0], 0.5), 3.0);
        assert_eq!(percentile(&[], 0.5), 0.0);
    }

    #[test]
    fn a_steady_tone_reads_its_pitch() {
        let p = pitches(&tone(1.0));
        assert_eq!(p.len(), 49);
        assert!(p.iter().all(|&f| f == 200.0));
        assert_eq!(median_pitch(&tone(1.0)), 200.0);
    }

    #[test]
    fn silence_has_no_pitch() {
        assert!(pitches(&[0.0; 48000]).is_empty());
        assert_eq!(median_pitch(&[0.0; 48000]), 0.0);
        assert_eq!(heat(&[0.0; 48000], 100.0), 0.0);
    }

    #[test]
    fn heat_is_the_peak_over_the_voice_median() {
        let x = tone(1.0);
        assert_eq!(heat(&x, 100.0), 2.0);
        assert!(heat(&x, 100.0) > HEATED);
        assert_eq!(heat(&x, 0.0), 0.0);
        assert_eq!(heat(&x, f64::NAN), 0.0);
        // 24 readings are too few to judge.
        assert_eq!(heat(&tone(0.5), 100.0), 0.0);
    }
}
