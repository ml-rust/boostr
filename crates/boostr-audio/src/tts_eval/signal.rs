//! Signal-level statistics of one rendered clip, at its native rate.
//!
//! One call joins three estimators this crate already ships:
//! [`measure_quality`] (peak, RMS, floor, SNR, clipping), [`integrated_lufs`]
//! (BS.1770-4 loudness) and [`estimate_pitch`] (YIN F0 and HNR). None of them
//! needs a model or a reference clip, so every render gets these numbers
//! whatever else the gate can or cannot load.
//!
//! What they tell you: level and headroom (`peak_dbfs`, `lufs`), noise under
//! the voice (`hnr_db`) as opposed to noise in the pauses (`floor_dbfs`,
//! `snr_db`), and pitch placement (`f0_mean_hz`, `f0_std_hz`). What they
//! cannot tell you: whether the words are right (see `intelligibility`) or
//! whether the voice is the right speaker (see `timbre`).

use crate::enhance::integrated_lufs;
use crate::error::Result;
use crate::pitch::{PitchOptions, estimate_pitch};
use crate::quality::measure_quality;

/// Level, noise and pitch statistics of one clip.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SignalStats {
    /// Clip length in seconds.
    pub duration_s: f64,
    /// `20 * log10(max |s|)`.
    pub peak_dbfs: f64,
    /// `20 * log10(rms)` over the whole clip.
    pub rms_dbfs: f64,
    /// Integrated loudness, BS.1770-4 gated. `-inf` for silence.
    pub lufs: f64,
    /// 5th-percentile 100 ms block RMS, dBFS: the noise in the pauses.
    pub floor_dbfs: f64,
    /// `rms_dbfs - floor_dbfs`.
    pub snr_db: f64,
    /// Samples at or beyond full scale.
    pub clipped_samples: usize,
    /// Mean F0 over voiced frames. `None` when nothing is voiced.
    pub f0_mean_hz: Option<f64>,
    /// Population standard deviation of F0 over voiced frames.
    pub f0_std_hz: Option<f64>,
    /// Fraction of analysis frames judged voiced.
    pub voiced_fraction: f64,
    /// Mean harmonic-to-noise ratio over voiced frames, dB: noise UNDER the
    /// voice, which `floor_dbfs` cannot see.
    pub hnr_db: Option<f64>,
}

/// Measure `samples` (mono, `[-1, 1]`) at `sample_rate`.
///
/// Pitch runs with [`PitchOptions::default`], the adult-speech range every
/// other caller in this crate uses. Errors come from the three estimators:
/// an empty clip or a zero rate.
pub fn signal_stats(samples: &[f32], sample_rate: u32) -> Result<SignalStats> {
    let quality = measure_quality(samples, sample_rate)?;
    let lufs = integrated_lufs(samples, sample_rate)?;
    let pitch = estimate_pitch(samples, sample_rate, PitchOptions::default())?;
    Ok(SignalStats {
        duration_s: quality.duration_s,
        peak_dbfs: quality.peak_dbfs,
        rms_dbfs: quality.rms_dbfs,
        lufs,
        floor_dbfs: quality.floor_dbfs,
        snr_db: quality.snr_db,
        clipped_samples: quality.clipped_samples,
        f0_mean_hz: pitch.mean_hz,
        f0_std_hz: pitch.std_hz,
        voiced_fraction: pitch.voiced_fraction,
        hnr_db: pitch.mean_hnr_db,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 16_000;

    fn sine(amplitude: f32, freq: f64, len: usize) -> Vec<f32> {
        (0..len)
            .map(|n| {
                amplitude * (std::f64::consts::TAU * freq * n as f64 / RATE as f64).sin() as f32
            })
            .collect()
    }

    #[test]
    fn steady_tone_is_voiced_at_its_frequency() {
        let samples = sine(0.5, 150.0, RATE as usize * 2);
        let stats = signal_stats(&samples, RATE).expect("stats");
        assert!((stats.duration_s - 2.0).abs() < 1e-6);
        assert!(
            (stats.peak_dbfs - (-6.02)).abs() < 0.1,
            "{}",
            stats.peak_dbfs
        );
        assert!(stats.lufs.is_finite());
        let f0 = stats.f0_mean_hz.expect("voiced");
        assert!((f0 - 150.0).abs() < 3.0, "f0 = {f0}");
        assert!(stats.voiced_fraction > 0.9);
        assert!(stats.hnr_db.expect("hnr") > 20.0);
        assert_eq!(stats.clipped_samples, 0);
    }

    #[test]
    fn silence_has_no_pitch_and_no_loudness() {
        let samples = vec![0.0f32; RATE as usize];
        let stats = signal_stats(&samples, RATE).expect("stats");
        assert_eq!(stats.f0_mean_hz, None);
        assert_eq!(stats.hnr_db, None);
        assert_eq!(stats.voiced_fraction, 0.0);
        assert_eq!(stats.lufs, f64::NEG_INFINITY);
    }

    #[test]
    fn empty_input_is_an_error() {
        assert!(signal_stats(&[], RATE).is_err());
        assert!(signal_stats(&[0.1], 0).is_err());
    }
}
