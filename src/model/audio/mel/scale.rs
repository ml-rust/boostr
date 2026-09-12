//! Hz↔mel warping and filterbank edge placement.

/// Which Hz↔mel warping to use when placing the filterbank edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MelScale {
    /// HTK formula: `2595 * log10(1 + hz / 700)`. librosa's `htk=True`.
    Htk,
    /// Slaney formula: linear below 1 kHz, logarithmic above. librosa's
    /// `htk=False`, and what Whisper uses.
    Slaney,
}

impl MelScale {
    /// Hz to mel under this warping.
    #[inline]
    pub fn to_mel(self, hz: f64) -> f64 {
        match self {
            Self::Htk => hz_to_mel_htk64(hz),
            Self::Slaney => hz_to_mel_slaney(hz),
        }
    }

    /// Mel back to Hz under this warping.
    #[inline]
    pub fn to_hz(self, mel: f64) -> f64 {
        match self {
            Self::Htk => mel_to_hz_htk64(mel),
            Self::Slaney => mel_to_hz_slaney(mel),
        }
    }
}

/// Convert frequency in Hz to mel scale (HTK formula).
#[inline]
pub fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert mel scale value back to Hz.
#[inline]
pub fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
}

/// Slope of the linear part of the Slaney scale: 1 mel per `200/3` Hz.
const SLANEY_F_SP: f64 = 200.0 / 3.0;
/// Frequency at which the Slaney scale switches from linear to logarithmic.
const SLANEY_MIN_LOG_HZ: f64 = 1000.0;
/// The mel value of [`SLANEY_MIN_LOG_HZ`]; exactly 15.
const SLANEY_MIN_LOG_MEL: f64 = SLANEY_MIN_LOG_HZ / SLANEY_F_SP;

/// Natural-log step per mel above 1 kHz: 6.4x in frequency over 27 mels.
#[inline]
fn slaney_logstep() -> f64 {
    6.4f64.ln() / 27.0
}

/// Convert Hz to mel on the Slaney scale (librosa's `htk=False`).
#[inline]
pub fn hz_to_mel_slaney(hz: f64) -> f64 {
    if hz >= SLANEY_MIN_LOG_HZ {
        SLANEY_MIN_LOG_MEL + (hz / SLANEY_MIN_LOG_HZ).ln() / slaney_logstep()
    } else {
        hz / SLANEY_F_SP
    }
}

/// Convert a Slaney-scale mel value back to Hz.
#[inline]
pub fn mel_to_hz_slaney(mel: f64) -> f64 {
    if mel >= SLANEY_MIN_LOG_MEL {
        SLANEY_MIN_LOG_HZ * (slaney_logstep() * (mel - SLANEY_MIN_LOG_MEL)).exp()
    } else {
        SLANEY_F_SP * mel
    }
}

/// Convert Hz to mel on the HTK scale, in f64.
#[inline]
fn hz_to_mel_htk64(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert an HTK mel value back to Hz, in f64.
#[inline]
fn mel_to_hz_htk64(mel: f64) -> f64 {
    700.0 * (10.0f64.powf(mel / 2595.0) - 1.0)
}

/// Compute `num_mel_bins + 2` linearly spaced mel frequencies, converted back to Hz.
pub fn mel_frequencies(num_mel_bins: usize, fmin: f32, fmax: f32) -> Vec<f32> {
    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);
    let n = num_mel_bins + 2;
    (0..n)
        .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f32 / (n - 1) as f32))
        .collect()
}

/// Compute `num_mel_bins + 2` filterbank edge frequencies in Hz, evenly
/// spaced on the chosen mel scale.
///
/// Done in f64: the Slaney edges go through `ln`/`exp`, and the filter
/// normalization then divides by small differences between adjacent edges,
/// where f32 rounding is visible in the resulting energies.
pub fn mel_frequencies_with(
    num_mel_bins: usize,
    fmin: f64,
    fmax: f64,
    scale: MelScale,
) -> Vec<f64> {
    let mel_min = scale.to_mel(fmin);
    let mel_max = scale.to_mel(fmax);
    let n = num_mel_bins + 2;
    (0..n)
        .map(|i| scale.to_hz(mel_min + (mel_max - mel_min) * i as f64 / (n - 1) as f64))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hz_mel_roundtrip() {
        let hz = 1000.0;
        let mel = hz_to_mel(hz);
        let recovered = mel_to_hz(mel);
        assert!(
            (recovered - hz).abs() < 0.01,
            "roundtrip failed: {recovered}"
        );
    }

    #[test]
    fn test_mel_frequencies_count() {
        let freqs = mel_frequencies(80, 0.0, 8000.0);
        assert_eq!(freqs.len(), 82); // num_mel_bins + 2
        assert!((freqs[0] - 0.0).abs() < 1.0);
    }

    // --- Slaney mel scale -------------------------------------------------------

    #[test]
    fn slaney_linear_region_is_hz_over_f_sp() {
        // Below 1 kHz the scale is exactly linear: 200/3 Hz per mel.
        assert!((hz_to_mel_slaney(200.0) - 3.0).abs() < 1e-12);
        assert!((hz_to_mel_slaney(0.0)).abs() < 1e-12);
        assert!((mel_to_hz_slaney(3.0) - 200.0).abs() < 1e-9);
    }

    #[test]
    fn slaney_breakpoint_is_15_mels() {
        // 1000 Hz sits exactly on the linear/log join, at mel 15.
        assert!((hz_to_mel_slaney(1000.0) - 15.0).abs() < 1e-12);
        assert!((mel_to_hz_slaney(15.0) - 1000.0).abs() < 1e-9);
    }

    #[test]
    fn slaney_log_region_spans_6p4x_over_27_mels() {
        // By construction 6.4 kHz is 27 mels above the 1 kHz breakpoint.
        assert!((hz_to_mel_slaney(6400.0) - 42.0).abs() < 1e-9);
        assert!((mel_to_hz_slaney(42.0) - 6400.0).abs() < 1e-6);
    }

    #[test]
    fn slaney_roundtrips_across_both_regions() {
        for hz in [0.0, 50.0, 500.0, 999.0, 1000.0, 2000.0, 8000.0] {
            let back = mel_to_hz_slaney(hz_to_mel_slaney(hz));
            assert!((back - hz).abs() < 1e-6, "roundtrip {hz} -> {back}");
        }
    }

    #[test]
    fn slaney_and_htk_edges_differ() {
        let htk = mel_frequencies_with(80, 0.0, 8000.0, MelScale::Htk);
        let slaney = mel_frequencies_with(80, 0.0, 8000.0, MelScale::Slaney);
        assert_eq!(htk.len(), 82);
        assert_eq!(slaney.len(), 82);
        // Endpoints coincide; the interior warping does not.
        assert!((htk[0] - slaney[0]).abs() < 1e-9);
        assert!((htk[81] - slaney[81]).abs() < 1e-6);
        assert!(
            (htk[40] - slaney[40]).abs() > 1.0,
            "htk {} vs slaney {}",
            htk[40],
            slaney[40]
        );
    }
}
