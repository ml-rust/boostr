//! The Povey analysis window.

use super::constants::{FRAME_LENGTH, POVEY_EXPONENT};

/// The Povey analysis window: a *symmetric* Hann window raised to 0.85.
///
/// `w[n] = (0.5 - 0.5 * cos(2*pi*n / 399))^0.85` for `n` in `0..400`.
///
/// The denominator is `FRAME_LENGTH - 1` = 399, i.e. plain `np.hanning(400)`,
/// the **symmetric** form with a zero at both ends. This is what
/// `SeamlessM4TFeatureExtractor` builds — it calls
/// `window_function(400, "povey", periodic=False)`, and `periodic=False` means
/// no `np.hanning(401)[:-1]` trim — and it is also what Kaldi's
/// `FeatureWindowFunction` uses. The periodic form (denominator 400) is the
/// tempting alternative: it is what `window_function` produces by default and
/// what most STFT code wants, but here it shifts every tap by a fraction of a
/// sample and changes the frame-edge taper.
///
/// Settled numerically, not by reading: the parity test against upstream's own
/// extractor agrees to `max|d| = 4.8e-7` with this symmetric form.
///
/// Raising to 0.85 is what distinguishes Povey from plain Hann; using plain
/// Hann changes the effective bandwidth of every FFT bin.
#[must_use]
pub fn povey_window() -> Vec<f64> {
    let denom = (FRAME_LENGTH - 1) as f64;
    (0..FRAME_LENGTH)
        .map(|n| {
            let hann = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / denom).cos();
            hann.powf(POVEY_EXPONENT)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn povey_window_shape_and_endpoints() {
        let w = povey_window();
        assert_eq!(w.len(), FRAME_LENGTH);
        // Symmetric Hann (denominator 399): BOTH endpoints are exactly zero.
        // Under the periodic form the last tap would be ~0.0001 instead.
        assert!(w[0].abs() < 1e-12);
        assert!(w[FRAME_LENGTH - 1].abs() < 1e-12);
        // Mirror symmetry about the midpoint between taps 199 and 200.
        for (n, (a, b)) in w.iter().zip(w.iter().rev()).enumerate() {
            assert!((a - b).abs() < 1e-12, "tap {n}");
        }
        // Peak straddles 199/200; neither reaches 1.0 exactly.
        assert!(w[199] > 0.9999 && w[199] < 1.0);
        assert!(w.iter().all(|v| (0.0..=1.0).contains(v)));
    }
}
