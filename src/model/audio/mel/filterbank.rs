//! The triangular mel filterbank and its normalization.

use super::scale::{MelScale, mel_frequencies_with};

/// Post-construction scaling applied to each triangular filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MelNorm {
    /// Leave the unit-peak triangles as built.
    None,
    /// Scale filter `m` by `2 / (edge[m + 2] - edge[m])` so each filter has
    /// approximately unit area. librosa's `norm="slaney"`.
    Slaney,
}

/// Build the `[num_mel_bins, n_fft / 2 + 1]` triangular filterbank, row-major.
///
/// Uses librosa's ramp formulation so the shared edges between neighbouring
/// filters agree exactly rather than by a floating-point coincidence.
pub(super) fn mel_filterbank(
    num_mel_bins: usize,
    n_fft: usize,
    sample_rate: f64,
    fmin: f64,
    fmax: f64,
    scale: MelScale,
    normalize: MelNorm,
) -> Vec<f64> {
    let num_fft_bins = n_fft / 2 + 1;
    let edges = mel_frequencies_with(num_mel_bins, fmin, fmax, scale);
    let fft_freqs: Vec<f64> = (0..num_fft_bins)
        .map(|k| k as f64 * sample_rate / n_fft as f64)
        .collect();

    let mut filterbank = vec![0.0f64; num_mel_bins * num_fft_bins];
    for m in 0..num_mel_bins {
        let lower_width = edges[m + 1] - edges[m];
        let upper_width = edges[m + 2] - edges[m + 1];
        for k in 0..num_fft_bins {
            let freq = fft_freqs[k];
            // Rising edge of filter m, and falling edge of the same filter.
            let lower = if lower_width > 0.0 {
                (freq - edges[m]) / lower_width
            } else {
                0.0
            };
            let upper = if upper_width > 0.0 {
                (edges[m + 2] - freq) / upper_width
            } else {
                0.0
            };
            filterbank[m * num_fft_bins + k] = lower.min(upper).max(0.0);
        }
        if normalize == MelNorm::Slaney {
            let span = edges[m + 2] - edges[m];
            if span > 0.0 {
                let enorm = 2.0 / span;
                for k in 0..num_fft_bins {
                    filterbank[m * num_fft_bins + k] *= enorm;
                }
            }
        }
    }
    filterbank
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slaney_norm_scales_filters_by_inverse_span() {
        let plain = mel_filterbank(
            80,
            400,
            16000.0,
            0.0,
            8000.0,
            MelScale::Slaney,
            MelNorm::None,
        );
        let normed = mel_filterbank(
            80,
            400,
            16000.0,
            0.0,
            8000.0,
            MelScale::Slaney,
            MelNorm::Slaney,
        );
        let edges = mel_frequencies_with(80, 0.0, 8000.0, MelScale::Slaney);
        let bins = 400 / 2 + 1;
        assert_eq!(plain.len(), 80 * bins);
        for m in 0..80 {
            let enorm = 2.0 / (edges[m + 2] - edges[m]);
            for k in 0..bins {
                let a = plain[m * bins + k] * enorm;
                let b = normed[m * bins + k];
                assert!((a - b).abs() < 1e-12, "filter {m} bin {k}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn filterbank_weights_are_non_negative_and_bounded() {
        let fb = mel_filterbank(80, 400, 16000.0, 0.0, 8000.0, MelScale::Htk, MelNorm::None);
        for w in &fb {
            assert!((0.0..=1.0).contains(w), "weight out of range: {w}");
        }
    }
}
