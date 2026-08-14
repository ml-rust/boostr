//! Kaldi mel scale and the triangular mel filterbank.

use super::constants::{FFT_LENGTH, HIGH_FREQ, LOW_FREQ, NUM_FFT_BINS, NUM_MEL_BINS, SAMPLE_RATE};

/// Kaldi mel scale: `mel(f) = 1127 * ln(1 + f / 700)`.
///
/// NOT HTK (`2595 * log10(1 + f/700)`) and NOT Slaney (linear below 1 kHz).
/// HTK is a constant multiple of this one, so substituting it rescales the mel
/// axis and moves every triangle edge.
#[must_use]
pub fn hz_to_mel(hz: f64) -> f64 {
    1127.0 * (1.0 + hz / 700.0).ln()
}

/// Inverse of [`hz_to_mel`].
#[must_use]
pub fn mel_to_hz(mel: f64) -> f64 {
    700.0 * ((mel / 1127.0).exp() - 1.0)
}

/// Triangular mel filterbank mapping 257 FFT bins to 80 mel channels.
///
/// Returns 80 rows of `NUM_FFT_BINS` weights.
///
/// Triangulation happens **in mel space**: each FFT bin's centre frequency is
/// converted to mel and the triangle evaluated against mel-domain edges. The
/// common alternative — converting the mel edges back to Hz and triangulating
/// in Hz — gives different (asymmetric-in-mel) filters, worst in the low
/// channels where the mel curve bends most.
///
/// No Slaney/area normalization (`norm = None`); adding it would scale each
/// channel by `2 / (right_hz - left_hz)` and change every energy.
#[must_use]
pub fn mel_filterbank() -> Vec<Vec<f64>> {
    let mel_low = hz_to_mel(LOW_FREQ);
    let mel_high = hz_to_mel(HIGH_FREQ);

    // 82 edges = 80 filters + 2; filter `m` spans edges (m, m+1, m+2).
    let num_edges = NUM_MEL_BINS + 2;
    let step = (mel_high - mel_low) / (num_edges - 1) as f64;
    let edges: Vec<f64> = (0..num_edges).map(|i| mel_low + step * i as f64).collect();

    // FFT bin centre frequencies, expressed in mel. The reference spells the
    // bin width `sampling_rate / ((num_frequency_bins - 1) * 2)` with
    // `num_frequency_bins = 257` — the same 31.25 Hz, not a `linspace` over a
    // different bin count.
    let bin_hz = SAMPLE_RATE as f64 / FFT_LENGTH as f64;
    let bin_mel: Vec<f64> = (0..NUM_FFT_BINS)
        .map(|k| hz_to_mel(k as f64 * bin_hz))
        .collect();

    (0..NUM_MEL_BINS)
        .map(|m| {
            let left = edges.get(m).copied().unwrap_or(0.0);
            let center = edges.get(m + 1).copied().unwrap_or(0.0);
            let right = edges.get(m + 2).copied().unwrap_or(0.0);
            bin_mel
                .iter()
                .map(|&x| {
                    let rising = (x - left) / (center - left);
                    let falling = (right - x) / (right - center);
                    rising.min(falling).max(0.0)
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mel_filterbank_rows_are_contiguous_and_non_negative() {
        let fb = mel_filterbank();
        assert_eq!(fb.len(), NUM_MEL_BINS);
        for (m, row) in fb.iter().enumerate() {
            assert_eq!(row.len(), NUM_FFT_BINS);
            assert!(row.iter().all(|&w| w >= 0.0), "row {m} has negative weight");

            let support: Vec<usize> = row
                .iter()
                .enumerate()
                .filter(|&(_, &w)| w > 0.0)
                .map(|(k, _)| k)
                .collect();
            assert!(!support.is_empty(), "row {m} is empty");
            let first = support[0];
            let last = support[support.len() - 1];
            assert_eq!(
                support.len(),
                last - first + 1,
                "row {m} support is not contiguous"
            );
        }
        // Bin 0 is DC (0 Hz), below the 20 Hz low edge — no filter touches it.
        assert!(fb.iter().all(|row| row[0] == 0.0));
    }
}
