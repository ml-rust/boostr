//! Radix-2 Cooley-Tukey FFT for audio preprocessing.
//!
//! Pure CPU, in-place, iterative. Size must be a power of 2 — the mel
//! spectrogram path pads its 400-sample windows up to 512.
//!
//! This exists because Whisper's mel pipeline currently operates on raw `&[f32]`
//! slices and numr's `FftAlgorithms` requires a `Tensor<R>` round-trip per
//! frame, which would dominate preprocessing cost. Audio preprocessing runs
//! once per utterance; a small CPU kernel is the right tool.

use std::f32::consts::PI;

/// In-place radix-2 Cooley-Tukey FFT on interleaved complex data.
///
/// `data.len()` must equal `2 * n` where `n` is a power of 2. Layout is
/// `[re_0, im_0, re_1, im_1, ...]`.
///
/// Panics if `n` is not a power of 2.
pub fn fft_inplace_radix2(data: &mut [f32]) {
    debug_assert!(
        data.len().is_multiple_of(2),
        "fft data must be interleaved complex"
    );
    let n = data.len() / 2;
    assert!(
        n.is_power_of_two(),
        "fft size must be a power of 2, got {n}"
    );
    if n < 2 {
        return;
    }

    // Bit-reversal permutation.
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = reverse_bits(i, bits);
        if j > i {
            data.swap(2 * i, 2 * j);
            data.swap(2 * i + 1, 2 * j + 1);
        }
    }

    // Iterative butterflies.
    let mut size = 2usize;
    while size <= n {
        let half = size / 2;
        let theta = -2.0 * PI / size as f32;
        let w_step_re = theta.cos();
        let w_step_im = theta.sin();

        let mut i = 0;
        while i < n {
            // Twiddle factor — rolled incrementally to avoid trig per butterfly.
            let mut w_re = 1.0f32;
            let mut w_im = 0.0f32;
            for k in 0..half {
                let a = 2 * (i + k);
                let b = 2 * (i + k + half);
                let t_re = w_re * data[b] - w_im * data[b + 1];
                let t_im = w_re * data[b + 1] + w_im * data[b];
                data[b] = data[a] - t_re;
                data[b + 1] = data[a + 1] - t_im;
                data[a] += t_re;
                data[a + 1] += t_im;
                // w *= w_step (complex multiply)
                let nw_re = w_re * w_step_re - w_im * w_step_im;
                let nw_im = w_re * w_step_im + w_im * w_step_re;
                w_re = nw_re;
                w_im = nw_im;
                let _ = k;
            }
            i += size;
        }
        size *= 2;
    }
}

/// Compute the power spectrum `|X[k]|^2` for `k = 0..=n/2` of a real-valued
/// windowed frame using the radix-2 FFT.
///
/// `windowed` must contain `fft_size` real samples already multiplied by the
/// analysis window (caller's responsibility). `fft_size` must be a power of 2.
/// The returned vector has `fft_size/2 + 1` entries.
pub fn power_spectrum_rfft(windowed: &[f32], fft_size: usize) -> Vec<f32> {
    assert_eq!(windowed.len(), fft_size, "input length must equal fft_size");
    let mut buf = vec![0.0f32; 2 * fft_size];
    for (i, &v) in windowed.iter().enumerate() {
        buf[2 * i] = v;
        // imag already 0
    }
    fft_inplace_radix2(&mut buf);
    let out_bins = fft_size / 2 + 1;
    let mut out = Vec::with_capacity(out_bins);
    for k in 0..out_bins {
        let re = buf[2 * k];
        let im = buf[2 * k + 1];
        out.push(re * re + im * im);
    }
    out
}

fn reverse_bits(x: usize, bits: u32) -> usize {
    let mut x = x;
    let mut r = 0usize;
    for _ in 0..bits {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn naive_dft_power(x: &[f32]) -> Vec<f32> {
        let n = x.len();
        let bins = n / 2 + 1;
        let mut out = Vec::with_capacity(bins);
        for k in 0..bins {
            let mut re = 0.0f32;
            let mut im = 0.0f32;
            for (i, &v) in x.iter().enumerate() {
                let angle = -2.0 * PI * k as f32 * i as f32 / n as f32;
                re += v * angle.cos();
                im += v * angle.sin();
            }
            out.push(re * re + im * im);
        }
        out
    }

    #[test]
    fn dc_signal() {
        let x = vec![1.0f32; 8];
        let p = power_spectrum_rfft(&x, 8);
        // All energy in bin 0: |sum|^2 = 64
        assert!((p[0] - 64.0).abs() < 1e-3);
        for v in &p[1..] {
            assert!(v.abs() < 1e-3);
        }
    }

    #[test]
    fn matches_naive_dft_small() {
        // Arbitrary waveform, size 16 (power of 2).
        let x: Vec<f32> = (0..16)
            .map(|i| (0.3 * i as f32).sin() + 0.5 * (0.7 * i as f32).cos())
            .collect();
        let fft = power_spectrum_rfft(&x, 16);
        let dft = naive_dft_power(&x);
        assert_eq!(fft.len(), dft.len());
        for (a, b) in fft.iter().zip(dft.iter()) {
            assert!((a - b).abs() < 1e-3, "fft {a} != dft {b}");
        }
    }

    #[test]
    fn matches_naive_dft_512() {
        let x: Vec<f32> = (0..512)
            .map(|i| (0.01 * i as f32).sin() + 0.3 * ((0.05 * i as f32).cos()))
            .collect();
        let fft = power_spectrum_rfft(&x, 512);
        let dft = naive_dft_power(&x);
        assert_eq!(fft.len(), 257);
        // Relative tolerance per-bin since absolute magnitudes vary.
        for (a, b) in fft.iter().zip(dft.iter()) {
            let denom = b.abs().max(1.0);
            assert!((a - b).abs() / denom < 1e-3, "fft {a} vs dft {b}");
        }
    }

    #[test]
    #[should_panic(expected = "must be a power of 2")]
    fn rejects_non_power_of_two() {
        let mut data = vec![0.0f32; 2 * 6]; // n=6 not power of 2
        fft_inplace_radix2(&mut data);
    }
}
