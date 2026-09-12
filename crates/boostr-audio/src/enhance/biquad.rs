//! Biquad IIR filters, RBJ cookbook coefficients.
//!
//! Direct Form II transposed: one state pair per filter, numerically better
//! behaved than Direct Form I at f32 and cheaper than DF1 in state.
//!
//! These exist for two callers. Loudness measurement needs the two K-weighting
//! stages of ITU-R BS.1770, and reference-audio cleanup needs a rumble
//! high-pass and a low-shelf for body.

/// A second-order IIR section with its own state.
///
/// Coefficients are stored already normalized by `a0`, so `process` is four
/// multiplies and four adds.
#[derive(Debug, Clone, Copy)]
pub struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    z1: f64,
    z2: f64,
}

impl Biquad {
    /// Build from unnormalized coefficients, dividing through by `a0`.
    fn new(b0: f64, b1: f64, b2: f64, a0: f64, a1: f64, a2: f64) -> Self {
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    /// Coefficients passed through verbatim, already normalized.
    ///
    /// BS.1770 specifies the K-weighting filters as literal coefficient tables
    /// rather than as parametric shapes, so those cannot go through the
    /// cookbook constructors without reproducing rounding differences.
    pub fn from_normalized(b0: f64, b1: f64, b2: f64, a1: f64, a2: f64) -> Self {
        Self {
            b0,
            b1,
            b2,
            a1,
            a2,
            z1: 0.0,
            z2: 0.0,
        }
    }

    /// 12 dB/octave high-pass at `freq` Hz.
    ///
    /// `q` of `1/sqrt(2)` is Butterworth — maximally flat, no passband ripple.
    pub fn highpass(rate: f64, freq: f64, q: f64) -> Self {
        let w0 = std::f64::consts::TAU * freq / rate;
        let (sin, cos) = w0.sin_cos();
        let alpha = sin / (2.0 * q);
        Self::new(
            (1.0 + cos) / 2.0,
            -(1.0 + cos),
            (1.0 + cos) / 2.0,
            1.0 + alpha,
            -2.0 * cos,
            1.0 - alpha,
        )
    }

    /// Low shelf: `gain_db` applied below `freq`, unity above.
    pub fn low_shelf(rate: f64, freq: f64, q: f64, gain_db: f64) -> Self {
        let a = 10f64.powf(gain_db / 40.0);
        let w0 = std::f64::consts::TAU * freq / rate;
        let (sin, cos) = w0.sin_cos();
        let alpha = sin / (2.0 * q);
        let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;
        Self::new(
            a * ((a + 1.0) - (a - 1.0) * cos + two_sqrt_a_alpha),
            2.0 * a * ((a - 1.0) - (a + 1.0) * cos),
            a * ((a + 1.0) - (a - 1.0) * cos - two_sqrt_a_alpha),
            (a + 1.0) + (a - 1.0) * cos + two_sqrt_a_alpha,
            -2.0 * ((a - 1.0) + (a + 1.0) * cos),
            (a + 1.0) + (a - 1.0) * cos - two_sqrt_a_alpha,
        )
    }

    /// High shelf: `gain_db` applied above `freq`, unity below.
    pub fn high_shelf(rate: f64, freq: f64, q: f64, gain_db: f64) -> Self {
        let a = 10f64.powf(gain_db / 40.0);
        let w0 = std::f64::consts::TAU * freq / rate;
        let (sin, cos) = w0.sin_cos();
        let alpha = sin / (2.0 * q);
        let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;
        Self::new(
            a * ((a + 1.0) + (a - 1.0) * cos + two_sqrt_a_alpha),
            -2.0 * a * ((a - 1.0) + (a + 1.0) * cos),
            a * ((a + 1.0) + (a - 1.0) * cos - two_sqrt_a_alpha),
            (a + 1.0) - (a - 1.0) * cos + two_sqrt_a_alpha,
            2.0 * ((a - 1.0) - (a + 1.0) * cos),
            (a + 1.0) - (a - 1.0) * cos - two_sqrt_a_alpha,
        )
    }

    /// One sample through the filter, updating state.
    #[inline]
    pub fn process(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }

    /// Filter a buffer in place. State carries across the buffer.
    pub fn process_buffer(&mut self, samples: &mut [f32]) {
        for s in samples.iter_mut() {
            *s = self.process(*s as f64) as f32;
        }
    }

    /// Clear the filter state without touching coefficients.
    ///
    /// Needed between independent signals: leftover state from a previous
    /// buffer is a transient at the start of the next one.
    pub fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }

    /// Magnitude response at `freq`, as a linear gain.
    ///
    /// Evaluates `|H(e^{jw})|` directly from the coefficients, so a test can
    /// check a filter shape without running a signal through it.
    pub fn magnitude_at(&self, rate: f64, freq: f64) -> f64 {
        let w = std::f64::consts::TAU * freq / rate;
        let (s1, c1) = w.sin_cos();
        let (s2, c2) = (2.0 * w).sin_cos();
        let num_re = self.b0 + self.b1 * c1 + self.b2 * c2;
        let num_im = -(self.b1 * s1 + self.b2 * s2);
        let den_re = 1.0 + self.a1 * c1 + self.a2 * c2;
        let den_im = -(self.a1 * s1 + self.a2 * s2);
        ((num_re * num_re + num_im * num_im) / (den_re * den_re + den_im * den_im)).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: f64 = 48_000.0;

    #[test]
    fn a_highpass_cuts_below_its_corner_and_passes_above() {
        let hp = Biquad::highpass(RATE, 80.0, std::f64::consts::FRAC_1_SQRT_2);
        // Butterworth is -3 dB at the corner: 10^(-3/20) = 0.708.
        let at_corner = hp.magnitude_at(RATE, 80.0);
        assert!(
            (at_corner - 0.7079).abs() < 0.01,
            "corner gain {at_corner:.4} should be -3 dB"
        );
        assert!(hp.magnitude_at(RATE, 20.0) < 0.1, "20 Hz must be well down");
        assert!(
            hp.magnitude_at(RATE, 1000.0) > 0.99,
            "passband must be flat"
        );
    }

    #[test]
    fn shelves_apply_their_gain_on_the_correct_side() {
        let low = Biquad::low_shelf(RATE, 200.0, 0.707, 6.0);
        // +6 dB is a linear gain of 2.
        assert!(
            (low.magnitude_at(RATE, 20.0) - 2.0).abs() < 0.05,
            "low shelf lifts lows"
        );
        assert!(
            (low.magnitude_at(RATE, 8000.0) - 1.0).abs() < 0.05,
            "and leaves highs"
        );

        let high = Biquad::high_shelf(RATE, 2000.0, 0.707, 6.0);
        assert!(
            (high.magnitude_at(RATE, 16000.0) - 2.0).abs() < 0.05,
            "high shelf lifts highs"
        );
        assert!(
            (high.magnitude_at(RATE, 100.0) - 1.0).abs() < 0.05,
            "and leaves lows"
        );
    }

    #[test]
    fn a_shelf_at_zero_gain_is_a_pass_through() {
        let flat = Biquad::low_shelf(RATE, 200.0, 0.707, 0.0);
        for f in [20.0, 100.0, 1000.0, 10000.0] {
            let g = flat.magnitude_at(RATE, f);
            assert!((g - 1.0).abs() < 1e-9, "{f} Hz gain {g}");
        }
    }

    #[test]
    fn filtering_a_tone_matches_the_computed_magnitude() {
        // The response function and the actual filter must agree, or tests that
        // use `magnitude_at` prove nothing about what `process` does.
        let freq = 1000.0;
        let mut hp = Biquad::highpass(RATE, 300.0, 0.707);
        let n = 48_000;
        let mut buf: Vec<f32> = (0..n)
            .map(|i| (std::f64::consts::TAU * freq * i as f64 / RATE).sin() as f32)
            .collect();
        hp.process_buffer(&mut buf);

        // Skip the transient; measure amplitude over the steady-state tail.
        let tail = &buf[n / 2..];
        let peak = tail.iter().fold(0.0f32, |a, &b| a.max(b.abs())) as f64;
        let expected = hp.magnitude_at(RATE, freq);
        assert!(
            (peak - expected).abs() < 0.02,
            "measured {peak:.4} vs computed {expected:.4}"
        );
    }

    #[test]
    fn reset_clears_the_tail_of_a_previous_signal() {
        let mut hp = Biquad::highpass(RATE, 100.0, 0.707);
        let mut loud = vec![1.0f32; 1000];
        hp.process_buffer(&mut loud);

        // Without reset the decaying state leaks into the next buffer as a click.
        let mut dirty = hp;
        let mut quiet_dirty = vec![0.0f32; 100];
        dirty.process_buffer(&mut quiet_dirty);
        let leak = quiet_dirty.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        assert!(leak > 1e-6, "state should leak without reset, got {leak}");

        hp.reset();
        let mut quiet_clean = vec![0.0f32; 100];
        hp.process_buffer(&mut quiet_clean);
        assert!(
            quiet_clean.iter().all(|&s| s == 0.0),
            "reset must silence it"
        );
    }
}
