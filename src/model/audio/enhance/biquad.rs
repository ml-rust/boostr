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
mod tests;
