//! Dynamic temperature from the entropy of host logits.

/// The lowest temperature [`DynaTemp::temperature`] returns, so no logit divides by 0.
pub const MIN_DYNAMIC_TEMPERATURE: f32 = 0.01;

/// The config of entropy-driven temperature.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DynaTemp {
    /// The temperature at half the maximum entropy, with exponent 1.
    pub base: f32,
    /// The temperature moves up to this far from `base` either way.
    pub range: f32,
    /// The normalized entropy rises to this power before it scales.
    pub exponent: f32,
}

impl DynaTemp {
    /// The temperature for `logits`, in `[base - range, base + range]`.
    ///
    /// The normalized entropy is `H(softmax(logits)) / ln(len)`, clamped to
    /// `[0, 1]`, or 0.5 for fewer than 2 logits. The result is
    /// `base - range + 2 * range * normalized^exponent`, floored at
    /// [`MIN_DYNAMIC_TEMPERATURE`]. A confident row gets `base - range`,
    /// a uniform row `base + range`.
    pub fn temperature(&self, logits: &[f32]) -> f32 {
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let probs: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
        let sum: f32 = probs.iter().sum();

        let mut entropy = 0.0f32;
        for &p in &probs {
            let normalized = p / sum;
            if normalized > 0.0 {
                entropy -= normalized * normalized.ln();
            }
        }

        let max_entropy = (logits.len() as f32).ln();
        let normalized_entropy = if max_entropy > 0.0 {
            (entropy / max_entropy).clamp(0.0, 1.0)
        } else {
            0.5
        };
        let mapped = normalized_entropy.powf(self.exponent);
        let dynamic = self.base - self.range + 2.0 * self.range * mapped;
        dynamic.max(MIN_DYNAMIC_TEMPERATURE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: DynaTemp = DynaTemp {
        base: 0.8,
        range: 0.5,
        exponent: 1.0,
    };

    #[test]
    fn a_uniform_row_gets_base_plus_range() {
        let t = DT.temperature(&[0.3f32; 64]);
        assert!((t - 1.3).abs() < 1e-5, "{t}");
    }

    #[test]
    fn a_one_hot_row_gets_base_minus_range() {
        let mut logits = vec![-1e4f32; 32];
        logits[7] = 10.0;
        assert_eq!(DT.temperature(&logits), 0.8 - 0.5);
    }

    #[test]
    fn the_result_never_drops_below_the_floor() {
        let wide = DynaTemp { range: 2.0, ..DT };
        let mut logits = vec![-1e4f32; 32];
        logits[0] = 0.0;
        assert_eq!(wide.temperature(&logits), MIN_DYNAMIC_TEMPERATURE);
    }

    #[test]
    fn a_single_logit_takes_the_midpoint() {
        assert_eq!(DT.temperature(&[4.0]), DT.base);
        assert_eq!(DT.temperature(&[]), DT.base);
    }
}
