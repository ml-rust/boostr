//! The two 16-entry reconstruction codebooks under test, and nothing else.
//!
//! Both are 16 values in `[-1.0, 1.0]`, normalized so the maximum absolute
//! level is exactly `1.0` — the scale means the same thing for either one,
//! so a candidate scale search built against `qmax = 1.0` (see
//! [`super::quantize`]) works unmodified for both.
//!
//! # Why an exact zero level is mandatory
//!
//! A weight that is genuinely zero (pruned, padding, a dead channel) MUST
//! decode back to exactly zero. A codebook missing `0.0` would inject
//! nonzero reconstruction noise into every such weight, corrupting sparsity
//! and biasing the tensor's mean — a cost paid on every zero weight in the
//! model, not just the ones near a decision boundary.

/// Which 16-level reconstruction codebook a group quantizes against.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Codebook {
    /// Evenly spaced symmetric 4-bit grid — the CONTROL. Signed 4-bit codes
    /// `-8..=7` (the full 16-code range, zero at code `0`) divided by 8.
    Uniform,
    /// NormalFloat4 (QLoRA): information-theoretically spaced quantiles of
    /// a standard normal, information-theoretically optimal for
    /// normally-distributed weights rather than evenly spaced.
    Nf4,
}

/// `-8..=7` divided by 8: the full 16-code signed range, no separate
/// zero-point. Step `0.125`, zero at index 8, max abs `1.0` at index 0.
pub const UNIFORM_LEVELS: [f32; 16] = [
    -1.0, -0.875, -0.75, -0.625, -0.5, -0.375, -0.25, -0.125, 0.0, 0.125, 0.25, 0.375, 0.5, 0.625,
    0.75, 0.875,
];

/// The standard NF4 codebook (Dettmers et al., QLoRA / bitsandbytes):
/// quantiles of a standard normal split into two halves (one per sign, plus
/// an exact zero), so density concentrates near zero where weight mass
/// concentrates. Already normalized to max abs `1.0` at the source.
pub const NF4_LEVELS: [f32; 16] = [
    -1.0,
    -0.696_192_8,
    -0.525_073_05,
    -0.394_917_5,
    -0.284_441_38,
    -0.184_773_43,
    -0.091_050_036,
    0.0,
    0.079_580_3,
    0.160_930_2,
    0.246_112_3,
    0.337_915_24,
    0.440_709_83,
    0.562_617,
    0.722_956_84,
    1.0,
];

impl Codebook {
    /// The 16 reconstruction levels for this codebook, ascending, in
    /// normalized units — the value a group reconstructs to is `d * level`.
    pub fn levels(self) -> &'static [f32; 16] {
        match self {
            Codebook::Uniform => &UNIFORM_LEVELS,
            Codebook::Nf4 => &NF4_LEVELS,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_16_with_zero_and_unit_max(levels: &[f32; 16]) {
        assert_eq!(levels.len(), 16);
        assert!(levels.contains(&0.0), "no exact zero level");
        let max_abs = levels.iter().fold(0.0f32, |acc, &l| acc.max(l.abs()));
        assert_eq!(max_abs, 1.0, "max abs level was {max_abs}, expected 1.0");
    }

    #[test]
    fn uniform_has_16_entries_zero_and_unit_max() {
        assert_16_with_zero_and_unit_max(&UNIFORM_LEVELS);
    }

    #[test]
    fn nf4_has_16_entries_zero_and_unit_max() {
        assert_16_with_zero_and_unit_max(&NF4_LEVELS);
    }

    #[test]
    fn uniform_levels_are_evenly_spaced() {
        let gaps: Vec<f32> = UNIFORM_LEVELS.windows(2).map(|w| w[1] - w[0]).collect();
        let first = gaps[0];
        for &gap in &gaps {
            assert!((gap - first).abs() < 1e-6, "gap {gap} != {first}: {gaps:?}");
        }
    }

    #[test]
    fn nf4_levels_are_not_evenly_spaced() {
        let gaps: Vec<f32> = NF4_LEVELS.windows(2).map(|w| w[1] - w[0]).collect();
        let first = gaps[0];
        assert!(
            gaps.iter().any(|&gap| (gap - first).abs() > 1e-4),
            "NF4 gaps were all equal: {gaps:?}"
        );
    }

    #[test]
    fn levels_accessor_matches_constants() {
        assert_eq!(Codebook::Uniform.levels(), &UNIFORM_LEVELS);
        assert_eq!(Codebook::Nf4.levels(), &NF4_LEVELS);
    }
}
