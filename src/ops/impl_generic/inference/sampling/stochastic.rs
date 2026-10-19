//! The stochastic branch of the generic `sample_token` and `logits_to_token`.
//!
//! The filter is the canonical device composition in [`super::probs`]. The
//! draw runs on the host, as the rest of the generic sampler does.

use super::probs::sampling_probs_impl;
use crate::error::Result;
use crate::ops::traits::inference::sampling_probs::FilterRule;
use numr::dtype::DType;
use numr::ops::{CompareOps, ScalarOps, TensorOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// The canonical filtered probabilities of one `[vocab]` logits row, on the host.
pub(super) fn filtered_probs<R: Runtime<DType = DType>>(
    client: &R::Client,
    logits: &Tensor<R>,
    rule: &FilterRule,
) -> Result<Vec<f32>>
where
    R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R>,
{
    Ok(sampling_probs_impl::<R>(client, logits, rule)?.to_vec())
}

/// The token whose cumulative probability, in id order, first exceeds `random_val`.
///
/// A token with probability 0 is never drawn. Float rounding at the tail
/// falls back to the last kept token.
pub(super) fn sample_from_probs(probs: &[f32], random_val: f32) -> usize {
    let total: f32 = probs.iter().sum();
    let target = random_val * total;
    let mut cumsum = 0.0f32;
    let mut last_kept = 0;
    for (i, &p) in probs.iter().enumerate() {
        if p <= 0.0 {
            continue;
        }
        cumsum += p;
        last_kept = i;
        if cumsum > target {
            return i;
        }
    }
    last_kept
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn the_draw_skips_removed_tokens() {
        let probs = [0.0f32, 0.5, 0.0, 0.5, 0.0];
        assert_eq!(sample_from_probs(&probs, 0.0), 1);
        assert_eq!(sample_from_probs(&probs, 0.49), 1);
        assert_eq!(sample_from_probs(&probs, 0.5), 3);
        assert_eq!(sample_from_probs(&probs, 0.999_999_9), 3);
        assert_eq!(
            sample_from_probs(&probs, 1.0),
            3,
            "the tail falls back to a kept token"
        );
    }

    /// The sampler draws from the canonical filter: the tie at the top-k
    /// threshold stays drawable.
    #[test]
    fn the_sampler_keeps_the_top_k_tie() {
        let (client, device) = cpu_setup();
        let logits = [0.4f32.ln(), 0.3f32.ln(), 0.3f32.ln()];
        let t = Tensor::<CpuRuntime>::from_slice(&logits, &[3], &device).expect("logits");
        let rule = FilterRule {
            temperature: 1.0,
            top_k: 2,
            top_p: 1.0,
            min_p: 0.0,
        };
        let probs = filtered_probs::<CpuRuntime>(&client, &t, &rule).expect("probs");
        assert!(probs.iter().all(|&p| p > 0.0), "{probs:?}");
        assert_eq!(sample_from_probs(&probs, 0.95), 2);
    }
}
