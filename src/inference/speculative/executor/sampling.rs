//! Softmax, top-p / categorical sampling, and the executor's seeded RNG.

use crate::error::Result;
use numr::runtime::Runtime;

use super::super::types::{SpeculativeModel, TokenId};
use super::generate::SpeculativeExecutor;

impl<R: Runtime, D, T> SpeculativeExecutor<R, D, T>
where
    D: SpeculativeModel<R>,
    T: SpeculativeModel<R>,
{
    pub(crate) fn apply_temperature_and_softmax(
        &self,
        logits: &[f32],
        temperature: f32,
    ) -> Vec<f32> {
        let temp = temperature.max(1e-8);
        let scaled: Vec<f32> = logits.iter().map(|&x| x / temp).collect();
        let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = scaled.iter().map(|&x| (x - max_val).exp()).sum();
        let log_sum = max_val + exp_sum.ln();
        scaled.iter().map(|&x| x - log_sum).collect()
    }

    pub(super) fn sample_from_logprobs(
        &mut self,
        log_probs: &[f32],
        top_p: f32,
    ) -> Result<TokenId> {
        let probs: Vec<f32> = log_probs.iter().map(|&lp| lp.exp()).collect();

        let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut cumsum = 0.0;
        let mut cutoff_idx = indexed.len();

        for (i, (_, p)) in indexed.iter().enumerate() {
            cumsum += p;
            if cumsum >= top_p {
                cutoff_idx = i + 1;
                break;
            }
        }

        let top_p_tokens: Vec<(usize, f32)> = indexed[..cutoff_idx].to_vec();
        let sum: f32 = top_p_tokens.iter().map(|(_, p)| p).sum();

        let r = self.random_uniform() * sum;
        let mut cumsum = 0.0;

        for (token_id, p) in top_p_tokens {
            cumsum += p;
            if r < cumsum {
                return Ok(token_id as TokenId);
            }
        }

        Ok(indexed[0].0 as TokenId)
    }

    pub(super) fn sample_from_probs(&mut self, probs: &[f32]) -> Result<TokenId> {
        let sum: f32 = probs.iter().sum();
        if sum < 1e-8 {
            return Ok(0);
        }

        let r = self.random_uniform() * sum;
        let mut cumsum = 0.0;

        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return Ok(i as TokenId);
            }
        }

        Ok(probs.len() as TokenId - 1)
    }

    pub(super) fn random_uniform(&mut self) -> f32 {
        const A: u64 = 48271;
        const M: u64 = 2147483647;
        self.rng_state = (A.wrapping_mul(self.rng_state)) % M;
        self.rng_state as f32 / M as f32
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::types::SpeculativeConfig;
    use super::super::generate::tests::MockModel;
    use super::*;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_apply_temperature_and_softmax() {
        let draft = MockModel::new(4, "draft");
        let target = MockModel::new(4, "target");
        let config = SpeculativeConfig::default();
        let executor = SpeculativeExecutor::<CpuRuntime, _, _>::new(draft, target, config);

        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let log_probs = executor.apply_temperature_and_softmax(&logits, 1.0);

        let sum: f32 = log_probs.iter().map(|&lp| lp.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-5);

        assert!(log_probs[3] > log_probs[2]);
        assert!(log_probs[2] > log_probs[1]);
        assert!(log_probs[1] > log_probs[0]);
    }
}
