//! Draft speculation, target verification, and rejection sampling.

use crate::error::Result;
use numr::runtime::Runtime;

use super::super::types::{
    DraftOutput, SpeculativeModel, TargetOutput, TokenId, VerificationResult,
};
use super::generate::SpeculativeExecutor;

impl<R: Runtime, D, T> SpeculativeExecutor<R, D, T>
where
    D: SpeculativeModel<R>,
    T: SpeculativeModel<R>,
{
    pub(super) fn draft_speculate(&mut self, context: &[TokenId]) -> Result<DraftOutput> {
        let vocab_size = self.draft_model.vocab_size();
        let mut tokens = Vec::with_capacity(self.current_depth);
        let mut log_probs = Vec::with_capacity(self.current_depth * vocab_size);

        let mut current_context = context.to_vec();

        for i in 0..self.current_depth {
            self.stats.draft_forward_passes += 1;

            let logits = self
                .draft_model
                .forward(&current_context, context.len() + i)?;

            let log_prob =
                self.apply_temperature_and_softmax(&logits, self.config.draft_temperature);

            let token = self.sample_from_logprobs(&log_prob, self.config.draft_top_p)?;

            tokens.push(token);
            log_probs.extend(log_prob);
            current_context.push(token);
        }

        Ok(DraftOutput {
            tokens,
            log_probs,
            vocab_size,
        })
    }

    pub(super) fn target_verify(
        &mut self,
        context: &[TokenId],
        draft_tokens: &[TokenId],
    ) -> Result<TargetOutput> {
        self.stats.target_forward_passes += 1;

        let vocab_size = self.target_model.vocab_size();
        let mut all_log_probs = Vec::with_capacity((draft_tokens.len() + 1) * vocab_size);

        let mut current_context = context.to_vec();

        for (i, &draft_token) in draft_tokens.iter().enumerate() {
            let logits = self
                .target_model
                .forward(&current_context, context.len() + i)?;
            let log_prob =
                self.apply_temperature_and_softmax(&logits, self.config.target_temperature);
            all_log_probs.extend(log_prob);
            current_context.push(draft_token);
        }

        let logits = self
            .target_model
            .forward(&current_context, context.len() + draft_tokens.len())?;
        let log_prob = self.apply_temperature_and_softmax(&logits, self.config.target_temperature);
        all_log_probs.extend(log_prob);

        Ok(TargetOutput {
            log_probs: all_log_probs,
            vocab_size,
        })
    }

    pub(super) fn verify_and_sample(
        &mut self,
        draft: &DraftOutput,
        target: &TargetOutput,
    ) -> Result<VerificationResult> {
        let vocab_size = draft.vocab_size;
        let mut accepted_tokens = Vec::new();
        let mut first_rejection_pos = -1i32;

        for (i, &draft_token) in draft.tokens.iter().enumerate() {
            let draft_prob = draft.log_probs[i * vocab_size + draft_token as usize].exp();
            let target_prob = target.log_probs[i * vocab_size + draft_token as usize].exp();

            let accept_prob = if draft_prob > 0.0 {
                (target_prob / draft_prob).min(1.0)
            } else {
                1.0
            };

            let r = self.random_uniform();

            if r < accept_prob {
                accepted_tokens.push(draft_token);
            } else {
                first_rejection_pos = i as i32;
                break;
            }
        }

        let bonus_token = if first_rejection_pos >= 0 {
            let i = first_rejection_pos as usize;
            let mut residual_probs = Vec::with_capacity(vocab_size);

            for t in 0..vocab_size {
                let draft_prob = draft.log_probs[i * vocab_size + t].exp();
                let target_prob = target.log_probs[i * vocab_size + t].exp();
                residual_probs.push((target_prob - draft_prob).max(0.0));
            }

            let sum: f32 = residual_probs.iter().sum();
            if sum > 1e-8 {
                for p in &mut residual_probs {
                    *p /= sum;
                }
                Some(self.sample_from_probs(&residual_probs)?)
            } else {
                let target_probs: Vec<f32> = (0..vocab_size)
                    .map(|t| target.log_probs[i * vocab_size + t].exp())
                    .collect();
                Some(self.sample_from_probs(&target_probs)?)
            }
        } else {
            let last_pos = draft.tokens.len();
            let target_probs: Vec<f32> = (0..vocab_size)
                .map(|t| target.log_probs[last_pos * vocab_size + t].exp())
                .collect();
            Some(self.sample_from_probs(&target_probs)?)
        };

        Ok(VerificationResult {
            num_accepted: accepted_tokens.len(),
            accepted_tokens,
            bonus_token,
            first_rejection_pos,
        })
    }
}
