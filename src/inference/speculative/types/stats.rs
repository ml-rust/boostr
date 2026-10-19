//! Speculative generation statistics and finish reason.

/// Why generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FinishReason {
    /// The last emitted token is an EOS id.
    Eos,
    /// The output reached `max_tokens`.
    Length,
}

/// Counters of one generation. Prefill is not counted.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpeculativeStats {
    /// Steps run.
    pub steps: usize,
    /// Drafts fed to the target, after the clamp and the vocabulary cut.
    pub drafted: usize,
    /// Drafts the target accepted, before the EOS cut.
    pub accepted: usize,
    /// Tokens emitted to the caller.
    pub emitted: usize,
    /// Verify forwards during steps: calls of
    /// [`SpeculativeModel::extend_greedy`](crate::inference::speculative::SpeculativeModel::extend_greedy)
    /// or [`SpeculativeModel::extend_sampled`](crate::inference::speculative::SpeculativeModel::extend_sampled).
    /// Each step makes exactly one.
    pub target_forwards: usize,
}

impl SpeculativeStats {
    /// `accepted / drafted`, or 0.0 when nothing was drafted.
    pub fn acceptance_rate(&self) -> f64 {
        if self.drafted == 0 {
            0.0
        } else {
            self.accepted as f64 / self.drafted as f64
        }
    }

    /// `emitted / target_forwards`, or 0.0 before the first step.
    pub fn tokens_per_forward(&self) -> f64 {
        if self.target_forwards == 0 {
            0.0
        } else {
            self.emitted as f64 / self.target_forwards as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rates_handle_zero_counts() {
        let stats = SpeculativeStats::default();
        assert_eq!(stats.acceptance_rate(), 0.0);
        assert_eq!(stats.tokens_per_forward(), 0.0);
        let stats = SpeculativeStats {
            steps: 2,
            drafted: 4,
            accepted: 3,
            emitted: 5,
            target_forwards: 2,
        };
        assert_eq!(stats.acceptance_rate(), 0.75);
        assert_eq!(stats.tokens_per_forward(), 2.5);
    }
}
