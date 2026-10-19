//! The contracts of a drafter that samples: its proposal and its draft model.

use crate::error::Result;
use crate::inference::speculative::model::SpeculativeModel;
use crate::inference::speculative::types::{Draw, SampleRule};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// One proposal of a [`ProbDrafter`].
#[derive(Debug, Clone)]
pub struct ProbDraft<R: Runtime> {
    /// The draft ids, at most `k`.
    pub tokens: Vec<u32>,
    /// `[1, tokens.len(), V]` F32. Row `i` is the distribution `tokens[i]`
    /// was drawn from. `None` for an argmax draft.
    pub probs: Option<Tensor<R>>,
}

impl<R: Runtime> ProbDraft<R> {
    /// A proposal with no draft.
    pub fn empty() -> Self {
        Self {
            tokens: Vec::new(),
            probs: None,
        }
    }
}

/// Proposes the next tokens of a sequence, with the distributions they came from.
///
/// The generator verifies every draft against the target by rejection
/// sampling. Its output follows the target distribution exactly when each
/// draft is an exact sample of the row of `probs` that names it.
pub trait ProbDrafter<R: Runtime> {
    /// Up to `k` tokens that are likely to follow `context`.
    ///
    /// `context` is the prompt plus every token emitted so far. `rule` is the
    /// generator's verify rule, or `None` for plain greedy decoding.
    ///
    /// - A rule with a temperature above 0: draw `tokens[i]` from row `i` of
    ///   `probs`. Build each row by the canonical filter of `rule.filter`
    ///   over logits with the penalties of `rule`, as the target does. Draw
    ///   from a Philox stream independent of the verify uniforms of `draw`.
    /// - `None` or temperature 0: take the argmax of each row, penalties
    ///   included, and return `probs` as `None`.
    ///
    /// # Errors
    ///
    /// An error when the draft forward or a device op fails.
    fn propose(
        &mut self,
        context: &[u32],
        k: usize,
        rule: Option<&SampleRule>,
        draw: Draw,
    ) -> Result<ProbDraft<R>>;
}

/// A causal draft model whose forward returns its logits.
///
/// The target contract [`SpeculativeModel`] stays unchanged. A draft model
/// adds only this forward.
pub trait DraftModel<R: Runtime>: SpeculativeModel {
    /// Feed `tokens` at positions `len()..len() + tokens.len()` in one forward.
    ///
    /// Returns `[1, tokens.len(), V]` logits in any float dtype. Row `j`
    /// scores the prefix that ends with `tokens[j]`. The logits carry the
    /// same logit bias the target applies. The sampled path needs the
    /// target's vocabulary size `V`.
    ///
    /// # Errors
    ///
    /// An error when the forward fails, for example past the cache capacity.
    fn forward_logits(&mut self, tokens: &[u32]) -> Result<Tensor<R>>;
}
