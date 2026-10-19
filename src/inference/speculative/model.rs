//! The target model contract of speculative generation.

use super::types::{Draw, SampleRule, Verdict};
use crate::error::{Error, Result};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// A causal target model with a rollback-capable cache.
///
/// The model holds a cache of `len()` positions. Position `i` holds the
/// `i`-th fed token. Every call of [`extend_greedy`](Self::extend_greedy) or
/// [`extend_sampled`](Self::extend_sampled) appends at the end, and
/// [`truncate`](Self::truncate) cuts the end off.
///
/// # Greedy rule
///
/// The id after a position is the argmax of that position's logits. Ties
/// break to the lowest id. numr's CPU argmax and the sampler's greedy path
/// both scan ids upward and keep the first strict maximum. An implementation
/// must pick exactly the id plain greedy decoding picks for the same prefix.
/// Speculative output equals plain greedy output only under that condition.
///
/// The logits of one position must not depend on how many tokens one call
/// feeds. A causal mask aligned to the cache end gives that. Floating-point
/// reduction order can still move a near-tie between a one-token and a
/// multi-token call. The output is exact only where no such flip occurs.
pub trait SpeculativeModel {
    /// Token ids the model scores, `0..vocab_size()`.
    fn vocab_size(&self) -> usize;

    /// Positions in the cache now.
    fn len(&self) -> usize;

    /// The cache holds no position.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Cut the cache back to its first `len` positions.
    ///
    /// The next [`extend_greedy`](Self::extend_greedy) writes at `len`.
    ///
    /// # Errors
    ///
    /// An error when `len` is past [`len`](Self::len).
    fn truncate(&mut self, len: usize) -> Result<()>;

    /// Feed `tokens` at positions `len()..len() + tokens.len()` in one forward.
    ///
    /// The cache grows by `tokens.len()` positions. The result holds one id
    /// per fed token. Id `j` is the greedy next token after the prefix that
    /// ends with `tokens[j]`. The argmax runs where the logits live, so only
    /// `tokens.len()` ids cross to the host, never a row of logits.
    ///
    /// # Errors
    ///
    /// An error when the forward fails, for example past the cache capacity.
    fn extend_greedy(&mut self, tokens: &[u32]) -> Result<Vec<u32>>;

    /// Feed `tokens` in one forward and verify the drafts under `rule`.
    ///
    /// The generator calls it for a temperature above 0, or for temperature 0
    /// with a penalty. `tokens` is the last committed token, then the drafts.
    /// The cache grows by `tokens.len()` positions. The generator truncates it
    /// afterwards.
    ///
    /// Row `i` scores the prefix that ends with `tokens[i]`. Its penalties
    /// count the penalty window of that prefix, the cached tokens included.
    /// An implementation passes its logits to
    /// [`verify_logits`](crate::inference::speculative::verify_logits).
    /// Only the verdict or the argmax ids cross to the host.
    ///
    /// The default refuses the call, for a greedy-only model.
    ///
    /// # Errors
    ///
    /// An error when the forward or the verify fails. The default always
    /// returns [`Error::InferenceError`].
    fn extend_sampled(&mut self, tokens: &[u32], rule: &SampleRule, draw: Draw) -> Result<Verdict> {
        let _ = (tokens, rule, draw);
        Err(Error::InferenceError {
            reason: "this model implements only greedy verification; implement \
                     SpeculativeModel::extend_sampled, or set temperature 0 with no penalty"
                .into(),
        })
    }
}

/// A target model that verifies drafts against their draft distributions.
///
/// A draft model drafts by sampling, so each draft distribution `q` is not
/// one-hot. Rejection sampling then needs `q` for an unbiased output.
pub trait SpeculativeProbModel<R: Runtime>: SpeculativeModel {
    /// [`extend_sampled`](SpeculativeModel::extend_sampled) with the draft distributions.
    ///
    /// `draft_probs` is `[1, tokens.len() - 1, V]` F32. Row `i` is the
    /// distribution draft `tokens[i + 1]` was drawn from. An implementation
    /// passes it to
    /// [`verify_logits`](crate::inference::speculative::verify_logits).
    /// `None` means one-hot draft distributions, as for `extend_sampled`.
    ///
    /// The default runs `extend_sampled` for `None` and refuses `Some`.
    ///
    /// # Errors
    ///
    /// An error when the forward or the verify fails. The default returns
    /// [`Error::InferenceError`] for `Some`.
    fn extend_sampled_with_probs(
        &mut self,
        tokens: &[u32],
        rule: &SampleRule,
        draw: Draw,
        draft_probs: Option<&Tensor<R>>,
    ) -> Result<Verdict> {
        match draft_probs {
            None => self.extend_sampled(tokens, rule, draw),
            Some(_) => Err(Error::InferenceError {
                reason: "this model ignores draft distributions; implement \
                         SpeculativeProbModel::extend_sampled_with_probs to verify a draft model"
                    .into(),
            }),
        }
    }
}
