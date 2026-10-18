//! The target model contract of speculative generation.

use crate::error::Result;

/// A causal target model with a rollback-capable cache, scored greedily.
///
/// The model holds a cache of `len()` positions. Position `i` holds the
/// `i`-th fed token. Every call of [`extend_greedy`](Self::extend_greedy)
/// appends at the end, and [`truncate`](Self::truncate) cuts the end off.
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
}
