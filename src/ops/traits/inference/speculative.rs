//! Sampled verification for speculative decoding (Leviathan et al., 2023).
//!
//! One call checks K draft tokens against the target distributions of one step.
//! The result stays on the device.

use crate::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Device-side sampled verification of draft tokens.
///
/// The output distribution equals the target distribution, whatever the drafter.
/// The sampled path of `boostr::inference::speculative::SpeculativeGenerator` calls it
/// through `verify_logits`. The greedy path does not.
pub trait SpeculativeOps<R: Runtime> {
    /// Accepts the longest valid draft prefix and draws the token after it.
    ///
    /// # Layout
    ///
    /// - `target_probs`: `[B, K+1, V]` F32. Each row is already filtered and normalized.
    /// - `draft_probs`: `[B, K, V]` F32, or `None` for a deterministic drafter.
    ///   `None` means each draft distribution is one-hot on its draft token.
    /// - `draft_tokens`: `[B, K]` I32 draft token ids. I32 is native on every backend.
    /// - `uniforms`: `[B, K+1]` F32 draws in `[0, 1)`.
    ///
    /// K can be 0. Then the call draws one token from target row 0.
    ///
    /// # Algorithm
    ///
    /// - Draft `i` is accepted iff `u_i * q_i(d_i) < p_i(d_i)`. No division runs.
    /// - A draft with `q_i(d_i) == 0` is rejected. The drafter cannot emit it.
    /// - A draft id outside `[0, V)` is rejected. Checking it on the host needs a device read.
    /// - `m` counts the leading accepted drafts. The scan stops at the first reject.
    /// - If `m < K`, `next` is drawn from `max(0, p_m - q_m)`, normalized.
    ///   A one-hot `q_m` zeroes `p_m` at `d_m`. A zero residual falls back to `p_m`.
    /// - If `m == K`, `next` is drawn from `p_K`.
    ///
    /// The single categorical draw always uses `u_K`. No accept test reads `u_K`.
    /// So the draw is independent of every accept decision.
    ///
    /// The draw takes the first id, in id order, whose running sum exceeds `u_K * sum(row)`.
    /// It never returns a zero-probability id. Float rounding at the tail returns
    /// the last id with mass.
    ///
    /// The caller must keep `uniforms` in `[0, 1)` and give each target row positive mass.
    /// Checking either on the host needs a device read.
    ///
    /// # Returns
    ///
    /// `[B, 2]` I32. Row `b` holds `(m, next)`. The caller emits
    /// `d_0 .. d_{m-1}`, then `next`.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidArgument` naming the argument for a wrong rank,
    /// shape or dtype, or for `V` above `2^24`.
    fn speculative_verify(
        &self,
        target_probs: &Tensor<R>,
        draft_probs: Option<&Tensor<R>>,
        draft_tokens: &Tensor<R>,
        uniforms: &Tensor<R>,
    ) -> Result<Tensor<R>>;
}
