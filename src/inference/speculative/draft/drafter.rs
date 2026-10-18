//! The drafter contract.

/// Proposes the next tokens of a sequence.
///
/// A draft only changes speed. The generator checks every draft against the
/// target, so a wrong draft never changes the output.
pub trait Drafter {
    /// Up to `k` tokens that are likely to follow `context`.
    ///
    /// `context` is the prompt plus every token emitted so far. The generator
    /// keeps the first `k` ids and cuts the proposal at the first id outside
    /// the model vocabulary. An empty proposal is valid.
    fn propose(&mut self, context: &[u32], k: usize) -> Vec<u32>;
}
