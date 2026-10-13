//! CUDA implementation of SamplingOps — fused GPU kernels.
//!
//! Penalties: one thread per unique token, modifies logits in-place.
//! Sampling: single-block fused kernel for temperature → softmax → top-k → top-p → min-p → multinomial.
//! Random value generated on-device via numr's RandomOps::rand_into, into the
//! caller-owned scratch buffer — no CPU RNG, no per-call allocation.

mod logits_to_token;
mod ops;
#[cfg(test)]
mod test_support;
