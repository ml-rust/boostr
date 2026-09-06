//! Tensor-core MMQ dispatch, feature-major tiling.
//!
//! The output-feature dimension gets the fixed 128 tile and the weight is the
//! MMA operand A. `quant_mmq_q8_0_q8_1_mma` fixes the token tile instead and
//! makes the activation operand A; this path swaps those roles. One entry
//! point is compiled per (weight format, token tile), in three roles:
//! tile-parallel, stream-k, and the stream-k fixup. This module owns the rules
//! that choose among them. The kernels themselves live in
//! `src/quant/cuda/kernels/quant_mmq_mma.cu`.
//!
//! The kernel family is parameterized over the weight format; everything that
//! differs per format is a field of `FeatMajorFormat`. Q8_0, Q4_0, Q4_K,
//! Q5_K, Q6_K, Q3_K and Q2_K are the formats compiled today.
//!
//! This path needs sm_80 and its own repacked activation layout, so the caller
//! gates on `caps.int8_mma_m16n8k32` and falls back to `quant_mmq_q8_0_q8_1_mma`
//! when this returns `Ok(None)`.

mod dispatch;
mod formats;

pub(super) use dispatch::dispatch;
pub(super) use formats::{Q2_K, Q3_K, Q4_0, Q4_K, Q5_K, Q6_K, Q8_0};
