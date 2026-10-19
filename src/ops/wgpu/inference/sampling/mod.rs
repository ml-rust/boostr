//! WebGPU implementation of SamplingOps — WGSL kernels, F32 only.
//!
//! - Penalties: one invocation per unique token, in place.
//! - `sample_token` and `logits_to_token`: one fused single-workgroup
//!   kernel. It copies the last position, applies penalties, then takes the
//!   argmax or runs the canonical filter of `FilterRule`, then the id-order
//!   draw. The random value comes from `RandomOps::rand_into` on the device.
//!
//! Every token stays on the device. The caller reads it back once.

mod dispatch;
mod ops;

pub(super) use dispatch::{tensor_buffer, uniform_buffer};
