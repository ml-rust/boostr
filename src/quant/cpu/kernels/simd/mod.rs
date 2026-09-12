//! SIMD helpers for quantized matmul kernels

#[cfg(target_arch = "aarch64")]
pub mod aarch64;

pub mod dot_f32;
pub mod fused_q2k_dot;
pub mod fused_q2k_q8k_dot;
pub mod fused_q3k_dot;
pub mod fused_q3k_q8k_dot;
pub mod fused_q4k_dot;
pub mod fused_q4k_q8k_dot;
pub mod fused_q5k_dot;
pub mod fused_q5k_q8k_dot;
pub mod fused_q6k_dot;
pub mod fused_q6k_q8k_dot;
pub mod fused_q8_0_q8k_dot;
pub mod int4_unpack;
pub mod quantize_act_q8k;
