//! CPU dequantization kernels
//!
//! Re-exports from format-specific modules:
//! - `dequant_simple` — Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1
//! - `dequant_k_quants` — Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K

pub use super::dequant_k_quants::{
    dequant_q2k, dequant_q3k, dequant_q4k, dequant_q5k, dequant_q6k, dequant_q8k,
};
pub use super::dequant_simple::{
    dequant_q4_0, dequant_q4_1, dequant_q5_0, dequant_q5_1, dequant_q8_0, dequant_q8_1,
};
