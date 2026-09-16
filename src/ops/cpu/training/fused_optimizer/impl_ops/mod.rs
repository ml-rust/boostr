//! CPU implementation of FusedOptimizerOps
//!
//! Single-pass parameter updates using raw pointer arithmetic.
//! Each fused kernel reads all inputs and writes all outputs in one loop,
//! reducing memory traffic by 4-8x vs composing individual ops.

mod dispatch;
