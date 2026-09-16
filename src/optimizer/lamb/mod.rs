//! LAMB optimizer (Layer-wise Adaptive Moments for Batch training)
//!
//! You et al., "Large Batch Optimization for Deep Learning: Training BERT in 76 Minutes", 2020.
//! Layer-wise adaptive scaling enables stable training at very large batch sizes (32K+).
//! Used by Google for BERT pre-training and applicable to frontier-scale LLM training.
//!
//! Also supports LARS mode (Layer-wise Adaptive Rate Scaling, You et al., 2017)
//! by setting `use_adam = false`, which uses SGD-style momentum instead of Adam moments.

mod apply;
mod step;
mod types;

pub use types::{Lamb, LambConfig};
