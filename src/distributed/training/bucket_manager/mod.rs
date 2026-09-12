//! Gradient bucket manager for overlapping allreduce with backward pass
//!
//! Groups model parameters into fixed-size buckets and fires allreduce
//! on each bucket as soon as all its gradients are ready, enabling
//! communication/computation overlap during the backward pass.

mod allreduce;
mod bucket;
mod param_order;

pub use bucket::GradientBucketManager;
pub use param_order::param_order_from_graph;
