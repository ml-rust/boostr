//! The Kev pointer readout: each option's logit is its closing token's key
//! dotted with the question's decide-token query (`head`), scored as a
//! [`DecisionHead`](crate::inference::decision::DecisionHead) (`logits`).

pub mod head;
pub mod logits;

pub use head::{K_BIAS, K_WEIGHT, PointerHead, PointerItem, Q_BIAS, Q_WEIGHT};
