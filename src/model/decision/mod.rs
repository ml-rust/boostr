//! Decision-model readouts: the [`DecisionHead`](crate::inference::decision::DecisionHead)
//! implementations that turn gathered hidden rows into option logits.
//!
//! - `label_token`: one label token per option, read at the answer slot

pub mod label_token;

pub use label_token::{LabelTokenHead, LabelTokenItem};
