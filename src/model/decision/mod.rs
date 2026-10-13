//! Decision-model readouts: the [`DecisionHead`](crate::inference::decision::DecisionHead)
//! implementations that turn gathered hidden rows into option logits.
//!
//! - `label_token`: one label token per option, read at the answer slot
//! - `pointer`: the Kev pointer head, a decide-token query against each
//!   option's closing-token key

pub mod label_token;
pub mod pointer;

pub use label_token::{LabelTokenHead, LabelTokenItem};
pub use pointer::{PointerHead, PointerItem};
