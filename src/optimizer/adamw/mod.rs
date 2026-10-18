//! AdamW optimizer
//!
//! Implements decoupled weight decay regularization (Loshchilov & Hutter, 2019).
//! Uses numr tensor ops directly — works on any backend without GPU↔CPU transfers.

mod apply;
mod groups;
mod state_dict;
mod step;
mod types;

#[cfg(test)]
mod test_support;

pub use groups::ParamGroup;
pub use types::{AdamW, AdamWConfig};
