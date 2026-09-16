//! AdamW optimizer
//!
//! Implements decoupled weight decay regularization (Loshchilov & Hutter, 2019).
//! Uses numr tensor ops directly — works on any backend without GPU↔CPU transfers.

mod apply;
mod step;
mod types;

#[cfg(test)]
mod test_support;

pub use types::{AdamW, AdamWConfig};
