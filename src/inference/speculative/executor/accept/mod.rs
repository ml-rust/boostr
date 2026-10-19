//! Acceptance of one verify forward: greedy match or rejection sampling.

pub(crate) mod contract;
pub(crate) mod greedy;
pub(crate) mod rejection;

pub(crate) use contract::{check_extend, check_growth};
