//! Shared plumbing for the optimizers' `state_dict` and `load_state_dict`.

mod check;
mod export;
mod group;
mod keys;
mod moments;
mod step;
#[cfg(test)]
pub(crate) mod test_support;

pub(crate) use check::{check_matches, check_state_dtype};
pub(crate) use export::{param_name, put};
pub(crate) use group::group_entries;
pub(crate) use keys::{EXP_AVG, EXP_AVG_SQ, MASTER, MOMENTUM_BUFFER, SUM_SQ};
pub(crate) use moments::{put_moments, take_moments};
pub(crate) use step::{put_step, read_step, reject_step};
