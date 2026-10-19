//! Checks of the target model contract after one forward.

use crate::error::{Error, Result};

/// A forward over `fed` tokens moved the cache from `len_before` to `len_after` positions.
///
/// # Errors
///
/// [`Error::InferenceError`] when the cache did not grow by exactly `fed`.
pub(crate) fn check_growth(fed: usize, len_before: usize, len_after: usize) -> Result<()> {
    if len_after != len_before + fed {
        return Err(Error::InferenceError {
            reason: format!(
                "the model cache holds {len_after} positions after {fed} tokens fed at \
                 {len_before}; it must hold {}",
                len_before + fed
            ),
        });
    }
    Ok(())
}

/// One `extend_greedy` call over `fed` tokens returned `got` ids and moved the
/// cache from `len_before` to `len_after` positions.
///
/// # Errors
///
/// [`Error::InferenceError`] when the id count or the new cache length breaks
/// the [`SpeculativeModel`](crate::inference::speculative::SpeculativeModel) contract.
pub(crate) fn check_extend(
    fed: usize,
    got: usize,
    len_before: usize,
    len_after: usize,
) -> Result<()> {
    if got != fed {
        return Err(Error::InferenceError {
            reason: format!(
                "extend_greedy returned {got} ids for {fed} fed tokens; the model must return \
                 one id per fed token"
            ),
        });
    }
    check_growth(fed, len_before, len_after)
}
