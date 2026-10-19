//! Greedy acceptance: keep the drafts equal to the target's argmax ids.

use super::contract::check_extend;
use crate::error::{Error, Result};
use crate::inference::speculative::model::SpeculativeModel;
use crate::inference::speculative::types::Verdict;

/// Feed `feed` through `extend_greedy` and keep the longest matching draft prefix.
///
/// `feed` is the last committed token, then the drafts. The forward returns
/// the target ids `a_0..a_k`. The verdict keeps the longest prefix with
/// `d_{i+1} == a_i`, of length `m`, and `next = a_m`.
///
/// # Errors
///
/// [`Error::InferenceError`] when `feed` is empty or the model breaks its
/// contract. A model error returns as is.
pub(crate) fn accept<M: SpeculativeModel>(
    model: &mut M,
    feed: &[u32],
    len_before: usize,
) -> Result<Verdict> {
    let vocab_size = model.vocab_size();
    let drafts = feed.get(1..).ok_or_else(|| Error::InferenceError {
        reason: "a verify forward needs the last committed token".into(),
    })?;
    let ids = model.extend_greedy(feed)?;
    check_extend(feed.len(), ids.len(), len_before, model.len())?;
    if let Some((j, &id)) = ids
        .iter()
        .enumerate()
        .find(|&(_, &id)| id as usize >= vocab_size)
    {
        return Err(Error::InferenceError {
            reason: format!(
                "extend_greedy returned id {id} at row {j}, outside the model vocabulary \
                 of {vocab_size} tokens"
            ),
        });
    }
    let accepted = drafts.iter().zip(&ids).take_while(|(d, a)| d == a).count();
    let next = *ids.get(accepted).ok_or_else(|| Error::InferenceError {
        reason: format!("extend_greedy returned no id for row {accepted}"),
    })?;
    Ok(Verdict { accepted, next })
}
