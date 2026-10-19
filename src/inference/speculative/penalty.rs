//! Per-row penalties of one verify forward over the penalty window.

use super::types::SampleRule;
use crate::error::Result;
use crate::ops::traits::inference::SamplingOps;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ShapeOps, TypeConversionOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::collections::BTreeMap;

/// `[1, M, V]` F32 logits with row `i` penalized by the window counts of `context ++ drafts[..i]`.
pub(crate) fn penalized_logits<R, C>(
    client: &C,
    logits: &Tensor<R>,
    context: &[u32],
    drafts: &[u32],
    rule: &SampleRule,
    vocab: usize,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: SamplingOps<R> + BinaryOps<R> + ShapeOps<R> + TypeConversionOps<R>,
{
    let rows = drafts.len() + 1;
    let device = logits.device();
    let f32_logits = if logits.dtype() == DType::F32 {
        logits.contiguous()?
    } else {
        client.cast(logits, DType::F32)?.contiguous()?
    };
    let flat = f32_logits.reshape(&[rows, vocab])?;

    let mut penalized = Vec::with_capacity(rows);
    for (i, counts) in window_counts(context, drafts, rule.penalty_last_n)
        .into_iter()
        .enumerate()
    {
        // A fresh buffer per row: the penalty writes in place, never into the caller's logits.
        let row = Tensor::<R>::empty(&[vocab], DType::F32, device)?;
        client.copy_into(&row, &flat.narrow(0, i, 1)?.reshape(&[vocab])?)?;
        let (ids, row_counts): (Vec<i64>, Vec<i32>) = counts
            .into_iter()
            .filter(|&(t, _)| (t as usize) < vocab)
            .map(|(t, c)| (i64::from(t), c))
            .unzip();
        if !ids.is_empty() {
            let ids_t = Tensor::<R>::from_slice(&ids, &[ids.len()], device)?;
            let counts_t = Tensor::<R>::from_slice(&row_counts, &[row_counts.len()], device)?;
            client.apply_sampling_penalties(
                &row,
                &ids_t,
                &counts_t,
                rule.repeat_penalty,
                rule.frequency_penalty,
                rule.presence_penalty,
            )?;
        }
        penalized.push(row.reshape(&[1, vocab])?);
    }
    let refs: Vec<&Tensor<R>> = penalized.iter().collect();
    Ok(client.cat(&refs, 0)?.reshape(&[1, rows, vocab])?)
}

/// The token counts of each verify row, ascending by id.
///
/// Row `i` counts the last `last_n` tokens of `context ++ drafts[..i]`, or all
/// of them for `last_n == 0`. The result holds `drafts.len() + 1` rows.
fn window_counts(context: &[u32], drafts: &[u32], last_n: usize) -> Vec<Vec<(u32, i32)>> {
    let seq: Vec<u32> = context.iter().chain(drafts).copied().collect();
    let start = |len: usize| {
        if last_n == 0 {
            0
        } else {
            len.saturating_sub(last_n)
        }
    };
    let mut counts: BTreeMap<u32, i32> = BTreeMap::new();
    for &t in &seq[start(context.len())..context.len()] {
        *counts.entry(t).or_insert(0) += 1;
    }
    let mut rows = Vec::with_capacity(drafts.len() + 1);
    rows.push(counts.iter().map(|(&t, &c)| (t, c)).collect());
    for len in context.len() + 1..=seq.len() {
        *counts.entry(seq[len - 1]).or_insert(0) += 1;
        for gone in &seq[start(len - 1)..start(len)] {
            if let Some(c) = counts.get_mut(gone) {
                *c -= 1;
            }
        }
        counts.retain(|_, c| *c > 0);
        rows.push(counts.iter().map(|(&t, &c)| (t, c)).collect());
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute force: count the window of each row from scratch.
    fn brute(context: &[u32], drafts: &[u32], last_n: usize) -> Vec<Vec<(u32, i32)>> {
        (0..=drafts.len())
            .map(|i| {
                let prefix: Vec<u32> = context.iter().chain(&drafts[..i]).copied().collect();
                let from = if last_n == 0 || last_n >= prefix.len() {
                    0
                } else {
                    prefix.len() - last_n
                };
                let mut counts = BTreeMap::<u32, i32>::new();
                for &t in &prefix[from..] {
                    *counts.entry(t).or_insert(0) += 1;
                }
                counts.into_iter().collect()
            })
            .collect()
    }

    #[test]
    fn window_counts_match_a_recount_of_every_row() {
        let contexts: [&[u32]; 4] = [&[3], &[1, 1, 2], &[4, 4, 4, 4, 4, 4], &[0, 1, 2, 3, 0, 1]];
        let drafts: [&[u32]; 4] = [&[], &[1], &[2, 1, 1], &[0, 0, 5, 4, 0]];
        for context in contexts {
            for d in drafts {
                for n in [0, 1, 2, 3, 5, 64] {
                    assert_eq!(
                        window_counts(context, d, n),
                        brute(context, d, n),
                        "context {context:?} drafts {d:?} window {n}"
                    );
                }
            }
        }
    }

    #[test]
    fn window_two_drops_the_older_tokens() {
        let rows = window_counts(&[7, 7, 8], &[9, 8], 2);
        assert_eq!(rows[0], [(7u32, 1i32), (8, 1)]);
        assert_eq!(rows[1], [(8u32, 1i32), (9, 1)]);
        assert_eq!(rows[2], [(8u32, 1i32), (9, 1)]);
    }
}
