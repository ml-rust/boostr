//! A [`DecisionPlan`] from full prompt rows: the shared prefix is the
//! rows' longest common token prefix, and each answer slot becomes one
//! gather.
//!
//! Coordinates: a [`Gather`]'s `pos` is a position in the sequence
//! `prefix + row`. Each plan row is a full row with the prefix cut off, so
//! that sequence is the full row and a slot's position in its full row is
//! its gather `pos` unchanged. The runner reads suffix index `pos -
//! prefix_len` of the row's lane.

use crate::error::{Error, Result};
use crate::inference::decision::run::{DecisionPlan, Gather, PrefixInput};
use std::collections::HashMap;

/// One answer slot: the full-row position whose hidden row a scoring row
/// reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RowSlot {
    /// The row index.
    pub row: usize,
    /// The position in the full row, prefix included.
    pub pos: usize,
}

/// A plan built by [`plan_from_rows`] and the gather each slot reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowsPlan {
    /// The plan: shared prefix, row suffixes, one gather per distinct slot.
    pub plan: DecisionPlan,
    /// The plan gather of each slot, in slot order. Repeated slots share
    /// one gather. A head item that names a gather by its plan index, such
    /// as `LabelTokenItem::slot`, takes this value.
    pub slot_gathers: Vec<usize>,
}

fn invalid(arg: &'static str, reason: String) -> Error {
    Error::InvalidArgument { arg, reason }
}

/// The longest token prefix every row shares.
fn common_prefix(rows: &[Vec<u32>]) -> usize {
    let Some((first, rest)) = rows.split_first() else {
        return 0;
    };
    rest.iter().fold(first.len(), |n, row| {
        first[..n]
            .iter()
            .zip(row)
            .take_while(|(a, b)| a == b)
            .count()
    })
}

/// The plan of `rows` (full prompt token ids) read at `slots`.
///
/// The prefix length is the longest common prefix of the rows when there
/// are two or more, capped at the earliest slot so every slot reads a row
/// token, and capped so every row keeps at least one token. A capped
/// length below `min_prefix`, or a single row, gives no prefix. Rows no
/// slot reads stay in the plan and the runner skips them.
///
/// # Errors
///
/// [`Error::InvalidArgument`] when there are no rows or no slots, when a
/// row is empty, or when a slot names a row past `rows` or a position past
/// its row's last token, naming the row, slot and value.
pub fn plan_from_rows(
    rows: Vec<Vec<u32>>,
    slots: &[RowSlot],
    min_prefix: usize,
) -> Result<RowsPlan> {
    if rows.is_empty() {
        return Err(invalid("rows", "no rows".into()));
    }
    if slots.is_empty() {
        return Err(invalid(
            "slots",
            "no slots: a plan reads at least one row".into(),
        ));
    }
    if let Some(r) = rows.iter().position(Vec::is_empty) {
        return Err(invalid("rows", format!("row {r} has no tokens")));
    }
    for (s, slot) in slots.iter().enumerate() {
        let Some(row) = rows.get(slot.row) else {
            return Err(invalid(
                "slots",
                format!(
                    "slot {s} names row {} but there are {} rows",
                    slot.row,
                    rows.len()
                ),
            ));
        };
        if slot.pos >= row.len() {
            return Err(invalid(
                "slots",
                format!(
                    "slot {s} reads position {} of row {}, which has {} tokens",
                    slot.pos,
                    slot.row,
                    row.len()
                ),
            ));
        }
    }

    let shortest = rows.iter().map(Vec::len).min().unwrap_or(1);
    let earliest = slots.iter().map(|s| s.pos).min().unwrap_or(0);
    let shared = if rows.len() >= 2 {
        common_prefix(&rows).min(earliest).min(shortest - 1)
    } else {
        0
    };
    let p = if shared < min_prefix { 0 } else { shared };

    let prefix = rows[0][..p].to_vec();
    let suffixes = rows.into_iter().map(|mut row| row.split_off(p)).collect();
    let mut seen: HashMap<RowSlot, usize> = HashMap::with_capacity(slots.len());
    let mut gathers = Vec::with_capacity(slots.len());
    let slot_gathers = slots
        .iter()
        .map(|&slot| {
            *seen.entry(slot).or_insert_with(|| {
                gathers.push(Gather {
                    row: Some(slot.row),
                    pos: slot.pos,
                });
                gathers.len() - 1
            })
        })
        .collect();
    let plan = DecisionPlan {
        prefix: PrefixInput::Tokens(prefix),
        rows: suffixes,
        gathers,
    };
    plan.validate()?;
    Ok(RowsPlan { plan, slot_gathers })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slot(row: usize, pos: usize) -> RowSlot {
        RowSlot { row, pos }
    }

    fn at(row: usize, pos: usize) -> Gather {
        Gather {
            row: Some(row),
            pos,
        }
    }

    fn build(rows: &[&[u32]], slots: &[RowSlot], min_prefix: usize) -> Result<RowsPlan> {
        plan_from_rows(rows.iter().map(|r| r.to_vec()).collect(), slots, min_prefix)
    }

    #[test]
    fn single_row_has_no_prefix() {
        let out = build(&[&[1, 2, 3, 4]], &[slot(0, 3), slot(0, 1)], 0).unwrap();
        assert!(out.plan.prefix.is_empty());
        assert_eq!(out.plan.rows, vec![vec![1, 2, 3, 4]]);
        assert_eq!(out.plan.gathers, vec![at(0, 3), at(0, 1)]);
        assert_eq!(out.slot_gathers, vec![0, 1]);
    }

    /// Common prefix 4, earliest slot 3: the prefix is 3 tokens and the
    /// slot at 3 reads its row's first suffix token.
    #[test]
    fn prefix_is_capped_at_the_earliest_slot() {
        let rows: [&[u32]; 3] = [
            &[1, 2, 3, 4, 5, 6],
            &[1, 2, 3, 4, 7],
            &[1, 2, 3, 4, 8, 9, 10],
        ];
        let out = build(&rows, &[slot(0, 5), slot(1, 3), slot(2, 6)], 0).unwrap();
        let plan = &out.plan;
        assert_eq!(plan.prefix.tokens(), &[1, 2, 3]);
        assert_eq!(
            plan.rows,
            vec![vec![4, 5, 6], vec![4, 7], vec![4, 8, 9, 10]]
        );
        assert_eq!(plan.gathers, vec![at(0, 5), at(1, 3), at(2, 6)]);
        // Uncapped, the prefix is the full common prefix of 4.
        let out = build(&rows, &[slot(0, 5), slot(1, 4), slot(2, 6)], 0).unwrap();
        assert_eq!(out.plan.prefix_len(), 4);
        assert_eq!(out.plan.rows[1], vec![7]);
    }

    #[test]
    fn short_prefix_below_minimum_is_dropped() {
        let rows: [&[u32]; 2] = [&[1, 2, 3, 9], &[1, 2, 4, 9]];
        let kept = build(&rows, &[slot(0, 3), slot(1, 3)], 2).unwrap();
        assert_eq!(kept.plan.prefix_len(), 2);
        let dropped = build(&rows, &[slot(0, 3), slot(1, 3)], 3).unwrap();
        assert_eq!(dropped.plan.prefix_len(), 0);
        assert_eq!(dropped.plan.rows[1], vec![1, 2, 4, 9]);
        assert_eq!(dropped.plan.gathers, vec![at(0, 3), at(1, 3)]);
    }

    /// Identical rows keep one token each; a repeated slot shares a gather.
    #[test]
    fn identical_rows_keep_their_last_token() {
        let rows: [&[u32]; 2] = [&[5, 6, 7], &[5, 6, 7]];
        let out = build(&rows, &[slot(1, 2), slot(0, 2), slot(1, 2)], 0).unwrap();
        assert_eq!(out.plan.prefix.tokens(), &[5, 6]);
        assert_eq!(out.plan.rows, vec![vec![7], vec![7]]);
        assert_eq!(out.plan.gathers, vec![at(1, 2), at(0, 2)]);
        assert_eq!(out.slot_gathers, vec![0, 1, 0]);
    }

    /// A slot inside the common prefix pulls the prefix back to it.
    #[test]
    fn slot_inside_the_common_prefix() {
        let rows: [&[u32]; 2] = [&[1, 2, 3, 4, 5], &[1, 2, 3, 4, 6]];
        let out = build(&rows, &[slot(0, 4), slot(1, 1)], 0).unwrap();
        assert_eq!(out.plan.prefix.tokens(), &[1]);
        assert_eq!(out.plan.rows, vec![vec![2, 3, 4, 5], vec![2, 3, 4, 6]]);
        assert_eq!(out.plan.gathers, vec![at(0, 4), at(1, 1)]);
        let zero = build(&rows, &[slot(0, 0)], 0).unwrap();
        assert!(zero.plan.prefix.is_empty());
    }

    /// Rows of 2, 6 and 4 tokens share 2; the shortest row keeps 1 token,
    /// so the prefix is 1. The unread row stays in the plan.
    #[test]
    fn ragged_rows() {
        let rows: [&[u32]; 3] = [&[8, 9], &[8, 9, 1, 2, 3, 4], &[8, 9, 5, 6]];
        let out = build(&rows, &[slot(1, 5), slot(2, 3)], 0).unwrap();
        assert_eq!(out.plan.prefix.tokens(), &[8]);
        assert_eq!(
            out.plan.rows,
            vec![vec![9], vec![9, 1, 2, 3, 4], vec![9, 5, 6]]
        );
        assert_eq!(out.plan.gathers, vec![at(1, 5), at(2, 3)]);
        let rows: [&[u32]; 2] = [&[8, 9, 7], &[8, 9, 1, 2, 3, 4]];
        let out = build(&rows, &[slot(1, 5), slot(0, 2)], 0).unwrap();
        assert_eq!(out.plan.prefix.tokens(), &[8, 9]);
        assert_eq!(out.plan.rows, vec![vec![7], vec![1, 2, 3, 4]]);
    }

    #[test]
    fn bad_input_names_the_value() {
        let e = |r: Result<RowsPlan>| r.unwrap_err().to_string();
        assert!(e(build(&[], &[slot(0, 0)], 0)).contains("no rows"));
        assert!(e(build(&[&[1]], &[], 0)).contains("no slots"));
        assert!(e(build(&[&[1], &[]], &[slot(0, 0)], 0)).contains("row 1 has no tokens"));
        let msg = e(build(&[&[1, 2]], &[slot(0, 1), slot(3, 0)], 0));
        assert!(
            msg.contains("slot 1 names row 3 but there are 1 rows"),
            "{msg}"
        );
        let msg = e(build(&[&[1, 2], &[1]], &[slot(1, 1)], 0));
        assert!(
            msg.contains("slot 0 reads position 1 of row 1, which has 1 tokens"),
            "{msg}"
        );
    }
}
