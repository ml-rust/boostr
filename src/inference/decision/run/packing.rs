//! Packing a plan's rows into forward passes of lanes.
//!
//! Rows that some gather reads run longest first. A pass takes the next
//! `min(max_lanes, max(1, pass_tokens / width), max_lane_bytes /
//! lane_bytes(width))` rows, where `width` is the length of its first
//! (longest) row, and right-pads the rest to `width`.

use super::cancel::CancelToken;
use super::plan::DecisionPlan;
use crate::error::{Error, Result};

/// The token id that fills a lane past its row's last real token.
///
/// Any id in the vocabulary works: positions past a row's end are never
/// gathered and nothing in a lane is reused after its pass.
pub const PAD_ID: i64 = 0;

/// The default token budget of one pass, `lanes * width`.
pub const DEFAULT_PASS_TOKENS: usize = 16384;

/// The default largest number of lanes in one pass.
pub const DEFAULT_MAX_LANES: usize = 64;

/// How a decision run splits its rows into passes, and when it stops.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunnerConfig {
    /// The most lanes one pass runs. At least 1.
    pub max_lanes: usize,
    /// The token budget of one pass, `lanes * width`. At least 1. A row
    /// longer than the budget still runs, alone.
    pub pass_tokens: usize,
    /// The byte budget of one pass's lane state,
    /// `lanes * DecisionBackbone::lane_bytes(width)`. `None` sets no
    /// budget. A pass whose single lane needs more is an error.
    pub max_lane_bytes: Option<usize>,
    /// A stop flag the run reads before the prefix, before each pass and
    /// before scoring. `None` never stops.
    pub cancel: Option<CancelToken>,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            max_lanes: DEFAULT_MAX_LANES,
            pass_tokens: DEFAULT_PASS_TOKENS,
            max_lane_bytes: None,
            cancel: None,
        }
    }
}

impl RunnerConfig {
    /// `Ok` unless the cancel flag is set.
    ///
    /// # Errors
    ///
    /// [`Error::Cancelled`] carrying `at` once the flag is set.
    pub fn check_cancel(&self, at: impl FnOnce() -> String) -> Result<()> {
        match &self.cancel {
            Some(token) => token.check(at),
            None => Ok(()),
        }
    }
}

/// One forward pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pass {
    /// The plan row of each lane, longest first.
    pub rows: Vec<usize>,
    /// The padded suffix length of every lane: the first row's length.
    pub width: usize,
    /// The plan gathers this pass returns, in plan order.
    pub gathers: Vec<usize>,
    /// The flat index `lane * width + (pos - prefix_len)` of each gather.
    pub flat: Vec<i64>,
}

impl Pass {
    /// The `[lanes, width]` token ids, row-major, right-padded with [`PAD_ID`].
    pub fn ids(&self, plan: &DecisionPlan) -> Vec<i64> {
        let mut out = Vec::with_capacity(self.rows.len() * self.width);
        for &r in &self.rows {
            let row = &plan.rows[r];
            out.extend(row.iter().map(|&t| i64::from(t)));
            out.extend(std::iter::repeat_n(PAD_ID, self.width - row.len()));
        }
        out
    }
}

/// The lanes of a pass `width` wide: the token and lane caps, then the
/// byte budget.
fn pass_lanes(
    cfg: &RunnerConfig,
    width: usize,
    lane_bytes: &impl Fn(usize) -> usize,
) -> Result<usize> {
    let lanes = cfg.max_lanes.min((cfg.pass_tokens / width).max(1));
    let Some(budget) = cfg.max_lane_bytes else {
        return Ok(lanes);
    };
    let per_lane = lane_bytes(width);
    let fit = budget.checked_div(per_lane).unwrap_or(lanes);
    if fit == 0 {
        return Err(Error::InvalidArgument {
            arg: "max_lane_bytes",
            reason: format!(
                "one lane of a row {width} tokens wide needs {per_lane} bytes, the budget is \
                 {budget} bytes"
            ),
        });
    }
    Ok(lanes.min(fit))
}

/// The passes of a validated `plan`. Rows no gather reads are not run.
/// `lane_bytes(width)` is the lane state one lane `width` wide needs, as
/// [`DecisionBackbone::lane_bytes`](super::backbone::DecisionBackbone::lane_bytes)
/// gives it; it is read only when `cfg.max_lane_bytes` is set.
///
/// # Errors
///
/// [`Error::InvalidArgument`] when `max_lanes` or `pass_tokens` is 0, or
/// when one lane of a pass needs more than `max_lane_bytes`, naming both
/// sizes.
pub fn pack_passes(
    plan: &DecisionPlan,
    cfg: &RunnerConfig,
    lane_bytes: impl Fn(usize) -> usize,
) -> Result<Vec<Pass>> {
    for (arg, value) in [
        ("max_lanes", cfg.max_lanes),
        ("pass_tokens", cfg.pass_tokens),
    ] {
        if value == 0 {
            return Err(Error::InvalidArgument {
                arg,
                reason: "must be at least 1, got 0".into(),
            });
        }
    }
    let p = plan.prefix_len();
    let mut read = vec![false; plan.rows.len()];
    for r in plan.gathers.iter().filter_map(|g| g.row) {
        read[r] = true;
    }
    let mut order: Vec<usize> = (0..plan.rows.len()).filter(|&r| read[r]).collect();
    order.sort_by_key(|&r| std::cmp::Reverse(plan.rows[r].len()));

    let mut seat = vec![(0usize, 0usize); plan.rows.len()];
    let mut passes: Vec<Pass> = Vec::new();
    let mut next = 0;
    while next < order.len() {
        let width = plan.rows[order[next]].len();
        let lanes = pass_lanes(cfg, width, &lane_bytes)?;
        let end = (next + lanes).min(order.len());
        for (lane, &r) in order[next..end].iter().enumerate() {
            seat[r] = (passes.len(), lane);
        }
        passes.push(Pass {
            rows: order[next..end].to_vec(),
            width,
            gathers: Vec::new(),
            flat: Vec::new(),
        });
        next = end;
    }
    for (i, g) in plan.gathers.iter().enumerate() {
        if let Some(r) = g.row {
            let (pass, lane) = seat[r];
            let pass = &mut passes[pass];
            pass.gathers.push(i);
            pass.flat.push((lane * pass.width + g.pos - p) as i64);
        }
    }
    Ok(passes)
}

#[cfg(test)]
mod tests {
    use super::super::plan::{Gather, PrefixInput};
    use super::*;

    fn plan(prefix: usize, lens: &[usize], gathers: &[(Option<usize>, usize)]) -> DecisionPlan {
        DecisionPlan {
            prefix: PrefixInput::Tokens((1..=prefix as u32).collect()),
            rows: lens
                .iter()
                .enumerate()
                .map(|(r, &n)| (0..n as u32).map(|t| 10 * r as u32 + t + 1).collect())
                .collect(),
            gathers: gathers
                .iter()
                .map(|&(row, pos)| Gather { row, pos })
                .collect(),
        }
    }

    fn cfg(max_lanes: usize, pass_tokens: usize) -> RunnerConfig {
        RunnerConfig {
            max_lanes,
            pass_tokens,
            ..RunnerConfig::default()
        }
    }

    /// Lane state of 50 fixed bytes plus 100 bytes per token.
    fn bytes(width: usize) -> usize {
        50 + 100 * width
    }

    #[test]
    fn longest_first_within_budget() {
        // Prefix 2; rows of 3, 6, 1, 6; row 2 is never read.
        let p = plan(
            2,
            &[3, 6, 1, 6],
            &[(Some(0), 4), (Some(3), 7), (Some(1), 2)],
        );
        let passes = pack_passes(&p, &cfg(8, 12), bytes).unwrap();
        assert_eq!(passes.len(), 2);
        assert_eq!(passes[0].rows, vec![1, 3]);
        assert_eq!(passes[0].width, 6);
        assert_eq!(passes[0].gathers, vec![1, 2]);
        // Row 3 is lane 1: 1 * 6 + (7 - 2); row 1 is lane 0: 0 * 6 + 0.
        assert_eq!(passes[0].flat, vec![11, 0]);
        assert_eq!(passes[1].rows, vec![0]);
        assert_eq!(passes[1].flat, vec![2]);
        let ids = passes[1].ids(&p);
        assert_eq!(ids, vec![1, 2, 3]);
        let first = passes[0].ids(&p);
        assert_eq!(first.len(), 12);
    }

    #[test]
    fn padding_and_lane_caps() {
        let p = plan(0, &[2, 4, 1], &[(Some(0), 1), (Some(1), 0), (Some(2), 0)]);
        let one = pack_passes(&p, &cfg(1, 1000), bytes).unwrap();
        assert_eq!(one.len(), 3);
        let all = pack_passes(&p, &cfg(64, 1000), bytes).unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].rows, vec![1, 0, 2]);
        assert_eq!(
            all[0].ids(&p),
            vec![
                11, 12, 13, 14, 1, 2, PAD_ID, PAD_ID, 21, PAD_ID, PAD_ID, PAD_ID
            ]
        );
        // A budget below the longest row still runs it, alone.
        let tight = pack_passes(&p, &cfg(64, 3), bytes).unwrap();
        assert_eq!(tight[0].rows, vec![1]);
        assert_eq!(tight[1].rows, vec![0]);
        assert!(pack_passes(&p, &cfg(0, 3), bytes).is_err());
        assert!(pack_passes(&p, &cfg(1, 0), bytes).is_err());
    }

    /// Rows of 4, 4, 2, 2, 2: a lane 4 wide needs 450 bytes and a lane 2
    /// wide needs 250, so a 1000-byte budget seats 2 and then 4 lanes.
    #[test]
    fn byte_budget_limits_lanes() {
        let p = plan(
            1,
            &[4, 2, 4, 2, 2],
            &[
                (Some(0), 1),
                (Some(1), 1),
                (Some(2), 1),
                (Some(3), 1),
                (Some(4), 1),
            ],
        );
        let mut c = cfg(64, 1000);
        let free = pack_passes(&p, &c, bytes).unwrap();
        assert_eq!(free.len(), 1);
        c.max_lane_bytes = Some(1000);
        let capped = pack_passes(&p, &c, bytes).unwrap();
        assert_eq!(capped.len(), 2);
        assert_eq!((capped[0].rows.clone(), capped[0].width), (vec![0, 2], 4));
        assert_eq!(
            (capped[1].rows.clone(), capped[1].width),
            (vec![1, 3, 4], 2)
        );
        // The lane cap still applies under a byte budget.
        c.max_lanes = 1;
        assert_eq!(pack_passes(&p, &c, bytes).unwrap().len(), 5);
        // 500 bytes seat one lane 4 wide and two lanes 2 wide.
        let c = RunnerConfig {
            max_lane_bytes: Some(500),
            ..cfg(64, 1000)
        };
        let rows: Vec<Vec<usize>> = pack_passes(&p, &c, bytes)
            .unwrap()
            .into_iter()
            .map(|pass| pass.rows)
            .collect();
        assert_eq!(rows, vec![vec![0], vec![2], vec![1, 3], vec![4]]);
        // One lane past the budget names both sizes.
        let c = RunnerConfig {
            max_lane_bytes: Some(449),
            ..cfg(64, 1000)
        };
        let e = pack_passes(&p, &c, bytes).unwrap_err().to_string();
        assert!(
            e.contains("4 tokens wide needs 450 bytes, the budget is 449 bytes"),
            "{e}"
        );
    }

    #[test]
    fn prefix_only_plan_has_no_passes() {
        let p = plan(3, &[], &[(None, 0), (None, 2)]);
        assert!(
            pack_passes(&p, &RunnerConfig::default(), bytes)
                .unwrap()
                .is_empty()
        );
    }
}
