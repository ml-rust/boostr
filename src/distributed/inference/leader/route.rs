//! Where the leader sends a request frame and where its reply comes from.

use anyhow::{Result, anyhow, bail};
use nexar::Rank;

use super::super::pipeline::PipelineSchedule;

/// The remote part of a pipeline, as the leader sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Route {
    /// Rank of the first stage that does not run on the leader.
    pub(super) entry: Rank,
    /// Whether that stage holds the embedding and takes token ids.
    pub(super) entry_takes_tokens: bool,
    /// Rank of the last stage, which sends the logits.
    pub(super) last: Rank,
    /// Every rank that runs a stage, other than the leader.
    pub(super) workers: Vec<Rank>,
}

/// The route of `schedule`, with `rank_of` mapping a stage's node id to its
/// rank. `None` when every stage runs on `leader`.
///
/// # Errors
///
/// A stage with no rank, a leader stage after a remote stage, or a last
/// stage on the leader.
pub(super) fn route(
    schedule: &PipelineSchedule,
    leader: Rank,
    rank_of: impl Fn(&str) -> Option<Rank>,
) -> Result<Option<Route>> {
    let mut ranks = Vec::with_capacity(schedule.stages().len());
    for stage in schedule.stages() {
        let rank = rank_of(&stage.node_id)
            .ok_or_else(|| anyhow!("no rank runs pipeline stage {}", stage.node_id))?;
        ranks.push((rank, stage));
    }
    let Some(first_remote) = ranks.iter().position(|&(rank, _)| rank != leader) else {
        return Ok(None);
    };
    if ranks[first_remote..]
        .iter()
        .any(|&(rank, _)| rank == leader)
    {
        bail!(
            "the leader runs a stage after a worker stage. The leader runs only the leading \
             stages, and workers run the rest"
        );
    }
    let (entry, entry_stage) = ranks[first_remote];
    let (last, _) = ranks[ranks.len() - 1];
    let mut workers: Vec<Rank> = ranks[first_remote..].iter().map(|&(r, _)| r).collect();
    workers.dedup();
    Ok(Some(Route {
        entry,
        entry_takes_tokens: entry_stage.has_embedding,
        last,
        workers,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::inference::pipeline::rank_node_id;
    use std::collections::HashMap;

    fn by_rank(world: u32, layers: usize) -> PipelineSchedule {
        PipelineSchedule::by_rank(world, layers).unwrap()
    }

    fn ranks(id: &str) -> Option<Rank> {
        id.strip_prefix("rank-").and_then(|r| r.parse().ok())
    }

    #[test]
    fn test_by_rank_routes_past_the_leader_stage() {
        let r = route(&by_rank(3, 6), 0, ranks).unwrap().unwrap();
        assert_eq!(
            r,
            Route {
                entry: 1,
                entry_takes_tokens: false,
                last: 2,
                workers: vec![1, 2],
            }
        );
        assert_eq!(route(&by_rank(1, 6), 0, ranks).unwrap(), None);
    }

    #[test]
    fn test_a_leader_with_no_stage_sends_token_ids() {
        let schedule = by_rank(2, 4);
        let shifted = |id: &str| ranks(id).map(|r| r + 1);
        let r = route(&schedule, 0, shifted).unwrap().unwrap();
        assert_eq!((r.entry, r.entry_takes_tokens, r.last), (1, true, 2));
    }

    #[test]
    fn test_a_leader_stage_after_a_worker_stage_is_refused() {
        let mut layers = HashMap::new();
        layers.insert(rank_node_id(0), (2, 4));
        layers.insert(rank_node_id(1), (0, 2));
        let schedule = PipelineSchedule::new(&layers, 4);
        assert!(route(&schedule, 0, ranks).is_err());
        assert!(route(&schedule, 0, |_| None).is_err());
    }
}
