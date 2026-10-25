//! Cluster layout of disaggregated inference.

use anyhow::{Result, anyhow, bail};
use nexar::Rank;
use std::time::Duration;

/// Default of [`DisaggConfig::handoff_timeout`].
pub const DEFAULT_HANDOFF_TIMEOUT: Duration = Duration::from_secs(30);

/// Role of a disaggregated inference rank.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisaggRole {
    Prefill,
    Decode,
    Router,
}

/// Ranks and limits of a disaggregated cluster.
///
/// One nexar client hosts one role. nexar gives each `(source, tag)` pair one
/// receiver, so two roles on one client steal each other's frames.
#[derive(Debug, Clone)]
pub struct DisaggConfig {
    /// Ranks of all prefill workers.
    pub prefill_workers: Vec<Rank>,
    /// Ranks of all decode workers.
    pub decode_workers: Vec<Rank>,
    /// Rank of the router node.
    pub router_rank: Rank,
    /// Largest serialized KV cache one request can transfer. A larger cache
    /// ends the request with an error event.
    pub max_kv_transfer_bytes: usize,
    /// Longest wait for a KV handoff to reach its decode worker.
    ///
    /// The router starts the clock at the first token. A request whose
    /// decode worker has not confirmed the handoff by then ends with an
    /// error. A prefill worker that stops waits this long for the
    /// confirmations of its last handoffs. Default [`DEFAULT_HANDOFF_TIMEOUT`].
    pub handoff_timeout: Duration,
}

impl DisaggConfig {
    /// The standard layout: the router on rank 0, prefill workers on ranks
    /// `1..=prefill_count`, decode workers on the ranks after them.
    ///
    /// Every process of a cluster derives the same layout from the two
    /// counts, so each worker can request its rank with [`Self::rank_of`].
    ///
    /// # Errors
    ///
    /// A count of 0, or more ranks than a `u32` holds.
    pub fn standard(
        prefill_count: u32,
        decode_count: u32,
        max_kv_transfer_bytes: usize,
    ) -> Result<Self> {
        if prefill_count == 0 || decode_count == 0 {
            bail!(
                "a disaggregated cluster needs at least one prefill and one decode worker, \
                 and the counts are {prefill_count} prefill and {decode_count} decode"
            );
        }
        let world = prefill_count
            .checked_add(decode_count)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| anyhow!("{prefill_count} + {decode_count} workers overflow a rank"))?;
        Ok(Self {
            prefill_workers: (1..=prefill_count).collect(),
            decode_workers: (prefill_count + 1..world).collect(),
            router_rank: 0,
            max_kv_transfer_bytes,
            handoff_timeout: DEFAULT_HANDOFF_TIMEOUT,
        })
    }

    /// Ranks in the cluster: the router and every worker.
    pub fn world_size(&self) -> u32 {
        (1 + self.prefill_workers.len() + self.decode_workers.len()) as u32
    }

    /// The rank of the `index`-th worker of `role`, counted from 0.
    ///
    /// The router has one rank, index 0.
    ///
    /// # Errors
    ///
    /// An index at or past the role's worker count.
    pub fn rank_of(&self, role: DisaggRole, index: u32) -> Result<Rank> {
        let (name, ranks) = match role {
            DisaggRole::Router => ("router", std::slice::from_ref(&self.router_rank)),
            DisaggRole::Prefill => ("prefill", self.prefill_workers.as_slice()),
            DisaggRole::Decode => ("decode", self.decode_workers.as_slice()),
        };
        ranks.get(index as usize).copied().ok_or_else(|| {
            anyhow!(
                "{name} index {index} is out of range: the cluster has {} {name} rank(s), \
                 indices 0..{}",
                ranks.len(),
                ranks.len()
            )
        })
    }

    /// Checks the layout against a cluster of `world_size` ranks.
    ///
    /// # Errors
    ///
    /// No prefill or no decode worker, a rank at or past `world_size`, a
    /// rank listed twice across the roles, or a zero `handoff_timeout`.
    pub fn check(&self, world_size: u32) -> Result<()> {
        if self.handoff_timeout.is_zero() {
            bail!(
                "disaggregated config has a zero handoff_timeout: every request with a decode \
                 phase would fail. Use a positive duration, for example {DEFAULT_HANDOFF_TIMEOUT:?}"
            );
        }
        if self.prefill_workers.is_empty() {
            bail!("disaggregated config lists no prefill workers: add at least one rank");
        }
        if self.decode_workers.is_empty() {
            bail!("disaggregated config lists no decode workers: add at least one rank");
        }
        let mut seen = std::collections::HashSet::new();
        let all = self
            .prefill_workers
            .iter()
            .chain(&self.decode_workers)
            .chain(std::iter::once(&self.router_rank));
        for &rank in all {
            if rank >= world_size {
                bail!(
                    "disaggregated config names rank {rank}, but the cluster has {world_size} ranks"
                );
            }
            if !seen.insert(rank) {
                bail!("disaggregated config lists rank {rank} twice: one rank hosts one role");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> DisaggConfig {
        DisaggConfig {
            prefill_workers: vec![1],
            decode_workers: vec![2, 3],
            router_rank: 0,
            max_kv_transfer_bytes: 1 << 20,
            handoff_timeout: DEFAULT_HANDOFF_TIMEOUT,
        }
    }

    #[test]
    fn check_accepts_a_valid_layout_and_names_each_error() {
        config().check(4).unwrap();
        assert!(
            config()
                .check(3)
                .unwrap_err()
                .to_string()
                .contains("rank 3")
        );
        let dup = DisaggConfig {
            decode_workers: vec![1],
            ..config()
        };
        assert!(dup.check(4).unwrap_err().to_string().contains("twice"));
        let none = DisaggConfig {
            prefill_workers: vec![],
            ..config()
        };
        assert!(none.check(4).unwrap_err().to_string().contains("prefill"));
        let none = DisaggConfig {
            decode_workers: vec![],
            ..config()
        };
        assert!(none.check(4).unwrap_err().to_string().contains("decode"));
        let zero = DisaggConfig {
            handoff_timeout: Duration::ZERO,
            ..config()
        };
        let err = zero.check(4).unwrap_err().to_string();
        assert!(err.contains("handoff_timeout"), "{err}");
    }

    #[test]
    fn standard_puts_the_router_first_then_prefill_then_decode() {
        let c = DisaggConfig::standard(2, 3, 1 << 20).unwrap();
        assert_eq!(c.router_rank, 0);
        assert_eq!(c.prefill_workers, vec![1, 2]);
        assert_eq!(c.decode_workers, vec![3, 4, 5]);
        assert_eq!(c.world_size(), 6);
        c.check(c.world_size()).unwrap();
        assert!(DisaggConfig::standard(0, 1, 1).is_err());
        assert!(DisaggConfig::standard(1, 0, 1).is_err());
        assert!(DisaggConfig::standard(u32::MAX, 1, 1).is_err());
    }

    #[test]
    fn rank_of_maps_each_role_and_index_and_refuses_one_past_the_count() {
        let c = DisaggConfig::standard(2, 3, 1 << 20).unwrap();
        assert_eq!(c.rank_of(DisaggRole::Router, 0).unwrap(), 0);
        assert_eq!(c.rank_of(DisaggRole::Prefill, 0).unwrap(), 1);
        assert_eq!(c.rank_of(DisaggRole::Prefill, 1).unwrap(), 2);
        assert_eq!(c.rank_of(DisaggRole::Decode, 0).unwrap(), 3);
        assert_eq!(c.rank_of(DisaggRole::Decode, 2).unwrap(), 5);
        let err = c.rank_of(DisaggRole::Prefill, 2).unwrap_err().to_string();
        assert!(
            err.contains("prefill index 2") && err.contains("0..2"),
            "{err}"
        );
        assert!(c.rank_of(DisaggRole::Decode, 3).is_err());
        assert!(c.rank_of(DisaggRole::Router, 1).is_err());
    }
}
