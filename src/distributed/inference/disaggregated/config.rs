//! Cluster layout of disaggregated inference.

use anyhow::{Result, bail};
use nexar::Rank;

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
}

impl DisaggConfig {
    /// Checks the layout against a cluster of `world_size` ranks.
    ///
    /// # Errors
    ///
    /// No prefill or no decode worker, a rank at or past `world_size`, or a
    /// rank listed twice across the roles.
    pub fn check(&self, world_size: u32) -> Result<()> {
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
    }
}
