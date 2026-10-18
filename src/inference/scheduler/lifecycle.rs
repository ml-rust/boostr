//! Sequence lifecycle transitions: prefill completion, token append,
//! finish, abort, and preemption.

use crate::error::{Error, Result};
use crate::inference::memory::BlockAllocator;

use super::core::SequenceScheduler;
use super::types::{SequenceId, SequenceState};

impl<A: BlockAllocator> SequenceScheduler<A> {
    pub fn prefill_complete(&mut self, seq_id: SequenceId) -> Result<()> {
        if let Some(seq) = self.sequences.get_mut(&seq_id) {
            if seq.state != SequenceState::Running {
                return Err(Error::SchedulerError {
                    reason: format!("Sequence {} is not running", seq_id),
                });
            }
            Ok(())
        } else {
            Err(Error::SchedulerError {
                reason: format!("Sequence {} not found", seq_id),
            })
        }
    }

    pub fn append_token(&mut self, seq_id: SequenceId, token: u32) -> Result<bool> {
        if let Some(seq) = self.sequences.get_mut(&seq_id) {
            if seq.state != SequenceState::Running {
                return Err(Error::SchedulerError {
                    reason: format!("Sequence {} is not running", seq_id),
                });
            }

            // Ensure a physical block exists for the slot where the new token lands
            // before committing any state change.  The new token occupies position
            // `total_tokens` (0-indexed), which belongs to block
            // `total_tokens / block_size`.
            let new_token_block_idx = seq.total_tokens / seq.block_table.block_size;
            if new_token_block_idx >= seq.block_table.blocks.len() {
                if !Self::can_allocate_blocks(&self.prefix_cache, &self.allocator, 1) {
                    return Err(Error::SchedulerError {
                        reason: format!(
                            "Cannot append token to sequence {}: no free blocks available",
                            seq_id
                        ),
                    });
                }
                let blk = Self::allocate_blocks(&mut self.prefix_cache, &self.allocator, 1)?;
                seq.block_table.append_blocks(blk);
            }

            seq.generated_tokens.push(token);
            seq.total_tokens += 1;
            seq.block_table.num_tokens = seq.total_tokens;
            self.stats.total_tokens_generated += 1;

            // Best-effort preallocation for the token after next (non-fatal if it fails).
            // Preallocation takes free blocks only and never evicts a cached prefix.
            let blocks_needed = seq.blocks_needed_for_next_token();
            if blocks_needed > 0
                && self.allocator.can_allocate(blocks_needed)
                && let Ok(new_blocks) = self.allocator.allocate(blocks_needed)
            {
                seq.block_table.append_blocks(new_blocks);
            }

            if seq.is_finished() || seq.total_tokens >= self.config.max_seq_len {
                self.finish_sequence(seq_id)?;
                return Ok(true);
            }

            Ok(false)
        } else {
            Err(Error::SchedulerError {
                reason: format!("Sequence {} not found", seq_id),
            })
        }
    }

    pub fn finish_sequence(&mut self, seq_id: SequenceId) -> Result<()> {
        if let Some(seq) = self.sequences.get_mut(&seq_id) {
            if !seq.block_table.blocks.is_empty() {
                Self::release_blocks(
                    &mut self.prefix_cache,
                    &self.allocator,
                    seq_id,
                    &seq.block_table.blocks,
                )?;
            }

            seq.state = SequenceState::Finished;
            seq.block_table.blocks.clear();

            self.running_set.retain(|&id| id != seq_id);
            self.stats.running_count -= 1;
            self.stats.finished_count += 1;

            Ok(())
        } else {
            Err(Error::SchedulerError {
                reason: format!("Sequence {} not found", seq_id),
            })
        }
    }

    pub fn abort_sequence(&mut self, seq_id: SequenceId) -> Result<()> {
        if let Some(seq) = self.sequences.get_mut(&seq_id) {
            // Idempotent: already terminal, nothing to do.
            if seq.state == SequenceState::Finished {
                return Ok(());
            }

            // Free physical blocks first, then clear to prevent double-free on
            // a second call before cleanup_finished() runs.
            if !seq.block_table.blocks.is_empty() {
                Self::release_blocks(
                    &mut self.prefix_cache,
                    &self.allocator,
                    seq_id,
                    &seq.block_table.blocks,
                )?;
                seq.block_table.blocks.clear();
            }

            match seq.state {
                SequenceState::Waiting => {
                    self.waiting_queue.retain(|&id| id != seq_id);
                    self.stats.waiting_count -= 1;
                }
                SequenceState::Running => {
                    self.running_set.retain(|&id| id != seq_id);
                    self.stats.running_count -= 1;
                }
                SequenceState::Preempted => {
                    self.preempted_set.retain(|&id| id != seq_id);
                    self.stats.preempted_count -= 1;
                }
                SequenceState::Finished => unreachable!("guarded above"),
            }

            seq.state = SequenceState::Finished;
            self.stats.finished_count += 1;

            Ok(())
        } else {
            Err(Error::SchedulerError {
                reason: format!("Sequence {} not found", seq_id),
            })
        }
    }

    pub(super) fn preempt_sequence(&mut self, seq_id: SequenceId) -> Result<()> {
        if let Some(seq) = self.sequences.get_mut(&seq_id) {
            if !seq.block_table.blocks.is_empty() {
                Self::release_blocks(
                    &mut self.prefix_cache,
                    &self.allocator,
                    seq_id,
                    &seq.block_table.blocks,
                )?;
            }

            seq.state = SequenceState::Preempted;
            seq.block_table.blocks.clear();

            self.running_set.retain(|&id| id != seq_id);
            self.preempted_set.push(seq_id);
            self.stats.running_count -= 1;
            self.stats.preempted_count += 1;
            self.stats.total_preemptions += 1;

            Ok(())
        } else {
            Err(Error::SchedulerError {
                reason: format!("Sequence {} not found", seq_id),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::{SchedulerConfig, SequenceRequest};
    use super::*;
    use crate::inference::memory::CpuBlockAllocator;
    use crate::inference::prefix_cache::{PrefixCache, PrefixCacheConfig};

    const BLOCK: usize = 4;

    fn scheduler(total: usize, with_cache: bool) -> SequenceScheduler<CpuBlockAllocator> {
        let allocator = CpuBlockAllocator::new(total, BLOCK);
        let config = SchedulerConfig {
            max_batch_size: 4,
            max_batch_tokens: 256,
            max_seq_len: 64,
            block_size: BLOCK,
            enable_preemption: true,
            max_preempt_per_step: 2,
        };
        let sched = SequenceScheduler::new(allocator.clone(), config);
        if !with_cache {
            return sched;
        }
        let cache_config = PrefixCacheConfig {
            enabled: true,
            max_cached_blocks: 0,
            min_prefix_tokens: 1,
            block_size: BLOCK,
        };
        sched.with_prefix_cache(PrefixCache::new(allocator, cache_config))
    }

    /// Free blocks plus idle cached blocks: what the pool can hand out.
    fn reclaimable(sched: &SequenceScheduler<CpuBlockAllocator>) -> usize {
        let idle = sched.prefix_cache().map_or(0, |c| c.idle_block_count());
        sched.allocator().free_block_count() + idle
    }

    /// One engine step: every scheduled sequence emits one token.
    /// A sequence the pool cannot grow is aborted.
    fn step(sched: &mut SequenceScheduler<CpuBlockAllocator>) {
        let Some(batch) = sched.schedule().expect("schedule") else {
            return;
        };
        for id in batch.all_sequences() {
            if sched.get_sequence_state(id) == Some(SequenceState::Running)
                && sched.append_token(id, 7).is_err()
            {
                sched.abort_sequence(id).expect("abort");
            }
        }
    }

    fn run(prompts: &[Vec<u32>], with_cache: bool) {
        let total = 48;
        let mut sched = scheduler(total, with_cache);
        for (id, prompt) in prompts.iter().enumerate() {
            let request = SequenceRequest::new(id as u64, prompt.clone()).with_max_tokens(6);
            sched.add_request(request).expect("add");
        }
        let mut steps = 0;
        while sched.has_work() {
            steps += 1;
            assert!(steps < 1000, "the scheduler stalls");
            step(&mut sched);
            sched.check_block_invariants().expect("invariants");
        }
        sched.cleanup_finished();
        assert_eq!(
            reclaimable(&sched),
            total,
            "cache {with_cache}: a block leaked"
        );
        if let Some(cache) = sched.prefix_cache_mut() {
            cache.reset().expect("reset");
        }
        assert_eq!(sched.allocator().free_block_count(), total);
    }

    fn shared_prompts() -> Vec<Vec<u32>> {
        (0..8u32)
            .map(|i| (0..10).chain(100 * i..100 * i + i % 5 + 1).collect())
            .collect()
    }

    fn disjoint_prompts() -> Vec<Vec<u32>> {
        (0..8u32)
            .map(|i| (1000 * i..1000 * i + 3 + 2 * i).collect())
            .collect()
    }

    #[test]
    fn finished_sequences_return_every_block() {
        for with_cache in [false, true] {
            run(&shared_prompts(), with_cache);
            run(&disjoint_prompts(), with_cache);
        }
    }

    #[test]
    fn finish_keeps_only_full_prefix_blocks_cached() {
        let mut sched = scheduler(32, true);
        let prompt: Vec<u32> = (0..10).collect();
        sched
            .add_request(SequenceRequest::new(1, prompt).with_max_tokens(8))
            .expect("add");
        while sched.has_work() {
            step(&mut sched);
        }
        let cache = sched.prefix_cache().expect("cache");
        assert_eq!(cache.registered_block_count(), 2);
        assert_eq!(cache.idle_block_count(), 2);
        assert_eq!(sched.allocator().free_block_count(), 30);
    }

    struct Lcg(u64);

    impl Lcg {
        fn below(&mut self, n: u64) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 33) % n
        }
    }

    fn live_ids(sched: &SequenceScheduler<CpuBlockAllocator>, running_only: bool) -> Vec<u64> {
        let mut ids: Vec<u64> = sched
            .sequences
            .iter()
            .filter(|(_, seq)| match seq.state {
                SequenceState::Running => true,
                SequenceState::Finished => false,
                _ => !running_only,
            })
            .map(|(&id, _)| id)
            .collect();
        ids.sort_unstable();
        ids
    }

    fn stress(seed: u64, with_cache: bool) {
        let total = 24;
        let mut sched = scheduler(total, with_cache);
        let mut rng = Lcg(seed);
        let bases: [Vec<u32>; 3] = [(0..12).collect(), (50..62).collect(), (0..6).collect()];
        let mut next_id = 0u64;
        for op in 0..200 {
            match rng.below(4) {
                0 => {
                    let base = &bases[rng.below(3) as usize];
                    let keep = rng.below(base.len() as u64 + 1) as usize;
                    let extra = 1 + rng.below(6) as usize;
                    let mut prompt = base[..keep].to_vec();
                    prompt.extend((0..extra).map(|_| 200 + rng.below(3) as u32));
                    let max_new = 1 + rng.below(8) as usize;
                    let request = SequenceRequest::new(next_id, prompt).with_max_tokens(max_new);
                    sched.add_request(request).expect("add");
                    next_id += 1;
                }
                1 => step(&mut sched),
                2 => {
                    let ids = live_ids(&sched, true);
                    if !ids.is_empty() {
                        let id = ids[rng.below(ids.len() as u64) as usize];
                        sched.finish_sequence(id).expect("finish");
                    }
                }
                _ => {
                    let ids = live_ids(&sched, false);
                    if !ids.is_empty() {
                        let id = ids[rng.below(ids.len() as u64) as usize];
                        sched.abort_sequence(id).expect("cancel");
                    }
                }
            }
            if op % 16 == 0 {
                sched.cleanup_finished();
            }
            if let Err(e) = sched.check_block_invariants() {
                panic!("seed {seed}, cache {with_cache}, op {op}: {e}");
            }
        }

        for id in live_ids(&sched, false) {
            sched.abort_sequence(id).expect("drain");
        }
        sched.cleanup_finished();
        sched.check_block_invariants().expect("drained");
        assert!(!sched.has_work());
        assert_eq!(
            reclaimable(&sched),
            total,
            "seed {seed}, cache {with_cache}"
        );
        if let Some(cache) = sched.prefix_cache_mut() {
            cache.reset().expect("reset");
        }
        assert_eq!(sched.allocator().free_block_count(), total);
    }

    #[test]
    fn random_operations_keep_the_pool_consistent() {
        for seed in [1, 7, 42, 2024] {
            stress(seed, true);
            stress(seed, false);
        }
    }
}
