//! Sequence lifecycle transitions: prefill completion, token append,
//! finish, abort, and preemption.
//!
//! Each method checks the current state of the sequence first. A call in a
//! state it does not accept returns an error and changes nothing. Valid
//! states per method:
//!
//! | Method             | Valid state                 | Other states                   |
//! | ------------------ | --------------------------- | ------------------------------ |
//! | `prefill_complete` | Running                     | error                          |
//! | `append_token`     | Running                     | error                          |
//! | `finish_sequence`  | Running                     | Finished is a no-op, rest error |
//! | `abort_sequence`   | Waiting, Running, Preempted | Finished is a no-op            |
//! | `preempt_sequence` | Running                     | error                          |
//!
//! Every method returns an error for an id the scheduler does not track.

use crate::error::{Error, Result};
use crate::inference::memory::BlockAllocator;

use super::core::SequenceScheduler;
use super::types::{SequenceId, SequenceState};

impl<A: BlockAllocator> SequenceScheduler<A> {
    /// Release the blocks of a sequence and clear its table.
    ///
    /// On an error from the release the table keeps its blocks.
    fn release_sequence_blocks(&mut self, seq_id: SequenceId, op: &str) -> Result<()> {
        let Some(seq) = self.sequences.get_mut(&seq_id) else {
            return Err(Self::not_found(seq_id, op));
        };
        if !seq.block_table.blocks.is_empty() {
            Self::release_blocks(
                &mut self.prefix_cache,
                &self.allocator,
                seq_id,
                &seq.block_table.blocks,
            )?;
            seq.block_table.blocks.clear();
        }
        Ok(())
    }

    /// Check that prefill of a Running sequence is complete.
    ///
    /// Valid state: Running. Changes nothing.
    pub fn prefill_complete(&mut self, seq_id: SequenceId) -> Result<()> {
        self.require_state(seq_id, "complete prefill of", &[SequenceState::Running])?;
        Ok(())
    }

    /// Append a sampled token to a Running sequence.
    ///
    /// Returns true when the token ends the sequence, which then moves to
    /// Finished. Valid state: Running.
    pub fn append_token(&mut self, seq_id: SequenceId, token: u32) -> Result<bool> {
        self.require_state(seq_id, "append a token to", &[SequenceState::Running])?;
        if let Some(seq) = self.sequences.get_mut(&seq_id) {
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
            Err(Self::not_found(seq_id, "append a token to"))
        }
    }

    /// Finish a Running sequence and release its blocks.
    ///
    /// Valid state: Running. A Finished sequence returns Ok and changes
    /// nothing, so a finish that races a finish or an abort is harmless.
    /// A Waiting or Preempted sequence is an error: finish needs a decode
    /// step, so use [`abort_sequence`](Self::abort_sequence) to end it.
    pub fn finish_sequence(&mut self, seq_id: SequenceId) -> Result<()> {
        const OP: &str = "finish";
        let state = self.require_state(
            seq_id,
            OP,
            &[SequenceState::Running, SequenceState::Finished],
        )?;
        if state == SequenceState::Finished {
            return Ok(());
        }
        self.release_sequence_blocks(seq_id, OP)?;
        self.running_set.retain(|&id| id != seq_id);
        self.move_state(seq_id, OP, SequenceState::Running, SequenceState::Finished)
    }

    /// Abort a sequence and release its blocks.
    ///
    /// Valid states: Waiting, Running, Preempted. A Finished sequence
    /// returns Ok and changes nothing. The counter of the state the
    /// sequence leaves drops by 1.
    pub fn abort_sequence(&mut self, seq_id: SequenceId) -> Result<()> {
        const OP: &str = "abort";
        let state = self.require_state(
            seq_id,
            OP,
            &[
                SequenceState::Waiting,
                SequenceState::Running,
                SequenceState::Preempted,
                SequenceState::Finished,
            ],
        )?;
        if state == SequenceState::Finished {
            return Ok(());
        }
        // Release first, then clear: a second call before cleanup_finished
        // must not free the blocks twice.
        self.release_sequence_blocks(seq_id, OP)?;
        match state {
            SequenceState::Waiting => self.waiting_queue.retain(|&id| id != seq_id),
            SequenceState::Running => self.running_set.retain(|&id| id != seq_id),
            SequenceState::Preempted => self.preempted_set.retain(|&id| id != seq_id),
            SequenceState::Finished => {}
        }
        self.move_state(seq_id, OP, state, SequenceState::Finished)
    }

    /// Move a Running sequence to Preempted and release its blocks.
    ///
    /// Valid state: Running.
    pub(super) fn preempt_sequence(&mut self, seq_id: SequenceId) -> Result<()> {
        const OP: &str = "preempt";
        self.require_state(seq_id, OP, &[SequenceState::Running])?;
        self.release_sequence_blocks(seq_id, OP)?;
        self.running_set.retain(|&id| id != seq_id);
        self.preempted_set.push(seq_id);
        self.move_state(seq_id, OP, SequenceState::Running, SequenceState::Preempted)?;
        self.stats.total_preemptions += 1;
        Ok(())
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
            match rng.below(6) {
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
                3 => {
                    let ids = live_ids(&sched, false);
                    if !ids.is_empty() {
                        let id = ids[rng.below(ids.len() as u64) as usize];
                        sched.abort_sequence(id).expect("cancel");
                    }
                }
                4 => {
                    // Finish or preempt a sequence that is not running.
                    let ids: Vec<u64> = live_ids(&sched, false)
                        .into_iter()
                        .filter(|&id| sched.get_sequence_state(id) != Some(SequenceState::Running))
                        .collect();
                    if !ids.is_empty() {
                        let id = ids[rng.below(ids.len() as u64) as usize];
                        let before = snapshot(&sched);
                        let result = if rng.below(2) == 0 {
                            sched.finish_sequence(id)
                        } else {
                            sched.preempt_sequence(id)
                        };
                        assert!(result.is_err(), "op {op}: a wrong-state call succeeded");
                        assert_eq!(before, snapshot(&sched), "op {op}: a rejected call mutated");
                    }
                }
                _ => {
                    // Unknown id: every operation is an error and changes nothing.
                    let unknown = next_id + 1000;
                    let before = snapshot(&sched);
                    assert!(sched.abort_sequence(unknown).is_err());
                    assert!(sched.finish_sequence(unknown).is_err());
                    assert!(sched.preempt_sequence(unknown).is_err());
                    assert!(sched.append_token(unknown, 1).is_err());
                    assert_eq!(before, snapshot(&sched), "op {op}: an unknown id mutated");
                }
            }
            if op % 16 == 0 {
                sched.cleanup_finished();
            }
            if let Err(e) = sched.check_block_invariants() {
                panic!("seed {seed}, cache {with_cache}, op {op}: {e}");
            }
            // A wrapped counter is far above the number of sequences created.
            let stats = sched.stats();
            for count in [
                stats.waiting_count,
                stats.running_count,
                stats.preempted_count,
                stats.finished_count,
            ] {
                assert!(
                    count as u64 <= next_id,
                    "seed {seed}, op {op}: counter wrapped"
                );
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

    /// Counters, queues, free blocks and every sequence's state and table.
    fn snapshot(sched: &SequenceScheduler<CpuBlockAllocator>) -> String {
        let mut seqs: Vec<_> = sched
            .sequences
            .iter()
            .map(|(&id, s)| (id, s.state, s.block_table.blocks.clone()))
            .collect();
        seqs.sort_by_key(|entry| entry.0);
        format!(
            "{:?} {:?} {:?} {:?} free {} {seqs:?}",
            sched.stats,
            sched.waiting_queue,
            sched.running_set,
            sched.preempted_set,
            sched.allocator().free_block_count()
        )
    }

    /// A scheduler with sequence 1 in `state`.
    fn in_state(state: SequenceState) -> SequenceScheduler<CpuBlockAllocator> {
        let mut sched = scheduler(32, false);
        let request = SequenceRequest::new(1, vec![1, 2, 3]).with_max_tokens(8);
        sched.add_request(request).expect("add");
        if state != SequenceState::Waiting {
            sched.schedule().expect("schedule").expect("batch");
        }
        match state {
            SequenceState::Preempted => sched.preempt_sequence(1).expect("preempt"),
            SequenceState::Finished => sched.finish_sequence(1).expect("finish"),
            _ => {}
        }
        assert_eq!(sched.get_sequence_state(1), Some(state));
        sched.check_block_invariants().expect("invariants");
        sched
    }

    fn counters(sched: &SequenceScheduler<CpuBlockAllocator>) -> [usize; 4] {
        let s = sched.stats();
        [
            s.waiting_count,
            s.running_count,
            s.preempted_count,
            s.finished_count,
        ]
    }

    #[test]
    fn wrong_state_operations_return_an_error_and_change_nothing() {
        use SequenceState::{Finished, Preempted, Waiting};
        fn apply(name: &str, s: &mut SequenceScheduler<CpuBlockAllocator>) -> Result<()> {
            match name {
                "finish" => s.finish_sequence(1),
                "preempt" => s.preempt_sequence(1),
                "append" => s.append_token(1, 5).map(drop),
                _ => s.prefill_complete(1),
            }
        }
        let cases = [
            ("finish", Waiting),
            ("finish", Preempted),
            ("preempt", Waiting),
            ("preempt", Preempted),
            ("preempt", Finished),
            ("append", Waiting),
            ("append", Preempted),
            ("append", Finished),
            ("prefill_complete", Waiting),
            ("prefill_complete", Preempted),
            ("prefill_complete", Finished),
        ];
        for (name, state) in cases {
            let mut sched = in_state(state);
            let before = snapshot(&sched);
            let err = apply(name, &mut sched).expect_err(name);
            let text = err.to_string();
            assert!(text.contains("sequence 1"), "{name} in {state:?}: {text}");
            assert!(
                text.contains(&format!("{state:?}")),
                "{name} in {state:?}: {text}"
            );
            assert_eq!(before, snapshot(&sched), "{name} in {state:?} mutated");
            sched.check_block_invariants().expect("invariants");
        }
    }

    #[test]
    fn an_unknown_sequence_is_an_error_for_every_operation() {
        let mut sched = in_state(SequenceState::Running);
        let before = snapshot(&sched);
        assert!(sched.finish_sequence(9).is_err());
        assert!(sched.abort_sequence(9).is_err());
        assert!(sched.preempt_sequence(9).is_err());
        assert!(sched.append_token(9, 1).is_err());
        assert!(sched.prefill_complete(9).is_err());
        assert_eq!(before, snapshot(&sched));
    }

    #[test]
    fn finishing_twice_counts_once() {
        let mut sched = in_state(SequenceState::Running);
        sched.finish_sequence(1).expect("first finish");
        let after_first = snapshot(&sched);
        sched.finish_sequence(1).expect("second finish");
        assert_eq!(after_first, snapshot(&sched));
        assert_eq!(counters(&sched), [0, 0, 0, 1]);
        sched.check_block_invariants().expect("invariants");
    }

    #[test]
    fn finish_after_abort_and_abort_after_finish_are_no_ops() {
        let mut sched = in_state(SequenceState::Running);
        sched.abort_sequence(1).expect("abort");
        sched.finish_sequence(1).expect("finish after abort");
        sched.abort_sequence(1).expect("abort again");
        assert_eq!(counters(&sched), [0, 0, 0, 1]);
        assert_eq!(sched.allocator().free_block_count(), 32);
        sched.check_block_invariants().expect("invariants");
    }

    #[test]
    fn abort_moves_the_counter_of_the_state_it_leaves() {
        let expected = [
            (SequenceState::Waiting, [1, 0, 0, 0]),
            (SequenceState::Running, [0, 1, 0, 0]),
            (SequenceState::Preempted, [0, 0, 1, 0]),
        ];
        for (state, before) in expected {
            let mut sched = in_state(state);
            assert_eq!(counters(&sched), before, "{state:?}");
            sched.abort_sequence(1).expect("abort");
            assert_eq!(counters(&sched), [0, 0, 0, 1], "{state:?}");
            assert!(!sched.has_work(), "{state:?}");
            assert_eq!(sched.allocator().free_block_count(), 32, "{state:?}");
            sched.check_block_invariants().expect("invariants");
        }
    }

    #[test]
    fn a_resumed_sequence_returns_to_running() {
        let mut sched = in_state(SequenceState::Preempted);
        sched.schedule().expect("schedule").expect("batch");
        assert_eq!(counters(&sched), [0, 1, 0, 0]);
        sched.check_block_invariants().expect("invariants");
    }
}
