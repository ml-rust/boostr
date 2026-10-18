//! One scheduling step: decode the running set, admit waiting sequences
//! for prefill, and resume preempted sequences, under the batch limits.

use crate::error::Result;
use crate::inference::memory::{BlockAllocator, BlockTable};
use std::collections::HashMap;

use super::core::SequenceScheduler;
use super::types::{ScheduledBatch, SequenceState};

impl<A: BlockAllocator> SequenceScheduler<A> {
    pub fn schedule(&mut self) -> Result<Option<ScheduledBatch>> {
        self.schedule_with_limit(usize::MAX)
    }

    /// [`schedule`](Self::schedule) with a cap on batch growth.
    ///
    /// Waiting and preempted sequences join the running set only while it
    /// holds fewer than `running_limit` sequences. Sequences already running
    /// keep decoding above the cap. `config.max_batch_size` still applies.
    ///
    /// A running sequence that `max_batch_size` or `max_batch_tokens` leaves
    /// out keeps its blocks and its Running state. It leads the next call.
    pub fn schedule_with_limit(&mut self, running_limit: usize) -> Result<Option<ScheduledBatch>> {
        let mut batch = ScheduledBatch {
            prefill_sequences: Vec::new(),
            decode_sequences: Vec::new(),
            block_tables: HashMap::new(),
            preempted_sequences: Vec::new(),
            cached_token_counts: HashMap::new(),
        };

        let mut batch_tokens = 0;

        // Step 1: decode the running set. A sequence the batch limits leave
        // out stays running and moves to the front for the next call, so
        // every running sequence decodes in turn.
        let mut scheduled = Vec::new();
        let mut deferred = Vec::new();
        let mut reserved = 0;
        for &seq_id in &self.running_set {
            let Some(seq) = self.sequences.get(&seq_id) else {
                continue;
            };
            let blocks_needed = seq.blocks_needed_for_next_token();
            if blocks_needed > 0
                && !Self::can_allocate_blocks(
                    &self.prefix_cache,
                    &self.allocator,
                    reserved + blocks_needed,
                )
                && self.config.enable_preemption
                && batch.preempted_sequences.len() < self.config.max_preempt_per_step
            {
                batch.preempted_sequences.push(seq_id);
                continue;
            }

            if batch.len() < self.config.max_batch_size
                && batch_tokens < self.config.max_batch_tokens
            {
                batch.decode_sequences.push(seq_id);
                batch
                    .block_tables
                    .insert(seq_id, seq.block_table.blocks.clone());
                batch_tokens += 1;
                reserved += blocks_needed;
                scheduled.push(seq_id);
            } else {
                deferred.push(seq_id);
            }
        }

        for seq_id in &batch.preempted_sequences {
            self.preempt_sequence(*seq_id)?;
        }

        deferred.extend(scheduled);
        self.running_set = deferred;

        // Step 2: Schedule waiting sequences for prefill
        let mut scheduled_waiting = Vec::new();
        for &seq_id in self.waiting_queue.iter() {
            if batch.len() >= self.config.max_batch_size
                || self.running_set.len() + scheduled_waiting.len() >= running_limit
            {
                break;
            }

            if let Some(seq) = self.sequences.get(&seq_id) {
                let prompt_tokens = seq.prompt_len;

                if batch_tokens + prompt_tokens > self.config.max_batch_tokens {
                    continue;
                }

                // Try prefix cache first for block reuse
                let (blocks, cached_count) = if let Some(ref mut cache) = self.prefix_cache {
                    let tokens = &seq.request.prompt_tokens;
                    match cache.get_or_allocate_blocks(seq_id, tokens) {
                        Ok(result) => {
                            // Always below the prompt length, so prefill
                            // computes at least the last prompt token.
                            let cached_tokens = result.cached_count() * self.config.block_size;
                            (result.into_blocks(), cached_tokens)
                        }
                        // The cache already evicted what it could, so the
                        // sequence waits for blocks to come back.
                        Err(_) => continue,
                    }
                } else {
                    let blocks_needed =
                        BlockTable::blocks_needed(prompt_tokens, self.config.block_size);
                    if !self.allocator.can_allocate(blocks_needed) {
                        continue;
                    }
                    (self.allocator.allocate(blocks_needed)?, 0)
                };

                if let Some(seq_mut) = self.sequences.get_mut(&seq_id) {
                    seq_mut.block_table.blocks = blocks.clone();
                    seq_mut.block_table.num_tokens = prompt_tokens;
                    seq_mut.cached_token_count = cached_count;
                }

                batch.prefill_sequences.push(seq_id);
                batch.block_tables.insert(seq_id, blocks);
                if cached_count > 0 {
                    batch.cached_token_counts.insert(seq_id, cached_count);
                }
                batch_tokens += prompt_tokens;
                scheduled_waiting.push(seq_id);
            }
        }

        self.waiting_queue
            .retain(|id| !scheduled_waiting.contains(id));

        for seq_id in scheduled_waiting {
            self.running_set.push(seq_id);
            self.move_state(
                seq_id,
                "admit",
                SequenceState::Waiting,
                SequenceState::Running,
            )?;
        }

        // Step 3: Resume preempted sequences
        let mut resumed = Vec::new();
        for &seq_id in &self.preempted_set {
            if batch.len() >= self.config.max_batch_size
                || self.running_set.len() + resumed.len() >= running_limit
            {
                break;
            }

            if let Some(seq) = self.sequences.get(&seq_id) {
                if batch_tokens + seq.total_tokens > self.config.max_batch_tokens {
                    continue;
                }

                let blocks_needed =
                    BlockTable::blocks_needed(seq.total_tokens, self.config.block_size);
                if !Self::can_allocate_blocks(&self.prefix_cache, &self.allocator, blocks_needed) {
                    continue;
                }

                let seq_tokens = seq.total_tokens;
                let blocks =
                    Self::allocate_blocks(&mut self.prefix_cache, &self.allocator, blocks_needed)?;

                if let Some(seq_mut) = self.sequences.get_mut(&seq_id) {
                    seq_mut.block_table.blocks = blocks.clone();
                    seq_mut.block_table.num_tokens = seq_mut.total_tokens;
                    // Fresh blocks hold no KV, so nothing is cached.
                    seq_mut.cached_token_count = 0;
                }

                batch.decode_sequences.push(seq_id);
                batch.block_tables.insert(seq_id, blocks);
                batch_tokens += seq_tokens;
                resumed.push(seq_id);
            }
        }

        self.preempted_set.retain(|id| !resumed.contains(id));
        for seq_id in resumed {
            self.running_set.push(seq_id);
            self.move_state(
                seq_id,
                "resume",
                SequenceState::Preempted,
                SequenceState::Running,
            )?;
        }

        if batch.is_empty() {
            Ok(None)
        } else {
            Ok(Some(batch))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::{SchedulerConfig, SequenceId, SequenceRequest};
    use super::*;
    use crate::inference::memory::CpuBlockAllocator;

    fn scheduler(max_batch_size: usize) -> SequenceScheduler<CpuBlockAllocator> {
        let config = SchedulerConfig {
            max_batch_size,
            block_size: 4,
            ..Default::default()
        };
        SequenceScheduler::new(CpuBlockAllocator::new(64, 4), config)
    }

    fn add(sched: &mut SequenceScheduler<CpuBlockAllocator>, id: SequenceId) {
        sched
            .add_request(SequenceRequest::new(id, vec![1, 2, 3]).with_max_tokens(8))
            .expect("add_request");
    }

    #[test]
    fn the_limit_stops_admission_and_keeps_running_sequences() {
        let mut sched = scheduler(8);
        add(&mut sched, 1);
        add(&mut sched, 2);
        let first = sched
            .schedule_with_limit(1)
            .expect("schedule")
            .expect("batch");
        assert_eq!(first.prefill_sequences, vec![1]);
        assert_eq!(sched.stats().waiting_count, 1);

        // The running set is at the cap: sequence 1 decodes, 2 waits.
        let held = sched
            .schedule_with_limit(1)
            .expect("schedule")
            .expect("batch");
        assert_eq!(held.decode_sequences, vec![1]);
        assert!(held.prefill_sequences.is_empty());
        assert_eq!(sched.stats().waiting_count, 1);

        // A cap below the running set never evicts a running sequence.
        let below = sched
            .schedule_with_limit(0)
            .expect("schedule")
            .expect("batch");
        assert_eq!(below.decode_sequences, vec![1]);

        let grown = sched.schedule().expect("schedule").expect("batch");
        assert_eq!(grown.prefill_sequences, vec![2]);
        assert_eq!(sched.stats().waiting_count, 0);
        assert_eq!(sched.stats().running_count, 2);
    }

    #[test]
    fn a_running_sequence_over_the_batch_limit_decodes_next() {
        let mut sched = scheduler(8);
        for id in 1..=3 {
            add(&mut sched, id);
        }
        let first = sched.schedule().expect("schedule").expect("batch");
        assert_eq!(first.prefill_sequences, vec![1, 2, 3]);

        // The batch limit drops below the running set.
        sched.config.max_batch_size = 2;
        let mut order = Vec::new();
        for _ in 0..3 {
            let batch = sched.schedule().expect("schedule").expect("batch");
            order.push(batch.decode_sequences.clone());
            assert_eq!(sched.stats().running_count, 3);
            sched.check_block_invariants().expect("invariants");
        }
        // Round robin: the left-out sequence leads the next batch.
        assert_eq!(order, vec![vec![1, 2], vec![3, 1], vec![2, 3]]);

        let mut steps = 0;
        while sched.has_work() {
            steps += 1;
            assert!(steps < 100, "a sequence starves");
            let batch = sched.schedule().expect("schedule").expect("batch");
            for id in batch.decode_sequences {
                sched.append_token(id, 7).expect("append");
            }
            sched.check_block_invariants().expect("invariants");
        }
        for id in 1..=3 {
            assert_eq!(sched.get_sequence_state(id), Some(SequenceState::Finished));
        }
        assert_eq!(sched.allocator().free_block_count(), 64);
    }

    #[test]
    fn the_token_limit_defers_without_dropping() {
        let mut sched = scheduler(8);
        for id in 1..=3 {
            add(&mut sched, id);
        }
        sched.schedule().expect("schedule").expect("batch");
        sched.config.max_batch_tokens = 1;
        let mut seen = Vec::new();
        for _ in 0..3 {
            let batch = sched.schedule().expect("schedule").expect("batch");
            seen.extend(batch.decode_sequences);
            sched.check_block_invariants().expect("invariants");
        }
        seen.sort_unstable();
        assert_eq!(
            seen,
            vec![1, 2, 3],
            "each sequence decodes once in three calls"
        );
    }

    #[test]
    fn max_batch_size_still_applies_under_a_higher_limit() {
        let mut sched = scheduler(1);
        add(&mut sched, 1);
        add(&mut sched, 2);
        let batch = sched
            .schedule_with_limit(4)
            .expect("schedule")
            .expect("batch");
        assert_eq!(batch.len(), 1);
        assert_eq!(sched.stats().waiting_count, 1);
    }
}
