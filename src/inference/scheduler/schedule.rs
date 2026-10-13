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
    pub fn schedule_with_limit(&mut self, running_limit: usize) -> Result<Option<ScheduledBatch>> {
        let mut batch = ScheduledBatch {
            prefill_sequences: Vec::new(),
            decode_sequences: Vec::new(),
            block_tables: HashMap::new(),
            preempted_sequences: Vec::new(),
            cached_token_counts: HashMap::new(),
        };

        let mut batch_tokens = 0;

        // Step 1: Add running sequences to decode batch
        let mut sequences_to_keep = Vec::new();
        for &seq_id in &self.running_set {
            if let Some(seq) = self.sequences.get(&seq_id) {
                let blocks_needed = seq.blocks_needed_for_next_token();
                if blocks_needed > 0
                    && !self.allocator.can_allocate(blocks_needed)
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
                    sequences_to_keep.push(seq_id);
                }
            }
        }

        for seq_id in &batch.preempted_sequences {
            self.preempt_sequence(*seq_id)?;
        }

        self.running_set = sequences_to_keep;

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
                            let cached = result.cached_count();
                            let cached_tokens = cached * self.config.block_size;
                            (result.into_blocks(), cached_tokens.min(prompt_tokens))
                        }
                        Err(_) => {
                            // Fallback to direct allocation
                            let blocks_needed =
                                BlockTable::blocks_needed(prompt_tokens, self.config.block_size);
                            if !self.allocator.can_allocate(blocks_needed) {
                                continue;
                            }
                            (self.allocator.allocate(blocks_needed)?, 0)
                        }
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
                    seq_mut.state = SequenceState::Running;
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

        for seq_id in &scheduled_waiting {
            self.running_set.push(*seq_id);
            self.stats.waiting_count -= 1;
            self.stats.running_count += 1;
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
                if !self.allocator.can_allocate(blocks_needed) {
                    continue;
                }

                let seq_tokens = seq.total_tokens;
                let blocks = self.allocator.allocate(blocks_needed)?;

                if let Some(seq_mut) = self.sequences.get_mut(&seq_id) {
                    seq_mut.block_table.blocks = blocks.clone();
                    seq_mut.block_table.num_tokens = seq_mut.total_tokens;
                    seq_mut.state = SequenceState::Running;
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
            self.stats.preempted_count -= 1;
            self.stats.running_count += 1;
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
