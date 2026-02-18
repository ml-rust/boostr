//! Continuous batching sequence scheduler

use crate::error::{Error, Result};
use crate::inference::memory::{BlockAllocator, BlockTable};
use std::collections::{HashMap, VecDeque};

use super::types::{
    ScheduledBatch, SchedulerConfig, SchedulerStats, SequenceData, SequenceId, SequenceRequest,
    SequenceState,
};

/// Continuous batching sequence scheduler
pub struct SequenceScheduler<A: BlockAllocator> {
    allocator: A,
    config: SchedulerConfig,
    sequences: HashMap<SequenceId, SequenceData>,
    waiting_queue: VecDeque<SequenceId>,
    running_set: Vec<SequenceId>,
    preempted_set: Vec<SequenceId>,
    stats: SchedulerStats,
}

impl<A: BlockAllocator> SequenceScheduler<A> {
    pub fn new(allocator: A, config: SchedulerConfig) -> Self {
        Self {
            allocator,
            config,
            sequences: HashMap::new(),
            waiting_queue: VecDeque::new(),
            running_set: Vec::new(),
            preempted_set: Vec::new(),
            stats: SchedulerStats::default(),
        }
    }

    pub fn add_request(&mut self, request: SequenceRequest) -> Result<()> {
        let id = request.id;

        if self.sequences.contains_key(&id) {
            return Err(Error::SchedulerError {
                reason: format!("Sequence {} already exists", id),
            });
        }

        if request.prompt_tokens.len() > self.config.max_seq_len {
            return Err(Error::SchedulerError {
                reason: format!(
                    "Prompt length {} exceeds max_seq_len {}",
                    request.prompt_tokens.len(),
                    self.config.max_seq_len
                ),
            });
        }

        let seq_data = SequenceData::new(request, self.config.block_size);
        self.sequences.insert(id, seq_data);
        self.waiting_queue.push_back(id);
        self.stats.total_requests += 1;
        self.stats.waiting_count += 1;

        Ok(())
    }

    pub fn schedule(&mut self) -> Result<Option<ScheduledBatch>> {
        let mut batch = ScheduledBatch {
            prefill_sequences: Vec::new(),
            decode_sequences: Vec::new(),
            block_tables: HashMap::new(),
            preempted_sequences: Vec::new(),
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
            if batch.len() >= self.config.max_batch_size {
                break;
            }

            if let Some(seq) = self.sequences.get(&seq_id) {
                let prompt_tokens = seq.prompt_len;

                if batch_tokens + prompt_tokens > self.config.max_batch_tokens {
                    continue;
                }

                let blocks_needed =
                    BlockTable::blocks_needed(prompt_tokens, self.config.block_size);
                if !self.allocator.can_allocate(blocks_needed) {
                    continue;
                }

                let blocks = self.allocator.allocate(blocks_needed)?;

                if let Some(seq_mut) = self.sequences.get_mut(&seq_id) {
                    seq_mut.block_table.blocks = blocks.clone();
                    seq_mut.block_table.num_tokens = prompt_tokens;
                    seq_mut.state = SequenceState::Running;
                }

                batch.prefill_sequences.push(seq_id);
                batch.block_tables.insert(seq_id, blocks);
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
            if batch.len() >= self.config.max_batch_size {
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

            seq.generated_tokens.push(token);
            seq.total_tokens += 1;
            seq.block_table.num_tokens = seq.total_tokens;
            self.stats.total_tokens_generated += 1;

            let blocks_needed = seq.blocks_needed_for_next_token();
            if blocks_needed > 0 && self.allocator.can_allocate(blocks_needed) {
                let new_blocks = self.allocator.allocate(blocks_needed)?;
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
                self.allocator.free(&seq.block_table.blocks)?;
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
            if !seq.block_table.blocks.is_empty() {
                self.allocator.free(&seq.block_table.blocks)?;
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
                SequenceState::Finished => {}
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

    fn preempt_sequence(&mut self, seq_id: SequenceId) -> Result<()> {
        if let Some(seq) = self.sequences.get_mut(&seq_id) {
            if !seq.block_table.blocks.is_empty() {
                self.allocator.free(&seq.block_table.blocks)?;
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

    pub fn stats(&self) -> SchedulerStats {
        self.stats
    }

    pub fn get_sequence_state(&self, seq_id: SequenceId) -> Option<SequenceState> {
        self.sequences.get(&seq_id).map(|s| s.state)
    }

    pub fn get_generated_tokens(&self, seq_id: SequenceId) -> Option<&[u32]> {
        self.sequences
            .get(&seq_id)
            .map(|s| s.generated_tokens.as_slice())
    }

    pub fn get_block_table(&self, seq_id: SequenceId) -> Option<&BlockTable> {
        self.sequences.get(&seq_id).map(|s| &s.block_table)
    }

    pub fn has_work(&self) -> bool {
        !self.waiting_queue.is_empty()
            || !self.running_set.is_empty()
            || !self.preempted_set.is_empty()
    }

    pub fn cleanup_finished(&mut self) {
        self.sequences
            .retain(|_, seq| seq.state != SequenceState::Finished);
        self.stats.finished_count = 0;
    }

    pub fn allocator(&self) -> &A {
        &self.allocator
    }

    pub fn allocator_mut(&mut self) -> &mut A {
        &mut self.allocator
    }

    pub fn has_sequence(&self, seq_id: SequenceId) -> bool {
        self.sequences.contains_key(&seq_id)
    }
}
