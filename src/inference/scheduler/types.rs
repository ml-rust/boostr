//! Scheduler types and configuration

use crate::inference::memory::BlockTable;
use std::time::Instant;

/// Unique identifier for a sequence
pub type SequenceId = u64;

/// Scheduler configuration
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub max_batch_size: usize,
    pub max_batch_tokens: usize,
    pub max_seq_len: usize,
    pub block_size: usize,
    pub enable_preemption: bool,
    pub max_preempt_per_step: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 256,
            max_batch_tokens: 8192,
            max_seq_len: 4096,
            block_size: 16,
            enable_preemption: true,
            max_preempt_per_step: 4,
        }
    }
}

/// Sequence state in the scheduler
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceState {
    Waiting,
    Running,
    Preempted,
    Finished,
}

/// Priority level for scheduling
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum SchedulingPriority {
    Low = 0,
    #[default]
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// A sequence request to be processed
#[derive(Debug, Clone)]
pub struct SequenceRequest {
    pub id: SequenceId,
    pub prompt_tokens: Vec<u32>,
    pub max_new_tokens: usize,
    pub priority: SchedulingPriority,
    pub arrival_time: Instant,
}

impl SequenceRequest {
    pub fn new(id: SequenceId, prompt_tokens: Vec<u32>) -> Self {
        Self {
            id,
            prompt_tokens,
            max_new_tokens: 256,
            priority: SchedulingPriority::Normal,
            arrival_time: Instant::now(),
        }
    }

    pub fn with_max_tokens(mut self, max_new_tokens: usize) -> Self {
        self.max_new_tokens = max_new_tokens;
        self
    }

    pub fn with_priority(mut self, priority: SchedulingPriority) -> Self {
        self.priority = priority;
        self
    }
}

#[derive(Debug)]
pub(crate) struct SequenceData {
    pub request: SequenceRequest,
    pub state: SequenceState,
    pub generated_tokens: Vec<u32>,
    pub block_table: BlockTable,
    pub total_tokens: usize,
    pub prompt_len: usize,
}

impl SequenceData {
    pub(crate) fn new(request: SequenceRequest, block_size: usize) -> Self {
        let prompt_len = request.prompt_tokens.len();
        Self {
            request,
            state: SequenceState::Waiting,
            generated_tokens: Vec::new(),
            block_table: BlockTable::new(block_size),
            total_tokens: prompt_len,
            prompt_len,
        }
    }

    pub(crate) fn is_finished(&self) -> bool {
        self.generated_tokens.len() >= self.request.max_new_tokens
    }

    pub(crate) fn blocks_needed_for_next_token(&self) -> usize {
        self.block_table.additional_blocks_needed(1)
    }
}

/// A batch of sequences to process
#[derive(Debug)]
pub struct ScheduledBatch {
    pub prefill_sequences: Vec<SequenceId>,
    pub decode_sequences: Vec<SequenceId>,
    pub block_tables: std::collections::HashMap<SequenceId, Vec<crate::inference::memory::BlockId>>,
    pub preempted_sequences: Vec<SequenceId>,
}

impl ScheduledBatch {
    pub fn is_empty(&self) -> bool {
        self.prefill_sequences.is_empty() && self.decode_sequences.is_empty()
    }

    pub fn len(&self) -> usize {
        self.prefill_sequences.len() + self.decode_sequences.len()
    }

    pub fn all_sequences(&self) -> Vec<SequenceId> {
        let mut all = self.prefill_sequences.clone();
        all.extend(&self.decode_sequences);
        all
    }
}

/// Scheduler statistics
#[derive(Debug, Clone, Copy, Default)]
pub struct SchedulerStats {
    pub total_requests: usize,
    pub waiting_count: usize,
    pub running_count: usize,
    pub preempted_count: usize,
    pub finished_count: usize,
    pub total_tokens_generated: usize,
    pub total_preemptions: usize,
}
