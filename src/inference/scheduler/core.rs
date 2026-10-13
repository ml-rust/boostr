//! Continuous batching sequence scheduler: the scheduler state, request
//! admission, and read accessors.

use crate::error::{Error, Result};
use crate::inference::memory::{BlockAllocator, BlockId, BlockTable};
use crate::inference::prefix_cache::PrefixCache;
use std::collections::{HashMap, VecDeque};

use super::types::{
    SchedulerConfig, SchedulerStats, SequenceData, SequenceId, SequenceRequest, SequenceState,
};

/// Continuous batching sequence scheduler
pub struct SequenceScheduler<A: BlockAllocator> {
    pub(super) allocator: A,
    pub(super) config: SchedulerConfig,
    pub(super) sequences: HashMap<SequenceId, SequenceData>,
    pub(super) waiting_queue: VecDeque<SequenceId>,
    pub(super) running_set: Vec<SequenceId>,
    pub(super) preempted_set: Vec<SequenceId>,
    pub(super) stats: SchedulerStats,
    /// Optional prefix cache for KV block reuse across sequences
    pub(super) prefix_cache: Option<PrefixCache<A>>,
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
            prefix_cache: None,
        }
    }

    /// Enable prefix caching with the given cache instance.
    /// The prefix cache should use a clone of the same allocator.
    pub fn with_prefix_cache(mut self, cache: PrefixCache<A>) -> Self {
        self.prefix_cache = Some(cache);
        self
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

    pub fn stats(&self) -> SchedulerStats {
        self.stats
    }

    pub fn block_stats(&self) -> crate::inference::memory::BlockAllocatorStats {
        self.allocator.stats()
    }

    pub fn get_sequence_state(&self, seq_id: SequenceId) -> Option<SequenceState> {
        self.sequences.get(&seq_id).map(|s| s.state)
    }

    /// Get the number of cached tokens for a sequence (prefix cache hit).
    pub fn get_cached_token_count(&self, seq_id: SequenceId) -> usize {
        self.sequences
            .get(&seq_id)
            .map(|s| s.cached_token_count)
            .unwrap_or(0)
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

    /// Release physical blocks for a sequence, delegating to the prefix cache
    /// when present or falling back to the block allocator directly.
    pub(super) fn release_blocks(
        prefix_cache: &mut Option<PrefixCache<A>>,
        allocator: &A,
        seq_id: SequenceId,
        blocks: &[BlockId],
    ) -> Result<()> {
        if let Some(cache) = prefix_cache {
            let _ = cache.release_blocks(seq_id, blocks);
        } else {
            allocator.free(blocks)?;
        }
        Ok(())
    }
}
