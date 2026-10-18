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
    ///
    /// The cache must wrap a clone of the scheduler's allocator, so both see
    /// one pool. With a cache, every block goes out and comes back through it.
    /// A finished sequence's shared prefix blocks stay cached, and every other
    /// block returns to the allocator.
    pub fn with_prefix_cache(mut self, cache: PrefixCache<A>) -> Self {
        self.prefix_cache = Some(cache);
        self
    }

    pub fn prefix_cache(&self) -> Option<&PrefixCache<A>> {
        self.prefix_cache.as_ref()
    }

    pub fn prefix_cache_mut(&mut self) -> Option<&mut PrefixCache<A>> {
        self.prefix_cache.as_mut()
    }

    pub fn add_request(&mut self, request: SequenceRequest) -> Result<()> {
        let id = request.id;

        if self.sequences.contains_key(&id) {
            return Err(Error::SchedulerError {
                reason: format!("Sequence {} already exists", id),
            });
        }

        // An empty prompt costs no batch tokens, so it overfills the
        // running set past max_batch_tokens. Prefill also needs one token.
        if request.prompt_tokens.is_empty() {
            return Err(Error::SchedulerError {
                reason: format!(
                    "Sequence {id} has an empty prompt. Submit at least one prompt token"
                ),
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

        if let Some(cache) = &self.prefix_cache
            && cache.config().block_size != self.config.block_size
        {
            return Err(Error::SchedulerError {
                reason: format!(
                    "prefix cache block_size {} differs from scheduler block_size {}. \
                     Build both from one block size",
                    cache.config().block_size,
                    self.config.block_size
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

    /// Release every block of a sequence's table.
    ///
    /// With a prefix cache, shared prefix blocks lose one reference and every
    /// private block returns to the allocator. Without one, all blocks return.
    pub(super) fn release_blocks(
        prefix_cache: &mut Option<PrefixCache<A>>,
        allocator: &A,
        seq_id: SequenceId,
        blocks: &[BlockId],
    ) -> Result<()> {
        match prefix_cache {
            Some(cache) => cache.release_blocks(seq_id, blocks),
            None => allocator.free(blocks),
        }
    }

    /// True when `count` blocks can be allocated, counting idle cached blocks.
    pub(super) fn can_allocate_blocks(
        prefix_cache: &Option<PrefixCache<A>>,
        allocator: &A,
        count: usize,
    ) -> bool {
        match prefix_cache {
            Some(cache) => cache.available_blocks() >= count,
            None => allocator.can_allocate(count),
        }
    }

    /// Allocate `count` private blocks, evicting idle cached blocks when short.
    pub(super) fn allocate_blocks(
        prefix_cache: &mut Option<PrefixCache<A>>,
        allocator: &A,
        count: usize,
    ) -> Result<Vec<BlockId>> {
        match prefix_cache {
            Some(cache) => cache.allocate(count),
            None => allocator.allocate(count),
        }
    }
}

#[cfg(test)]
impl<A: BlockAllocator> SequenceScheduler<A> {
    /// Check the block and queue accounting. The error names the first broken rule.
    pub(super) fn check_block_invariants(&self) -> std::result::Result<(), String> {
        let mut holders: HashMap<BlockId, usize> = HashMap::new();
        let mut running = std::collections::BTreeSet::new();
        for (&id, seq) in &self.sequences {
            if seq.state == SequenceState::Running {
                running.insert(id);
                let cached = seq.cached_token_count;
                if cached >= seq.prompt_len || cached % self.config.block_size != 0 {
                    return Err(format!(
                        "sequence {id}: {cached} cached of {}",
                        seq.prompt_len
                    ));
                }
            } else if !seq.block_table.blocks.is_empty() {
                return Err(format!("sequence {id} in {:?} holds blocks", seq.state));
            }
            let mut own = std::collections::HashSet::new();
            for &block in &seq.block_table.blocks {
                if !own.insert(block) {
                    return Err(format!("sequence {id} lists block {block} twice"));
                }
                *holders.entry(block).or_default() += 1;
            }
        }

        for (&block, &count) in &holders {
            let shared = self
                .prefix_cache
                .as_ref()
                .and_then(|c| c.block_ref_count(block));
            match shared {
                Some(refs) if refs != count => {
                    return Err(format!("block {block}: {refs} references, {count} tables"));
                }
                None if count != 1 => {
                    return Err(format!("private block {block} is in {count} tables"));
                }
                _ => {}
            }
        }

        let idle = self
            .prefix_cache
            .as_ref()
            .map_or(0, |c| c.idle_block_count());
        let free = self.allocator.free_block_count();
        let total = self.allocator.total_blocks();
        if free + holders.len() + idle != total {
            return Err(format!(
                "free {free} + held {} + idle {idle} != total {total}",
                holders.len()
            ));
        }

        let listed: std::collections::BTreeSet<_> = self.running_set.iter().copied().collect();
        if listed.len() != self.running_set.len() || listed != running {
            return Err(format!(
                "running set {:?} != running {running:?}",
                self.running_set
            ));
        }
        if self.stats.running_count != running.len()
            || self.stats.waiting_count != self.waiting_queue.len()
            || self.stats.preempted_count != self.preempted_set.len()
        {
            return Err(format!("stats {:?} disagree with the queues", self.stats));
        }
        Ok(())
    }
}
