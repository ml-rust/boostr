//! Prefix cache types and configuration

pub use crate::inference::scheduler::SequenceId;

use crate::inference::memory::BlockId;
use std::time::Instant;

/// 64-bit hash of a token sequence (block content)
pub type BlockHash = u64;

/// Configuration for prefix caching
#[derive(Debug, Clone)]
pub struct PrefixCacheConfig {
    pub enabled: bool,
    pub max_cached_blocks: usize,
    pub min_prefix_tokens: usize,
    pub block_size: usize,
}

impl Default for PrefixCacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_cached_blocks: 10000,
            min_prefix_tokens: 16,
            block_size: 16,
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct CachedBlockInfo {
    pub(super) block_id: BlockId,
    pub(super) ref_count: usize,
    pub(super) owners: Vec<SequenceId>,
    pub(super) last_access: Instant,
    pub(super) tokens: Vec<u32>,
}

/// Result of a cache lookup/allocation
#[derive(Debug)]
pub enum CacheResult {
    Hit {
        blocks: Vec<BlockId>,
        cached_blocks: usize,
        new_blocks: usize,
    },
    Miss {
        blocks: Vec<BlockId>,
    },
}

impl CacheResult {
    pub fn blocks(&self) -> &[BlockId] {
        match self {
            CacheResult::Hit { blocks, .. } => blocks,
            CacheResult::Miss { blocks } => blocks,
        }
    }

    pub fn is_hit(&self) -> bool {
        matches!(self, CacheResult::Hit { cached_blocks, .. } if *cached_blocks > 0)
    }

    pub fn cached_count(&self) -> usize {
        match self {
            CacheResult::Hit { cached_blocks, .. } => *cached_blocks,
            CacheResult::Miss { .. } => 0,
        }
    }
}

/// Statistics for prefix cache
#[derive(Debug, Clone, Copy, Default)]
pub struct PrefixCacheStats {
    pub total_lookups: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub blocks_from_cache: usize,
    pub blocks_allocated: usize,
    pub cached_blocks: usize,
    pub blocks_evicted: usize,
    pub hit_rate: f64,
    pub block_reuse_rate: f64,
}

/// Per-sequence prefix cache state
#[derive(Debug, Clone, Default)]
pub struct SequencePrefixState {
    pub blocks: Vec<BlockId>,
    pub cached_blocks: Vec<bool>,
    pub tokens_covered: usize,
}

impl SequencePrefixState {
    pub fn from_cache_result(result: &CacheResult, total_tokens: usize) -> Self {
        match result {
            CacheResult::Hit {
                blocks,
                cached_blocks,
                ..
            } => {
                let mut cached_flags = vec![true; *cached_blocks];
                cached_flags.resize(blocks.len(), false);
                Self {
                    blocks: blocks.clone(),
                    cached_blocks: cached_flags,
                    tokens_covered: total_tokens,
                }
            }
            CacheResult::Miss { blocks } => Self {
                blocks: blocks.clone(),
                cached_blocks: vec![false; blocks.len()],
                tokens_covered: total_tokens,
            },
        }
    }

    pub fn cached_count(&self) -> usize {
        self.cached_blocks.iter().filter(|&&c| c).count()
    }

    pub fn new_count(&self) -> usize {
        self.cached_blocks.iter().filter(|&&c| !c).count()
    }
}
