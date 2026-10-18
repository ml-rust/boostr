//! Prefix cache state, block ownership rules, hashing and read accessors.
//!
//! # Block ownership
//!
//! Three parties own the blocks of one pool, and each block has one owner:
//!
//! - **Allocator** — owns every free block.
//! - **Prefix cache** — owns every registered block: a full prompt block with
//!   a prefix hash and a reference count. Sequences share it read-only.
//!   An idle registered block (count 0) stays cached until eviction frees it.
//!   A block that [`PrefixCache::reset`] detaches while in use stays shared
//!   until its last release frees it.
//! - **Sequence** — owns its private blocks: the partial last prompt block,
//!   decode blocks, and every block of a lookup that skips the cache.
//!
//! [`PrefixCache::release_blocks`] takes a sequence's whole block table. It
//! drops one reference per registered block and frees every private block.

use crate::error::Result;
use crate::inference::memory::{BlockAllocator, BlockId};
use std::collections::{HashMap, VecDeque};

use super::types::{BlockHash, CachedBlockInfo, PrefixCacheConfig, PrefixCacheStats};

const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

/// Prefix cache for KV cache block reuse.
///
/// The cache shares only full prompt blocks. Decode never writes into one,
/// because decode positions start after the prompt.
pub struct PrefixCache<A: BlockAllocator> {
    pub(super) allocator: A,
    pub(super) config: PrefixCacheConfig,
    pub(super) hash_to_block: HashMap<BlockHash, CachedBlockInfo>,
    pub(super) block_to_hash: HashMap<BlockId, BlockHash>,
    /// Every registered block, least recently used first.
    pub(super) lru_queue: VecDeque<BlockId>,
    /// Registered blocks with a reference count of 0.
    pub(super) idle_blocks: usize,
    /// In-use blocks that [`Self::reset`] unregistered. The last release frees each.
    pub(super) detached: HashMap<BlockId, CachedBlockInfo>,
    pub(super) stats: PrefixCacheStats,
}

impl<A: BlockAllocator> PrefixCache<A> {
    /// A cache over `allocator`. A scheduler that uses the cache must hold a
    /// clone of the same allocator, so both see one pool.
    pub fn new(allocator: A, config: PrefixCacheConfig) -> Self {
        Self {
            allocator,
            config,
            hash_to_block: HashMap::new(),
            block_to_hash: HashMap::new(),
            lru_queue: VecDeque::new(),
            idle_blocks: 0,
            detached: HashMap::new(),
            stats: PrefixCacheStats::default(),
        }
    }

    /// FNV-1a hash of `tokens`.
    #[cfg(test)]
    pub(super) fn compute_hash(tokens: &[u32]) -> BlockHash {
        Self::extend_hash(FNV_OFFSET, tokens)
    }

    fn extend_hash(mut hash: BlockHash, tokens: &[u32]) -> BlockHash {
        for &token in tokens {
            for byte in token.to_le_bytes() {
                hash ^= byte as u64;
                hash = hash.wrapping_mul(FNV_PRIME);
            }
        }
        hash
    }

    /// Prefix hashes of the first `count` full blocks of `tokens`.
    ///
    /// Entry `i` hashes `tokens[..(i + 1) * block_size]`, so a hit on block
    /// `i` matches the whole prefix up to it.
    pub(super) fn prefix_hashes(&self, tokens: &[u32], count: usize) -> Vec<BlockHash> {
        let block_size = self.config.block_size;
        let mut hashes = Vec::with_capacity(count);
        if block_size == 0 {
            return hashes;
        }
        let mut hash = FNV_OFFSET;
        for chunk in tokens.chunks_exact(block_size).take(count) {
            hash = Self::extend_hash(hash, chunk);
            hashes.push(hash);
        }
        hashes
    }

    pub fn stats(&self) -> PrefixCacheStats {
        self.stats
    }

    /// Registered blocks, in use or idle.
    pub fn registered_block_count(&self) -> usize {
        self.hash_to_block.len()
    }

    /// Registered blocks that no sequence references. Eviction can free them.
    pub fn idle_block_count(&self) -> usize {
        self.idle_blocks
    }

    /// The reference count of a shared block, or `None` for a private or free block.
    ///
    /// A shared block is registered, or detached by [`Self::reset`] while in use.
    pub fn block_ref_count(&self, block_id: BlockId) -> Option<usize> {
        self.block_to_hash
            .get(&block_id)
            .and_then(|hash| self.hash_to_block.get(hash))
            .or_else(|| self.detached.get(&block_id))
            .map(|info| info.ref_count)
    }

    /// Blocks an allocation can get: free blocks plus idle registered blocks.
    pub fn available_blocks(&self) -> usize {
        self.allocator
            .free_block_count()
            .saturating_add(self.idle_blocks)
    }

    pub(super) fn update_rates(&mut self) {
        if self.stats.total_lookups > 0 {
            self.stats.hit_rate = self.stats.cache_hits as f64 / self.stats.total_lookups as f64;
        }

        let total_blocks = self.stats.blocks_from_cache + self.stats.blocks_allocated;
        if total_blocks > 0 {
            self.stats.block_reuse_rate = self.stats.blocks_from_cache as f64 / total_blocks as f64;
        }
    }

    /// Drop every registration and reset the stats.
    ///
    /// Idle blocks return to the allocator. A block still in use leaves the
    /// lookup but keeps its references. The last holder's release frees it.
    pub fn reset(&mut self) -> Result<()> {
        let idle: Vec<BlockId> = self
            .hash_to_block
            .values()
            .filter(|info| info.ref_count == 0)
            .map(|info| info.block_id)
            .collect();
        self.allocator.free(&idle)?;

        for (_, info) in self.hash_to_block.drain() {
            if info.ref_count > 0 {
                self.detached.insert(info.block_id, info);
            }
        }
        self.block_to_hash.clear();
        self.lru_queue.clear();
        self.idle_blocks = 0;
        self.stats = PrefixCacheStats::default();
        Ok(())
    }

    pub fn allocator(&self) -> &A {
        &self.allocator
    }

    pub fn allocator_mut(&mut self) -> &mut A {
        &mut self.allocator
    }

    pub fn config(&self) -> &PrefixCacheConfig {
        &self.config
    }
}

// Public API tests live in tests/prefix_cache.rs
#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::memory::CpuBlockAllocator;

    fn cache(total: usize) -> PrefixCache<CpuBlockAllocator> {
        let config = PrefixCacheConfig {
            enabled: true,
            max_cached_blocks: 0,
            min_prefix_tokens: 1,
            block_size: 4,
        };
        PrefixCache::new(CpuBlockAllocator::new(total, 4), config)
    }

    #[test]
    fn test_compute_hash() {
        let hash1 = PrefixCache::<CpuBlockAllocator>::compute_hash(&[1, 2, 3, 4]);
        let hash2 = PrefixCache::<CpuBlockAllocator>::compute_hash(&[1, 2, 3, 4]);
        let hash3 = PrefixCache::<CpuBlockAllocator>::compute_hash(&[1, 2, 3, 5]);

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn prefix_hashes_match_whole_prefix_hashes() {
        let c = cache(8);
        let tokens: Vec<u32> = (0..10).collect();
        let hashes = c.prefix_hashes(&tokens, 2);
        assert_eq!(hashes.len(), 2);
        assert_eq!(
            hashes[0],
            PrefixCache::<CpuBlockAllocator>::compute_hash(&tokens[..4])
        );
        assert_eq!(
            hashes[1],
            PrefixCache::<CpuBlockAllocator>::compute_hash(&tokens[..8])
        );
    }

    #[test]
    fn reset_frees_idle_blocks_and_detaches_held_ones() {
        let mut c = cache(8);
        let held: Vec<u32> = (0..9).collect();
        let idle: Vec<u32> = (100..109).collect();
        let held_blocks = c
            .get_or_allocate_blocks(1, &held)
            .expect("held")
            .into_blocks();
        let idle_blocks = c
            .get_or_allocate_blocks(2, &idle)
            .expect("idle")
            .into_blocks();
        c.release_blocks(2, &idle_blocks).expect("release idle");
        // Each prompt holds two registered full blocks and one private block.
        assert_eq!(c.allocator().free_block_count(), 8 - 3 - 2);
        assert_eq!(c.idle_block_count(), 2);

        c.reset().expect("reset");
        assert_eq!(c.registered_block_count(), 0);
        assert_eq!(c.allocator().free_block_count(), 8 - 3);
        assert_eq!(c.cached_block_count(&held), 0, "a reset block never hits");
        assert_eq!(c.block_ref_count(held_blocks[0]), Some(1));

        // The last release frees the detached and the private blocks.
        c.release_blocks(1, &held_blocks).expect("release held");
        assert_eq!(c.allocator().free_block_count(), 8);
        assert_eq!(c.block_ref_count(held_blocks[0]), None);
    }
}
