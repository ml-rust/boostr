//! Prefix cache core implementation

use crate::error::Result;
use crate::inference::memory::{BlockAllocator, BlockId};
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use super::types::{
    BlockHash, CacheResult, CachedBlockInfo, PrefixCacheConfig, PrefixCacheStats, SequenceId,
};

/// Prefix cache for KV cache block reuse
pub struct PrefixCache<A: BlockAllocator> {
    allocator: A,
    config: PrefixCacheConfig,
    hash_to_block: HashMap<BlockHash, CachedBlockInfo>,
    block_to_hash: HashMap<BlockId, BlockHash>,
    lru_queue: VecDeque<BlockId>,
    stats: PrefixCacheStats,
}

impl<A: BlockAllocator> PrefixCache<A> {
    pub fn new(allocator: A, config: PrefixCacheConfig) -> Self {
        Self {
            allocator,
            config,
            hash_to_block: HashMap::new(),
            block_to_hash: HashMap::new(),
            lru_queue: VecDeque::new(),
            stats: PrefixCacheStats::default(),
        }
    }

    fn compute_hash(tokens: &[u32]) -> BlockHash {
        const FNV_OFFSET: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x100000001b3;

        let mut hash = FNV_OFFSET;
        for &token in tokens {
            let bytes = token.to_le_bytes();
            for byte in bytes {
                hash ^= byte as u64;
                hash = hash.wrapping_mul(FNV_PRIME);
            }
        }
        hash
    }

    fn compute_block_hashes(&self, tokens: &[u32]) -> Vec<BlockHash> {
        let block_size = self.config.block_size;
        let num_blocks = tokens.len().div_ceil(block_size);

        let mut hashes = Vec::with_capacity(num_blocks);
        for block_idx in 0..num_blocks {
            let end = ((block_idx + 1) * block_size).min(tokens.len());
            let prefix = &tokens[..end];
            hashes.push(Self::compute_hash(prefix));
        }
        hashes
    }

    pub fn get_or_allocate_blocks(
        &mut self,
        seq_id: SequenceId,
        tokens: &[u32],
    ) -> Result<CacheResult> {
        if !self.config.enabled || tokens.len() < self.config.min_prefix_tokens {
            let num_blocks = tokens.len().div_ceil(self.config.block_size);
            let blocks = self.allocator.allocate(num_blocks)?;
            self.stats.blocks_allocated += blocks.len();
            self.stats.cache_misses += 1;
            self.stats.total_lookups += 1;
            self.update_rates();
            return Ok(CacheResult::Miss { blocks });
        }

        let block_hashes = self.compute_block_hashes(tokens);
        let mut result_blocks = Vec::with_capacity(block_hashes.len());
        let mut cached_count = 0;
        let mut new_count = 0;

        self.stats.total_lookups += 1;

        for (block_idx, &hash) in block_hashes.iter().enumerate() {
            let block_start = block_idx * self.config.block_size;
            let block_end = ((block_idx + 1) * self.config.block_size).min(tokens.len());
            let block_tokens = &tokens[block_start..block_end];

            let cache_hit = if let Some(cached) = self.hash_to_block.get(&hash) {
                if cached.tokens == block_tokens {
                    Some(cached.block_id)
                } else {
                    None
                }
            } else {
                None
            };

            if let Some(block_id) = cache_hit {
                if let Some(cached) = self.hash_to_block.get_mut(&hash) {
                    cached.ref_count += 1;
                    cached.owners.push(seq_id);
                    cached.last_access = Instant::now();
                }
                result_blocks.push(block_id);
                cached_count += 1;
                self.touch_lru(block_id);
                continue;
            }

            let new_blocks = self.allocate_with_eviction(1)?;
            let block_id = new_blocks[0];

            let info = CachedBlockInfo {
                block_id,
                ref_count: 1,
                owners: vec![seq_id],
                last_access: Instant::now(),
                tokens: block_tokens.to_vec(),
            };

            self.hash_to_block.insert(hash, info);
            self.block_to_hash.insert(block_id, hash);
            self.lru_queue.push_back(block_id);

            result_blocks.push(block_id);
            new_count += 1;
        }

        self.stats.blocks_from_cache += cached_count;
        self.stats.blocks_allocated += new_count;
        self.stats.cached_blocks = self.hash_to_block.len();

        if cached_count > 0 {
            self.stats.cache_hits += 1;
            self.update_rates();
            Ok(CacheResult::Hit {
                blocks: result_blocks,
                cached_blocks: cached_count,
                new_blocks: new_count,
            })
        } else {
            self.stats.cache_misses += 1;
            self.update_rates();
            Ok(CacheResult::Miss {
                blocks: result_blocks,
            })
        }
    }

    fn allocate_with_eviction(&mut self, count: usize) -> Result<Vec<BlockId>> {
        if self.allocator.can_allocate(count) {
            return self.allocator.allocate(count);
        }

        let mut evicted = 0;
        while !self.allocator.can_allocate(count) && !self.lru_queue.is_empty() {
            if let Some(block_id) = self.lru_queue.pop_front() {
                if let Some(&hash) = self.block_to_hash.get(&block_id) {
                    if let Some(info) = self.hash_to_block.get(&hash) {
                        if info.ref_count == 0 {
                            self.allocator.free(&[block_id])?;
                            self.hash_to_block.remove(&hash);
                            self.block_to_hash.remove(&block_id);
                            evicted += 1;
                            self.stats.blocks_evicted += 1;
                            continue;
                        }
                    }
                }
                self.lru_queue.push_back(block_id);
            }

            if evicted == 0 && self.lru_queue.len() == self.hash_to_block.len() {
                break;
            }
        }

        self.stats.cached_blocks = self.hash_to_block.len();
        self.allocator.allocate(count)
    }

    fn touch_lru(&mut self, block_id: BlockId) {
        self.lru_queue.retain(|&id| id != block_id);
        self.lru_queue.push_back(block_id);
    }

    pub fn release_blocks(&mut self, seq_id: SequenceId, blocks: &[BlockId]) -> Result<()> {
        for &block_id in blocks {
            if let Some(&hash) = self.block_to_hash.get(&block_id) {
                if let Some(info) = self.hash_to_block.get_mut(&hash) {
                    info.ref_count = info.ref_count.saturating_sub(1);
                    info.owners.retain(|&id| id != seq_id);
                }
            }
        }
        Ok(())
    }

    pub fn force_free_blocks(&mut self, blocks: &[BlockId]) -> Result<()> {
        for &block_id in blocks {
            if let Some(hash) = self.block_to_hash.remove(&block_id) {
                self.hash_to_block.remove(&hash);
                self.lru_queue.retain(|&id| id != block_id);
            }
            self.allocator.free(&[block_id])?;
        }
        self.stats.cached_blocks = self.hash_to_block.len();
        Ok(())
    }

    pub fn has_cached_prefix(&self, tokens: &[u32]) -> bool {
        if !self.config.enabled || tokens.len() < self.config.min_prefix_tokens {
            return false;
        }

        let block_hashes = self.compute_block_hashes(tokens);
        block_hashes
            .first()
            .map(|hash| self.hash_to_block.contains_key(hash))
            .unwrap_or(false)
    }

    pub fn cached_block_count(&self, tokens: &[u32]) -> usize {
        if !self.config.enabled {
            return 0;
        }

        let block_hashes = self.compute_block_hashes(tokens);
        block_hashes
            .iter()
            .take_while(|hash| self.hash_to_block.contains_key(hash))
            .count()
    }

    pub fn stats(&self) -> PrefixCacheStats {
        self.stats
    }

    fn update_rates(&mut self) {
        if self.stats.total_lookups > 0 {
            self.stats.hit_rate = self.stats.cache_hits as f64 / self.stats.total_lookups as f64;
        }

        let total_blocks = self.stats.blocks_from_cache + self.stats.blocks_allocated;
        if total_blocks > 0 {
            self.stats.block_reuse_rate = self.stats.blocks_from_cache as f64 / total_blocks as f64;
        }
    }

    pub fn reset(&mut self) -> Result<()> {
        let blocks: Vec<BlockId> = self.block_to_hash.keys().copied().collect();
        for block_id in blocks {
            if let Some(hash) = self.block_to_hash.remove(&block_id) {
                if let Some(info) = self.hash_to_block.remove(&hash) {
                    if info.ref_count == 0 {
                        self.allocator.free(&[block_id])?;
                    }
                }
            }
        }

        self.lru_queue.clear();
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

// Public API tests live in tests/inference/prefix_cache.rs
#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::memory::CpuBlockAllocator;

    #[test]
    fn test_compute_hash() {
        let hash1 = PrefixCache::<CpuBlockAllocator>::compute_hash(&[1, 2, 3, 4]);
        let hash2 = PrefixCache::<CpuBlockAllocator>::compute_hash(&[1, 2, 3, 4]);
        let hash3 = PrefixCache::<CpuBlockAllocator>::compute_hash(&[1, 2, 3, 5]);

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }
}
