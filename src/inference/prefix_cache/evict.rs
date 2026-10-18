//! Allocation that evicts idle registered blocks, least recently used first.
//!
//! Eviction only takes a block with a reference count of 0. A block a
//! sequence still references is never evicted.

use crate::error::Result;
use crate::inference::memory::{BlockAllocator, BlockId};

use super::core::PrefixCache;

impl<A: BlockAllocator> PrefixCache<A> {
    /// Allocate `count` private blocks for a sequence, such as decode blocks.
    ///
    /// Idle registered blocks are evicted when the pool is short. The caller
    /// owns the result and returns it through [`Self::release_blocks`].
    ///
    /// # Errors
    ///
    /// An error when free plus idle blocks are fewer than `count`.
    pub fn allocate(&mut self, count: usize) -> Result<Vec<BlockId>> {
        self.reclaim(count)?;
        self.allocator.allocate(count)
    }

    /// Evict idle blocks until `count` blocks are free or none is idle.
    pub(super) fn reclaim(&mut self, count: usize) -> Result<()> {
        while !self.allocator.can_allocate(count) && self.evict_one()? {}
        Ok(())
    }

    /// Evict idle blocks so `incoming` registrations fit `max_cached_blocks`.
    ///
    /// Returns how many of them fit. A cap of 0 means no cap.
    pub(super) fn make_room(&mut self, incoming: usize) -> Result<usize> {
        let cap = self.config.max_cached_blocks;
        if cap == 0 {
            return Ok(incoming);
        }
        while self.hash_to_block.len() + incoming > cap && self.evict_one()? {}
        Ok(cap.saturating_sub(self.hash_to_block.len()).min(incoming))
    }

    /// Evict the least recently used idle block. False when none is idle.
    fn evict_one(&mut self) -> Result<bool> {
        if self.idle_blocks == 0 {
            return Ok(false);
        }
        let idle_pos = self
            .lru_queue
            .iter()
            .position(|&id| self.block_ref_count(id) == Some(0));
        let Some(pos) = idle_pos else {
            return Ok(false);
        };
        let Some(&block_id) = self.lru_queue.get(pos) else {
            return Ok(false);
        };
        self.allocator.free(&[block_id])?;
        self.lru_queue.remove(pos);
        if let Some(hash) = self.block_to_hash.remove(&block_id) {
            self.hash_to_block.remove(&hash);
        }
        self.idle_blocks = self.idle_blocks.saturating_sub(1);
        self.stats.blocks_evicted += 1;
        self.stats.cached_blocks = self.hash_to_block.len();
        Ok(true)
    }

    /// Move a registered block to the most recently used end.
    pub(super) fn touch_lru(&mut self, block_id: BlockId) {
        self.lru_queue.retain(|&id| id != block_id);
        self.lru_queue.push_back(block_id);
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::PrefixCacheConfig;
    use super::*;
    use crate::inference::memory::CpuBlockAllocator;

    fn cache(total: usize, cap: usize) -> PrefixCache<CpuBlockAllocator> {
        let config = PrefixCacheConfig {
            enabled: true,
            max_cached_blocks: cap,
            min_prefix_tokens: 1,
            block_size: 4,
        };
        PrefixCache::new(CpuBlockAllocator::new(total, 4), config)
    }

    #[test]
    fn idle_blocks_are_evicted_to_the_allocator() {
        let mut c = cache(4, 0);
        let prompt: Vec<u32> = (0..9).collect();
        let blocks = c
            .get_or_allocate_blocks(1, &prompt)
            .expect("lookup")
            .into_blocks();
        c.release_blocks(1, &blocks).expect("release");
        assert_eq!(c.idle_block_count(), 2);
        assert_eq!(c.allocator().free_block_count(), 2);

        let all = c.allocate(4).expect("eviction frees the idle blocks");
        assert_eq!(all.len(), 4);
        assert_eq!(c.registered_block_count(), 0);
        assert_eq!(c.idle_block_count(), 0);
        assert_eq!(c.stats().blocks_evicted, 2);
    }

    #[test]
    fn a_referenced_block_is_never_evicted() {
        let mut c = cache(4, 0);
        let prompt: Vec<u32> = (0..9).collect();
        let blocks = c
            .get_or_allocate_blocks(1, &prompt)
            .expect("lookup")
            .into_blocks();
        assert!(c.allocate(2).is_err(), "one block is free and none is idle");
        assert_eq!(c.registered_block_count(), 2);
        assert_eq!(c.block_ref_count(blocks[0]), Some(1));
        assert_eq!(c.block_ref_count(blocks[1]), Some(1));
        assert_eq!(c.stats().blocks_evicted, 0);
    }

    #[test]
    fn eviction_takes_the_least_recently_used_idle_block() {
        let mut c = cache(4, 0);
        let first: Vec<u32> = (0..5).collect();
        let second: Vec<u32> = (100..105).collect();
        let a = c
            .get_or_allocate_blocks(1, &first)
            .expect("a")
            .into_blocks();
        let b = c
            .get_or_allocate_blocks(2, &second)
            .expect("b")
            .into_blocks();
        c.release_blocks(1, &a).expect("release a");
        c.release_blocks(2, &b).expect("release b");
        // A hit on `first` makes it the most recently used block.
        let again = c
            .get_or_allocate_blocks(3, &first)
            .expect("again")
            .into_blocks();
        c.release_blocks(3, &again).expect("release again");

        let taken = c.allocate(3).expect("one eviction");
        assert_eq!(c.stats().blocks_evicted, 1);
        assert_eq!(c.cached_block_count(&first), 1);
        assert_eq!(c.cached_block_count(&second), 0);
        c.release_blocks(9, &taken).expect("release taken");
    }

    #[test]
    fn the_cap_bounds_registrations_and_the_rest_stay_private() {
        let mut c = cache(16, 2);
        let prompt: Vec<u32> = (0..13).collect();
        let blocks = c
            .get_or_allocate_blocks(1, &prompt)
            .expect("lookup")
            .into_blocks();
        assert_eq!(blocks.len(), 4);
        assert_eq!(c.registered_block_count(), 2);
        assert_eq!(
            c.block_ref_count(blocks[2]),
            None,
            "past the cap stays private"
        );

        c.release_blocks(1, &blocks).expect("release");
        assert_eq!(c.allocator().free_block_count(), 16 - 2);

        // A new prompt evicts the idle entries to make room under the cap.
        let other: Vec<u32> = (100..109).collect();
        c.get_or_allocate_blocks(2, &other).expect("other");
        assert_eq!(c.registered_block_count(), 2);
        assert_eq!(c.stats().blocks_evicted, 2);
    }
}
