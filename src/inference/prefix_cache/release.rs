//! Return a sequence's blocks: shared blocks lose a reference, private
//! blocks go back to the allocator.

use crate::error::{Error, Result};
use crate::inference::memory::{BlockAllocator, BlockId};
use std::collections::HashSet;

use super::core::PrefixCache;
use super::types::{BlockHash, CachedBlockInfo, SequenceId};

/// Where a shared block of a release lives.
#[derive(Clone, Copy)]
enum Shared {
    Registered(BlockHash),
    Detached(BlockId),
}

impl<A: BlockAllocator> PrefixCache<A> {
    /// Release every block `seq_id` holds.
    ///
    /// `blocks` is the sequence's whole block table. A shared block loses the
    /// sequence's reference and stays cached when the count reaches 0. Every
    /// other block is private and returns to the allocator.
    ///
    /// # Errors
    ///
    /// An error, with nothing changed, when `blocks` lists a block twice, when
    /// a shared block holds no reference of `seq_id`, or when the allocator
    /// refuses a private block.
    pub fn release_blocks(&mut self, seq_id: SequenceId, blocks: &[BlockId]) -> Result<()> {
        let mut seen = HashSet::with_capacity(blocks.len());
        let mut private = Vec::new();
        let mut shared = Vec::new();
        for &block_id in blocks {
            if !seen.insert(block_id) {
                return Err(release_error(
                    seq_id,
                    block_id,
                    "the block list holds it twice",
                ));
            }
            let place = match self.block_to_hash.get(&block_id) {
                Some(&hash) => Some(Shared::Registered(hash)),
                None if self.detached.contains_key(&block_id) => Some(Shared::Detached(block_id)),
                None => None,
            };
            let Some(place) = place else {
                private.push(block_id);
                continue;
            };
            let held = self
                .shared_info(place)
                .is_some_and(|info| info.ref_count > 0 && info.owners.contains(&seq_id));
            if !held {
                return Err(release_error(
                    seq_id,
                    block_id,
                    "the shared block holds no reference of this sequence",
                ));
            }
            shared.push(place);
        }

        // A detached block whose last reference goes now returns with the
        // private blocks, in one all-or-nothing free.
        let mut to_free = private;
        for &place in &shared {
            if let Shared::Detached(block_id) = place
                && self
                    .detached
                    .get(&block_id)
                    .is_some_and(|info| info.ref_count == 1)
            {
                to_free.push(block_id);
            }
        }
        self.allocator.free(&to_free)?;

        for place in shared {
            match place {
                Shared::Registered(hash) => self.drop_reference(seq_id, hash),
                Shared::Detached(block_id) => self.drop_detached_reference(seq_id, block_id),
            }
        }
        Ok(())
    }

    /// Unregister and free blocks whose last holder gives them up.
    ///
    /// A registered block with at most one reference loses its registration
    /// and returns to the allocator. The caller is that one holder and must
    /// not release the block again. A private block returns to the allocator.
    ///
    /// # Errors
    ///
    /// An error, with nothing changed, when a block still has two or more
    /// references, or when the allocator refuses a block.
    pub fn force_free_blocks(&mut self, blocks: &[BlockId]) -> Result<()> {
        for &block_id in blocks {
            if let Some(count) = self.block_ref_count(block_id)
                && count > 1
            {
                return Err(Error::SchedulerError {
                    reason: format!(
                        "cannot force-free block {block_id}: {count} sequences reference it. \
                         Release it with release_blocks instead"
                    ),
                });
            }
        }
        self.allocator.free(blocks)?;
        for &block_id in blocks {
            if let Some(hash) = self.block_to_hash.remove(&block_id) {
                if let Some(info) = self.hash_to_block.remove(&hash)
                    && info.ref_count == 0
                {
                    self.idle_blocks = self.idle_blocks.saturating_sub(1);
                }
                self.lru_queue.retain(|&id| id != block_id);
            }
            self.detached.remove(&block_id);
        }
        self.stats.cached_blocks = self.hash_to_block.len();
        Ok(())
    }

    fn shared_info(&self, place: Shared) -> Option<&CachedBlockInfo> {
        match place {
            Shared::Registered(hash) => self.hash_to_block.get(&hash),
            Shared::Detached(block_id) => self.detached.get(&block_id),
        }
    }

    /// Drop one reference of `seq_id` on a registered block.
    pub(super) fn drop_reference(&mut self, seq_id: SequenceId, hash: BlockHash) {
        if let Some(info) = self.hash_to_block.get_mut(&hash)
            && Self::remove_owner(info, seq_id)
            && info.ref_count == 0
        {
            self.idle_blocks += 1;
        }
    }

    /// Drop one reference of `seq_id` on a detached block. The entry goes with
    /// its last reference. The caller has freed the block by then.
    fn drop_detached_reference(&mut self, seq_id: SequenceId, block_id: BlockId) {
        let emptied = self
            .detached
            .get_mut(&block_id)
            .is_some_and(|info| Self::remove_owner(info, seq_id) && info.ref_count == 0);
        if emptied {
            self.detached.remove(&block_id);
        }
    }

    /// Remove one `seq_id` owner entry and its reference. False when absent.
    fn remove_owner(info: &mut CachedBlockInfo, seq_id: SequenceId) -> bool {
        let Some(pos) = info.owners.iter().position(|&id| id == seq_id) else {
            return false;
        };
        info.owners.swap_remove(pos);
        info.ref_count = info.ref_count.saturating_sub(1);
        true
    }
}

fn release_error(seq_id: SequenceId, block_id: BlockId, why: &str) -> Error {
    Error::SchedulerError {
        reason: format!(
            "cannot release block {block_id} of sequence {seq_id}: {why}. \
             Pass each block of the sequence's own table once"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::PrefixCacheConfig;
    use super::*;
    use crate::inference::memory::CpuBlockAllocator;

    const TOTAL: usize = 16;

    fn cache() -> PrefixCache<CpuBlockAllocator> {
        let config = PrefixCacheConfig {
            enabled: true,
            max_cached_blocks: 0,
            min_prefix_tokens: 1,
            block_size: 4,
        };
        PrefixCache::new(CpuBlockAllocator::new(TOTAL, 4), config)
    }

    #[test]
    fn release_frees_private_blocks_and_keeps_registered_ones() {
        let mut c = cache();
        let prompt: Vec<u32> = (0..10).collect();
        let mut blocks = c
            .get_or_allocate_blocks(1, &prompt)
            .expect("lookup")
            .into_blocks();
        // Two decode blocks the cache never saw.
        blocks.extend(c.allocate(2).expect("decode blocks"));
        assert_eq!(c.allocator().free_block_count(), TOTAL - 5);

        c.release_blocks(1, &blocks).expect("release");
        assert_eq!(c.registered_block_count(), 2);
        assert_eq!(c.idle_block_count(), 2);
        assert_eq!(c.allocator().free_block_count(), TOTAL - 2);
        assert_eq!(c.available_blocks(), TOTAL);
    }

    #[test]
    fn release_with_the_cache_off_frees_everything() {
        let config = PrefixCacheConfig {
            enabled: false,
            ..Default::default()
        };
        let mut c = PrefixCache::new(CpuBlockAllocator::new(TOTAL, 16), config);
        let prompt: Vec<u32> = (0..40).collect();
        let blocks = c
            .get_or_allocate_blocks(7, &prompt)
            .expect("lookup")
            .into_blocks();
        assert_eq!(blocks.len(), 3);
        c.release_blocks(7, &blocks).expect("release");
        assert_eq!(c.allocator().free_block_count(), TOTAL);
    }

    #[test]
    fn a_shared_block_is_released_once_per_holder() {
        let mut c = cache();
        let prompt: Vec<u32> = (0..9).collect();
        let a = c
            .get_or_allocate_blocks(1, &prompt)
            .expect("a")
            .into_blocks();
        let b = c
            .get_or_allocate_blocks(2, &prompt)
            .expect("b")
            .into_blocks();
        assert_eq!(c.block_ref_count(a[0]), Some(2));

        c.release_blocks(1, &a).expect("release a");
        assert_eq!(c.block_ref_count(a[0]), Some(1));
        assert!(
            c.release_blocks(1, &a[..2]).is_err(),
            "a holds no reference now"
        );
        assert_eq!(
            c.block_ref_count(a[0]),
            Some(1),
            "the refused call changes nothing"
        );

        c.release_blocks(2, &b).expect("release b");
        assert_eq!(c.block_ref_count(a[0]), Some(0));
        assert_eq!(
            c.allocator().free_block_count() + c.idle_block_count(),
            TOTAL
        );
    }

    #[test]
    fn a_duplicate_block_is_refused_without_changes() {
        let mut c = cache();
        let prompt: Vec<u32> = (0..9).collect();
        let blocks = c
            .get_or_allocate_blocks(1, &prompt)
            .expect("lookup")
            .into_blocks();
        let free = c.allocator().free_block_count();
        assert!(c.release_blocks(1, &[blocks[2], blocks[2]]).is_err());
        assert_eq!(c.allocator().free_block_count(), free);
        assert!(c.release_blocks(1, &[blocks[0], blocks[0]]).is_err());
        assert_eq!(c.block_ref_count(blocks[0]), Some(1));
    }

    #[test]
    fn force_free_refuses_a_block_another_sequence_holds() {
        let mut c = cache();
        let prompt: Vec<u32> = (0..9).collect();
        let a = c
            .get_or_allocate_blocks(1, &prompt)
            .expect("a")
            .into_blocks();
        c.get_or_allocate_blocks(2, &prompt).expect("b");
        assert!(c.force_free_blocks(&a).is_err());
        assert_eq!(c.registered_block_count(), 2);
    }
}
