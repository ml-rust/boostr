//! Prompt lookup: reuse cached full blocks, allocate the rest.
//!
//! Only full prompt blocks are hashed, registered and shared. The block that
//! holds the last prompt token is never shared, so prefill always computes
//! at least that token. The cached token count is therefore a multiple of
//! the block size and always below the prompt length.

use crate::error::{Error, Result};
use crate::inference::memory::{BlockAllocator, BlockId};

use super::core::PrefixCache;
use super::types::{BlockHash, CacheResult, CachedBlockInfo, SequenceId};

impl<A: BlockAllocator> PrefixCache<A> {
    /// Leading blocks a prompt of `prompt_len` tokens can share.
    ///
    /// These are the full blocks before the block of the last prompt token.
    pub fn shareable_blocks(&self, prompt_len: usize) -> usize {
        match self.config.block_size {
            0 => 0,
            block_size => prompt_len.saturating_sub(1) / block_size,
        }
    }

    fn lookup_enabled(&self, tokens: &[u32]) -> bool {
        self.config.enabled
            && self.config.block_size > 0
            && tokens.len() >= self.config.min_prefix_tokens
    }

    /// Count of leading blocks in `hashes` that are registered with matching tokens.
    fn leading_hits(&self, tokens: &[u32], hashes: &[BlockHash]) -> usize {
        let block_size = self.config.block_size;
        hashes
            .iter()
            .enumerate()
            .take_while(|&(idx, hash)| {
                let block = &tokens[idx * block_size..(idx + 1) * block_size];
                // The token check guards against hash collisions.
                self.hash_to_block
                    .get(hash)
                    .is_some_and(|info| info.tokens == block)
            })
            .count()
    }

    /// The blocks of a prompt: cached full blocks first, then fresh ones.
    ///
    /// The first `cached_count()` blocks of the result already hold KV for
    /// `cached_count() * block_size` prompt tokens, always fewer than
    /// `tokens.len()`. Fresh full blocks are registered for later prompts,
    /// up to `max_cached_blocks`. The partial last block and the block of
    /// the last prompt token stay private when no entry exists for them.
    ///
    /// # Errors
    ///
    /// An error when `block_size` is 0, or when free plus idle blocks are too
    /// few. The error drops every reference the call took and holds no block.
    pub fn get_or_allocate_blocks(
        &mut self,
        seq_id: SequenceId,
        tokens: &[u32],
    ) -> Result<CacheResult> {
        let block_size = self.config.block_size;
        if block_size == 0 {
            return Err(Error::SchedulerError {
                reason: "prefix cache block_size is 0. Set PrefixCacheConfig::block_size \
                         to the KV block size"
                    .to_string(),
            });
        }
        let num_blocks = tokens.len().div_ceil(block_size);
        self.stats.total_lookups += 1;

        if !self.lookup_enabled(tokens) {
            let result = self.allocate(num_blocks).map(|blocks| {
                self.stats.blocks_allocated += blocks.len();
                CacheResult::Miss { blocks }
            });
            return self.finish_lookup(result);
        }

        let full = tokens.len() / block_size;
        let hashes = self.prefix_hashes(tokens, full);
        let shareable = self.shareable_blocks(tokens.len()).min(hashes.len());
        let hits = self.leading_hits(tokens, &hashes[..shareable]);

        let mut blocks = Vec::with_capacity(num_blocks);
        for &hash in &hashes[..hits] {
            if let Some(block_id) = self.take_reference(seq_id, hash) {
                blocks.push(block_id);
            }
        }

        let fresh_count = num_blocks - hits;
        let candidates: Vec<usize> = (hits..full)
            .filter(|&idx| !self.hash_to_block.contains_key(&hashes[idx]))
            .collect();
        let (room, fresh) = match self.take_fresh(candidates.len(), fresh_count) {
            Ok(taken) => taken,
            Err(e) => {
                for &hash in &hashes[..hits] {
                    self.drop_reference(seq_id, hash);
                }
                return self.finish_lookup(Err(e));
            }
        };

        for &idx in candidates.iter().take(room) {
            let block_id = fresh[idx - hits];
            let block_tokens = tokens[idx * block_size..(idx + 1) * block_size].to_vec();
            self.register(seq_id, hashes[idx], block_id, block_tokens);
        }
        blocks.extend(fresh);

        self.stats.blocks_from_cache += hits;
        self.stats.blocks_allocated += fresh_count;
        let result = if hits > 0 {
            CacheResult::Hit {
                blocks,
                cached_blocks: hits,
                new_blocks: fresh_count,
            }
        } else {
            CacheResult::Miss { blocks }
        };
        self.finish_lookup(Ok(result))
    }

    /// Make room for `registrations` under the cap, then allocate `count` blocks.
    fn take_fresh(&mut self, registrations: usize, count: usize) -> Result<(usize, Vec<BlockId>)> {
        let room = self.make_room(registrations)?;
        self.reclaim(count)?;
        let fresh = self.allocator.allocate(count)?;
        Ok((room, fresh))
    }

    /// Add one reference of `seq_id` to a registered block and return its id.
    fn take_reference(&mut self, seq_id: SequenceId, hash: BlockHash) -> Option<BlockId> {
        let info = self.hash_to_block.get_mut(&hash)?;
        if info.ref_count == 0 {
            self.idle_blocks = self.idle_blocks.saturating_sub(1);
        }
        info.ref_count += 1;
        info.owners.push(seq_id);
        let block_id = info.block_id;
        self.touch_lru(block_id);
        Some(block_id)
    }

    fn register(
        &mut self,
        seq_id: SequenceId,
        hash: BlockHash,
        block_id: BlockId,
        tokens: Vec<u32>,
    ) {
        // Two prefixes with one hash keep the first entry. The later block stays private.
        if self.hash_to_block.contains_key(&hash) {
            return;
        }
        let info = CachedBlockInfo {
            block_id,
            ref_count: 1,
            owners: vec![seq_id],
            tokens,
        };
        self.hash_to_block.insert(hash, info);
        self.block_to_hash.insert(block_id, hash);
        self.lru_queue.push_back(block_id);
    }

    fn finish_lookup(&mut self, result: Result<CacheResult>) -> Result<CacheResult> {
        if matches!(result, Ok(CacheResult::Hit { .. })) {
            self.stats.cache_hits += 1;
        } else {
            self.stats.cache_misses += 1;
        }
        self.stats.cached_blocks = self.hash_to_block.len();
        self.update_rates();
        result
    }

    /// True when a lookup of `tokens` reuses at least one block.
    pub fn has_cached_prefix(&self, tokens: &[u32]) -> bool {
        self.cached_block_count(tokens) > 0
    }

    /// Blocks a lookup of `tokens` reuses now. It takes no reference.
    pub fn cached_block_count(&self, tokens: &[u32]) -> usize {
        if !self.lookup_enabled(tokens) {
            return 0;
        }
        let hashes = self.prefix_hashes(tokens, self.shareable_blocks(tokens.len()));
        self.leading_hits(tokens, &hashes)
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::PrefixCacheConfig;
    use super::*;
    use crate::inference::memory::{BlockTable, CpuBlockAllocator};
    use std::collections::HashMap;

    const BLOCK: usize = 4;

    fn cache(total: usize) -> PrefixCache<CpuBlockAllocator> {
        let config = PrefixCacheConfig {
            enabled: true,
            max_cached_blocks: 0,
            min_prefix_tokens: 1,
            block_size: BLOCK,
        };
        PrefixCache::new(CpuBlockAllocator::new(total, BLOCK), config)
    }

    fn table(blocks: &[BlockId], tokens: usize) -> BlockTable {
        let mut table = BlockTable::new(BLOCK);
        table.blocks = blocks.to_vec();
        table.num_tokens = tokens;
        table
    }

    #[test]
    fn a_shared_prompt_reuses_full_blocks_only() {
        let mut c = cache(16);
        let prompt: Vec<u32> = (0..10).collect();
        let first = c.get_or_allocate_blocks(1, &prompt).expect("first");
        assert!(!first.is_hit());
        let first = first.into_blocks();

        let second = c.get_or_allocate_blocks(2, &prompt).expect("second");
        assert_eq!(second.cached_count(), 2);
        let second = second.into_blocks();
        assert_eq!(first[..2], second[..2]);
        assert_ne!(first[2], second[2], "the partial block is private");
        assert_eq!(c.block_ref_count(first[2]), None);
        assert_eq!(c.block_ref_count(second[2]), None);

        // Decode writes the next position into each sequence's own partial block.
        let mut kv: HashMap<(BlockId, usize), u64> = HashMap::new();
        let decode_pos = prompt.len();
        let slot_a = table(&first, decode_pos)
            .get_slot(decode_pos)
            .expect("slot a");
        let slot_b = table(&second, decode_pos)
            .get_slot(decode_pos)
            .expect("slot b");
        kv.insert(slot_a, 1);
        assert_eq!(
            kv.get(&slot_b),
            None,
            "sequence 2 never sees sequence 1's decode KV"
        );
        kv.insert(slot_b, 2);
        assert_eq!(kv.get(&slot_a), Some(&1));
    }

    #[test]
    fn an_exact_multiple_prompt_keeps_its_last_block_private() {
        let mut c = cache(16);
        let prompt: Vec<u32> = (0..8).collect();
        let first = c
            .get_or_allocate_blocks(1, &prompt)
            .expect("first")
            .into_blocks();
        let second = c.get_or_allocate_blocks(2, &prompt).expect("second");
        let cached_tokens = second.cached_count() * BLOCK;
        assert_eq!(cached_tokens, 4);
        assert!(cached_tokens < prompt.len());
        let second = second.into_blocks();
        assert_eq!(first[0], second[0]);
        assert_ne!(first[1], second[1]);
        assert_eq!(c.block_ref_count(second[1]), None);

        // A longer prompt reuses both full blocks of the first prompt.
        let longer: Vec<u32> = (0..9).collect();
        assert_eq!(c.cached_block_count(&longer), 2);
    }

    #[test]
    fn cached_tokens_stay_below_the_prompt_length() {
        for len in 1..=20 {
            let mut c = cache(32);
            let prompt: Vec<u32> = (0..len as u32).collect();
            c.get_or_allocate_blocks(1, &prompt).expect("first");
            let again = c.get_or_allocate_blocks(2, &prompt).expect("again");
            let cached_tokens = again.cached_count() * BLOCK;
            assert!(cached_tokens < len, "len {len}: {cached_tokens} cached");
            assert_eq!(again.cached_count(), (len - 1) / BLOCK, "len {len}");
            assert_eq!(c.cached_block_count(&prompt), again.cached_count());
        }
    }

    #[test]
    fn a_hit_needs_every_earlier_block() {
        let mut c = cache(4);
        let prompt: Vec<u32> = (0..12).collect();
        let blocks = c
            .get_or_allocate_blocks(1, &prompt)
            .expect("lookup")
            .into_blocks();
        assert_eq!(c.registered_block_count(), 3);
        c.release_blocks(1, &blocks).expect("release");

        // One eviction drops block 0. Blocks 1 and 2 stay registered.
        let taken = c.allocate(2).expect("evict one");
        assert_eq!(c.registered_block_count(), 2);
        assert_eq!(
            c.cached_block_count(&prompt),
            0,
            "block 1 alone is no prefix"
        );
        c.release_blocks(9, &taken).expect("release taken");

        let again = c.get_or_allocate_blocks(2, &prompt).expect("again");
        assert_eq!(again.cached_count(), 0);
    }

    #[test]
    fn a_failed_lookup_changes_nothing() {
        let mut c = cache(3);
        let prompt: Vec<u32> = (0..9).collect();
        let held = c
            .get_or_allocate_blocks(1, &prompt)
            .expect("held")
            .into_blocks();
        let free = c.allocator().free_block_count();
        let longer: Vec<u32> = (0..20).collect();
        assert!(c.get_or_allocate_blocks(2, &longer).is_err());
        assert_eq!(c.allocator().free_block_count(), free);
        assert_eq!(
            c.block_ref_count(held[0]),
            Some(1),
            "the hit reference is rolled back"
        );
        assert_eq!(c.registered_block_count(), 2);
    }

    #[test]
    fn a_short_prompt_skips_the_cache_and_frees_on_release() {
        let config = PrefixCacheConfig {
            enabled: true,
            max_cached_blocks: 0,
            min_prefix_tokens: 16,
            block_size: BLOCK,
        };
        let mut c = PrefixCache::new(CpuBlockAllocator::new(8, BLOCK), config);
        let prompt: Vec<u32> = (0..9).collect();
        let blocks = c
            .get_or_allocate_blocks(1, &prompt)
            .expect("lookup")
            .into_blocks();
        assert_eq!(c.registered_block_count(), 0);
        c.release_blocks(1, &blocks).expect("release");
        assert_eq!(c.allocator().free_block_count(), 8);
    }

    #[test]
    fn a_zero_block_size_is_an_error() {
        let config = PrefixCacheConfig {
            block_size: 0,
            ..Default::default()
        };
        let mut c = PrefixCache::new(CpuBlockAllocator::new(8, 4), config);
        assert!(c.get_or_allocate_blocks(1, &[1, 2, 3]).is_err());
        assert_eq!(c.cached_block_count(&[1, 2, 3]), 0);
    }
}
