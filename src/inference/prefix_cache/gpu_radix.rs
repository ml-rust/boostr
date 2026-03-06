//! GPU-resident radix tree for O(1) amortised prefix matching.
//!
//! # Architecture
//!
//! The radix tree maps block-aligned token-sequence hashes to cached `BlockId`
//! values.  The hash table is stored as three parallel arrays on the CPU and —
//! when the `cuda` feature is enabled — mirrored into GPU global memory so that
//! batch prefix lookups can be executed entirely on the device, avoiding a
//! CPU scheduling bottleneck.
//!
//! CPU layout (one entry per slot):
//! ```text
//! keys:      [u64; capacity]   — FNV hash of the token subsequence
//! values:    [i32; capacity]   — BlockId (-1 = empty slot)
//! metadata:  [u64; capacity]   — packed (ref_count: u32, lru_timestamp: u32)
//! ```
//!
//! Probing strategy: open addressing with linear probing.  The capacity is
//! always a power of two so that `hash & (capacity - 1)` gives the slot index.
//!
//! # Tiering
//!
//! VRAM (hot) → RAM (warm) → NVMe (cold, future work).
//! Eviction promotes entries toward cold tiers; insertion starts in VRAM.
//!
//! # Feature gating
//!
//! The `GpuRadixTree` type and its CUDA-backed `PrefixLookup` implementation
//! are compiled only when the `cuda` feature is enabled.  A pure-CPU fallback
//! is provided unconditionally so that the `PrefixLookup` trait can be used
//! in feature-agnostic scheduling code.

use crate::inference::memory::BlockId;

// ---------------------------------------------------------------------------
// PrefixLookup trait — feature-agnostic
// ---------------------------------------------------------------------------

/// Batch prefix-cache lookup interface.
///
/// Given a slice of per-block token-sequence hashes (one hash per block,
/// covering the entire sequence from the start), returns a parallel
/// `Vec<Option<BlockId>>` where `None` indicates a cache miss for that block.
///
/// Implementations are expected to be fast: the hot path in the batch
/// scheduler calls this before every decode step.
pub trait PrefixLookup: Send + Sync {
    /// Look up cached block IDs for the given block hashes.
    ///
    /// `token_block_hashes` must have one entry per logical KV block, computed
    /// as the FNV-1a hash of the full token prefix up to and including that
    /// block (i.e. hashes are chained/cumulative, not per-block-only).
    ///
    /// Returns `None` for each block that is not in the cache.
    fn lookup_blocks(&self, token_block_hashes: &[u64]) -> Vec<Option<BlockId>>;
}

// ---------------------------------------------------------------------------
// CPU-only radix tree (always compiled)
// ---------------------------------------------------------------------------

/// Statistics reported by `GpuRadixTree`.
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuRadixStats {
    /// Number of successful lookups (cache hits) since last reset.
    pub hits: usize,
    /// Number of failed lookups (cache misses) since last reset.
    pub misses: usize,
    /// Number of entries currently in the table.
    pub num_entries: usize,
    /// Table capacity (number of slots).
    pub capacity: usize,
    /// Occupancy in `[0.0, 1.0]`.
    pub occupancy: f64,
    /// Hit rate in `[0.0, 1.0]`.
    pub hit_rate: f64,
}

/// Sentinel value used in the `values` array to mark an empty slot.
const EMPTY_SLOT: i32 = -1;

/// FNV-1a offset basis (64-bit).
const FNV_OFFSET: u64 = 0xcbf29ce484222325;
/// FNV-1a prime (64-bit).
const FNV_PRIME: u64 = 0x100000001b3;

/// Round `n` up to the next power of two (minimum 16).
fn next_power_of_two(n: usize) -> usize {
    if n <= 16 {
        return 16;
    }
    let mut p = 1usize;
    while p < n {
        p <<= 1;
    }
    p
}

/// Pack `(ref_count, lru_timestamp)` into a single `u64`.
fn pack_meta(ref_count: u32, lru_timestamp: u32) -> u64 {
    ((ref_count as u64) << 32) | (lru_timestamp as u64)
}

/// Unpack `(ref_count, lru_timestamp)` from a `u64`.
fn unpack_meta(meta: u64) -> (u32, u32) {
    ((meta >> 32) as u32, meta as u32)
}

/// GPU-resident radix tree for prefix-cache lookup.
///
/// When compiled without the `cuda` feature the GPU mirror is absent; all
/// operations run on the CPU.  When the `cuda` feature is enabled the table
/// arrays are also uploaded to the device and batch lookups use the
/// `prefix_cache_lookup` CUDA kernel (see
/// `ops/cuda/inference/prefix_cache.rs`).
pub struct GpuRadixTree {
    /// Hash keys — one `u64` per slot.
    keys: Vec<u64>,
    /// Block IDs — one `i32` per slot (`EMPTY_SLOT` means empty).
    values: Vec<i32>,
    /// Packed metadata — one `u64` per slot.
    metadata: Vec<u64>,
    /// Number of live entries.
    num_entries: usize,
    /// Capacity (power of two).
    capacity: usize,
    /// Number of tokens per block (used for documentation / validation).
    block_size: usize,
    /// Monotonically increasing logical clock for LRU ordering.
    clock: u32,
    /// Cumulative hit counter.
    hits: usize,
    /// Cumulative miss counter.
    misses: usize,
}

impl GpuRadixTree {
    /// Create a new radix tree with at least `capacity` slots.
    ///
    /// The actual capacity will be rounded up to the next power of two.
    ///
    /// # Arguments
    /// * `capacity` — initial number of hash table slots (minimum 16).
    /// * `block_size` — number of tokens per KV block.
    pub fn new(capacity: usize, block_size: usize) -> Self {
        let cap = next_power_of_two(capacity);
        Self {
            keys: vec![0u64; cap],
            values: vec![EMPTY_SLOT; cap],
            metadata: vec![0u64; cap],
            num_entries: 0,
            capacity: cap,
            block_size,
            clock: 0,
            hits: 0,
            misses: 0,
        }
    }

    /// Probe the table for `key`.  Returns `Some(slot_index)` on hit, `None`
    /// on miss.  Uses linear probing.
    fn probe(&self, key: u64) -> Option<usize> {
        if self.num_entries == 0 {
            return None;
        }
        let mask = self.capacity - 1;
        let mut slot = (key as usize) & mask;
        for _ in 0..self.capacity {
            let v = self.values[slot];
            if v == EMPTY_SLOT {
                return None; // empty slot → definite miss
            }
            if self.keys[slot] == key {
                return Some(slot);
            }
            slot = (slot + 1) & mask;
        }
        None
    }

    /// Find the slot where `key` should be inserted (may be a new empty slot or
    /// an existing slot with the same key).  Returns `None` if the table is
    /// full and the key is not already present.
    fn probe_for_insert(&self, key: u64) -> Option<usize> {
        let mask = self.capacity - 1;
        let mut slot = (key as usize) & mask;
        for _ in 0..self.capacity {
            let v = self.values[slot];
            if v == EMPTY_SLOT || self.keys[slot] == key {
                return Some(slot);
            }
            slot = (slot + 1) & mask;
        }
        None
    }

    /// Compute the FNV-1a hash of a token sequence.
    pub fn hash_tokens(tokens: &[u32]) -> u64 {
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

    /// Compute block-aligned cumulative hashes for `tokens`.
    ///
    /// Returns one hash per block, where each hash covers the prefix of
    /// `tokens` up to and including that block.
    pub fn compute_block_hashes(tokens: &[u32], block_size: usize) -> Vec<u64> {
        if block_size == 0 || tokens.is_empty() {
            return Vec::new();
        }
        let num_blocks = tokens.len().div_ceil(block_size);
        let mut hashes = Vec::with_capacity(num_blocks);
        for block_idx in 0..num_blocks {
            let end = ((block_idx + 1) * block_size).min(tokens.len());
            hashes.push(Self::hash_tokens(&tokens[..end]));
        }
        hashes
    }

    /// Batch lookup: returns `Some(block_id)` for each hash in `token_block_hashes`
    /// that is present in the table, `None` for misses.
    ///
    /// Also updates hit/miss counters and refreshes LRU timestamps for hits.
    pub fn lookup(&mut self, token_block_hashes: &[u64]) -> Vec<Option<BlockId>> {
        let ts = self.clock;
        self.clock = self.clock.wrapping_add(1);

        token_block_hashes
            .iter()
            .map(|&key| {
                if let Some(slot) = self.probe(key) {
                    // Update LRU timestamp.
                    let (ref_count, _) = unpack_meta(self.metadata[slot]);
                    self.metadata[slot] = pack_meta(ref_count, ts);
                    self.hits += 1;
                    Some(self.values[slot] as BlockId)
                } else {
                    self.misses += 1;
                    None
                }
            })
            .collect()
    }

    /// Insert or update a mapping from `token_hash` to `block_id`.
    ///
    /// If the table is full the insertion is silently dropped — the caller
    /// should call `evict_lru` first to make room.
    ///
    /// Returns `true` if the entry was inserted (or updated), `false` if the
    /// table had no room.
    pub fn insert(&mut self, token_hash: u64, block_id: BlockId) -> bool {
        // Refuse to exceed 75 % load factor.
        if self.num_entries * 4 >= self.capacity * 3 {
            return false;
        }
        if let Some(slot) = self.probe_for_insert(token_hash) {
            let is_new = self.values[slot] == EMPTY_SLOT;
            self.keys[slot] = token_hash;
            self.values[slot] = block_id as i32;
            let ts = self.clock;
            self.clock = self.clock.wrapping_add(1);
            self.metadata[slot] = pack_meta(0, ts);
            if is_new {
                self.num_entries += 1;
            }
            true
        } else {
            false
        }
    }

    /// Increment the reference count for a cached block identified by its hash.
    ///
    /// Does nothing if the hash is not in the table.
    pub fn inc_ref(&mut self, token_hash: u64) {
        if let Some(slot) = self.probe(token_hash) {
            let (ref_count, ts) = unpack_meta(self.metadata[slot]);
            self.metadata[slot] = pack_meta(ref_count.saturating_add(1), ts);
        }
    }

    /// Decrement the reference count for a cached block identified by its hash.
    ///
    /// Does nothing if the hash is not in the table.
    pub fn dec_ref(&mut self, token_hash: u64) {
        if let Some(slot) = self.probe(token_hash) {
            let (ref_count, ts) = unpack_meta(self.metadata[slot]);
            self.metadata[slot] = pack_meta(ref_count.saturating_sub(1), ts);
        }
    }

    /// Evict up to `num_to_evict` least-recently-used entries whose reference
    /// count is zero.
    ///
    /// Returns the `BlockId`s that were evicted so the caller can return them
    /// to the block allocator.
    pub fn evict_lru(&mut self, num_to_evict: usize) -> Vec<BlockId> {
        if num_to_evict == 0 || self.num_entries == 0 {
            return Vec::new();
        }

        // Collect candidate slots: live entries with ref_count == 0.
        let mut candidates: Vec<(u32, usize)> = self // (lru_ts, slot)
            .values
            .iter()
            .enumerate()
            .filter_map(|(slot, &v)| {
                if v == EMPTY_SLOT {
                    return None;
                }
                let (ref_count, ts) = unpack_meta(self.metadata[slot]);
                if ref_count == 0 {
                    Some((ts, slot))
                } else {
                    None
                }
            })
            .collect();

        // Sort ascending by timestamp: smallest timestamp = least recently used.
        candidates.sort_unstable_by_key(|&(ts, _)| ts);

        let evict_count = num_to_evict.min(candidates.len());
        let mut evicted = Vec::with_capacity(evict_count);

        for (_, slot) in candidates.into_iter().take(evict_count) {
            evicted.push(self.values[slot] as BlockId);
            // Clear the slot.
            self.values[slot] = EMPTY_SLOT;
            self.keys[slot] = 0;
            self.metadata[slot] = 0;
            self.num_entries -= 1;
        }

        // Rehash remaining entries to close gaps created by deletion.
        // This is required for correctness with linear probing.
        self.rehash();

        evicted
    }

    /// Full rehash: rebuild the table in place to remove tombstone-style gaps.
    fn rehash(&mut self) {
        // Snapshot live entries.
        let live: Vec<(u64, i32, u64)> = (0..self.capacity)
            .filter_map(|slot| {
                if self.values[slot] != EMPTY_SLOT {
                    Some((self.keys[slot], self.values[slot], self.metadata[slot]))
                } else {
                    None
                }
            })
            .collect();

        // Clear the table.
        self.keys.iter_mut().for_each(|k| *k = 0);
        self.values.iter_mut().for_each(|v| *v = EMPTY_SLOT);
        self.metadata.iter_mut().for_each(|m| *m = 0);
        self.num_entries = 0;

        // Re-insert.
        let mask = self.capacity - 1;
        for (key, value, meta) in live {
            let mut slot = (key as usize) & mask;
            loop {
                if self.values[slot] == EMPTY_SLOT {
                    self.keys[slot] = key;
                    self.values[slot] = value;
                    self.metadata[slot] = meta;
                    self.num_entries += 1;
                    break;
                }
                slot = (slot + 1) & mask;
            }
        }
    }

    /// Return current statistics.
    pub fn stats(&self) -> GpuRadixStats {
        let occupancy = if self.capacity == 0 {
            0.0
        } else {
            self.num_entries as f64 / self.capacity as f64
        };
        let total_lookups = self.hits + self.misses;
        let hit_rate = if total_lookups == 0 {
            0.0
        } else {
            self.hits as f64 / total_lookups as f64
        };
        GpuRadixStats {
            hits: self.hits,
            misses: self.misses,
            num_entries: self.num_entries,
            capacity: self.capacity,
            occupancy,
            hit_rate,
        }
    }

    /// Number of tokens per KV block (informational).
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Number of live entries in the table.
    pub fn num_entries(&self) -> usize {
        self.num_entries
    }

    /// Read-only view of the key array (used for GPU upload).
    #[cfg(feature = "cuda")]
    pub fn keys(&self) -> &[u64] {
        &self.keys
    }

    /// Read-only view of the value array (used for GPU upload).
    #[cfg(feature = "cuda")]
    pub fn values(&self) -> &[i32] {
        &self.values
    }
}

impl PrefixLookup for GpuRadixTree {
    fn lookup_blocks(&self, token_block_hashes: &[u64]) -> Vec<Option<BlockId>> {
        // Read-only probe (no LRU update) — the mutable `lookup` method above
        // should be preferred when mutation is acceptable.
        token_block_hashes
            .iter()
            .map(|&key| self.probe(key).map(|slot| self.values[slot] as BlockId))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_tokens_deterministic() {
        let h1 = GpuRadixTree::hash_tokens(&[1, 2, 3, 4]);
        let h2 = GpuRadixTree::hash_tokens(&[1, 2, 3, 4]);
        let h3 = GpuRadixTree::hash_tokens(&[1, 2, 3, 5]);
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_compute_block_hashes_length() {
        // 10 tokens, block_size 4 → 3 blocks (4, 4, 2)
        let hashes = GpuRadixTree::compute_block_hashes(&[0u32; 10], 4);
        assert_eq!(hashes.len(), 3);
    }

    #[test]
    fn test_insert_and_lookup() {
        let mut tree = GpuRadixTree::new(64, 16);
        let key = GpuRadixTree::hash_tokens(&[1, 2, 3, 4]);
        assert!(tree.insert(key, 42));
        let results = tree.lookup(&[key]);
        assert_eq!(results, vec![Some(42)]);
    }

    #[test]
    fn test_miss_returns_none() {
        let mut tree = GpuRadixTree::new(64, 16);
        let key = GpuRadixTree::hash_tokens(&[9, 9, 9]);
        let results = tree.lookup(&[key]);
        assert_eq!(results, vec![None]);
    }

    #[test]
    fn test_evict_lru_removes_unreferenced() {
        let mut tree = GpuRadixTree::new(64, 16);
        let k1 = GpuRadixTree::hash_tokens(&[1]);
        let k2 = GpuRadixTree::hash_tokens(&[2]);
        tree.insert(k1, 10);
        tree.insert(k2, 11);
        assert_eq!(tree.num_entries(), 2);

        let evicted = tree.evict_lru(1);
        assert_eq!(evicted.len(), 1);
        assert_eq!(tree.num_entries(), 1);
    }

    #[test]
    fn test_evict_lru_skips_referenced() {
        let mut tree = GpuRadixTree::new(64, 16);
        let k1 = GpuRadixTree::hash_tokens(&[1]);
        tree.insert(k1, 10);
        tree.inc_ref(k1);
        // ref_count = 1, must not be evicted
        let evicted = tree.evict_lru(10);
        assert!(evicted.is_empty());
        assert_eq!(tree.num_entries(), 1);
    }

    #[test]
    fn test_prefix_lookup_trait() {
        let mut tree = GpuRadixTree::new(64, 16);
        let k = GpuRadixTree::hash_tokens(&[5, 6, 7]);
        tree.insert(k, 99);
        let lookup: &dyn PrefixLookup = &tree;
        let result = lookup.lookup_blocks(&[k, 0xdeadbeefdeadbeef]);
        assert_eq!(result[0], Some(99));
        assert_eq!(result[1], None);
    }

    #[test]
    fn test_stats_hit_rate() {
        let mut tree = GpuRadixTree::new(64, 16);
        let k = GpuRadixTree::hash_tokens(&[1, 2]);
        tree.insert(k, 7);
        tree.lookup(&[k]);
        tree.lookup(&[0xdeadbeef]);
        let stats = tree.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_load_factor_limit() {
        // Table capacity 16, max entries at 75 % = 12.
        let mut tree = GpuRadixTree::new(16, 1);
        let mut inserted = 0usize;
        for i in 0u32..20 {
            if tree.insert(GpuRadixTree::hash_tokens(&[i]), i as BlockId) {
                inserted += 1;
            }
        }
        // Must not exceed 75 % of 16 = 12.
        assert!(inserted <= 12);
    }
}
