//! `PrefixLookup` trait, `GpuRadixTree`'s data layout, hashing, and
//! construction. The lookup/insert/eviction algorithm lives in `super::ops`.

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
pub(super) const EMPTY_SLOT: i32 = -1;

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
pub(super) fn pack_meta(ref_count: u32, lru_timestamp: u32) -> u64 {
    ((ref_count as u64) << 32) | (lru_timestamp as u64)
}

/// Unpack `(ref_count, lru_timestamp)` from a `u64`.
pub(super) fn unpack_meta(meta: u64) -> (u32, u32) {
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
    pub(super) keys: Vec<u64>,
    /// Block IDs — one `i32` per slot (`EMPTY_SLOT` means empty).
    pub(super) values: Vec<i32>,
    /// Packed metadata — one `u64` per slot.
    pub(super) metadata: Vec<u64>,
    /// Number of live entries.
    pub(super) num_entries: usize,
    /// Capacity (power of two).
    pub(super) capacity: usize,
    /// Number of tokens per block (used for documentation / validation).
    pub(super) block_size: usize,
    /// Monotonically increasing logical clock for LRU ordering.
    pub(super) clock: u32,
    /// Cumulative hit counter.
    pub(super) hits: usize,
    /// Cumulative miss counter.
    pub(super) misses: usize,
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
}
