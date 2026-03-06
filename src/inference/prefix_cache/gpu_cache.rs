//! GPU-accelerated prefix cache with two-tier storage (VRAM → RAM).
//!
//! `GpuPrefixCache` combines a CPU-side [`GpuRadixTree`] (hash table bookkeeping)
//! with device-mirrored tensors, enabling batch prefix lookups via a CUDA kernel
//! without CPU round-trips.
//!
//! Tier layout:
//! - **VRAM (hot):** tracked by `GpuRadixTree`, mirrored to device tensors.
//!   Batch lookups on this tier run entirely on GPU.
//! - **RAM (warm):** entries evicted from GPU tier are demoted here.
//!   Lookups fall back to CPU hash map.  Entries promoted back to GPU on hit.

#[cfg(feature = "cuda")]
mod inner {
    use std::collections::{HashMap, VecDeque};

    use numr::dtype::DType;
    use numr::runtime::Device;
    use numr::runtime::cuda::{CudaClient, CudaRuntime};
    use numr::tensor::Tensor;

    use crate::error::{Error, Result};
    use crate::inference::memory::BlockId;
    use crate::inference::prefix_cache::gpu_radix::{GpuRadixStats, GpuRadixTree};
    use crate::ops::cuda::inference::prefix_cache::{gpu_prefix_cache_lookup, result_to_options};

    /// GPU-accelerated prefix cache with VRAM (hot) and RAM (warm) tiers.
    pub struct GpuPrefixCache {
        /// CPU-side hash table (source of truth for the VRAM tier).
        tree: GpuRadixTree,
        /// Device tensor mirroring `tree.keys()` — shape `[capacity]`, dtype I64.
        device_keys: Option<Tensor<CudaRuntime>>,
        /// Device tensor mirroring `tree.values()` — shape `[capacity]`, dtype I32.
        device_values: Option<Tensor<CudaRuntime>>,
        /// Whether CPU state has diverged from the GPU mirror.
        dirty: bool,
        /// CUDA device for tensor creation.
        device: <CudaRuntime as numr::runtime::Runtime>::Device,
        /// RAM tier: evicted-from-GPU entries kept warm for possible re-promotion.
        /// Key = token-sequence hash, Value = block ID.
        ram_tier: HashMap<u64, BlockId>,
        /// LRU ordering for the RAM tier (front = oldest).
        ram_lru: VecDeque<u64>,
        /// Maximum entries in RAM tier before permanent eviction.
        ram_tier_capacity: usize,
    }

    impl GpuPrefixCache {
        /// Create a new GPU prefix cache.
        ///
        /// # Arguments
        /// * `capacity` — hash table capacity (rounded up to power of two internally).
        /// * `block_size` — tokens per KV block.
        /// * `device` — CUDA device for tensor allocation.
        /// * `ram_tier_capacity` — maximum warm entries in RAM (0 = disable RAM tier).
        pub fn new(
            capacity: usize,
            block_size: usize,
            device: <CudaRuntime as numr::runtime::Runtime>::Device,
            ram_tier_capacity: usize,
        ) -> Self {
            Self {
                tree: GpuRadixTree::new(capacity, block_size),
                device_keys: None,
                device_values: None,
                dirty: true, // force initial upload
                device,
                ram_tier: HashMap::with_capacity(ram_tier_capacity.min(4096)),
                ram_lru: VecDeque::with_capacity(ram_tier_capacity.min(4096)),
                ram_tier_capacity,
            }
        }

        /// Insert a hash → block mapping into the GPU (hot) tier.
        ///
        /// Returns `true` if inserted, `false` if table is full (call `evict` first).
        pub fn insert(&mut self, token_hash: u64, block_id: BlockId) -> bool {
            // If the entry is in the RAM tier, remove it (it's being promoted to GPU).
            self.ram_tier.remove(&token_hash);
            self.ram_lru.retain(|&h| h != token_hash);

            let ok = self.tree.insert(token_hash, block_id);
            if ok {
                self.dirty = true;
            }
            ok
        }

        /// Increment reference count for a cached block.
        pub fn inc_ref(&mut self, token_hash: u64) {
            self.tree.inc_ref(token_hash);
        }

        /// Decrement reference count for a cached block.
        pub fn dec_ref(&mut self, token_hash: u64) {
            self.tree.dec_ref(token_hash);
        }

        /// Evict LRU entries from the GPU tier.
        ///
        /// Returns block IDs of evicted entries. The caller is responsible for
        /// returning these to the block allocator.
        ///
        /// **Note:** This does NOT demote to RAM tier — the hash is lost during
        /// tree eviction. To keep entries warm in RAM, call [`demote_to_ram`]
        /// with the known hash *before* eviction.
        pub fn evict_lru(&mut self, count: usize) -> Vec<BlockId> {
            let evicted_block_ids = self.tree.evict_lru(count);
            if !evicted_block_ids.is_empty() {
                self.dirty = true;
            }
            evicted_block_ids
        }

        /// Demote a specific entry from GPU tier to RAM tier.
        ///
        /// Use this before eviction when you know the hash and want to keep it warm.
        pub fn demote_to_ram(&mut self, token_hash: u64, block_id: BlockId) {
            if self.ram_tier_capacity == 0 {
                return;
            }

            // Evict oldest RAM entry if at capacity
            while self.ram_tier.len() >= self.ram_tier_capacity {
                if let Some(oldest_hash) = self.ram_lru.pop_front() {
                    self.ram_tier.remove(&oldest_hash);
                } else {
                    break;
                }
            }

            self.ram_tier.insert(token_hash, block_id);
            self.ram_lru.push_back(token_hash);
        }

        /// Promote an entry from RAM tier back to GPU tier.
        ///
        /// Returns the block ID if found and successfully promoted, `None` otherwise.
        pub fn promote_from_ram(&mut self, token_hash: u64) -> Option<BlockId> {
            if let Some(block_id) = self.ram_tier.remove(&token_hash) {
                self.ram_lru.retain(|&h| h != token_hash);
                if self.tree.insert(token_hash, block_id) {
                    self.dirty = true;
                    Some(block_id)
                } else {
                    // GPU tier full, put back in RAM
                    self.ram_tier.insert(token_hash, block_id);
                    self.ram_lru.push_back(token_hash);
                    Some(block_id) // still usable from RAM
                }
            } else {
                None
            }
        }

        /// Synchronize CPU-side hash table to GPU device tensors.
        ///
        /// Creates (or replaces) device tensors from the tree's key/value arrays.
        /// No-op if the table hasn't changed since the last sync.
        pub fn sync_to_device(&mut self) -> Result<()> {
            if !self.dirty {
                return Ok(());
            }

            let keys = self.tree.keys();
            let values = self.tree.values();

            // Keys: &[u64] → Tensor<CudaRuntime> with dtype I64, shape [capacity].
            // SAFETY: u64 and i64 have identical size, alignment, and bit representation.
            // We reinterpret the buffer for device upload only — no CPU arithmetic on the result.
            let keys_as_i64: &[i64] =
                unsafe { std::slice::from_raw_parts(keys.as_ptr() as *const i64, keys.len()) };

            self.device_keys = Some(Tensor::<CudaRuntime>::from_slice(
                keys_as_i64,
                &[keys.len()],
                &self.device,
            ));

            self.device_values = Some(Tensor::<CudaRuntime>::from_slice(
                values,
                &[values.len()],
                &self.device,
            ));

            self.dirty = false;
            Ok(())
        }

        /// Batch lookup on GPU.
        ///
        /// Returns a device tensor of shape `[num_queries]` with block IDs (-1 = miss).
        /// Automatically syncs to device if dirty.
        pub fn lookup_gpu(
            &mut self,
            client: &CudaClient,
            query_hashes: &Tensor<CudaRuntime>,
        ) -> Result<Tensor<CudaRuntime>> {
            self.sync_to_device()?;

            let device_keys = self.device_keys.as_ref().ok_or_else(|| Error::ModelError {
                reason: "GPU prefix cache: device_keys not initialized after sync".into(),
            })?;
            let device_values = self
                .device_values
                .as_ref()
                .ok_or_else(|| Error::ModelError {
                    reason: "GPU prefix cache: device_values not initialized after sync".into(),
                })?;

            gpu_prefix_cache_lookup(client, query_hashes, device_keys, device_values)
        }

        /// Batch lookup returning host-side results, with RAM tier fallback.
        ///
        /// 1. Runs GPU kernel for all queries.
        /// 2. For misses, checks the RAM tier (CPU hash map).
        /// 3. Promotes RAM hits back to GPU tier.
        pub fn lookup_with_fallback(
            &mut self,
            client: &CudaClient,
            token_block_hashes: &[u64],
        ) -> Result<Vec<Option<BlockId>>> {
            if token_block_hashes.is_empty() {
                return Ok(Vec::new());
            }

            // Create query tensor on device
            let hashes_i64: Vec<i64> = token_block_hashes.iter().map(|&h| h as i64).collect();
            let query_tensor =
                Tensor::<CudaRuntime>::from_slice(&hashes_i64, &[hashes_i64.len()], &self.device);

            // GPU lookup
            let gpu_result = self.lookup_gpu(client, &query_tensor)?;
            let mut results = result_to_options(&gpu_result);

            // RAM tier fallback for misses
            for (i, result) in results.iter_mut().enumerate() {
                if result.is_none() {
                    let hash = token_block_hashes[i];
                    if let Some(&block_id) = self.ram_tier.get(&hash) {
                        *result = Some(block_id);
                        // Promote to GPU tier (lazy — will sync on next GPU lookup)
                        self.ram_tier.remove(&hash);
                        self.ram_lru.retain(|&h| h != hash);
                        if self.tree.insert(hash, block_id) {
                            self.dirty = true;
                        }
                    }
                }
            }

            Ok(results)
        }

        /// CPU-only lookup (no GPU kernel). Checks GPU tree then RAM tier.
        pub fn lookup_cpu(&mut self, token_block_hashes: &[u64]) -> Vec<Option<BlockId>> {
            let mut results = self.tree.lookup(token_block_hashes);

            // Check RAM tier for misses
            for (i, result) in results.iter_mut().enumerate() {
                if result.is_none() {
                    if let Some(&block_id) = self.ram_tier.get(&token_block_hashes[i]) {
                        *result = Some(block_id);
                    }
                }
            }

            results
        }

        /// Compute block-aligned cumulative hashes for a token sequence.
        pub fn compute_block_hashes(tokens: &[u32], block_size: usize) -> Vec<u64> {
            GpuRadixTree::compute_block_hashes(tokens, block_size)
        }

        /// Return current statistics for the GPU tier.
        pub fn stats(&self) -> GpuRadixStats {
            self.tree.stats()
        }

        /// Number of entries in the RAM (warm) tier.
        pub fn ram_tier_count(&self) -> usize {
            self.ram_tier.len()
        }

        /// Number of entries in the GPU (hot) tier.
        pub fn gpu_tier_count(&self) -> usize {
            self.tree.num_entries()
        }

        /// Block size (tokens per KV block).
        pub fn block_size(&self) -> usize {
            self.tree.block_size()
        }

        /// Whether the CPU state has diverged from the GPU mirror.
        pub fn is_dirty(&self) -> bool {
            self.dirty
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        // Note: these tests run without a real CUDA device. They test the CPU-side
        // logic (insert, evict, RAM tier, lookup_cpu). GPU tensor tests require
        // a CUDA device and are in integration tests.

        #[test]
        fn test_insert_and_cpu_lookup() {
            let device = <CudaRuntime as numr::runtime::Runtime>::Device::default();
            let mut cache = GpuPrefixCache::new(64, 16, device, 100);

            let h1 = GpuRadixTree::hash_tokens(&[1, 2, 3]);
            assert!(cache.insert(h1, 42));

            let results = cache.lookup_cpu(&[h1]);
            assert_eq!(results, vec![Some(42)]);
            assert!(cache.is_dirty());
        }

        #[test]
        fn test_ram_tier_demotion() {
            let device = <CudaRuntime as numr::runtime::Runtime>::Device::default();
            let mut cache = GpuPrefixCache::new(64, 16, device, 10);

            let h1 = GpuRadixTree::hash_tokens(&[1]);
            cache.insert(h1, 10);

            // Demote to RAM
            cache.demote_to_ram(h1, 10);
            assert_eq!(cache.ram_tier_count(), 1);

            // Should be findable via CPU lookup (RAM tier)
            let results = cache.lookup_cpu(&[h1]);
            // h1 is in both tree and RAM; tree lookup should find it
            assert_eq!(results, vec![Some(10)]);
        }

        #[test]
        fn test_ram_tier_capacity() {
            let device = <CudaRuntime as numr::runtime::Runtime>::Device::default();
            let mut cache = GpuPrefixCache::new(64, 16, device, 2);

            cache.demote_to_ram(100, 1);
            cache.demote_to_ram(200, 2);
            assert_eq!(cache.ram_tier_count(), 2);

            // Third entry should evict the oldest (100)
            cache.demote_to_ram(300, 3);
            assert_eq!(cache.ram_tier_count(), 2);
            assert!(cache.ram_tier.get(&100).is_none());
            assert!(cache.ram_tier.get(&200).is_some());
            assert!(cache.ram_tier.get(&300).is_some());
        }

        #[test]
        fn test_promote_from_ram() {
            let device = <CudaRuntime as numr::runtime::Runtime>::Device::default();
            let mut cache = GpuPrefixCache::new(64, 16, device, 10);

            cache.demote_to_ram(500, 77);
            assert_eq!(cache.gpu_tier_count(), 0);

            let block = cache.promote_from_ram(500);
            assert_eq!(block, Some(77));
            assert_eq!(cache.gpu_tier_count(), 1);
            assert_eq!(cache.ram_tier_count(), 0);
        }

        #[test]
        fn test_insert_removes_from_ram() {
            let device = <CudaRuntime as numr::runtime::Runtime>::Device::default();
            let mut cache = GpuPrefixCache::new(64, 16, device, 10);

            cache.demote_to_ram(999, 50);
            assert_eq!(cache.ram_tier_count(), 1);

            // Inserting same hash into GPU should remove from RAM
            cache.insert(999, 50);
            assert_eq!(cache.ram_tier_count(), 0);
            assert_eq!(cache.gpu_tier_count(), 1);
        }
    }
}

#[cfg(feature = "cuda")]
pub use inner::GpuPrefixCache;
