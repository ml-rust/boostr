//! GPU-resident block allocator for paged KV cache.
//!
//! Tracks block allocation state on CPU but the blocks themselves reside in GPU
//! memory.  The design mirrors `CpuBlockAllocator` exactly — same `Arc<Mutex>`
//! interior mutability, same `BlockAllocator` trait — so callers can swap
//! allocators transparently.
//!
//! Gate: all types and impls are compiled only when the `cuda` feature is
//! enabled, because this allocator is only meaningful when GPU memory exists.

#[cfg(feature = "cuda")]
mod inner {
    use crate::error::{Error, Result};
    use crate::inference::memory::block_allocator::{BlockAllocator, BlockAllocatorStats, BlockId};
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    struct GpuBlockAllocatorState {
        total_blocks: usize,
        block_size: usize,
        free_list: VecDeque<BlockId>,
        /// Per-block allocated flag.
        allocated_flags: Vec<bool>,
        total_allocations: usize,
        total_frees: usize,
        peak_usage: usize,
    }

    impl GpuBlockAllocatorState {
        fn new(total_blocks: usize, block_size: usize) -> Self {
            let free_list: VecDeque<BlockId> = (0..total_blocks as BlockId).collect();
            Self {
                total_blocks,
                block_size,
                free_list,
                allocated_flags: vec![false; total_blocks],
                total_allocations: 0,
                total_frees: 0,
                peak_usage: 0,
            }
        }

        fn allocated_count(&self) -> usize {
            self.total_blocks - self.free_list.len()
        }
    }

    /// GPU-resident block allocator for paged KV cache.
    ///
    /// Allocation state is tracked on the CPU (a free-list of `BlockId`s).
    /// The actual KV cache data for each block lives in a pre-allocated GPU
    /// buffer; the allocator does not own that buffer — the KV cache manager
    /// does.  `BlockId` values index into that buffer.
    #[derive(Clone)]
    pub struct GpuBlockAllocator {
        state: Arc<Mutex<GpuBlockAllocatorState>>,
    }

    impl GpuBlockAllocator {
        /// Create a new GPU block allocator.
        ///
        /// # Arguments
        /// * `total_blocks` — total number of fixed-size blocks available in
        ///   the GPU KV cache buffer.
        /// * `block_size` — number of tokens that fit in one block.
        pub fn new(total_blocks: usize, block_size: usize) -> Self {
            Self {
                state: Arc::new(Mutex::new(GpuBlockAllocatorState::new(
                    total_blocks,
                    block_size,
                ))),
            }
        }

        /// Convenience constructor: derive `total_blocks` from a memory budget.
        ///
        /// The formula matches the one used by `CpuBlockAllocator::from_memory_budget`.
        pub fn from_memory_budget(
            memory_budget_bytes: usize,
            block_size: usize,
            num_heads: usize,
            head_dim: usize,
            dtype_size: usize,
        ) -> Self {
            let block_memory = block_size * num_heads * head_dim * dtype_size * 2;
            let total_blocks = if block_memory == 0 {
                0
            } else {
                memory_budget_bytes / block_memory
            };
            Self::new(total_blocks, block_size)
        }
    }

    impl BlockAllocator for GpuBlockAllocator {
        fn allocate(&self, count: usize) -> Result<Vec<BlockId>> {
            if count == 0 {
                return Ok(Vec::new());
            }

            let mut state = self.state.lock().map_err(|e| Error::SchedulerError {
                reason: format!("gpu block allocator mutex poisoned: {e}"),
            })?;

            if state.free_list.len() < count {
                return Err(Error::SchedulerError {
                    reason: format!(
                        "Cannot allocate {} GPU blocks: only {} free (total: {})",
                        count,
                        state.free_list.len(),
                        state.total_blocks
                    ),
                });
            }

            let blocks: Vec<BlockId> = state.free_list.drain(0..count).collect();

            for &block_id in &blocks {
                state.allocated_flags[block_id as usize] = true;
            }

            state.total_allocations += 1;

            let current_usage = state.allocated_count();
            if current_usage > state.peak_usage {
                state.peak_usage = current_usage;
            }

            Ok(blocks)
        }

        fn free(&self, blocks: &[BlockId]) -> Result<()> {
            if blocks.is_empty() {
                return Ok(());
            }

            let mut state = self.state.lock().map_err(|e| Error::SchedulerError {
                reason: format!("gpu block allocator mutex poisoned: {e}"),
            })?;

            // Validate all IDs before mutating state.
            for &block_id in blocks {
                if block_id as usize >= state.total_blocks {
                    return Err(Error::SchedulerError {
                        reason: format!(
                            "Invalid GPU block ID {} (max: {})",
                            block_id,
                            state.total_blocks.saturating_sub(1)
                        ),
                    });
                }
                if !state.allocated_flags[block_id as usize] {
                    return Err(Error::SchedulerError {
                        reason: format!("Double-free of GPU block {}", block_id),
                    });
                }
            }

            // Check for duplicates within this call.
            let mut seen = std::collections::HashSet::new();
            for &block_id in blocks {
                if !seen.insert(block_id) {
                    return Err(Error::SchedulerError {
                        reason: format!("Duplicate GPU block ID {} in free call", block_id),
                    });
                }
            }

            for &block_id in blocks {
                state.allocated_flags[block_id as usize] = false;
                state.free_list.push_back(block_id);
            }

            state.total_frees += 1;

            Ok(())
        }

        fn block_size(&self) -> usize {
            self.state
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .block_size
        }

        fn total_blocks(&self) -> usize {
            self.state
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .total_blocks
        }

        fn free_block_count(&self) -> usize {
            self.state
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .free_list
                .len()
        }

        fn stats(&self) -> BlockAllocatorStats {
            let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            BlockAllocatorStats {
                total_blocks: state.total_blocks,
                allocated_blocks: state.allocated_count(),
                free_blocks: state.free_list.len(),
                block_size: state.block_size,
                total_allocations: state.total_allocations,
                total_frees: state.total_frees,
                peak_usage: state.peak_usage,
            }
        }

        fn reset(&self) -> Result<()> {
            let mut state = self.state.lock().map_err(|e| Error::SchedulerError {
                reason: format!("gpu block allocator mutex poisoned: {e}"),
            })?;
            state.free_list.clear();
            for i in 0..state.total_blocks as BlockId {
                state.free_list.push_back(i);
            }
            for flag in &mut state.allocated_flags {
                *flag = false;
            }
            state.total_allocations = 0;
            state.total_frees = 0;
            state.peak_usage = 0;
            Ok(())
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_gpu_block_allocator_create() {
            let alloc = GpuBlockAllocator::new(100, 16);
            assert_eq!(alloc.total_blocks(), 100);
            assert_eq!(alloc.block_size(), 16);
            assert_eq!(alloc.free_block_count(), 100);
        }

        #[test]
        fn test_gpu_block_allocator_allocate_and_free() {
            let alloc = GpuBlockAllocator::new(100, 16);
            let blocks = alloc.allocate(5).unwrap();
            assert_eq!(blocks.len(), 5);
            assert_eq!(alloc.free_block_count(), 95);
            alloc.free(&blocks).unwrap();
            assert_eq!(alloc.free_block_count(), 100);
        }

        #[test]
        fn test_gpu_block_allocator_exhaustion() {
            let alloc = GpuBlockAllocator::new(10, 16);
            let blocks = alloc.allocate(10).unwrap();
            assert_eq!(alloc.free_block_count(), 0);
            assert!(alloc.allocate(1).is_err());
            alloc.free(&blocks[..5]).unwrap();
            let more = alloc.allocate(3).unwrap();
            assert_eq!(more.len(), 3);
        }

        #[test]
        fn test_gpu_block_allocator_double_free_detected() {
            let alloc = GpuBlockAllocator::new(10, 16);
            let blocks = alloc.allocate(2).unwrap();
            alloc.free(&blocks).unwrap();
            assert!(alloc.free(&blocks).is_err());
        }

        #[test]
        fn test_gpu_block_allocator_duplicate_in_free_call_detected() {
            let alloc = GpuBlockAllocator::new(10, 16);
            let blocks = alloc.allocate(2).unwrap();
            assert!(alloc.free(&[blocks[0], blocks[0]]).is_err());
        }

        #[test]
        fn test_gpu_block_allocator_shared_state() {
            let alloc1 = GpuBlockAllocator::new(100, 16);
            let alloc2 = alloc1.clone();
            alloc1.allocate(10).unwrap();
            assert_eq!(alloc1.free_block_count(), 90);
            assert_eq!(alloc2.free_block_count(), 90);
        }

        #[test]
        fn test_gpu_block_allocator_reset() {
            let alloc = GpuBlockAllocator::new(100, 16);
            alloc.allocate(50).unwrap();
            assert_eq!(alloc.free_block_count(), 50);
            alloc.reset().unwrap();
            assert_eq!(alloc.free_block_count(), 100);
            let stats = alloc.stats();
            assert_eq!(stats.total_allocations, 0);
            assert_eq!(stats.peak_usage, 0);
        }

        #[test]
        fn test_gpu_block_allocator_stats() {
            let alloc = GpuBlockAllocator::new(100, 16);
            alloc.allocate(10).unwrap();
            alloc.allocate(20).unwrap();
            let stats = alloc.stats();
            assert_eq!(stats.total_blocks, 100);
            assert_eq!(stats.allocated_blocks, 30);
            assert_eq!(stats.free_blocks, 70);
            assert_eq!(stats.block_size, 16);
            assert_eq!(stats.total_allocations, 2);
            assert_eq!(stats.peak_usage, 30);
        }

        #[test]
        fn test_gpu_block_allocator_from_memory_budget() {
            let budget = 1024 * 1024 * 1024_usize;
            let alloc = GpuBlockAllocator::from_memory_budget(budget, 16, 32, 128, 2);
            // 16 * 32 * 128 * 2 * 2 = 262_144 bytes per block
            let expected_blocks = budget / 262_144;
            assert_eq!(alloc.total_blocks(), expected_blocks);
        }

        #[test]
        fn test_gpu_block_allocator_fifo_order() {
            let alloc = GpuBlockAllocator::new(10, 16);
            let blocks1 = alloc.allocate(5).unwrap();
            assert_eq!(blocks1, vec![0, 1, 2, 3, 4]);
            let blocks2 = alloc.allocate(5).unwrap();
            assert_eq!(blocks2, vec![5, 6, 7, 8, 9]);
            alloc.free(&blocks1).unwrap();
            let blocks3 = alloc.allocate(3).unwrap();
            assert_eq!(blocks3, vec![0, 1, 2]);
        }
    }
}

#[cfg(feature = "cuda")]
pub use inner::GpuBlockAllocator;
