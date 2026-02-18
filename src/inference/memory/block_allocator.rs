//! Block-based memory allocator for KV cache management
//!
//! Implements the vLLM PagedAttention pattern where KV cache is divided into
//! fixed-size blocks that can be allocated/freed independently.

use crate::error::{Error, Result};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// Statistics for block allocator
#[derive(Debug, Clone, Copy, Default)]
pub struct BlockAllocatorStats {
    pub total_blocks: usize,
    pub allocated_blocks: usize,
    pub free_blocks: usize,
    pub block_size: usize,
    pub total_allocations: usize,
    pub total_frees: usize,
    pub peak_usage: usize,
}

/// Block ID type - represents a physical block in the pool
pub type BlockId = u32;

/// Block allocator trait for fixed-size block management
pub trait BlockAllocator: Send + Sync + Clone {
    fn allocate(&self, count: usize) -> Result<Vec<BlockId>>;
    fn free(&self, blocks: &[BlockId]) -> Result<()>;
    fn block_size(&self) -> usize;
    fn total_blocks(&self) -> usize;
    fn free_block_count(&self) -> usize;
    fn stats(&self) -> BlockAllocatorStats;
    fn reset(&self) -> Result<()>;

    fn can_allocate(&self, count: usize) -> bool {
        self.free_block_count() >= count
    }

    fn block_memory_size(&self, num_heads: usize, head_dim: usize, dtype_size: usize) -> usize {
        self.block_size() * num_heads * head_dim * dtype_size * 2
    }

    fn total_memory_size(&self, num_heads: usize, head_dim: usize, dtype_size: usize) -> usize {
        self.total_blocks() * self.block_memory_size(num_heads, head_dim, dtype_size)
    }
}

struct BlockAllocatorState {
    total_blocks: usize,
    block_size: usize,
    free_list: VecDeque<BlockId>,
    total_allocations: usize,
    total_frees: usize,
    peak_usage: usize,
}

impl BlockAllocatorState {
    fn new(total_blocks: usize, block_size: usize) -> Self {
        let free_list: VecDeque<BlockId> = (0..total_blocks as BlockId).collect();
        Self {
            total_blocks,
            block_size,
            free_list,
            total_allocations: 0,
            total_frees: 0,
            peak_usage: 0,
        }
    }

    fn allocated_count(&self) -> usize {
        self.total_blocks - self.free_list.len()
    }
}

/// CPU-based block allocator implementation
#[derive(Clone)]
pub struct CpuBlockAllocator {
    state: Arc<Mutex<BlockAllocatorState>>,
}

impl CpuBlockAllocator {
    pub fn new(total_blocks: usize, block_size: usize) -> Self {
        Self {
            state: Arc::new(Mutex::new(BlockAllocatorState::new(
                total_blocks,
                block_size,
            ))),
        }
    }

    pub fn from_memory_budget(
        memory_budget_bytes: usize,
        block_size: usize,
        num_heads: usize,
        head_dim: usize,
        dtype_size: usize,
    ) -> Self {
        let block_memory = block_size * num_heads * head_dim * dtype_size * 2;
        let total_blocks = memory_budget_bytes / block_memory;
        Self::new(total_blocks, block_size)
    }
}

impl BlockAllocator for CpuBlockAllocator {
    fn allocate(&self, count: usize) -> Result<Vec<BlockId>> {
        if count == 0 {
            return Ok(Vec::new());
        }

        let mut state = self.state.lock().map_err(|e| Error::SchedulerError {
            reason: format!("block allocator mutex poisoned: {e}"),
        })?;

        if state.free_list.len() < count {
            return Err(Error::SchedulerError {
                reason: format!(
                    "Cannot allocate {} blocks: only {} free (total: {})",
                    count,
                    state.free_list.len(),
                    state.total_blocks
                ),
            });
        }

        let blocks: Vec<BlockId> = state.free_list.drain(0..count).collect();

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
            reason: format!("block allocator mutex poisoned: {e}"),
        })?;

        for &block_id in blocks {
            if block_id as usize >= state.total_blocks {
                return Err(Error::SchedulerError {
                    reason: format!(
                        "Invalid block ID {} (max: {})",
                        block_id,
                        state.total_blocks - 1
                    ),
                });
            }
        }

        for &block_id in blocks {
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
            reason: format!("block allocator mutex poisoned: {e}"),
        })?;
        state.free_list.clear();
        for i in 0..state.total_blocks as BlockId {
            state.free_list.push_back(i);
        }
        state.total_allocations = 0;
        state.total_frees = 0;
        state.peak_usage = 0;
        Ok(())
    }
}

/// No-op block allocator for backends that don't support block management
#[derive(Clone, Default)]
pub struct NoOpBlockAllocator {
    block_size: usize,
}

impl NoOpBlockAllocator {
    pub fn new(block_size: usize) -> Self {
        Self { block_size }
    }
}

impl BlockAllocator for NoOpBlockAllocator {
    fn allocate(&self, count: usize) -> Result<Vec<BlockId>> {
        Ok((0..count as BlockId).collect())
    }

    fn free(&self, _blocks: &[BlockId]) -> Result<()> {
        Ok(())
    }

    fn block_size(&self) -> usize {
        self.block_size
    }

    fn total_blocks(&self) -> usize {
        usize::MAX
    }

    fn free_block_count(&self) -> usize {
        usize::MAX
    }

    fn stats(&self) -> BlockAllocatorStats {
        BlockAllocatorStats {
            block_size: self.block_size,
            ..Default::default()
        }
    }

    fn reset(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_allocator_create() {
        let alloc = CpuBlockAllocator::new(100, 16);
        assert_eq!(alloc.total_blocks(), 100);
        assert_eq!(alloc.block_size(), 16);
        assert_eq!(alloc.free_block_count(), 100);
    }

    #[test]
    fn test_block_allocator_allocate() {
        let alloc = CpuBlockAllocator::new(100, 16);
        let blocks = alloc.allocate(5).unwrap();
        assert_eq!(blocks.len(), 5);
        assert_eq!(alloc.free_block_count(), 95);

        let mut sorted = blocks.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 5);
    }

    #[test]
    fn test_block_allocator_free() {
        let alloc = CpuBlockAllocator::new(100, 16);
        let blocks = alloc.allocate(5).unwrap();
        assert_eq!(alloc.free_block_count(), 95);
        alloc.free(&blocks).unwrap();
        assert_eq!(alloc.free_block_count(), 100);
    }

    #[test]
    fn test_block_allocator_exhaustion() {
        let alloc = CpuBlockAllocator::new(10, 16);
        let blocks = alloc.allocate(10).unwrap();
        assert_eq!(alloc.free_block_count(), 0);

        let result = alloc.allocate(1);
        assert!(result.is_err());

        alloc.free(&blocks[..5]).unwrap();
        assert_eq!(alloc.free_block_count(), 5);

        let more_blocks = alloc.allocate(3).unwrap();
        assert_eq!(more_blocks.len(), 3);
    }

    #[test]
    fn test_block_allocator_stats() {
        let alloc = CpuBlockAllocator::new(100, 16);
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
    fn test_block_allocator_reset() {
        let alloc = CpuBlockAllocator::new(100, 16);
        alloc.allocate(50).unwrap();
        assert_eq!(alloc.free_block_count(), 50);

        alloc.reset().unwrap();
        assert_eq!(alloc.free_block_count(), 100);

        let stats = alloc.stats();
        assert_eq!(stats.total_allocations, 0);
        assert_eq!(stats.peak_usage, 0);
    }

    #[test]
    fn test_block_allocator_can_allocate() {
        let alloc = CpuBlockAllocator::new(10, 16);
        assert!(alloc.can_allocate(10));
        assert!(!alloc.can_allocate(11));

        alloc.allocate(5).unwrap();
        assert!(alloc.can_allocate(5));
        assert!(!alloc.can_allocate(6));
    }

    #[test]
    fn test_block_allocator_memory_size() {
        let alloc = CpuBlockAllocator::new(100, 16);
        let block_size = alloc.block_memory_size(32, 128, 2);
        assert_eq!(block_size, 16 * 32 * 128 * 2 * 2);

        let total_size = alloc.total_memory_size(32, 128, 2);
        assert_eq!(total_size, 100 * block_size);
    }

    #[test]
    fn test_block_allocator_from_memory_budget() {
        let budget = 1024 * 1024 * 1024;
        let alloc = CpuBlockAllocator::from_memory_budget(budget, 16, 32, 128, 2);
        let expected_blocks = budget / 262_144;
        assert_eq!(alloc.total_blocks(), expected_blocks);
    }

    #[test]
    fn test_noop_allocator() {
        let alloc = NoOpBlockAllocator::new(16);
        let blocks = alloc.allocate(1000).unwrap();
        assert_eq!(blocks.len(), 1000);
        alloc.free(&blocks).unwrap();
        assert!(alloc.can_allocate(usize::MAX));
    }

    #[test]
    fn test_block_allocator_allocate_zero() {
        let alloc = CpuBlockAllocator::new(100, 16);
        let blocks = alloc.allocate(0).unwrap();
        assert!(blocks.is_empty());
        assert_eq!(alloc.free_block_count(), 100);
    }

    #[test]
    fn test_block_allocator_free_invalid() {
        let alloc = CpuBlockAllocator::new(10, 16);
        let result = alloc.free(&[100]);
        assert!(result.is_err());
    }

    #[test]
    fn test_block_allocator_shared_state() {
        let alloc1 = CpuBlockAllocator::new(100, 16);
        let alloc2 = alloc1.clone();

        alloc1.allocate(10).unwrap();

        assert_eq!(alloc1.free_block_count(), 90);
        assert_eq!(alloc2.free_block_count(), 90);
    }

    #[test]
    fn test_block_allocator_fifo_order() {
        let alloc = CpuBlockAllocator::new(10, 16);

        let blocks1 = alloc.allocate(5).unwrap();
        assert_eq!(blocks1, vec![0, 1, 2, 3, 4]);

        let blocks2 = alloc.allocate(5).unwrap();
        assert_eq!(blocks2, vec![5, 6, 7, 8, 9]);

        alloc.free(&blocks1).unwrap();

        let blocks3 = alloc.allocate(3).unwrap();
        assert_eq!(blocks3, vec![0, 1, 2]);
    }
}
