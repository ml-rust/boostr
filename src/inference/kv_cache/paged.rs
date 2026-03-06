//! PagedKvCache and LayeredPagedKvCache - Block-based paged KV cache for PagedAttention

use crate::error::{Error, Result};
use crate::inference::memory::{BlockAllocator, BlockId, BlockTable};
use crate::ops::traits::KvCacheOps;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Paged KV cache for a single attention layer.
///
/// Uses block-based storage with slot mapping for PagedAttention.
/// Cache layout: `[num_blocks, block_size, num_heads, head_dim]`
pub struct PagedKvCache<R: Runtime> {
    k_cache: Tensor<R>,
    v_cache: Tensor<R>,
    num_blocks: usize,
    block_size: usize,
    num_heads: usize,
    head_dim: usize,
    dtype: DType,
}

impl<R: Runtime<DType = DType>> PagedKvCache<R> {
    /// Create a new paged KV cache.
    pub fn new(
        num_blocks: usize,
        block_size: usize,
        num_heads: usize,
        head_dim: usize,
        dtype: DType,
        device: &R::Device,
    ) -> Self {
        let shape = [num_blocks, block_size, num_heads, head_dim];
        let k_cache = Tensor::<R>::zeros(&shape, dtype, device);
        let v_cache = Tensor::<R>::zeros(&shape, dtype, device);

        Self {
            k_cache,
            v_cache,
            num_blocks,
            block_size,
            num_heads,
            head_dim,
            dtype,
        }
    }

    /// Write new K/V tokens into cache blocks using slot_mapping.
    ///
    /// - `key`, `value`: `[num_tokens, num_heads, head_dim]`
    /// - `slot_mapping`: `[num_tokens]` (I64) — maps each token to a slot
    ///
    /// Slot `s` maps to block `s / block_size`, offset `s % block_size`.
    pub fn update(
        &self,
        key: &Tensor<R>,
        value: &Tensor<R>,
        slot_mapping: &Tensor<R>,
        client: &R::Client,
    ) -> Result<()>
    where
        R::Client: KvCacheOps<R>,
    {
        client.reshape_and_cache(
            key,
            value,
            &self.k_cache,
            &self.v_cache,
            slot_mapping,
            self.block_size,
        )
    }

    pub fn k_cache(&self) -> &Tensor<R> {
        &self.k_cache
    }

    pub fn v_cache(&self) -> &Tensor<R> {
        &self.v_cache
    }

    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

/// Multi-layer paged KV cache for a full transformer model.
///
/// Wraps `Vec<PagedKvCache<R>>` (one per layer) with per-sequence `BlockTable`s.
/// All layers share the same block allocator and block size.
pub struct LayeredPagedKvCache<R: Runtime> {
    layers: Vec<PagedKvCache<R>>,
    block_tables: Vec<BlockTable>,
    block_size: usize,
    seq_len: usize,
}

impl<R: Runtime<DType = DType>> LayeredPagedKvCache<R> {
    /// Create a new layered paged KV cache.
    ///
    /// Allocates `num_blocks_per_layer` blocks for each of `num_layers` layers.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num_layers: usize,
        num_blocks_per_layer: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype: DType,
        device: &R::Device,
    ) -> Self {
        let mut layers = Vec::with_capacity(num_layers);
        let mut block_tables = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(PagedKvCache::new(
                num_blocks_per_layer,
                block_size,
                num_kv_heads,
                head_dim,
                dtype,
                device,
            ));
            block_tables.push(BlockTable::new(block_size));
        }
        Self {
            layers,
            block_tables,
            block_size,
            seq_len: 0,
        }
    }

    pub fn layer(&self, idx: usize) -> &PagedKvCache<R> {
        &self.layers[idx]
    }

    pub fn block_table(&self, idx: usize) -> &BlockTable {
        &self.block_tables[idx]
    }

    pub fn block_table_mut(&mut self, idx: usize) -> &mut BlockTable {
        &mut self.block_tables[idx]
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    pub fn set_seq_len(&mut self, seq_len: usize) {
        self.seq_len = seq_len;
        for bt in &mut self.block_tables {
            bt.set_num_tokens(seq_len);
        }
    }

    /// Allocate blocks for `additional_tokens` new tokens across all layers.
    ///
    /// Uses the same allocator for all layers so each layer gets the same
    /// logical→physical mapping (block tables are independent per layer but
    /// the physical block pool is shared).
    pub fn allocate_blocks<A: BlockAllocator>(
        &mut self,
        additional_tokens: usize,
        allocator: &A,
    ) -> Result<()> {
        // All layers share the same logical→physical mapping.
        // Allocate once, reuse same block IDs for every layer.
        let needed = self.block_tables[0].additional_blocks_needed(additional_tokens);
        if needed == 0 {
            return Ok(());
        }

        let blocks = allocator.allocate(needed)?;
        for bt in &mut self.block_tables {
            bt.append_blocks(blocks.clone());
        }
        Ok(())
    }

    /// Compute slot mapping for token positions `start..start+count`.
    ///
    /// Returns a Vec<i32> suitable for uploading as a tensor.
    /// Uses layer 0's block table (all layers have the same logical mapping).
    pub fn compute_slot_mapping(&self, start: usize, count: usize) -> Result<Vec<i32>> {
        let bt = &self.block_tables[0];
        let mut slots = Vec::with_capacity(count);
        for pos in start..start + count {
            let (block_id, offset) = bt.get_slot(pos).ok_or_else(|| Error::InferenceError {
                reason: format!(
                    "no block allocated for token position {} (have {} blocks of size {})",
                    pos,
                    bt.num_blocks(),
                    self.block_size
                ),
            })?;
            slots.push((block_id as i32) * (self.block_size as i32) + (offset as i32));
        }
        Ok(slots)
    }

    /// Get block table in device format (i32 vec) for layer `idx`.
    pub fn block_table_device_format(&self, idx: usize) -> Vec<i32> {
        self.block_tables[idx].to_device_format()
    }

    /// Set pre-existing block IDs on all layers' block tables.
    ///
    /// Used when the prefix cache returns already-allocated blocks that should
    /// be reused directly instead of allocating new ones.
    pub fn set_blocks(&mut self, blocks: Vec<BlockId>) {
        for bt in &mut self.block_tables {
            bt.blocks = blocks.clone();
        }
    }

    /// Reset all block tables and sequence length (does NOT free blocks from allocator).
    pub fn reset(&mut self) {
        self.seq_len = 0;
        for bt in &mut self.block_tables {
            bt.blocks.clear();
            bt.num_tokens = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_paged_kv_cache_update() {
        let (client, device) = crate::test_utils::cpu_setup();

        let num_blocks = 4;
        let block_size = 8;
        let num_heads = 2;
        let head_dim = 4;

        let cache = PagedKvCache::<CpuRuntime>::new(
            num_blocks,
            block_size,
            num_heads,
            head_dim,
            DType::F32,
            &device,
        );

        // 3 tokens to write
        let num_tokens = 3;
        let k_data: Vec<f32> = (0..num_tokens * num_heads * head_dim)
            .map(|i| i as f32 * 0.1)
            .collect();
        let v_data: Vec<f32> = (0..num_tokens * num_heads * head_dim)
            .map(|i| i as f32 * 0.2)
            .collect();
        let key =
            Tensor::<CpuRuntime>::from_slice(&k_data, &[num_tokens, num_heads, head_dim], &device);
        let value =
            Tensor::<CpuRuntime>::from_slice(&v_data, &[num_tokens, num_heads, head_dim], &device);

        // slot_mapping: token 0 → slot 0, token 1 → slot 1, token 2 → slot 9
        let slots: Vec<i32> = vec![0, 1, 9];
        let slot_mapping = Tensor::<CpuRuntime>::from_slice(&slots, &[num_tokens], &device);

        cache.update(&key, &value, &slot_mapping, &client).unwrap();

        // Verify cache shape
        assert_eq!(
            cache.k_cache().shape(),
            &[num_blocks, block_size, num_heads, head_dim]
        );
        assert_eq!(
            cache.v_cache().shape(),
            &[num_blocks, block_size, num_heads, head_dim]
        );

        // Verify written data: slot 0 = block 0, offset 0
        let kc = cache.k_cache().to_vec::<f32>();
        // k_data[0] = 0.0 (token 0, head 0, dim 0)
        assert!((kc[0] - 0.0).abs() < 1e-6);
        // k_data for token 0, head 0, dim 1 = 0.1
        assert!((kc[1] - 0.1).abs() < 1e-6);
    }
}
