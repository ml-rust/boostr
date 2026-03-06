//! Paged KV cache serialization for disaggregated prefill/decode transfer.
//!
//! # Wire format — `LayeredPagedKvCache`
//!
//! ```text
//! [num_layers:  u32 LE]
//! [block_size:  u32 LE]
//! [seq_len:     u32 LE]
//! For each layer:
//!   [num_blocks:    u32 LE]
//!   [num_heads:     u32 LE]
//!   [head_dim:      u32 LE]
//!   [k_data:        num_blocks * block_size * num_heads * head_dim * 4 bytes (f32 LE)]
//!   [v_data:        num_blocks * block_size * num_heads * head_dim * 4 bytes (f32 LE)]
//!   [block_table_len: u32 LE]
//!   [block_ids:     block_table_len * 4 bytes (u32 LE)]   — BlockId = u32
//! ```

use crate::inference::{BlockTable, LayeredPagedKvCache};
use crate::{DType, Runtime};
use anyhow::{Result, anyhow};

/// Per-layer deserialized K/V data that could not be directly loaded into the
/// paged cache (the public API does not expose mutable raw-block writes).
///
/// The caller is responsible for feeding this data into the paged cache via
/// whatever mechanism is available (e.g. by using the data in model forward
/// passes directly, or by writing via `update()` with proper slot mappings).
#[derive(Debug)]
pub struct PagedLayerData {
    pub k_data: Vec<f32>,
    pub v_data: Vec<f32>,
    pub block_ids: Vec<u32>,
}

/// Serialize a `LayeredPagedKvCache` and its associated `BlockTable` into bytes.
pub fn serialize_paged_kv_cache<R>(
    cache: &LayeredPagedKvCache<R>,
    _block_table: &BlockTable,
) -> Vec<u8>
where
    R: Runtime<DType = DType>,
{
    let num_layers = cache.num_layers() as u32;
    let block_size = cache.block_size() as u32;
    let seq_len = cache.seq_len() as u32;

    let mut buf: Vec<u8> = Vec::new();

    buf.extend_from_slice(&num_layers.to_le_bytes());
    buf.extend_from_slice(&block_size.to_le_bytes());
    buf.extend_from_slice(&seq_len.to_le_bytes());

    for layer_idx in 0..num_layers as usize {
        let layer = cache.layer(layer_idx);
        let bt = cache.block_table(layer_idx);

        let num_blocks = layer.num_blocks() as u32;
        let num_heads = layer.num_heads() as u32;
        let head_dim = layer.head_dim() as u32;

        buf.extend_from_slice(&num_blocks.to_le_bytes());
        buf.extend_from_slice(&num_heads.to_le_bytes());
        buf.extend_from_slice(&head_dim.to_le_bytes());

        let k_data: Vec<f32> = layer.k_cache().to_vec::<f32>();
        let v_data: Vec<f32> = layer.v_cache().to_vec::<f32>();
        buf.extend_from_slice(bytemuck::cast_slice::<f32, u8>(&k_data));
        buf.extend_from_slice(bytemuck::cast_slice::<f32, u8>(&v_data));

        let block_ids = &bt.blocks;
        let bt_len = block_ids.len() as u32;
        buf.extend_from_slice(&bt_len.to_le_bytes());
        for &id in block_ids {
            buf.extend_from_slice(&id.to_le_bytes());
        }
    }

    buf
}

/// Deserialize bytes into a `LayeredPagedKvCache`, per-layer K/V data, and
/// per-layer block tables.
///
/// Returns `(cache, layer_data, block_tables)`. The `layer_data` contains the
/// deserialized K/V float data for each layer — the caller must write this into
/// the cache's backing tensors (the paged cache API does not expose raw block
/// writes, so this data is returned separately).
pub fn deserialize_paged_kv_cache<R>(
    bytes: &[u8],
    device: &R::Device,
) -> Result<(LayeredPagedKvCache<R>, Vec<PagedLayerData>, Vec<BlockTable>)>
where
    R: Runtime<DType = DType>,
{
    if bytes.len() < 12 {
        return Err(anyhow!(
            "Paged KV cache buffer too short: need 12 bytes, got {}",
            bytes.len()
        ));
    }

    let num_layers = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let block_size = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let seq_len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;

    let mut cursor = 12usize;

    struct RawLayerParams {
        num_blocks: usize,
        num_heads: usize,
        head_dim: usize,
        k_data: Vec<f32>,
        v_data: Vec<f32>,
        block_ids: Vec<u32>,
    }

    let mut raw_layers: Vec<RawLayerParams> = Vec::with_capacity(num_layers);

    for layer_idx in 0..num_layers {
        if cursor + 12 > bytes.len() {
            return Err(anyhow!(
                "Paged KV cache buffer truncated at layer {} header",
                layer_idx
            ));
        }
        let num_blocks = u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap()) as usize;
        let num_heads =
            u32::from_le_bytes(bytes[cursor + 4..cursor + 8].try_into().unwrap()) as usize;
        let head_dim =
            u32::from_le_bytes(bytes[cursor + 8..cursor + 12].try_into().unwrap()) as usize;
        cursor += 12;

        let numel = num_blocks * block_size * num_heads * head_dim;
        let data_bytes = numel * 4;

        if cursor + data_bytes * 2 > bytes.len() {
            return Err(anyhow!(
                "Paged KV cache buffer truncated at layer {} data",
                layer_idx
            ));
        }

        let k_data: Vec<f32> =
            bytemuck::cast_slice::<u8, f32>(&bytes[cursor..cursor + data_bytes]).to_vec();
        cursor += data_bytes;
        let v_data: Vec<f32> =
            bytemuck::cast_slice::<u8, f32>(&bytes[cursor..cursor + data_bytes]).to_vec();
        cursor += data_bytes;

        if cursor + 4 > bytes.len() {
            return Err(anyhow!(
                "Paged KV cache buffer truncated at layer {} block table length",
                layer_idx
            ));
        }
        let bt_len = u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap()) as usize;
        cursor += 4;

        if cursor + bt_len * 4 > bytes.len() {
            return Err(anyhow!(
                "Paged KV cache buffer truncated at layer {} block table data",
                layer_idx
            ));
        }
        let mut block_ids = Vec::with_capacity(bt_len);
        for i in 0..bt_len {
            let offset = cursor + i * 4;
            let id = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
            block_ids.push(id);
        }
        cursor += bt_len * 4;

        raw_layers.push(RawLayerParams {
            num_blocks,
            num_heads,
            head_dim,
            k_data,
            v_data,
            block_ids,
        });
    }

    if raw_layers.is_empty() {
        let cache = LayeredPagedKvCache::<R>::new(0, 0, block_size, 1, 64, DType::F32, device);
        return Ok((cache, Vec::new(), Vec::new()));
    }

    let first = &raw_layers[0];
    let mut paged_cache = LayeredPagedKvCache::<R>::new(
        num_layers,
        first.num_blocks,
        block_size,
        first.num_heads,
        first.head_dim,
        DType::F32,
        device,
    );
    paged_cache.set_seq_len(seq_len);

    let mut block_tables: Vec<BlockTable> = Vec::with_capacity(num_layers);
    let mut layer_data: Vec<PagedLayerData> = Vec::with_capacity(num_layers);

    for params in &raw_layers {
        let mut bt = BlockTable::new(block_size);
        bt.blocks = params.block_ids.clone();
        bt.num_tokens = seq_len;
        block_tables.push(bt);

        layer_data.push(PagedLayerData {
            k_data: params.k_data.clone(),
            v_data: params.v_data.clone(),
            block_ids: params.block_ids.clone(),
        });
    }

    Ok((paged_cache, layer_data, block_tables))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_too_short_buffer() {
        let bytes = [0u8; 4];
        let result =
            deserialize_paged_kv_cache::<crate::CpuRuntime>(&bytes, &crate::CpuDevice::new());
        assert!(result.is_err());
    }

    #[test]
    fn test_deserialize_zero_layers() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0u32.to_le_bytes()); // num_layers = 0
        bytes.extend_from_slice(&16u32.to_le_bytes()); // block_size
        bytes.extend_from_slice(&0u32.to_le_bytes()); // seq_len

        let (cache, layer_data, block_tables) =
            deserialize_paged_kv_cache::<crate::CpuRuntime>(&bytes, &crate::CpuDevice::new())
                .unwrap();
        assert_eq!(cache.num_layers(), 0);
        assert!(layer_data.is_empty());
        assert!(block_tables.is_empty());
    }

    #[test]
    fn test_deserialize_truncated_layer_header() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1u32.to_le_bytes()); // num_layers = 1
        bytes.extend_from_slice(&16u32.to_le_bytes()); // block_size
        bytes.extend_from_slice(&0u32.to_le_bytes()); // seq_len
        // Missing layer header → should error

        let result =
            deserialize_paged_kv_cache::<crate::CpuRuntime>(&bytes, &crate::CpuDevice::new());
        assert!(result.is_err());
    }
}
