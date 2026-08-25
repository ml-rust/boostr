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
//!
//! The format carries **no dtype tag**: element data is f32 little-endian and nothing
//! else. A cache of any other dtype is refused by [`serialize_paged_kv_cache`] rather
//! than reinterpreted or converted.

use crate::inference::{BlockTable, LayeredPagedKvCache};
use crate::{DType, Runtime};
use anyhow::{Result, anyhow};

/// The only element type this wire format can carry.
///
/// The header records dimensions and block ids but no dtype tag, so both sides are
/// hard-wired to f32.
const WIRE_DTYPE: DType = DType::F32;

/// Read a little-endian `u32` at `offset`, erroring rather than panicking when the
/// buffer is too short.
///
/// The callers bounds-check before each read, so this is defence in depth: it keeps a
/// truncated or hostile payload from turning into a slice-index panic inside a server.
fn read_u32_le(bytes: &[u8], offset: usize) -> Result<u32> {
    let end = offset.checked_add(4).ok_or_else(|| {
        anyhow!(
            "Paged KV cache read offset {} overflows the address space",
            offset
        )
    })?;
    let slice = bytes.get(offset..end).ok_or_else(|| {
        anyhow!(
            "Paged KV cache buffer too short: need {} bytes, got {}",
            end,
            bytes.len()
        )
    })?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

/// Decode a little-endian `f32` run without requiring 4-byte alignment.
///
/// `len` is a multiple of 4 and already bounds-checked by the caller, so the trailing
/// partial chunk `chunks_exact` would leave is always empty.
pub(super) fn read_f32_le_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&c| f32::from_le_bytes(c))
        .collect()
}

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
) -> Result<Vec<u8>>
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

        // `try_to_vec::<f32>()` does not check dtype: on a bf16 layer it copies
        // `numel * 4` bytes out of a `numel * 2` byte allocation and reinterprets the
        // result as f32 — a silent wrong answer at the peer. Converting to f32 is no
        // better, since the peer has no dtype tag telling it the values changed.
        let layer_dtype = layer.dtype();
        if layer_dtype != WIRE_DTYPE {
            return Err(anyhow!(
                "Paged KV cache layer {layer_idx} has dtype {layer_dtype}, but the paged KV \
                 cache wire format carries {WIRE_DTYPE} only (the header has no dtype tag). \
                 Refusing to transfer: reinterpreting {layer_dtype} bytes as {WIRE_DTYPE} \
                 corrupts the cache, and converting would change the values the peer \
                 receives without telling it. Allocate the cache with \
                 DType::{WIRE_DTYPE:?} for disaggregated transfer."
            ));
        }

        let k_data: Vec<f32> = layer.k_cache().try_to_vec::<f32>()?;
        let v_data: Vec<f32> = layer.v_cache().try_to_vec::<f32>()?;
        // Explicit little-endian, matching the documented wire format and the reader.
        // `cast_slice` would emit native-endian bytes, which disagree on a big-endian host.
        for &x in &k_data {
            buf.extend_from_slice(&x.to_le_bytes());
        }
        for &x in &v_data {
            buf.extend_from_slice(&x.to_le_bytes());
        }

        let block_ids = &bt.blocks;
        let bt_len = block_ids.len() as u32;
        buf.extend_from_slice(&bt_len.to_le_bytes());
        for &id in block_ids {
            buf.extend_from_slice(&id.to_le_bytes());
        }
    }

    Ok(buf)
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

    let num_layers = read_u32_le(bytes, 0)? as usize;
    let block_size = read_u32_le(bytes, 4)? as usize;
    let seq_len = read_u32_le(bytes, 8)? as usize;

    let mut cursor = 12usize;

    struct RawLayerParams {
        num_blocks: usize,
        num_heads: usize,
        head_dim: usize,
        k_data: Vec<f32>,
        v_data: Vec<f32>,
        block_ids: Vec<u32>,
    }

    // `num_layers` is attacker-controlled: reserve lazily rather than trusting it.
    let mut raw_layers: Vec<RawLayerParams> = Vec::new();

    for layer_idx in 0..num_layers {
        if cursor + 12 > bytes.len() {
            return Err(anyhow!(
                "Paged KV cache buffer truncated at layer {} header",
                layer_idx
            ));
        }
        let num_blocks = read_u32_le(bytes, cursor)? as usize;
        let num_heads = read_u32_le(bytes, cursor + 4)? as usize;
        let head_dim = read_u32_le(bytes, cursor + 8)? as usize;
        cursor += 12;

        // Every factor here comes off the wire. In release builds `*` wraps, so an
        // unchecked product can land back inside the buffer and slip past the
        // truncation check below while the header still claims enormous dimensions.
        let data_bytes = num_blocks
            .checked_mul(block_size)
            .and_then(|n| n.checked_mul(num_heads))
            .and_then(|n| n.checked_mul(head_dim))
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| {
                anyhow!(
                    "Paged KV cache layer {} dimensions overflow: \
                     num_blocks={} block_size={} num_heads={} head_dim={}",
                    layer_idx,
                    num_blocks,
                    block_size,
                    num_heads,
                    head_dim
                )
            })?;

        let both = data_bytes
            .checked_mul(2)
            .and_then(|n| n.checked_add(cursor))
            .ok_or_else(|| {
                anyhow!(
                    "Paged KV cache layer {} data size overflows the address space",
                    layer_idx
                )
            })?;
        if both > bytes.len() {
            return Err(anyhow!(
                "Paged KV cache buffer truncated at layer {} data",
                layer_idx
            ));
        }

        // Read element-wise rather than `bytemuck::cast_slice`: a received buffer
        // carries no alignment guarantee, and `cast_slice` panics when `&bytes[cursor..]`
        // is not 4-aligned. This also makes the f32s explicitly little-endian, matching
        // the documented wire format.
        let k_data = read_f32_le_vec(&bytes[cursor..cursor + data_bytes]);
        cursor += data_bytes;
        let v_data = read_f32_le_vec(&bytes[cursor..cursor + data_bytes]);
        cursor += data_bytes;

        if cursor + 4 > bytes.len() {
            return Err(anyhow!(
                "Paged KV cache buffer truncated at layer {} block table length",
                layer_idx
            ));
        }
        let bt_len = read_u32_le(bytes, cursor)? as usize;
        cursor += 4;

        let bt_end = bt_len
            .checked_mul(4)
            .and_then(|n| n.checked_add(cursor))
            .ok_or_else(|| {
                anyhow!(
                    "Paged KV cache layer {} block table length {} overflows",
                    layer_idx,
                    bt_len
                )
            })?;
        if bt_end > bytes.len() {
            return Err(anyhow!(
                "Paged KV cache buffer truncated at layer {} block table data",
                layer_idx
            ));
        }
        // `bt_len` is bounded by the buffer now, so reserving it cannot be used to
        // force a large allocation from a 4-byte field.
        let mut block_ids = Vec::with_capacity(bt_len);
        for i in 0..bt_len {
            block_ids.push(read_u32_le(bytes, cursor + i * 4)?);
        }
        cursor = bt_end;

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
        let cache = LayeredPagedKvCache::<R>::new(0, 0, block_size, 1, 64, WIRE_DTYPE, device)?;
        return Ok((cache, Vec::new(), Vec::new()));
    }

    let first = &raw_layers[0];
    let mut paged_cache = LayeredPagedKvCache::<R>::new(
        num_layers,
        first.num_blocks,
        block_size,
        first.num_heads,
        first.head_dim,
        WIRE_DTYPE,
        device,
    )?;
    paged_cache.set_seq_len(seq_len);

    // The bytes carry no dtype tag, so the reader decodes them as `WIRE_DTYPE` and
    // requested a `WIRE_DTYPE` cache above. State that symmetry as a check rather than
    // leaving it implied.
    let built_dtype = paged_cache.layer(0).dtype();
    if built_dtype != WIRE_DTYPE {
        return Err(anyhow!(
            "Paged KV cache wire format decodes {WIRE_DTYPE} only, but the reconstructed \
             cache has dtype {built_dtype}"
        ));
    }

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

    /// A hostile header whose dimensions multiply past `usize::MAX` must be rejected.
    ///
    /// Without checked arithmetic the product wraps in release builds to a small
    /// `data_bytes`, the truncation check then passes, and the enormous `num_blocks` /
    /// `num_heads` / `head_dim` reach the cache constructor anyway. The assertion pins
    /// the error, not merely "did not panic".
    #[test]
    fn test_deserialize_rejects_dimension_overflow() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1u32.to_le_bytes()); // num_layers
        bytes.extend_from_slice(&u32::MAX.to_le_bytes()); // block_size
        bytes.extend_from_slice(&0u32.to_le_bytes()); // seq_len
        bytes.extend_from_slice(&u32::MAX.to_le_bytes()); // num_blocks
        bytes.extend_from_slice(&u32::MAX.to_le_bytes()); // num_heads
        bytes.extend_from_slice(&u32::MAX.to_le_bytes()); // head_dim

        // `expect_err` would require `Debug` on the success type, which the cache
        // does not implement.
        let Err(err) =
            deserialize_paged_kv_cache::<crate::CpuRuntime>(&bytes, &crate::CpuDevice::new())
        else {
            panic!("overflowing dimensions must be rejected");
        };
        assert!(
            err.to_string().contains("overflow"),
            "expected an overflow error, got: {err}"
        );
    }

    /// The reader must not require the input buffer to be 4-byte aligned.
    ///
    /// A buffer arriving off the wire carries no alignment guarantee, and
    /// `bytemuck::cast_slice::<u8, f32>` panics outright on a misaligned slice. Reading
    /// a deliberately offset copy reproduces that panic if the element-wise decode is
    /// reverted.
    #[test]
    fn test_deserialize_accepts_misaligned_buffer() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&1u32.to_le_bytes()); // num_layers
        payload.extend_from_slice(&1u32.to_le_bytes()); // block_size
        payload.extend_from_slice(&1u32.to_le_bytes()); // seq_len
        payload.extend_from_slice(&1u32.to_le_bytes()); // num_blocks
        payload.extend_from_slice(&1u32.to_le_bytes()); // num_heads
        payload.extend_from_slice(&2u32.to_le_bytes()); // head_dim
        for x in [1.0f32, 2.0] {
            payload.extend_from_slice(&x.to_le_bytes()); // k_data
        }
        for x in [3.0f32, 4.0] {
            payload.extend_from_slice(&x.to_le_bytes()); // v_data
        }
        payload.extend_from_slice(&0u32.to_le_bytes()); // block_table_len

        // Shift by one byte so the f32 runs start at an odd address.
        let mut shifted = vec![0u8];
        shifted.extend_from_slice(&payload);

        let (_cache, layer_data, _tables) = deserialize_paged_kv_cache::<crate::CpuRuntime>(
            &shifted[1..],
            &crate::CpuDevice::new(),
        )
        .expect("a misaligned buffer must deserialize");
        assert_eq!(layer_data[0].k_data, vec![1.0, 2.0]);
        assert_eq!(layer_data[0].v_data, vec![3.0, 4.0]);
    }

    /// The paged serializer carries the same f32-only wire format and the same
    /// unchecked `try_to_vec::<f32>()`, so it must refuse a non-f32 cache by name.
    #[test]
    fn test_serialize_rejects_non_f32_cache() {
        let device = crate::CpuDevice::new();
        let cache =
            LayeredPagedKvCache::<crate::CpuRuntime>::new(1, 2, 16, 2, 4, DType::BF16, &device)
                .unwrap();
        let block_table = BlockTable::new(16);

        let Err(err) = serialize_paged_kv_cache(&cache, &block_table) else {
            panic!("a BF16 paged KV cache must be refused, not serialized as f32");
        };
        let msg = err.to_string();
        // `DType` displays lowercase (`bf16`, `f32`), so compare case-insensitively
        // rather than against a spelling the type does not produce.
        let lower = msg.to_lowercase();
        assert!(lower.contains("bf16"), "error must name the dtype: {msg}");
        assert!(
            lower.contains("f32") && lower.contains("only"),
            "error must state the wire format is f32-only: {msg}"
        );
    }

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
