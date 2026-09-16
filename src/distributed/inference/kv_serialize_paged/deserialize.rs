//! Deserializing bytes produced by [`super::serialize::serialize_paged_kv_cache`]
//! back into a `LayeredPagedKvCache`.

use super::codec::{HEADER_LEN, PAGED_MAGIC, read_header, read_le_f32_widened, read_u32_le};
use crate::inference::{BlockTable, LayeredPagedKvCache};
use crate::{DType, Runtime};
use anyhow::{Result, anyhow};

/// Per-layer deserialized K/V data that could not be directly loaded into the
/// paged cache (the public API does not expose mutable raw-block writes).
///
/// The caller is responsible for feeding this data into the paged cache via
/// whatever mechanism is available (e.g. by using the data in model forward
/// passes directly, or by writing via `update()` with proper slot mappings).
///
/// `k_data` and `v_data` are always host `f32`: f16 and bf16 widen to f32 exactly, so
/// nothing is lost. `dtype` records the dtype the bytes arrived as — the same dtype the
/// returned cache was built with — so the caller narrows back before writing.
#[derive(Debug)]
pub struct PagedLayerData {
    pub k_data: Vec<f32>,
    pub v_data: Vec<f32>,
    pub block_ids: Vec<u32>,
    pub dtype: DType,
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
    // Magic, version and dtype tag are checked before any dimension is read, so a foreign
    // or stale buffer errors on its first four bytes rather than on an absurd dimension.
    let wire_dtype = read_header(bytes, PAGED_MAGIC, "Paged KV cache")?;
    let elem_size = wire_dtype.size_in_bytes();

    if bytes.len() < HEADER_LEN + 12 {
        return Err(anyhow!(
            "Paged KV cache buffer too short: need {} bytes, got {}",
            HEADER_LEN + 12,
            bytes.len()
        ));
    }

    let num_layers = read_u32_le(bytes, HEADER_LEN)? as usize;
    let block_size = read_u32_le(bytes, HEADER_LEN + 4)? as usize;
    let seq_len = read_u32_le(bytes, HEADER_LEN + 8)? as usize;

    let mut cursor = HEADER_LEN + 12;

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
            .and_then(|n| n.checked_mul(elem_size))
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
        // is not aligned for the element type. This also makes the decode explicitly
        // little-endian, matching the documented wire format. f16 and bf16 widen to f32
        // exactly, so the returned host values are lossless whatever the wire dtype.
        let k_data = read_le_f32_widened(&bytes[cursor..cursor + data_bytes], wire_dtype)?;
        cursor += data_bytes;
        let v_data = read_le_f32_widened(&bytes[cursor..cursor + data_bytes], wire_dtype)?;
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
        let cache = LayeredPagedKvCache::<R>::new(0, 0, block_size, 1, 64, wire_dtype, device)?;
        return Ok((cache, Vec::new(), Vec::new()));
    }

    let first = &raw_layers[0];
    let mut paged_cache = LayeredPagedKvCache::<R>::new(
        num_layers,
        first.num_blocks,
        block_size,
        first.num_heads,
        first.head_dim,
        wire_dtype,
        device,
    )?;
    paged_cache.set_seq_len(seq_len);

    // The header's dtype tag was passed to the constructor above. State the symmetry as a
    // check rather than leaving it implied: if the constructor ever stops honouring the
    // requested dtype, this errors instead of returning a cache the peer's tag misdescribes.
    let built_dtype = paged_cache.layer(0).dtype();
    if built_dtype != wire_dtype {
        return Err(anyhow!(
            "Paged KV cache header declares dtype {wire_dtype}, but the reconstructed \
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
            dtype: wire_dtype,
        });
    }

    Ok((paged_cache, layer_data, block_tables))
}

#[cfg(test)]
mod tests {
    use super::super::codec::{FLAT_MAGIC, WIRE_VERSION};
    use super::super::serialize::serialize_paged_kv_cache;
    use super::super::test_support::paged_header;
    use super::*;
    use crate::inference::LayeredPagedKvCache;

    /// A hostile header whose dimensions multiply past `usize::MAX` must be rejected.
    ///
    /// Without checked arithmetic the product wraps in release builds to a small
    /// `data_bytes`, the truncation check then passes, and the enormous `num_blocks` /
    /// `num_heads` / `head_dim` reach the cache constructor anyway. The assertion pins
    /// the error, not merely "did not panic".
    #[test]
    fn test_deserialize_rejects_dimension_overflow() {
        let mut bytes = paged_header(DType::F32);
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
        let mut payload = paged_header(DType::F32);
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
        assert_eq!(layer_data[0].dtype, DType::F32);
    }

    /// The declared dtype must survive the round trip: bf16 in, bf16 cache out, and the
    /// same bf16 values on the wire.
    #[test]
    fn test_roundtrip_bf16_cache() {
        let device = crate::CpuDevice::new();
        let cache =
            LayeredPagedKvCache::<crate::CpuRuntime>::new(1, 2, 4, 2, 4, DType::BF16, &device)
                .unwrap();
        let block_table = BlockTable::new(4);

        let bytes = serialize_paged_kv_cache(&cache, &block_table).unwrap();
        assert_eq!(read_u32_le(&bytes, 0).unwrap(), PAGED_MAGIC);
        assert_eq!(read_u32_le(&bytes, 4).unwrap(), WIRE_VERSION);
        assert_eq!(read_u32_le(&bytes, 8).unwrap(), DType::BF16 as u32);

        // 2 bytes per element, not 4: the layer holds 2 blocks * 4 tokens * 2 heads * 4 dims.
        let elems = 2 * 4 * 2 * 4;
        assert_eq!(bytes.len(), HEADER_LEN + 12 + 12 + elems * 2 * 2 + 4);

        let (restored, layer_data, _tables) =
            deserialize_paged_kv_cache::<crate::CpuRuntime>(&bytes, &device).unwrap();
        assert_eq!(restored.layer(0).dtype(), DType::BF16);
        assert_eq!(layer_data[0].dtype, DType::BF16);
        assert_eq!(layer_data[0].k_data, vec![0.0f32; elems]);
        assert_eq!(layer_data[0].v_data, vec![0.0f32; elems]);
    }

    /// f32 stays the 4-byte-per-element format it always was.
    #[test]
    fn test_roundtrip_f32_cache() {
        let device = crate::CpuDevice::new();
        let cache =
            LayeredPagedKvCache::<crate::CpuRuntime>::new(1, 2, 4, 2, 4, DType::F32, &device)
                .unwrap();
        let block_table = BlockTable::new(4);

        let bytes = serialize_paged_kv_cache(&cache, &block_table).unwrap();
        assert_eq!(read_u32_le(&bytes, 8).unwrap(), DType::F32 as u32);

        let elems = 2 * 4 * 2 * 4;
        assert_eq!(bytes.len(), HEADER_LEN + 12 + 12 + elems * 4 * 2 + 4);

        let (restored, layer_data, _tables) =
            deserialize_paged_kv_cache::<crate::CpuRuntime>(&bytes, &device).unwrap();
        assert_eq!(restored.layer(0).dtype(), DType::F32);
        assert_eq!(layer_data[0].dtype, DType::F32);
        assert_eq!(layer_data[0].k_data.len(), elems);
    }

    /// A buffer shorter than the 12-byte prefix errors in the header, naming the length.
    #[test]
    fn test_deserialize_too_short_buffer() {
        let bytes = [0u8; 4];
        let Err(err) =
            deserialize_paged_kv_cache::<crate::CpuRuntime>(&bytes, &crate::CpuDevice::new())
        else {
            panic!("a 4-byte buffer must be refused");
        };
        let msg = err.to_string();
        assert!(
            msg.contains("truncated in the header"),
            "expected a header truncation error, got: {msg}"
        );
        assert!(
            msg.contains("need 12 bytes, got 4"),
            "error must name both lengths: {msg}"
        );
    }

    /// A flat buffer must be refused by the paged reader on its magic.
    #[test]
    fn test_deserialize_rejects_flat_magic() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&FLAT_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&WIRE_VERSION.to_le_bytes());
        bytes.extend_from_slice(&(DType::F32 as u32).to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes()); // num_layers
        bytes.extend_from_slice(&16u32.to_le_bytes()); // block_size
        bytes.extend_from_slice(&0u32.to_le_bytes()); // seq_len

        let Err(err) =
            deserialize_paged_kv_cache::<crate::CpuRuntime>(&bytes, &crate::CpuDevice::new())
        else {
            panic!("a paged KV cache buffer must be refused by the flat reader");
        };
        let msg = err.to_string();
        assert!(
            msg.contains("magic 0xB0057B01") && msg.contains("expected 0xB0057B02"),
            "error must name both magics: {msg}"
        );
        assert!(
            msg.contains("that is the flat KV cache magic"),
            "error must name the sibling format: {msg}"
        );
    }

    /// An unknown wire version errors, naming the version it got.
    #[test]
    fn test_deserialize_rejects_unknown_version() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&PAGED_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&7u32.to_le_bytes());
        bytes.extend_from_slice(&(DType::F32 as u32).to_le_bytes());

        let Err(err) =
            deserialize_paged_kv_cache::<crate::CpuRuntime>(&bytes, &crate::CpuDevice::new())
        else {
            panic!("an unknown wire version must be refused");
        };
        let msg = err.to_string();
        assert!(
            msg.contains("wire version 7") && msg.contains("version 1 only"),
            "error must name the version it got and the one it reads: {msg}"
        );
    }

    /// An unknown dtype tag errors, naming the tag.
    #[test]
    fn test_deserialize_rejects_unknown_dtype_tag() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&PAGED_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&WIRE_VERSION.to_le_bytes());
        bytes.extend_from_slice(&99u32.to_le_bytes());

        let Err(err) =
            deserialize_paged_kv_cache::<crate::CpuRuntime>(&bytes, &crate::CpuDevice::new())
        else {
            panic!("an unknown dtype tag must be refused");
        };
        let msg = err.to_string();
        assert!(
            msg.contains("unknown dtype tag 99"),
            "error must name the tag: {msg}"
        );
        assert!(
            msg.contains("bf16 (tag 3)"),
            "error must name the tags it does carry: {msg}"
        );
    }

    #[test]
    fn test_deserialize_zero_layers() {
        let mut bytes = paged_header(DType::F32);
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
        let mut bytes = paged_header(DType::F32);
        bytes.extend_from_slice(&1u32.to_le_bytes()); // num_layers = 1
        bytes.extend_from_slice(&16u32.to_le_bytes()); // block_size
        bytes.extend_from_slice(&0u32.to_le_bytes()); // seq_len
        // Missing layer header → should error

        let Err(err) =
            deserialize_paged_kv_cache::<crate::CpuRuntime>(&bytes, &crate::CpuDevice::new())
        else {
            panic!("a missing layer header must be refused");
        };
        assert!(
            err.to_string().contains("truncated at layer 0 header"),
            "expected a layer-header truncation error, got: {err}"
        );
    }
}
