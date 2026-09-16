//! Serializing a `LayeredPagedKvCache` into the wire format documented in `super`.

use super::codec::{PAGED_MAGIC, append_le_elements, write_header};
use crate::inference::{BlockTable, LayeredPagedKvCache};
use crate::{DType, Runtime};
use anyhow::{Result, anyhow};

/// Serialize a `LayeredPagedKvCache` and its associated `BlockTable` into bytes.
///
/// # Errors
///
/// Returns an error naming the dtype if the cache's element type is not one the wire
/// format carries (f32, f16, bf16), or if the layers do not all share one dtype — the
/// header declares a single dtype for the whole cache.
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

    // The header declares one dtype for the whole cache, taken from layer 0 and checked
    // against every other layer below. A cache with no layers carries no elements, so its
    // tag is the format's default rather than a property of absent data.
    let wire_dtype = if cache.num_layers() == 0 {
        DType::F32
    } else {
        cache.layer(0).dtype()
    };

    let mut buf: Vec<u8> = Vec::new();

    write_header(&mut buf, PAGED_MAGIC, wire_dtype)?;
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

        // The header declares one dtype for the whole cache. A layer that disagrees with
        // layer 0 would be written at layer 0's width and read back as the wrong numbers,
        // so refuse rather than emit a buffer no reader can parse.
        let layer_dtype = layer.dtype();
        if layer_dtype != wire_dtype {
            return Err(anyhow!(
                "Paged KV cache layer {layer_idx} has dtype {layer_dtype}, but the header \
                 declares {wire_dtype} from layer 0. The wire format carries one dtype for \
                 the whole cache; refusing to transfer a mixed-dtype cache."
            ));
        }

        append_le_elements(&mut buf, layer.k_cache())?;
        append_le_elements(&mut buf, layer.v_cache())?;

        let block_ids = &bt.blocks;
        let bt_len = block_ids.len() as u32;
        buf.extend_from_slice(&bt_len.to_le_bytes());
        for &id in block_ids {
            buf.extend_from_slice(&id.to_le_bytes());
        }
    }

    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A dtype the format cannot carry must be refused by name at serialize time.
    ///
    /// This replaces the old `test_serialize_rejects_non_f32_cache`, which asserted that
    /// BF16 was refused. BF16 now round-trips (see `test_roundtrip_bf16_cache` in
    /// `deserialize.rs`), so the refusal is pinned on F64, which the format still does not
    /// carry.
    #[test]
    fn test_serialize_rejects_unsupported_dtype() {
        let device = crate::CpuDevice::new();
        let cache =
            LayeredPagedKvCache::<crate::CpuRuntime>::new(1, 2, 16, 2, 4, DType::F64, &device)
                .unwrap();
        let block_table = BlockTable::new(16);

        let Err(err) = serialize_paged_kv_cache(&cache, &block_table) else {
            panic!("an F64 paged KV cache must be refused, not written at another width");
        };
        let msg = err.to_string();
        // `DType` displays lowercase (`f64`, `bf16`), so compare case-insensitively
        // rather than against a spelling the type does not produce.
        let lower = msg.to_lowercase();
        assert!(lower.contains("f64"), "error must name the dtype: {msg}");
        assert!(
            lower.contains("f32") && lower.contains("f16") && lower.contains("bf16"),
            "error must list what the format does carry: {msg}"
        );
    }
}
