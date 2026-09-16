//! Serializing a `LayeredKvCache` into the flat wire format documented in `super`.

use crate::distributed::inference::kv_serialize_paged::{
    FLAT_MAGIC, HEADER_LEN, append_le_elements, unsupported_dtype_err, write_header,
};
use crate::inference::LayeredKvCache;
use crate::{DType, IndexingOps, Runtime};
use anyhow::{Result, anyhow};

/// Reject a cache tensor whose dtype disagrees with the one the header declares.
///
/// The header carries a single dtype for the whole cache. A layer that disagrees would
/// be written at the declared width and read back as the wrong numbers, so refuse.
fn check_wire_dtype(layer_idx: usize, which: &str, dtype: DType, wire_dtype: DType) -> Result<()> {
    if dtype != wire_dtype {
        return Err(anyhow!(
            "KV cache layer {layer_idx} {which} tensor has dtype {dtype}, but the header \
             declares {wire_dtype} from layer 0. The wire format carries one dtype for the \
             whole cache; refusing to transfer a mixed-dtype cache."
        ));
    }
    Ok(())
}

/// Serialize a `LayeredKvCache` into bytes for network transfer.
///
/// Extracts only the *used* portion of each layer's K/V tensors (up to
/// `seq_len` tokens). The resulting bytes contain enough metadata for the
/// receiving side to reconstruct a fresh cache with the same dimensions.
///
/// Note: This function copies each tensor from the device to CPU memory. For GPU
/// tensors this is an intentional transfer — disaggregated inference requires
/// moving the KV cache over the network, so a CPU copy is unavoidable.
///
/// # Errors
///
/// Returns an error naming the offending dtype if the cache's element type is not one
/// the wire format carries (f32, f16, bf16), or if the layers do not all share one
/// dtype — the header declares a single dtype for the whole cache.
pub fn serialize_kv_cache<R>(cache: &LayeredKvCache<R>) -> Result<Vec<u8>>
where
    R: Runtime<DType = DType>,
    R::Client: IndexingOps<R>,
{
    let num_layers = cache.num_layers() as u32;
    let seq_len = cache.seq_len() as u32;

    // The header declares one dtype for the whole cache, taken from layer 0 and checked
    // against every other layer below. A cache with no layers carries no elements, so its
    // tag is the format's default rather than a property of absent data.
    let wire_dtype = match cache.layer(0) {
        Some(layer) => layer.k_cache_raw().dtype(),
        None => DType::F32,
    };
    let elem_size = match wire_dtype {
        DType::F32 | DType::F16 | DType::BF16 => wire_dtype.size_in_bytes(),
        other => return Err(unsupported_dtype_err(other)),
    };

    let mut buf: Vec<u8> = Vec::with_capacity(
        HEADER_LEN + 8 + num_layers as usize * (12 + seq_len as usize * elem_size * 2 * 64 * 32),
    );

    write_header(&mut buf, FLAT_MAGIC, wire_dtype)?;
    buf.extend_from_slice(&num_layers.to_le_bytes());
    buf.extend_from_slice(&seq_len.to_le_bytes());

    for layer_idx in 0..num_layers as usize {
        let layer = match cache.layer(layer_idx) {
            Some(l) => l,
            None => {
                buf.extend_from_slice(&0u32.to_le_bytes());
                buf.extend_from_slice(&0u32.to_le_bytes());
                buf.extend_from_slice(&0u32.to_le_bytes());
                continue;
            }
        };

        let batch_size = layer.batch_size() as u32;
        let num_kv_heads = layer.num_kv_heads() as u32;
        let head_dim = layer.head_dim() as u32;

        buf.extend_from_slice(&batch_size.to_le_bytes());
        buf.extend_from_slice(&num_kv_heads.to_le_bytes());
        buf.extend_from_slice(&head_dim.to_le_bytes());

        // Checked before the `seq_len == 0` shortcut: a cache the format cannot carry
        // is refused whether or not it currently holds tokens.
        check_wire_dtype(layer_idx, "K", layer.k_cache_raw().dtype(), wire_dtype)?;
        check_wire_dtype(layer_idx, "V", layer.v_cache_raw().dtype(), wire_dtype)?;

        if seq_len == 0 {
            continue;
        }

        // A failure here used to be padded with zeros, which handed the peer a
        // silently blank cache. Propagate instead.
        let (k, v) = layer
            .get_kv()
            .map_err(|e| anyhow!("Failed to read K/V from layer {layer_idx}: {e}"))?;
        let k_c = k
            .contiguous()
            .map_err(|e| anyhow!("Failed to make layer {layer_idx} K contiguous: {e}"))?;
        let v_c = v
            .contiguous()
            .map_err(|e| anyhow!("Failed to make layer {layer_idx} V contiguous: {e}"))?;

        // Element-wise little-endian at the tensor's own width, matching the documented
        // wire format and the reader. `cast_slice` would emit native-endian bytes, which
        // disagree on a big-endian host.
        append_le_elements(&mut buf, &k_c)?;
        append_le_elements(&mut buf, &v_c)?;
    }

    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::super::deserialize::read_u32_le;
    use super::*;
    use crate::{CpuDevice, CpuRuntime};

    fn cpu_device() -> CpuDevice {
        CpuDevice::new()
    }

    #[test]
    fn test_serialize_empty_cache() {
        let device = cpu_device();
        let cache =
            LayeredKvCache::<CpuRuntime>::new_positional(2, 1, 2, 4, 64, 32, DType::F32, &device)
                .unwrap();

        let bytes = serialize_kv_cache(&cache).unwrap();
        // Header, then num_layers + seq_len, then a 12-byte dimension block per layer.
        assert_eq!(bytes.len(), HEADER_LEN + 8 + 2 * 12);
        assert_eq!(read_u32_le(&bytes, 0).unwrap(), FLAT_MAGIC);
        assert_eq!(
            read_u32_le(&bytes, 4).unwrap(),
            crate::distributed::inference::kv_serialize_paged::WIRE_VERSION
        );
        assert_eq!(read_u32_le(&bytes, 8).unwrap(), DType::F32 as u32);
    }

    /// A dtype the format cannot carry is refused by name, even with no tokens held, so
    /// a mismatched peer is rejected at the start of a transfer rather than after prefill.
    ///
    /// This replaces the old `test_serialize_rejects_non_f32_empty_cache`, which pinned
    /// the refusal on BF16. BF16 is now carried, so the refusal is pinned on F64.
    #[test]
    fn test_serialize_rejects_unsupported_dtype() {
        let device = cpu_device();
        let cache =
            LayeredKvCache::<CpuRuntime>::new_positional(2, 1, 2, 4, 64, 32, DType::F64, &device)
                .unwrap();

        assert_eq!(cache.seq_len(), 0);
        let Err(err) = serialize_kv_cache(&cache) else {
            panic!("an F64 KV cache must be refused, not written at another width");
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
