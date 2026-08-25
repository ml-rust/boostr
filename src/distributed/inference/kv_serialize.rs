//! KV cache serialization for disaggregated prefill/decode transfer.
//!
//! Converts a `LayeredKvCache` into a flat byte buffer suitable for sending
//! over nexar, and reconstructs it on the other side.
//!
//! # Wire format — `LayeredKvCache`
//!
//! ```text
//! [magic:       u32 LE = 0xB0057B01]   — flat (non-paged) KV cache
//! [version:     u32 LE = 1]
//! [dtype_tag:   u32 LE]                — numr DType discriminant: 1 = f32, 2 = f16, 3 = bf16
//! [num_layers:  u32 LE]
//! [seq_len:     u32 LE]       — used token count (same for all layers)
//! For each layer:
//!   [batch_size:    u32 LE]
//!   [num_kv_heads:  u32 LE]
//!   [head_dim:      u32 LE]
//!   [k_data:        seq_len * batch_size * num_kv_heads * head_dim * E bytes (LE)]
//!   [v_data:        seq_len * batch_size * num_kv_heads * head_dim * E bytes (LE)]
//! ```
//!
//! `E` is the element width the dtype tag names: 4 bytes for f32, 2 for f16 and bf16.
//! Elements are little-endian on every host and are decoded element-wise, so a received
//! buffer needs no alignment.
//!
//! The dtype tag round-trips: a bf16 cache is written as bf16 elements and comes back as
//! a bf16 cache. Only dtypes carried end to end are accepted — f32, f16 and bf16. Any
//! other dtype is refused by [`serialize_kv_cache`], naming itself, rather than
//! reinterpreted or converted, since either hands the peer numbers it cannot know are
//! wrong.
//!
//! The magic differs from the paged cache's `0xB0057B02`, so a paged buffer handed to
//! [`deserialize_kv_cache`] errors on the first four bytes.
//!
//! Both magics sit above `0xB0057B00` = 2_952_003_840. The pre-header format started
//! straight at `num_layers`, and no cache has billions of layers — every layer allocates
//! its own K and V tensors — so an old buffer can never present a matching magic. It is
//! rejected up front instead of being misparsed.

use super::kv_serialize_paged::{
    FLAT_MAGIC, HEADER_LEN, append_le_elements, read_header, tensor_from_le_wire,
    unsupported_dtype_err, write_header,
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

/// Read a little-endian `u32` at `offset`, returning an error instead of
/// panicking if the buffer is too short.
fn read_u32_le(bytes: &[u8], offset: usize) -> Result<u32> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| anyhow!("KV cache read offset {offset} overflows the address space"))?;
    let slice = bytes
        .get(offset..end)
        .ok_or_else(|| anyhow!("KV cache buffer truncated reading u32 at offset {offset}"))?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

/// Deserialize bytes (produced by [`serialize_kv_cache`]) into a fresh
/// `LayeredKvCache` on the given device.
pub fn deserialize_kv_cache<R>(bytes: &[u8], device: &R::Device) -> Result<LayeredKvCache<R>>
where
    R: Runtime<DType = DType>,
    R::Client: IndexingOps<R>,
{
    // Magic, version and dtype tag are checked before any dimension is read, so a foreign
    // or stale buffer errors on its first four bytes rather than on an absurd dimension.
    let wire_dtype = read_header(bytes, FLAT_MAGIC, "KV cache")?;
    let elem_size = wire_dtype.size_in_bytes();

    if bytes.len() < HEADER_LEN + 8 {
        return Err(anyhow!(
            "KV cache buffer too short: need at least {} bytes, got {}",
            HEADER_LEN + 8,
            bytes.len()
        ));
    }

    let num_layers = read_u32_le(bytes, HEADER_LEN)? as usize;
    let seq_len = read_u32_le(bytes, HEADER_LEN + 4)? as usize;

    let mut cursor = HEADER_LEN + 8;

    if num_layers == 0 {
        let cache = LayeredKvCache::<R>::new_positional(0, 1, 1, 1, 64, 1, wire_dtype, device)?;
        return Ok(cache);
    }

    if cursor + 12 > bytes.len() {
        return Err(anyhow!("KV cache buffer truncated in layer 0 header"));
    }

    let batch_size = read_u32_le(bytes, cursor)? as usize;
    let num_kv_heads = read_u32_le(bytes, cursor + 4)? as usize;
    let head_dim = read_u32_le(bytes, cursor + 8)? as usize;

    let initial_capacity = seq_len.max(1);
    let max_seq_len = (seq_len * 2).max(32768);

    // Every factor comes off the wire, and the allocation below multiplies all four.
    // `Tensor::zeros` computes that product itself, where an unchecked overflow wraps
    // in release builds and panics in debug builds. Reject the header first.
    batch_size
        .checked_mul(num_kv_heads)
        .and_then(|n| n.checked_mul(initial_capacity))
        .and_then(|n| n.checked_mul(head_dim))
        .ok_or_else(|| {
            anyhow!(
                "KV cache layer 0 dimensions overflow: batch={batch_size} \
                 heads={num_kv_heads} capacity={initial_capacity} head_dim={head_dim}"
            )
        })?;

    // Bound the allocation by the payload that must back it: a 20-byte buffer claiming
    // billion-element layers is rejected here rather than after a multi-gigabyte
    // allocation attempt.
    let layer0_bytes = batch_size
        .checked_mul(num_kv_heads)
        .and_then(|n| n.checked_mul(seq_len))
        .and_then(|n| n.checked_mul(head_dim))
        .and_then(|n| n.checked_mul(elem_size * 2))
        .and_then(|n| n.checked_add(cursor + 12))
        .ok_or_else(|| anyhow!("KV cache layer 0 data size overflows the address space"))?;
    if layer0_bytes > bytes.len() {
        return Err(anyhow!(
            "KV cache buffer truncated at layer 0 data (header claims {} bytes, buffer has {})",
            layer0_bytes,
            bytes.len()
        ));
    }

    let mut cache = LayeredKvCache::<R>::new_positional(
        num_layers,
        batch_size,
        num_kv_heads,
        initial_capacity,
        max_seq_len,
        head_dim,
        wire_dtype,
        device,
    )?;

    // The header's dtype tag was passed to the constructor above. State the symmetry as a
    // check rather than leaving it implied: if the constructor ever stops honouring the
    // requested dtype, this errors instead of writing elements of one width into a cache
    // of another.
    if let Some(layer) = cache.layer(0) {
        let got = layer.k_cache_raw().dtype();
        if got != wire_dtype {
            return Err(anyhow!(
                "KV cache header declares dtype {wire_dtype}, but the reconstructed cache \
                 has dtype {got}"
            ));
        }
    }

    for layer_idx in 0..num_layers {
        if cursor + 12 > bytes.len() {
            return Err(anyhow!(
                "KV cache buffer truncated at layer {} header (offset {})",
                layer_idx,
                cursor
            ));
        }

        let layer_batch = read_u32_le(bytes, cursor)? as usize;
        let layer_heads = read_u32_le(bytes, cursor + 4)? as usize;
        let layer_dim = read_u32_le(bytes, cursor + 8)? as usize;
        cursor += 12;

        if seq_len == 0 {
            continue;
        }

        // Every factor comes off the wire. In release builds `*` wraps, so an unchecked
        // product can land back inside the buffer, slip past the truncation check, and
        // hand enormous dimensions to `from_slice` anyway.
        let data_bytes = layer_batch
            .checked_mul(layer_heads)
            .and_then(|n| n.checked_mul(seq_len))
            .and_then(|n| n.checked_mul(layer_dim))
            .and_then(|n| n.checked_mul(elem_size))
            .ok_or_else(|| {
                anyhow!(
                    "KV cache layer {layer_idx} dimensions overflow: \
                     batch={layer_batch} heads={layer_heads} seq_len={seq_len} head_dim={layer_dim}"
                )
            })?;

        let both_end = data_bytes
            .checked_mul(2)
            .and_then(|n| n.checked_add(cursor))
            .ok_or_else(|| {
                anyhow!("KV cache layer {layer_idx} data size overflows the address space")
            })?;
        if both_end > bytes.len() {
            return Err(anyhow!(
                "KV cache buffer truncated at layer {} data (need {} bytes, have {})",
                layer_idx,
                data_bytes * 2,
                bytes.len() - cursor
            ));
        }

        // Decode element-wise rather than with `bytemuck::cast_slice`: a buffer arriving
        // off the wire carries no alignment guarantee, and `cast_slice` panics on a
        // misaligned slice. This also makes the decode explicitly little-endian, matching
        // the documented wire format, and keeps the elements at the header's dtype instead
        // of widening them.
        let shape = [layer_batch, layer_heads, seq_len, layer_dim];
        let k_tensor = tensor_from_le_wire::<R>(
            &bytes[cursor..cursor + data_bytes],
            wire_dtype,
            &shape,
            device,
        )?;
        cursor += data_bytes;

        let v_tensor = tensor_from_le_wire::<R>(
            &bytes[cursor..cursor + data_bytes],
            wire_dtype,
            &shape,
            device,
        )?;
        cursor += data_bytes;

        if let Some(layer) = cache.layer_mut(layer_idx) {
            layer
                .update(&k_tensor, &v_tensor)
                .map_err(|e| anyhow!("Failed to write K/V into layer {}: {}", layer_idx, e))?;
        }
    }

    Ok(cache)
}

#[cfg(test)]
mod tests {
    use super::super::kv_serialize_paged::{PAGED_MAGIC, WIRE_VERSION};
    use super::*;
    use crate::{CpuDevice, CpuRuntime, Tensor};

    fn cpu_device() -> CpuDevice {
        CpuDevice::new()
    }

    /// Build the 12-byte `[magic][version][dtype_tag]` prefix a flat buffer starts with.
    fn flat_header(dtype: DType) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&FLAT_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&WIRE_VERSION.to_le_bytes());
        bytes.extend_from_slice(&(dtype as u32).to_le_bytes());
        bytes
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
        assert_eq!(read_u32_le(&bytes, 4).unwrap(), WIRE_VERSION);
        assert_eq!(read_u32_le(&bytes, 8).unwrap(), DType::F32 as u32);
    }

    #[test]
    fn test_roundtrip_empty_cache() {
        let device = cpu_device();
        let cache =
            LayeredKvCache::<CpuRuntime>::new_positional(2, 1, 2, 4, 64, 32, DType::F32, &device)
                .unwrap();

        let bytes = serialize_kv_cache(&cache).unwrap();
        let restored = deserialize_kv_cache::<CpuRuntime>(&bytes, &device).unwrap();

        assert_eq!(restored.num_layers(), 2);
        assert_eq!(restored.seq_len(), 0);
    }

    #[test]
    fn test_roundtrip_with_data() {
        let device = cpu_device();
        let mut cache =
            LayeredKvCache::<CpuRuntime>::new_positional(1, 1, 2, 16, 64, 4, DType::F32, &device)
                .unwrap();

        let k_data: Vec<f32> = (0..24).map(|i| i as f32 * 0.1).collect();
        let v_data: Vec<f32> = (0..24).map(|i| i as f32 * 0.2).collect();
        let k = Tensor::<CpuRuntime>::from_slice(&k_data, &[1, 2, 3, 4], &device).unwrap();
        let v = Tensor::<CpuRuntime>::from_slice(&v_data, &[1, 2, 3, 4], &device).unwrap();

        cache.layer_mut(0).unwrap().update(&k, &v).unwrap();
        assert_eq!(cache.seq_len(), 3);

        let bytes = serialize_kv_cache(&cache).unwrap();
        assert_eq!(read_u32_le(&bytes, 8).unwrap(), DType::F32 as u32);
        let restored = deserialize_kv_cache::<CpuRuntime>(&bytes, &device).unwrap();

        assert_eq!(restored.num_layers(), 1);
        assert_eq!(restored.seq_len(), 3);
        let restored_dtype = restored.layer(0).unwrap().k_cache_raw().dtype();
        assert_eq!(restored_dtype, DType::F32);

        let (rk, rv) = restored.layer(0).unwrap().get_kv().unwrap();
        let rk_data: Vec<f32> = rk.contiguous().unwrap().to_vec::<f32>();
        let rv_data: Vec<f32> = rv.contiguous().unwrap().to_vec::<f32>();

        for (orig, got) in k_data.iter().zip(rk_data.iter()) {
            assert!((orig - got).abs() < 1e-6, "K mismatch: {} vs {}", orig, got);
        }
        for (orig, got) in v_data.iter().zip(rv_data.iter()) {
            assert!((orig - got).abs() < 1e-6, "V mismatch: {} vs {}", orig, got);
        }
    }

    /// A bf16 cache round-trips as bf16: bf16 elements on the wire, a bf16 cache back,
    /// and the same values.
    ///
    /// This replaces the old `test_serialize_rejects_non_f32_cache`, which asserted the
    /// serializer refused BF16. The header now carries a dtype tag, so BF16 is supported
    /// end to end and the refusal moved to `test_serialize_rejects_unsupported_dtype`.
    #[test]
    fn test_roundtrip_bf16_cache() {
        let device = cpu_device();
        let mut cache =
            LayeredKvCache::<CpuRuntime>::new_positional(1, 1, 1, 16, 64, 2, DType::BF16, &device)
                .unwrap();

        // Every value is exactly representable in bf16, so equality is the right check.
        let k_vals: Vec<half::bf16> = [0.5f32, -1.5, 2.0, 3.0]
            .iter()
            .map(|&x| half::bf16::from_f32(x))
            .collect();
        let v_vals: Vec<half::bf16> = [-0.25f32, 4.0, 6.0, -8.0]
            .iter()
            .map(|&x| half::bf16::from_f32(x))
            .collect();
        let k = Tensor::<CpuRuntime>::from_slice(&k_vals, &[1, 1, 2, 2], &device).unwrap();
        let v = Tensor::<CpuRuntime>::from_slice(&v_vals, &[1, 1, 2, 2], &device).unwrap();
        cache.layer_mut(0).unwrap().update(&k, &v).unwrap();

        let bytes = serialize_kv_cache(&cache).unwrap();
        assert_eq!(read_u32_le(&bytes, 8).unwrap(), DType::BF16 as u32);
        // 2 bytes per element, not 4: 4 elements each for K and V.
        assert_eq!(bytes.len(), HEADER_LEN + 8 + 12 + 4 * 2 * 2);

        let restored = deserialize_kv_cache::<CpuRuntime>(&bytes, &device).unwrap();
        let layer = restored.layer(0).unwrap();
        assert_eq!(layer.k_cache_raw().dtype(), DType::BF16);
        assert_eq!(layer.v_cache_raw().dtype(), DType::BF16);

        let (rk, rv) = layer.get_kv().unwrap();
        assert_eq!(rk.contiguous().unwrap().to_vec::<half::bf16>(), k_vals);
        assert_eq!(rv.contiguous().unwrap().to_vec::<half::bf16>(), v_vals);
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

    /// A buffer shorter than the 12-byte prefix errors in the header, naming the length.
    #[test]
    fn test_deserialize_rejects_truncated_header() {
        let device = cpu_device();
        let bytes = [0u8; 7];

        let Err(err) = deserialize_kv_cache::<CpuRuntime>(&bytes, &device) else {
            panic!("a 7-byte buffer must be refused");
        };
        let msg = err.to_string();
        assert!(
            msg.contains("truncated in the header"),
            "expected a header truncation error, got: {msg}"
        );
        assert!(
            msg.contains("need 12 bytes, got 7"),
            "error must name both lengths: {msg}"
        );
    }

    /// A paged buffer must be refused by the flat reader on its magic, not thirty lines
    /// later on a dimension field.
    #[test]
    fn test_deserialize_rejects_paged_magic() {
        let device = cpu_device();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&PAGED_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&WIRE_VERSION.to_le_bytes());
        bytes.extend_from_slice(&(DType::F32 as u32).to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes()); // num_layers
        bytes.extend_from_slice(&16u32.to_le_bytes()); // block_size
        bytes.extend_from_slice(&0u32.to_le_bytes()); // seq_len

        let Err(err) = deserialize_kv_cache::<CpuRuntime>(&bytes, &device) else {
            panic!("a paged KV cache buffer must be refused by the flat reader");
        };
        let msg = err.to_string();
        assert!(
            msg.contains("magic 0xB0057B02") && msg.contains("expected 0xB0057B01"),
            "error must name both magics: {msg}"
        );
        assert!(
            msg.contains("that is the paged KV cache magic"),
            "error must name the sibling format: {msg}"
        );
    }

    /// A buffer in the pre-header format starts at `num_layers`, which can never match a
    /// magic, so it lands as a magic error rather than as garbage dimensions.
    #[test]
    fn test_deserialize_rejects_pre_header_buffer() {
        let device = cpu_device();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&2u32.to_le_bytes()); // old num_layers
        bytes.extend_from_slice(&3u32.to_le_bytes()); // old seq_len
        for _ in 0..2 {
            bytes.extend_from_slice(&1u32.to_le_bytes()); // batch_size
            bytes.extend_from_slice(&2u32.to_le_bytes()); // num_kv_heads
            bytes.extend_from_slice(&4u32.to_le_bytes()); // head_dim
        }

        let Err(err) = deserialize_kv_cache::<CpuRuntime>(&bytes, &device) else {
            panic!("a pre-header buffer must be refused");
        };
        let msg = err.to_string();
        assert!(
            msg.contains("magic 0x00000002") && msg.contains("expected 0xB0057B01"),
            "error must name the magic it read: {msg}"
        );
    }

    /// An unknown wire version errors, naming the version it got.
    #[test]
    fn test_deserialize_rejects_unknown_version() {
        let device = cpu_device();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&FLAT_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&(DType::F32 as u32).to_le_bytes());

        let Err(err) = deserialize_kv_cache::<CpuRuntime>(&bytes, &device) else {
            panic!("an unknown wire version must be refused");
        };
        let msg = err.to_string();
        assert!(
            msg.contains("wire version 2") && msg.contains("version 1 only"),
            "error must name the version it got and the one it reads: {msg}"
        );
    }

    /// An unknown dtype tag errors, naming the tag.
    #[test]
    fn test_deserialize_rejects_unknown_dtype_tag() {
        let device = cpu_device();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&FLAT_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&WIRE_VERSION.to_le_bytes());
        bytes.extend_from_slice(&23u32.to_le_bytes()); // DType::U8, not carried

        let Err(err) = deserialize_kv_cache::<CpuRuntime>(&bytes, &device) else {
            panic!("an unknown dtype tag must be refused");
        };
        let msg = err.to_string();
        assert!(
            msg.contains("unknown dtype tag 23"),
            "error must name the tag: {msg}"
        );
        assert!(
            msg.contains("bf16 (tag 3)"),
            "error must name the tags it does carry: {msg}"
        );
    }

    /// The reader must not require the input buffer to be 4-byte aligned.
    ///
    /// A buffer arriving off the wire carries no alignment guarantee, and
    /// `bytemuck::cast_slice::<u8, f32>` panics outright on a misaligned slice.
    /// Deserializing a deliberately offset copy reproduces that panic if the
    /// element-wise decode is reverted.
    #[test]
    fn test_deserialize_accepts_misaligned_buffer() {
        let device = cpu_device();
        let mut cache =
            LayeredKvCache::<CpuRuntime>::new_positional(1, 1, 1, 16, 64, 2, DType::F32, &device)
                .unwrap();

        let k_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let v_data: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
        let k = Tensor::<CpuRuntime>::from_slice(&k_data, &[1, 1, 2, 2], &device).unwrap();
        let v = Tensor::<CpuRuntime>::from_slice(&v_data, &[1, 1, 2, 2], &device).unwrap();
        cache.layer_mut(0).unwrap().update(&k, &v).unwrap();

        let bytes = serialize_kv_cache(&cache).unwrap();
        let mut shifted = vec![0u8];
        shifted.extend_from_slice(&bytes);

        let restored = deserialize_kv_cache::<CpuRuntime>(&shifted[1..], &device)
            .expect("a misaligned buffer must deserialize");
        let (rk, _rv) = restored.layer(0).unwrap().get_kv().unwrap();
        assert_eq!(rk.contiguous().unwrap().to_vec::<f32>(), k_data);
    }

    /// A hostile header whose dimensions multiply past `usize::MAX` must be rejected.
    ///
    /// Without checked arithmetic the product wraps in release builds to a small
    /// `data_bytes`, the truncation check then passes, and the enormous dimensions
    /// reach `from_slice` anyway.
    #[test]
    fn test_deserialize_rejects_dimension_overflow() {
        let device = cpu_device();
        let mut bytes = flat_header(DType::F32);
        bytes.extend_from_slice(&1u32.to_le_bytes()); // num_layers
        bytes.extend_from_slice(&u32::MAX.to_le_bytes()); // seq_len
        bytes.extend_from_slice(&u32::MAX.to_le_bytes()); // batch_size
        bytes.extend_from_slice(&u32::MAX.to_le_bytes()); // num_kv_heads
        bytes.extend_from_slice(&u32::MAX.to_le_bytes()); // head_dim

        let Err(err) = deserialize_kv_cache::<CpuRuntime>(&bytes, &device) else {
            panic!("overflowing dimensions must be rejected");
        };
        assert!(
            err.to_string().contains("overflow"),
            "expected an overflow error, got: {err}"
        );
    }
}
