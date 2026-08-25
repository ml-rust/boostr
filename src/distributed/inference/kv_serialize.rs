//! KV cache serialization for disaggregated prefill/decode transfer.
//!
//! Converts a `LayeredKvCache` into a flat byte buffer suitable for sending
//! over nexar, and reconstructs it on the other side.
//!
//! # Wire format — `LayeredKvCache`
//!
//! ```text
//! [num_layers:  u32 LE]
//! [seq_len:     u32 LE]       — used token count (same for all layers)
//! For each layer:
//!   [batch_size:    u32 LE]
//!   [num_kv_heads:  u32 LE]
//!   [head_dim:      u32 LE]
//!   [k_data:        seq_len * batch_size * num_kv_heads * head_dim * 4 bytes (f32 LE)]
//!   [v_data:        seq_len * batch_size * num_kv_heads * head_dim * 4 bytes (f32 LE)]
//! ```
//!
//! The format carries **no dtype tag**: element data is f32 little-endian and nothing
//! else. A cache of any other dtype (bf16 in particular, the usual training dtype) is
//! refused by [`serialize_kv_cache`] rather than reinterpreted or converted, since
//! either would hand the peer numbers it cannot know are wrong.

use super::kv_serialize_paged::read_f32_le_vec;
use crate::inference::LayeredKvCache;
use crate::{DType, IndexingOps, Runtime, Tensor};
use anyhow::{Result, anyhow};

/// The only element type this wire format can carry.
///
/// The header records dimensions but no dtype tag, so both sides are hard-wired to
/// f32. Every read and write below goes through this constant so the assumption is
/// stated once instead of being implied by a bare `to_vec::<f32>()`.
const WIRE_DTYPE: DType = DType::F32;

/// Reject a cache tensor whose dtype the wire format cannot represent.
///
/// `try_to_vec::<f32>()` does not check dtype: on a bf16 cache it copies
/// `numel * 4` bytes out of a `numel * 2` byte allocation and reinterprets the
/// result as f32. That is a silent wrong answer at the peer, so refuse instead.
///
/// Converting to f32 here is not an option either — the peer has no dtype tag to
/// tell it the numbers were changed on the way.
fn check_wire_dtype(layer_idx: usize, which: &str, dtype: DType) -> Result<()> {
    if dtype != WIRE_DTYPE {
        return Err(anyhow!(
            "KV cache layer {layer_idx} {which} tensor has dtype {dtype}, but the KV cache \
             wire format carries {WIRE_DTYPE} only (the header has no dtype tag). Refusing \
             to transfer: reinterpreting {dtype} bytes as {WIRE_DTYPE} corrupts the cache, \
             and converting would change the values the peer receives without telling it. \
             Allocate the cache with DType::{WIRE_DTYPE:?} for disaggregated transfer."
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
/// Returns an error naming the offending dtype if any layer's K or V tensor is
/// not [`WIRE_DTYPE`]. The wire format carries no dtype tag, so a non-f32 cache
/// cannot be represented and is refused rather than silently mangled.
pub fn serialize_kv_cache<R>(cache: &LayeredKvCache<R>) -> Result<Vec<u8>>
where
    R: Runtime<DType = DType>,
    R::Client: IndexingOps<R>,
{
    let num_layers = cache.num_layers() as u32;
    let seq_len = cache.seq_len() as u32;

    let mut buf: Vec<u8> =
        Vec::with_capacity(8 + num_layers as usize * (12 + seq_len as usize * 4 * 2 * 64 * 32));

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
        check_wire_dtype(layer_idx, "K", layer.k_cache_raw().dtype())?;
        check_wire_dtype(layer_idx, "V", layer.v_cache_raw().dtype())?;

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

        let k_data: Vec<f32> = k_c.try_to_vec::<f32>()?;
        let v_data: Vec<f32> = v_c.try_to_vec::<f32>()?;
        // Explicit little-endian, matching the documented wire format and the reader.
        // `cast_slice` would emit native-endian bytes, which disagree on a big-endian host.
        for &x in &k_data {
            buf.extend_from_slice(&x.to_le_bytes());
        }
        for &x in &v_data {
            buf.extend_from_slice(&x.to_le_bytes());
        }
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
    if bytes.len() < 8 {
        return Err(anyhow!(
            "KV cache buffer too short: need at least 8 bytes, got {}",
            bytes.len()
        ));
    }

    let num_layers = read_u32_le(bytes, 0)? as usize;
    let seq_len = read_u32_le(bytes, 4)? as usize;

    let mut cursor = 8usize;

    if num_layers == 0 {
        let cache = LayeredKvCache::<R>::new_positional(0, 1, 1, 1, 64, 1, WIRE_DTYPE, device)?;
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
        .and_then(|n| n.checked_mul(8))
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
        WIRE_DTYPE,
        device,
    )?;

    // The bytes carry no dtype tag, so the reader decodes them as `WIRE_DTYPE` and
    // requested a `WIRE_DTYPE` cache above. State that symmetry as a check rather than
    // leaving it implied: if the constructor ever stops honouring the requested dtype,
    // this errors instead of writing f32 bytes into a differently-typed cache.
    if let Some(layer) = cache.layer(0) {
        let got = layer.k_cache_raw().dtype();
        if got != WIRE_DTYPE {
            return Err(anyhow!(
                "KV cache wire format decodes {WIRE_DTYPE} only, but the reconstructed \
                 cache has dtype {got}"
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
            .and_then(|n| n.checked_mul(4))
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

        // Read element-wise rather than `bytemuck::cast_slice`: a buffer arriving off
        // the wire carries no alignment guarantee, and `cast_slice::<u8, f32>` panics on
        // a misaligned slice. This also makes the decode explicitly little-endian,
        // matching the documented wire format.
        let k_f32: Vec<f32> = read_f32_le_vec(&bytes[cursor..cursor + data_bytes]);
        cursor += data_bytes;

        let v_f32: Vec<f32> = read_f32_le_vec(&bytes[cursor..cursor + data_bytes]);
        cursor += data_bytes;

        let k_tensor = Tensor::<R>::from_slice(
            &k_f32,
            &[layer_batch, layer_heads, seq_len, layer_dim],
            device,
        )?;
        let v_tensor = Tensor::<R>::from_slice(
            &v_f32,
            &[layer_batch, layer_heads, seq_len, layer_dim],
            device,
        )?;

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
        assert!(bytes.len() >= 8 + 2 * 12);
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
        let restored = deserialize_kv_cache::<CpuRuntime>(&bytes, &device).unwrap();

        assert_eq!(restored.num_layers(), 1);
        assert_eq!(restored.seq_len(), 3);

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

    /// A cache whose dtype the wire format cannot carry must be refused by name.
    ///
    /// `try_to_vec::<f32>()` performs no dtype check: for a BF16 cache it copies
    /// `numel * 4` bytes out of a `numel * 2` byte allocation and reinterprets them as
    /// f32, so without the guard this returns a buffer of garbage rather than an error.
    /// The assertion pins the dtype name and the f32-only statement, not merely
    /// "is_err".
    #[test]
    fn test_serialize_rejects_non_f32_cache() {
        let device = cpu_device();
        let cache =
            LayeredKvCache::<CpuRuntime>::new_positional(1, 1, 2, 4, 64, 32, DType::BF16, &device)
                .unwrap();

        let Err(err) = serialize_kv_cache(&cache) else {
            panic!("a BF16 KV cache must be refused, not serialized as f32");
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

    /// The dtype guard fires even when the cache holds no tokens, so a mismatched
    /// peer is rejected at the start of a transfer rather than after prefill.
    #[test]
    fn test_serialize_rejects_non_f32_empty_cache() {
        let device = cpu_device();
        let cache =
            LayeredKvCache::<CpuRuntime>::new_positional(2, 1, 2, 4, 64, 32, DType::BF16, &device)
                .unwrap();

        assert_eq!(cache.seq_len(), 0);
        let Err(err) = serialize_kv_cache(&cache) else {
            panic!("an empty BF16 KV cache must still be refused");
        };
        assert!(
            err.to_string().to_lowercase().contains("bf16"),
            "error must name the dtype: {err}"
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
        let mut bytes = Vec::new();
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
