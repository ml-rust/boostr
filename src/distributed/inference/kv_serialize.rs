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

use crate::inference::LayeredKvCache;
use crate::{DType, IndexingOps, Runtime, Tensor};
use anyhow::{Result, anyhow};

/// Serialize a `LayeredKvCache` into bytes for network transfer.
///
/// Extracts only the *used* portion of each layer's K/V tensors (up to
/// `seq_len` tokens). The resulting bytes contain enough metadata for the
/// receiving side to reconstruct a fresh cache with the same dimensions.
///
/// Note: This function calls `to_vec::<f32>()` on each tensor, which copies
/// data from the device to CPU memory. For GPU tensors this is an intentional
/// transfer — disaggregated inference requires moving the KV cache over the
/// network, so a CPU copy is unavoidable.
pub fn serialize_kv_cache<R>(cache: &LayeredKvCache<R>) -> Vec<u8>
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

        if seq_len == 0 {
            continue;
        }

        match layer.get_kv() {
            Ok((k, v)) => {
                let k_data: Vec<f32> = k.contiguous().to_vec::<f32>();
                let v_data: Vec<f32> = v.contiguous().to_vec::<f32>();
                buf.extend_from_slice(bytemuck::cast_slice::<f32, u8>(&k_data));
                buf.extend_from_slice(bytemuck::cast_slice::<f32, u8>(&v_data));
            }
            Err(_) => {
                let numel = batch_size as usize
                    * num_kv_heads as usize
                    * seq_len as usize
                    * head_dim as usize;
                let zeros = vec![0u8; numel * 4 * 2];
                buf.extend_from_slice(&zeros);
            }
        }
    }

    buf
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

    let num_layers = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let seq_len = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;

    let mut cursor = 8usize;

    if num_layers == 0 {
        let cache = LayeredKvCache::<R>::new_positional(0, 1, 1, 1, 64, 1, DType::F32, device)?;
        return Ok(cache);
    }

    if cursor + 12 > bytes.len() {
        return Err(anyhow!("KV cache buffer truncated in layer 0 header"));
    }

    let batch_size = u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap()) as usize;
    let num_kv_heads =
        u32::from_le_bytes(bytes[cursor + 4..cursor + 8].try_into().unwrap()) as usize;
    let head_dim = u32::from_le_bytes(bytes[cursor + 8..cursor + 12].try_into().unwrap()) as usize;

    let initial_capacity = seq_len.max(1);
    let max_seq_len = (seq_len * 2).max(32768);

    let mut cache = LayeredKvCache::<R>::new_positional(
        num_layers,
        batch_size,
        num_kv_heads,
        initial_capacity,
        max_seq_len,
        head_dim,
        DType::F32,
        device,
    )?;

    for layer_idx in 0..num_layers {
        if cursor + 12 > bytes.len() {
            return Err(anyhow!(
                "KV cache buffer truncated at layer {} header (offset {})",
                layer_idx,
                cursor
            ));
        }

        let layer_batch =
            u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap()) as usize;
        let layer_heads =
            u32::from_le_bytes(bytes[cursor + 4..cursor + 8].try_into().unwrap()) as usize;
        let layer_dim =
            u32::from_le_bytes(bytes[cursor + 8..cursor + 12].try_into().unwrap()) as usize;
        cursor += 12;

        if seq_len == 0 {
            continue;
        }

        let numel = layer_batch * layer_heads * seq_len * layer_dim;
        let data_bytes = numel * 4;

        if cursor + data_bytes * 2 > bytes.len() {
            return Err(anyhow!(
                "KV cache buffer truncated at layer {} data (need {} bytes, have {})",
                layer_idx,
                data_bytes * 2,
                bytes.len() - cursor
            ));
        }

        let k_f32: Vec<f32> =
            bytemuck::cast_slice::<u8, f32>(&bytes[cursor..cursor + data_bytes]).to_vec();
        cursor += data_bytes;

        let v_f32: Vec<f32> =
            bytemuck::cast_slice::<u8, f32>(&bytes[cursor..cursor + data_bytes]).to_vec();
        cursor += data_bytes;

        let k_tensor = Tensor::<R>::from_slice(
            &k_f32,
            &[layer_batch, layer_heads, seq_len, layer_dim],
            device,
        );
        let v_tensor = Tensor::<R>::from_slice(
            &v_f32,
            &[layer_batch, layer_heads, seq_len, layer_dim],
            device,
        );

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

        let bytes = serialize_kv_cache(&cache);
        assert!(bytes.len() >= 8 + 2 * 12);
    }

    #[test]
    fn test_roundtrip_empty_cache() {
        let device = cpu_device();
        let cache =
            LayeredKvCache::<CpuRuntime>::new_positional(2, 1, 2, 4, 64, 32, DType::F32, &device)
                .unwrap();

        let bytes = serialize_kv_cache(&cache);
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
        let k = Tensor::<CpuRuntime>::from_slice(&k_data, &[1, 2, 3, 4], &device);
        let v = Tensor::<CpuRuntime>::from_slice(&v_data, &[1, 2, 3, 4], &device);

        cache.layer_mut(0).unwrap().update(&k, &v).unwrap();
        assert_eq!(cache.seq_len(), 3);

        let bytes = serialize_kv_cache(&cache);
        let restored = deserialize_kv_cache::<CpuRuntime>(&bytes, &device).unwrap();

        assert_eq!(restored.num_layers(), 1);
        assert_eq!(restored.seq_len(), 3);

        let (rk, rv) = restored.layer(0).unwrap().get_kv().unwrap();
        let rk_data: Vec<f32> = rk.contiguous().to_vec::<f32>();
        let rv_data: Vec<f32> = rv.contiguous().to_vec::<f32>();

        for (orig, got) in k_data.iter().zip(rk_data.iter()) {
            assert!((orig - got).abs() < 1e-6, "K mismatch: {} vs {}", orig, got);
        }
        for (orig, got) in v_data.iter().zip(rv_data.iter()) {
            assert!((orig - got).abs() < 1e-6, "V mismatch: {} vs {}", orig, got);
        }
    }
}
