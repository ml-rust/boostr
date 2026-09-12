//! Streaming dequant path: chunked CPU dequantization uploaded into a pre-allocated device buffer.

use super::open::Gguf;
use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Tensors whose dequantized f32 output exceeds this byte threshold use the
/// streaming dequant path, which processes one chunk at a time and uploads
/// each chunk directly into the pre-allocated device buffer.  Tensors below
/// this threshold use the original one-shot allocation path.
///
/// 64 MiB keeps the transient CPU buffer well within L3 / RSS limits even on
/// memory-constrained systems, while keeping the number of upload calls low
/// (a 1 GiB embedding table becomes ~16 chunks).
const STREAMING_THRESHOLD: usize = 64 * 1024 * 1024; // 64 MiB in bytes

impl Gguf {
    /// Load an F32 tensor using a streaming dequant path for large tensors.
    ///
    /// For tensors whose dequantized size is ≤ `STREAMING_THRESHOLD` bytes this
    /// is identical to [`Gguf::load_tensor_f32`].  For larger tensors the method:
    ///
    /// 1. Allocates the destination `Tensor<R>` of the correct shape once on the
    ///    device via `Tensor::empty`.
    /// 2. Dequantizes `STREAMING_CHUNK_ELEMS` f32 elements at a time on the CPU.
    /// 3. Uploads each chunk directly into the pre-allocated device buffer at the
    ///    correct byte offset, then drops the chunk `Vec<f32>` before the next
    ///    iteration.
    ///
    /// This bounds the transient CPU heap usage to `STREAMING_THRESHOLD` bytes
    /// instead of `numel × 4` bytes, preventing heap OOM for large quantized
    /// weights such as the token-embedding table in bge-reranker-v2-m3.
    ///
    /// The GPU-side allocation is a single contiguous buffer of `numel × 4`
    /// bytes; if the device cannot satisfy that allocation `Tensor::empty`
    /// returns `OutOfMemory` immediately (no change from the non-streaming path).
    ///
    /// Non-quantized types (F32 / F16 / BF16 / F64) always use the one-shot
    /// path regardless of size because their source bytes are already in the
    /// correct format and a single `copy_to_device` is optimal.
    pub fn load_tensor_f32_streaming<R: Runtime<DType = DType>>(
        &mut self,
        name: &str,
        device: &R::Device,
    ) -> Result<Tensor<R>> {
        self.load_tensor_f32_streaming_impl::<R>(name, device, STREAMING_THRESHOLD)
    }

    /// Internal implementation shared by the public API and tests.
    ///
    /// `threshold_bytes` controls when to use the streaming path. Pass `0` in
    /// tests to force streaming even for small tensors.
    pub(super) fn load_tensor_f32_streaming_impl<R: Runtime<DType = DType>>(
        &mut self,
        name: &str,
        device: &R::Device,
        threshold_bytes: usize,
    ) -> Result<Tensor<R>> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::ModelError {
                reason: format!("GGUF tensor not found: {name}"),
            })?
            .clone();

        // For non-quantized types there is no block structure to chunk over;
        // the bytes map 1-to-1 (or 2-to-1) to the output f32 values.  Use the
        // existing one-shot path which is already optimal.
        // Delegates to `GgmlType::is_quantized` so this never drifts from the
        // enum's own IQ/Q variant list (previously restated here and missed
        // every IQ format when they were added).
        let is_quantized = info.ggml_type.is_quantized();

        let mut shape = info.shape.clone();
        shape.reverse();
        let numel: usize = shape.iter().product();

        // Fall back to the one-shot path for small tensors or non-quantized types.
        if !is_quantized || numel * 4 <= threshold_bytes {
            return self.load_tensor_f32(name, device);
        }

        // --- Streaming dequant path ---

        // All quantized formats use a 1-D row structure where shape[0] (before
        // reversal) is the innermost / fastest-varying dimension K.
        let format = info
            .ggml_type
            .to_quant_format()
            .ok_or_else(|| Error::ModelError {
                reason: format!(
                    "tensor '{name}' type {:?} cannot be dequantized",
                    info.ggml_type
                ),
            })?;

        let row_k = info.shape[0]; // innermost dim (GGML order, before reversal)
        let row_bytes = format.storage_bytes(row_k)?;
        let n_rows = numel / row_k;
        let expected_bytes = n_rows * row_bytes;

        // Read all quantized source bytes once (they are compressed; ~¼ the
        // size of the f32 output).
        let abs_offset = self.data_offset + info.offset;
        let raw = self.read_slice(abs_offset, expected_bytes, name)?;

        // Allocate the destination tensor on the device.
        let dst = Tensor::<R>::empty(&shape, DType::F32, device).map_err(Error::Numr)?;
        let base_ptr = dst.ptr();

        // How many rows fit in one chunk?  Use the chunk size that corresponds
        // to `threshold_bytes`; when threshold_bytes == 0 (test mode) use 1 row.
        let chunk_elems_limit = if threshold_bytes == 0 {
            row_k
        } else {
            threshold_bytes / 4
        };
        let chunk_rows = (chunk_elems_limit / row_k).max(1);

        let mut row_start = 0usize;
        while row_start < n_rows {
            let row_end = (row_start + chunk_rows).min(n_rows);
            let chunk_elems = (row_end - row_start) * row_k;

            // Dequantize chunk into a temporary CPU buffer.
            let mut chunk_f32 = vec![0.0f32; chunk_elems];
            for (local_row, abs_row) in (row_start..row_end).enumerate() {
                let src = &raw[abs_row * row_bytes..(abs_row + 1) * row_bytes];
                let dst_slice = &mut chunk_f32[local_row * row_k..(local_row + 1) * row_k];
                crate::quant::cpu::kernels::quant_matmul::dequant_row_f32(src, dst_slice, format);
            }

            // Upload this chunk to the device at the correct byte offset.
            // Flatten f32 values to little-endian bytes without requiring bytemuck.
            let chunk_bytes: Vec<u8> = chunk_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
            let byte_offset = (row_start * row_k * 4) as u64;
            R::copy_to_device(&chunk_bytes, base_ptr + byte_offset, device).map_err(Error::Numr)?;

            // `chunk_f32` drops here, freeing the CPU buffer before the next chunk.
            row_start = row_end;
        }

        Ok(dst)
    }
}

#[cfg(test)]
mod tests {
    use super::super::open::tests::create_test_gguf_bytes;
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    /// Verify that `load_tensor_f32_streaming` with a zero threshold (streaming
    /// forced for every quantized tensor) produces byte-identical results to the
    /// one-shot `load_tensor_f32` path.
    ///
    /// This test covers the correctness of the chunked dequant + partial upload
    /// logic using the Q4_0 tensor in the standard test fixture.
    #[test]
    fn test_streaming_dequant_matches_oneshot() {
        let (_, device) = cpu_setup();

        // One-shot path
        let buf = create_test_gguf_bytes();
        let mut gguf_oneshot = Gguf::from_bytes(buf).unwrap();
        let oneshot = gguf_oneshot
            .load_tensor_f32::<CpuRuntime>("weight_q4", &device)
            .unwrap();
        let oneshot_data = oneshot.to_vec::<f32>();

        // Streaming path (threshold = 0 forces streaming even for the 32-element tensor)
        let buf = create_test_gguf_bytes();
        let mut gguf_streaming = Gguf::from_bytes(buf).unwrap();
        let streaming = gguf_streaming
            .load_tensor_f32_streaming_impl::<CpuRuntime>("weight_q4", &device, 0)
            .unwrap();
        let streaming_data = streaming.to_vec::<f32>();

        assert_eq!(
            oneshot_data.len(),
            streaming_data.len(),
            "element count mismatch"
        );
        for (i, (a, b)) in oneshot_data.iter().zip(streaming_data.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-7,
                "mismatch at element {i}: one-shot={a} streaming={b}"
            );
        }
    }

    /// Verify that `load_tensor_f32_streaming` for an F32 tensor uses the one-shot
    /// path (non-quantized types are never streamed) and returns correct data.
    #[test]
    fn test_streaming_f32_tensor_uses_oneshot_path() {
        let (_, device) = cpu_setup();

        let buf = create_test_gguf_bytes();
        let mut gguf = Gguf::from_bytes(buf).unwrap();
        let tensor = gguf
            .load_tensor_f32_streaming::<CpuRuntime>("weight_f32", &device)
            .unwrap();

        assert_eq!(tensor.shape(), &[4]);
        let data = tensor.to_vec::<f32>();
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[3] - 4.0).abs() < 1e-6);
    }
}
