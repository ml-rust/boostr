//! KV cache quantization traits
//!
//! Compression of KV caches for long-context inference:
//! - FP8: 2x compression with per-token or per-tensor scaling
//! - INT4: 4x compression with per-group asymmetric scaling
//! - INT8: 2x compression with per-token scaling

use crate::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// KV cache quantization mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvQuantMode {
    /// Per-tensor: single scale for entire tensor (fastest, least accurate)
    PerTensor,
    /// Per-token: one scale per token across head_dim (balanced)
    PerToken,
}

/// INT4 group size for quantization
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Int4GroupSize {
    Group32 = 32,
    #[default]
    Group64 = 64,
    Group128 = 128,
}

/// KV cache quantization operations
///
/// Compress KV caches from FP16/BF16/F32 to lower precision formats
/// for memory-efficient long-context inference.
///
/// # Layout
/// - Input: `[num_tokens, head_dim]` or `[batch, num_kv_heads, seq_len, head_dim]`
/// - FP8 output: same shape, 1 byte per element + scales
/// - INT4 output: `[..., head_dim/2]` packed (2 values per byte) + scales + zeros
/// - INT8 output: same shape, 1 byte per element + scales
pub trait KvCacheQuantOps<R: Runtime> {
    /// Quantize KV cache to FP8 (E4M3) with per-token scaling
    ///
    /// Returns `(quantized, scales)` where scales is `[num_tokens]` F32.
    fn quantize_kv_fp8_per_token(
        &self,
        input: &Tensor<R>,
        num_tokens: usize,
        head_dim: usize,
    ) -> Result<(Tensor<R>, Tensor<R>)>;

    /// Dequantize FP8 KV cache back to original dtype
    fn dequantize_kv_fp8_per_token(
        &self,
        quantized: &Tensor<R>,
        scales: &Tensor<R>,
        num_tokens: usize,
        head_dim: usize,
        output_dtype: numr::dtype::DType,
    ) -> Result<Tensor<R>>;

    /// Quantize KV cache to INT4 with per-group asymmetric scaling
    ///
    /// Returns `(packed_int4, scales, zeros)`.
    /// - packed_int4: `[num_tokens, head_dim/2]` (2 values per byte)
    /// - scales: `[num_groups]` FP32
    /// - zeros: `[num_groups]` FP32
    fn quantize_kv_int4(
        &self,
        input: &Tensor<R>,
        num_tokens: usize,
        head_dim: usize,
        group_size: Int4GroupSize,
    ) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>)>;

    /// Dequantize INT4 KV cache back to F32
    fn dequantize_kv_int4(
        &self,
        packed: &Tensor<R>,
        scales: &Tensor<R>,
        zeros: &Tensor<R>,
        num_tokens: usize,
        head_dim: usize,
        group_size: Int4GroupSize,
    ) -> Result<Tensor<R>>;

    /// Quantize KV cache to INT8 with per-token scaling
    ///
    /// Returns `(quantized, scales)`.
    fn quantize_kv_int8(
        &self,
        input: &Tensor<R>,
        num_tokens: usize,
        head_dim: usize,
    ) -> Result<(Tensor<R>, Tensor<R>)>;

    /// Dequantize INT8 KV cache back to F32
    fn dequantize_kv_int8(
        &self,
        quantized: &Tensor<R>,
        scales: &Tensor<R>,
        num_tokens: usize,
        head_dim: usize,
    ) -> Result<Tensor<R>>;
}
