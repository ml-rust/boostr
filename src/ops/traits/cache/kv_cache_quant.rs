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

    /// Quantize KV cache to FP8 (E4M3) with per-head scaling
    ///
    /// One scale covers a head's whole `seq_len * head_dim` span, unlike
    /// `quantize_kv_fp8_per_token`, which produces one scale per token.
    /// Returns `(quantized, scales)` where scales is `[num_heads]` F32.
    fn quantize_kv_fp8_per_head(
        &self,
        input: &Tensor<R>,
        num_heads: usize,
        seq_len: usize,
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

    /// Dequantize INT4 KV cache back to `output_dtype`
    ///
    /// INT4 needs a zeros tensor and a group size on top of scales, so this
    /// carries more parameters than the symmetric INT8 and FP8 dequantizers.
    #[allow(clippy::too_many_arguments)]
    fn dequantize_kv_int4(
        &self,
        packed: &Tensor<R>,
        scales: &Tensor<R>,
        zeros: &Tensor<R>,
        num_tokens: usize,
        head_dim: usize,
        group_size: Int4GroupSize,
        output_dtype: numr::dtype::DType,
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

    /// Backward for FP8 fake-quantization with a single tensor-wide scale.
    ///
    /// `grad_kv` is a straight-through-estimator identity: `grad_kv =
    /// grad_output`. `grad_scale` differentiates the dequant `x_hat = c /
    /// scale` with the FP8 code `c` held constant, so `grad_scale =
    /// sum(grad_output * -c / scale^2)`, returned as a 1-element F32 tensor.
    ///
    /// `grad_output`'s dtype (F32/F16/BF16) selects the kernel. `kv_fp8` must
    /// be `DType::FP8E4M3`. Returns `(grad_kv, grad_scale)`.
    fn kv_fp8_bwd_per_tensor(
        &self,
        grad_output: &Tensor<R>,
        kv_fp8: &Tensor<R>,
        scale: f32,
    ) -> Result<(Tensor<R>, Tensor<R>)>;

    /// Backward for FP8 fake-quantization with one scale per token.
    ///
    /// Same STE identity and scale-gradient formula as
    /// `kv_fp8_bwd_per_tensor`, but reduced per token instead of over the
    /// whole tensor: `grad_scales[token] = sum_d(grad_output[d] * -c_d /
    /// scale[token]^2)`. Pairs with `quantize_kv_fp8_per_token`'s forward
    /// layout, where `scales` is flat `[num_tokens]` with `num_tokens ==
    /// batch * num_kv_heads * seq_len`. `scales` must be F32.
    ///
    /// Returns `(grad_kv, grad_scales)`.
    #[allow(clippy::too_many_arguments)]
    fn kv_fp8_bwd_per_token(
        &self,
        grad_output: &Tensor<R>,
        kv_fp8: &Tensor<R>,
        scales: &Tensor<R>,
        batch: usize,
        num_kv_heads: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> Result<(Tensor<R>, Tensor<R>)>;
}
