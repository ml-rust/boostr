//! Quantized matmul operations trait

use crate::error::Result;
use crate::quant::QuantTensor;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Mixed-precision matmul: standard activation × quantized weight
///
/// This is THE hot path in quantized inference. The weight matrix stays in
/// quantized format and is dequantized on-the-fly during the multiply,
/// avoiding full materialization of the dequantized weight.
///
/// # Contract
///
/// - `activation` shape: `[..., M, K]` where K matches weight's last-but-one dim
/// - `weight` shape: `[N, K]` (2D quantized weight — N output rows, K input cols; blocks packed along K)
/// - Output shape: `[..., M, N]` with same dtype as `activation`
///
/// The result should match `matmul(activation, dequantize(weight))` within
/// quantization tolerance (see Verification Standards).
pub trait QuantMatmulOps<R: Runtime> {
    fn quant_matmul(&self, activation: &Tensor<R>, weight: &QuantTensor<R>) -> Result<Tensor<R>>;

    /// Batched quant_matmul: same activation × multiple quantized weights.
    ///
    /// Enables implementations to amortize activation preprocessing (e.g. Q8_1
    /// quantization on CUDA) across multiple projections that share the same input.
    /// Default implementation just loops.
    fn quant_matmul_batch(
        &self,
        activation: &Tensor<R>,
        weights: &[&QuantTensor<R>],
    ) -> Result<Vec<Tensor<R>>> {
        weights
            .iter()
            .map(|w| self.quant_matmul(activation, w))
            .collect()
    }

    /// AWQ W4A16 GEMM: input × dequantized INT4 weight
    ///
    /// Weight is packed in AWQ format: 8 INT4 values per u32 with non-sequential
    /// bit shifts `[0,16,4,20,8,24,12,28]`.
    ///
    /// # Contract
    ///
    /// - `input` shape: `[..., M, K]`, dtype F32
    /// - `qweight` shape: `[K, N/8]` packed u32 (AWQ layout)
    /// - `scales` shape: `[K/group_size, N]` F32
    /// - `zeros` shape: `[K/group_size, N]` F32
    /// - Output shape: `[..., M, N]` F32
    ///
    /// Dequant formula: `w = (q - zero) * scale`
    fn int4_gemm(
        &self,
        input: &Tensor<R>,
        qweight: &Tensor<R>,
        scales: &Tensor<R>,
        zeros: &Tensor<R>,
        group_size: usize,
    ) -> Result<Tensor<R>>;

    /// GPTQ W4A16 GEMM: input × dequantized INT4 weight (GPTQ layout)
    ///
    /// GPTQ uses a different packing and dequantization formula from AWQ,
    /// plus a `g_idx` permutation tensor for column reordering.
    ///
    /// # Contract
    ///
    /// - `input` shape: `[..., M, K]`, dtype F32
    /// - `qweight` shape: `[K/8, N]` packed u32 (sequential 4-bit packing)
    /// - `qzeros` shape: `[K/group_size, N/8]` packed u32
    /// - `scales` shape: `[K/group_size, N]` F32
    /// - `g_idx` shape: `[K]` I32 — maps each input column to its group index
    /// - Output shape: `[..., M, N]` F32
    ///
    /// Dequant formula: `w = q * scale + zero`
    fn int4_gemm_gptq(
        &self,
        input: &Tensor<R>,
        qweight: &Tensor<R>,
        qzeros: &Tensor<R>,
        scales: &Tensor<R>,
        g_idx: &Tensor<R>,
    ) -> Result<Tensor<R>>;

    /// Marlin-format W4A16 GEMM: tensor-core-friendly sequential INT4 packing
    ///
    /// Marlin uses sequential 4-bit packing (not AWQ order) optimized for
    /// tensor core access patterns.
    ///
    /// # Contract
    ///
    /// - `input` shape: `[..., M, K]`, dtype F32
    /// - `weight` shape: `[K/8, N]` packed u32 (sequential 4-bit)
    /// - `scales` shape: `[K/group_size, N]` F32
    /// - `zeros` shape: `[K/group_size, N]` F32
    /// - Output shape: `[..., M, N]` F32
    ///
    /// Dequant formula: `w = (q - 8) * scale + zero`
    fn marlin_gemm(
        &self,
        input: &Tensor<R>,
        weight: &Tensor<R>,
        scales: &Tensor<R>,
        zeros: &Tensor<R>,
        group_size: usize,
    ) -> Result<Tensor<R>>;
}
