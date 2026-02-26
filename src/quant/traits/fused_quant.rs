//! Fused quantized operation traits

use crate::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Fused quantized operations that combine multiple INT4 GEMMs with activations.
///
/// These avoid redundant activation reads by computing multiple projections
/// in a single pass over the input tensor.
#[allow(clippy::too_many_arguments)]
pub trait FusedQuantOps<R: Runtime> {
    /// Fused INT4 dual-GEMM + SwiGLU: `silu(input @ gate_w) * (input @ up_w)`
    ///
    /// Reads input once and computes both gate and up projections simultaneously,
    /// then applies SwiGLU activation. All weights in AWQ INT4 format.
    ///
    /// # Contract
    ///
    /// - `input` shape: `[..., M, K]`, dtype F32
    /// - `gate_qweight`, `up_qweight` shape: `[K, N/8]` packed u32 (AWQ)
    /// - `gate_scales`, `up_scales` shape: `[K/group_size, N]` F32
    /// - `gate_zeros`, `up_zeros` shape: `[K/group_size, N]` F32
    /// - Output shape: `[..., M, N]` F32
    fn fused_int4_swiglu(
        &self,
        input: &Tensor<R>,
        gate_qweight: &Tensor<R>,
        gate_scales: &Tensor<R>,
        gate_zeros: &Tensor<R>,
        up_qweight: &Tensor<R>,
        up_scales: &Tensor<R>,
        up_zeros: &Tensor<R>,
        group_size: usize,
    ) -> Result<Tensor<R>>;

    /// Fused INT4 triple-GEMM QKV projection: `(input@Wq, input@Wk, input@Wv)`
    ///
    /// Reads input once and computes Q, K, V projections simultaneously.
    /// All weights in AWQ INT4 format.
    ///
    /// # Contract
    ///
    /// - `input` shape: `[..., M, K]`, dtype F32
    /// - `qweight_q` shape: `[K, Nq/8]`, `qweight_k/v` shape: `[K, Nkv/8]`
    /// - `scales_q` shape: `[K/group_size, Nq]`, etc.
    /// - Returns `(Q, K, V)` with shapes `[..., M, Nq]`, `[..., M, Nkv]`, `[..., M, Nkv]`
    fn fused_int4_qkv(
        &self,
        input: &Tensor<R>,
        qweight_q: &Tensor<R>,
        scales_q: &Tensor<R>,
        zeros_q: &Tensor<R>,
        qweight_k: &Tensor<R>,
        scales_k: &Tensor<R>,
        zeros_k: &Tensor<R>,
        qweight_v: &Tensor<R>,
        scales_v: &Tensor<R>,
        zeros_v: &Tensor<R>,
        group_size: usize,
    ) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>)>;
}
