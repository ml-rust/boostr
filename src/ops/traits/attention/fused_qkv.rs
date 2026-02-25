//! Fused QKV projection operations trait
//!
//! Fuses Q/K/V linear projections into a single matmul + split, reducing kernel
//! launches and memory traffic. Also fuses output projection with residual addition.

use crate::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Fused QKV projection and output projection operations
///
/// These operations fuse multiple linear projections that are typically done
/// separately in attention layers (Q, K, V projections) into single matmul
/// operations, reducing kernel launch overhead and memory traffic.
///
/// # Layout contracts
///
/// ## `fused_qkv_projection`
///
/// - `input`: `[B, S, H]` — hidden states
/// - `weight`: `[(Hq + 2*Hkv), H]` — concatenated Q/K/V weight matrices
///   where `Hq = num_heads * head_dim`, `Hkv = num_kv_heads * head_dim`
/// - `bias`: optional `[(Hq + 2*Hkv)]`
/// - Returns `(Q, K, V)`:
///   - Q: `[B, num_heads, S, head_dim]`
///   - K: `[B, num_kv_heads, S, head_dim]`
///   - V: `[B, num_kv_heads, S, head_dim]`
///
/// ## `fused_output_projection_residual`
///
/// - `attn_out`: `[B, S, Hq*D]` — flattened attention output
/// - `weight`: `[H, Hq*D]` — output projection weight
/// - `bias`: optional `[H]`
/// - `residual`: `[B, S, H]` — residual connection input
/// - Returns: `[B, S, H]`
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub trait FusedQkvOps<R: Runtime> {
    /// Fused Q/K/V projection: single matmul + split into Q, K, V
    ///
    /// Equivalent to:
    /// ```text
    /// qkv = input @ weight.T + bias
    /// q, k, v = split(qkv, [Hq, Hkv, Hkv], dim=-1)
    /// q = reshape(q, [B, S, num_heads, D]).transpose(1, 2)
    /// k = reshape(k, [B, S, num_kv_heads, D]).transpose(1, 2)
    /// v = reshape(v, [B, S, num_kv_heads, D]).transpose(1, 2)
    /// ```
    fn fused_qkv_projection(
        &self,
        input: &Tensor<R>,
        weight: &Tensor<R>,
        bias: Option<&Tensor<R>>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>)>;

    /// Fused output projection with residual addition
    ///
    /// Equivalent to:
    /// ```text
    /// output = attn_out @ weight.T + bias + residual
    /// ```
    fn fused_output_projection_residual(
        &self,
        attn_out: &Tensor<R>,
        weight: &Tensor<R>,
        bias: Option<&Tensor<R>>,
        residual: &Tensor<R>,
    ) -> Result<Tensor<R>>;

    /// Backward pass for fused QKV projection
    ///
    /// Given gradients `dq`, `dk`, `dv` w.r.t. Q, K, V outputs, computes:
    /// - `d_input`: gradient w.r.t. input `[B, S, H]`
    /// - `d_weight`: gradient w.r.t. weight `[(Hq + 2*Hkv), H]`
    /// - `d_bias`: gradient w.r.t. bias `[(Hq + 2*Hkv)]` (if bias was used)
    fn fused_qkv_projection_bwd(
        &self,
        dq: &Tensor<R>,
        dk: &Tensor<R>,
        dv: &Tensor<R>,
        input: &Tensor<R>,
        weight: &Tensor<R>,
        has_bias: bool,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<(Tensor<R>, Tensor<R>, Option<Tensor<R>>)>;

    /// Backward pass for fused output projection with residual
    ///
    /// Given `d_output` gradient, computes:
    /// - `d_attn_out`: gradient w.r.t. attn_out `[B, S, Hq*D]`
    /// - `d_weight`: gradient w.r.t. weight `[H, Hq*D]`
    /// - `d_bias`: gradient w.r.t. bias `[H]` (if bias was used)
    /// - `d_residual`: gradient w.r.t. residual (equals d_output, identity)
    fn fused_output_projection_residual_bwd(
        &self,
        d_output: &Tensor<R>,
        attn_out: &Tensor<R>,
        weight: &Tensor<R>,
        has_bias: bool,
    ) -> Result<(Tensor<R>, Tensor<R>, Option<Tensor<R>>, Tensor<R>)>;
}
