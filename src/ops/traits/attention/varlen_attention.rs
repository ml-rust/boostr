//! Variable-length (ragged/packed) attention traits

use crate::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Variable-length Flash Attention — packed sequences with cu_seqlens indexing
///
/// Eliminates padding waste by packing sequences of different lengths into
/// a single 1D buffer. 30-50% memory savings for variable-length batches.
///
/// Supports both MHA (`num_kv_heads == num_heads`) and GQA
/// (`num_kv_heads < num_heads`, where `num_heads % num_kv_heads == 0`).
///
/// # Layout contract
///
/// - `q`: `[total_tokens_q, num_heads, head_dim]` — packed queries
/// - `k`: `[total_tokens_k, num_kv_heads, head_dim]` — packed keys (GQA: fewer heads)
/// - `v`: `[total_tokens_k, num_kv_heads, head_dim]` — packed values (GQA: fewer heads)
/// - `cu_seqlens_q`: `[batch_size + 1]` — cumulative query sequence lengths (I32)
/// - `cu_seqlens_k`: `[batch_size + 1]` — cumulative key sequence lengths (I32)
/// - Output: `[total_tokens_q, num_heads, head_dim]`
/// - Logsumexp: `[total_tokens_q, num_heads]` (F32)
///
/// For MHA pass `num_kv_heads == num_heads`; K/V layout is then identical to
/// the old MHA-only contract.
///
/// # GQA key/value head mapping
///
/// `kv_head_idx = q_head_idx / (num_heads / num_kv_heads)`
///
/// # Cumulative sequence lengths
///
/// `cu_seqlens[0] = 0`, `cu_seqlens[i] = sum of lengths for sequences 0..i-1`.
/// For batch `[512, 300, 128]`: `cu_seqlens = [0, 512, 812, 940]`.
#[allow(clippy::too_many_arguments)]
pub trait VarLenAttentionOps<R: Runtime> {
    /// Variable-length attention forward pass
    ///
    /// For GQA set `num_kv_heads < num_heads`; K/V must be shaped
    /// `[total_tokens_k, num_kv_heads, head_dim]`.  For MHA set
    /// `num_kv_heads == num_heads`.
    ///
    /// Returns `(output, logsumexp)`.
    fn varlen_attention_fwd(
        &self,
        q: &Tensor<R>,
        k: &Tensor<R>,
        v: &Tensor<R>,
        cu_seqlens_q: &Tensor<R>,
        cu_seqlens_k: &Tensor<R>,
        batch_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        max_seqlen_q: usize,
        max_seqlen_k: usize,
        head_dim: usize,
        causal: bool,
    ) -> Result<(Tensor<R>, Tensor<R>)>;

    /// Variable-length attention backward pass
    ///
    /// Supports full GQA (`num_kv_heads < num_heads`): `dk` and `dv` are shaped
    /// `[total_tokens_k, num_kv_heads, head_dim]`.  For MHA set
    /// `num_kv_heads == num_heads`.  `num_heads % num_kv_heads` must be zero.
    ///
    /// Returns `(dq, dk, dv)`.
    fn varlen_attention_bwd(
        &self,
        dout: &Tensor<R>,
        q: &Tensor<R>,
        k: &Tensor<R>,
        v: &Tensor<R>,
        output: &Tensor<R>,
        lse: &Tensor<R>,
        cu_seqlens_q: &Tensor<R>,
        cu_seqlens_k: &Tensor<R>,
        batch_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        max_seqlen_q: usize,
        max_seqlen_k: usize,
        head_dim: usize,
        causal: bool,
    ) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>)>;
}
