//! Per-shape kernel-family dispatch for CUDA Flash Attention v2 forward and
//! backward. Used by `impl_ops.rs`'s trait methods.
//!
//! [`dispatch_fwd`] tries the specialised paths in order and falls through
//! to the general `flash_v2.cu` kernel.
//!
//! `seq_len_q == 1` in F32/F16/BF16 always takes the decode kernel
//! (`decode_attention.cu`): every head_dim `validate_qkv` admits, any window,
//! either causal setting. The single query is the newest position, so the
//! causal mask admits every key and the flag is dropped; a window keeps the
//! last `min(window_size, seq_len_k)` keys, which the kernel walks as a key
//! range. FP8 decode has no such kernel and falls through.
//!
//! What reaches the general kernel:
//!
//! - head_dim 96, 192 or 256 at `seq_len_q > 1`, at every dtype, causal
//!   setting, window and head ratio (no dedicated prefill kernel instantiates
//!   them),
//! - `window_size > 0` at every head_dim and `seq_len_q > 1` (the prefill
//!   kernels have no window path),
//! - head_dim 32, 64 or 128 with `window_size == 0` on a device below sm_80
//!   (the MQA/GQA family is gated on `caps.bf16`).
//!
//! `validate_qkv` rejects `num_heads` not divisible by `num_kv_heads` before
//! any kernel is chosen, so that case reaches nothing.
//!
//! Key length (`kv_seq_len`): `KvSeqLen::Shared(n)` is the loop bound above,
//! with the tensor's dim-2 as the stride. `KvSeqLen::PerRow(ends)` reaches the
//! `seq_len_q == 1` decode kernel only, as a `[B]` I32 device pointer the
//! kernel reads once per block; row `b` then attends over
//! `[kv_start[b], ends[b])`. Every other tier here is refused — no kernel in
//! them reads a per-row end, and the CPU client is the path that takes one.
//!
//! Left padding (`kv_start`): every tier above takes the `[B]` start vector
//! as a device pointer (null for `None`) and reads it once per block; the
//! v3 launcher has no such argument, so a padded batch never reaches it.
//!
//! Output layout (`AttnOutLayout`): every kernel above takes it as a store
//! flag, so token-major output costs no extra launch. The decode kernel at
//! `seq_len_q == 1` stores the same bytes under either layout; the folded
//! short-query path passes `S_q` so the kernel unfolds the head axis at the
//! store. The v3 launcher predates the flag and is re-laid after the fact,
//! which only matters once `flash_v3::dispatch_enabled` is true.

use crate::error::Result;
use crate::ops::impl_generic::attention::{
    per_row_kv_seq_len_unsupported, validate_kv_start, validate_row_bounds,
};
use crate::ops::traits::{AttnOutLayout, KvSeqLen};
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::super::decode_split;
use super::super::mqa_gqa;
use super::flash_bwd;
use super::flash_decode;
use super::flash_fwd;
use super::flash_utils::validate_qkv;
use super::flash_v3;

/// Longest non-causal query sequence routed through
/// [`flash_decode::decode_attention_fwd_folded`] at a head dimension. The
/// fold re-reads K/V once per query row, so its cost grows with `S_q * S_k`,
/// while the tiled kernels reuse a staged K/V tile across every row of a
/// block and pay for the block's full row count even when most rows are past
/// `S_q`. The fold leads only while that row waste dominates.
///
/// The bound depends on which tiled kernel is the alternative: the MQA/GQA
/// kernel at 64 and 128, the general `flash_v2.cu` kernel at 96, 192 and 256,
/// and either at 32 depending on the head ratio. `examples/cuda_short_query_profile.rs`
/// (`--decode --d N --sq N` against the same shape at `S_q == 1`, or the
/// default sweep for 128) measures the crossover per head dimension; past
/// each bound the tiled kernel leads at both F32 and BF16.
fn short_query_fold_max(head_dim: usize) -> usize {
    match head_dim {
        64 | 128 => 16,
        96 => 8,
        _ => 4,
    }
}

/// Routes one forward call to the kernel family that serves its shape. See
/// the module doc for the full routing table.
#[allow(clippy::too_many_arguments)]
pub(super) fn dispatch_fwd(
    client: &CudaClient,
    q: &Tensor<CudaRuntime>,
    k: &Tensor<CudaRuntime>,
    v: &Tensor<CudaRuntime>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    causal: bool,
    window_size: usize,
    kv_seq_len: Option<KvSeqLen<'_, CudaRuntime>>,
    kv_start: Option<&Tensor<CudaRuntime>>,
    out_layout: AttnOutLayout,
) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
    let mut p = validate_qkv(q, k, v, num_heads, num_kv_heads, head_dim)?;

    // kv_seq_len override: a shared length is the loop bound, tensor dim-2
    // the stride. A per-row end leaves the scalar at the capacity and is
    // carried to the decode kernel below as a device pointer instead.
    let kv_seq_stride = p.seq_len_k; // memory stride = full capacity
    if let Some(KvSeqLen::Shared(seq_len)) = kv_seq_len {
        p.seq_len_k = seq_len;
    }

    // Per-row left padding reaches every kernel as a device pointer;
    // null means no padding and skips the per-row lookup.
    let kv_start_ptr = match kv_start {
        Some(t) => {
            validate_kv_start(t, p.batch_size)?;
            t.ptr()
        }
        None => 0,
    };

    // Decode path: S_q=1, use lightweight vec kernel (supports separate stride,
    // window, per-row key ends and every validated head_dim). Instantiated
    // for F32/F16/BF16; anything else falls through to the general kernel,
    // which tiles a one-row query and is far slower here. `causal` is not
    // forwarded: the one query is the newest position and sees every key
    // either way.
    if p.seq_len_q == 1
        && decode_split::decode_supports_dtype(q.dtype())
        && decode_split::decode_supports_head_dim(head_dim)
    {
        // Per-row ends reach the kernel as a device pointer beside the
        // starts; null means every row ends at `p.seq_len_k`.
        let kv_end_ptr = match kv_seq_len {
            Some(KvSeqLen::PerRow(t)) => {
                validate_row_bounds(t, p.batch_size, "kv_seq_len")?;
                t.ptr()
            }
            _ => 0,
        };
        return flash_decode::decode_attention_fwd(
            client,
            q,
            k,
            v,
            &p,
            kv_seq_stride,
            window_size,
            kv_start_ptr,
            kv_end_ptr,
            out_layout,
        );
    }

    // Every tier below reads one key length for the whole batch.
    if let Some(KvSeqLen::PerRow(_)) = kv_seq_len {
        return Err(per_row_kv_seq_len_unsupported("CUDA"));
    }

    // Short non-causal query sequence: run as decode with the query axis
    // folded into the head axis. The tiled kernels below pay for a whole
    // block of rows, most of them empty at this length;
    // `examples/cuda_short_query_profile.rs` measures the fold ahead at
    // every length up to the bound. It reads K/V with the stride the
    // decode path already takes, so it goes before the narrowing copy.
    if !causal
        && window_size == 0
        && p.seq_len_q <= short_query_fold_max(head_dim)
        && decode_split::decode_supports_dtype(q.dtype())
        && decode_split::decode_supports_head_dim(head_dim)
    {
        return flash_decode::decode_attention_fwd_folded(
            client,
            q,
            k,
            v,
            &p,
            kv_seq_stride,
            kv_start_ptr,
            out_layout,
        );
    }

    // Flash v2/v3 don't support separate kv_seq_stride — narrow if needed
    if kv_seq_stride != p.seq_len_k {
        let k_narrow = k.narrow(2, 0, p.seq_len_k)?.contiguous()?;
        let v_narrow = v.narrow(2, 0, p.seq_len_k)?.contiguous()?;
        return dispatch_fwd(
            client,
            q,
            &k_narrow,
            &v_narrow,
            num_heads,
            num_kv_heads,
            head_dim,
            causal,
            window_size,
            None,
            kv_start,
            out_layout,
        );
    }

    // Flash v3 on SM 90+ for supported configs, when v3 is dispatchable
    // at all. `flash_v3::dispatch_enabled` is the single decision point and
    // is currently false — see its doc comment. The v3 launcher takes no
    // per-row key start, so a padded batch stays on the v2 family.
    if kv_start.is_none()
        && num_kv_heads == num_heads
        && window_size == 0
        && flash_v3::dispatch_enabled(client, q.device())
        && let Some(result) = flash_v3::flash_v3_fwd(
            client,
            q,
            k,
            v,
            p.batch_size,
            p.num_heads,
            p.seq_len_q,
            p.seq_len_k,
            p.head_dim,
            causal,
        )?
    {
        let (out, lse) = result;
        return Ok((out_layout.from_head_major(out)?, lse));
    }

    // Dedicated MQA/GQA kernels, for the shapes they're capable of.
    //
    // `should_use_mqa_gqa` is a capability gate, not a performance
    // heuristic — see its doc comment. Gated on two more things the
    // kernels actually support: `window_size == 0` because they have no
    // sliding-window path, and F32/F16/BF16 because those are the only
    // dtype variants instantiated. Anything else falls through to the
    // general kernel below, which is what ran for every shape before
    // this was wired up.
    // `caps.bf16` gates this because `mqa_gqa_bwd.cu` needs sm_80 for its
    // native bf16 backward kernels. The forward runs fine below sm_80, but
    // gating it here too keeps forward and backward on the same kernel
    // family. Falling through reaches the general flash kernel.
    if window_size == 0
        && matches!(q.dtype(), DType::F32 | DType::F16 | DType::BF16)
        && numr::runtime::cuda::CudaDevice::new(q.device().id())
            .profile()
            .caps
            .bf16
        && mqa_gqa::should_use_mqa_gqa(num_heads, num_kv_heads, head_dim)
    {
        return mqa_gqa::mqa_gqa_fwd(
            client,
            q,
            k,
            v,
            num_heads,
            num_kv_heads,
            head_dim,
            causal,
            kv_start_ptr,
            out_layout,
        );
    }

    flash_fwd::flash_attention_fwd_impl(
        client,
        q,
        k,
        v,
        &p,
        causal,
        window_size,
        kv_start_ptr,
        out_layout,
    )
}

/// Routes one backward call to the kernel family that serves its shape.
/// Mirrors [`dispatch_fwd`]'s v3 and MQA/GQA gates: the forward and backward
/// of one call must land on the same kernel family, since a v3 forward with
/// a v2 backward would mix causal conventions, and forward/backward pairs
/// outside their tested combination were never parity-tested together.
#[allow(clippy::too_many_arguments)]
pub(super) fn dispatch_bwd(
    client: &CudaClient,
    dout: &Tensor<CudaRuntime>,
    q: &Tensor<CudaRuntime>,
    k: &Tensor<CudaRuntime>,
    v: &Tensor<CudaRuntime>,
    output: &Tensor<CudaRuntime>,
    lse: &Tensor<CudaRuntime>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    causal: bool,
    window_size: usize,
) -> Result<(
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
)> {
    let p = validate_qkv(q, k, v, num_heads, num_kv_heads, head_dim)?;

    // Same v3 gate as the forward, through the same single decision
    // point: a v3 forward with a v2 backward would mix causal conventions.
    if num_kv_heads == num_heads
        && window_size == 0
        && flash_v3::dispatch_enabled(client, q.device())
        && let Some(result) = flash_v3::flash_v3_bwd(
            client,
            dout,
            q,
            k,
            v,
            output,
            lse,
            p.batch_size,
            p.num_heads,
            p.seq_len_q,
            p.seq_len_k,
            p.head_dim,
            causal,
        )?
    {
        return Ok(result);
    }

    // Same gate as the forward. Both halves must agree: routing the forward
    // to the MQA/GQA kernel and the backward to the general one would pair
    // kernels that were never parity-tested together.
    // `caps.bf16` gates this because `mqa_gqa_bwd.cu` needs sm_80 for its
    // native bf16 backward kernels. Falling through reaches the general
    // flash kernel.
    if window_size == 0
        && matches!(q.dtype(), DType::F32 | DType::F16 | DType::BF16)
        && numr::runtime::cuda::CudaDevice::new(q.device().id())
            .profile()
            .caps
            .bf16
        && mqa_gqa::should_use_mqa_gqa(num_heads, num_kv_heads, head_dim)
    {
        return mqa_gqa::mqa_gqa_bwd(
            client,
            dout,
            q,
            k,
            v,
            output,
            lse,
            num_heads,
            num_kv_heads,
            head_dim,
            causal,
        );
    }

    flash_bwd::flash_attention_bwd_impl(client, dout, q, k, v, output, lse, &p, causal, window_size)
}
