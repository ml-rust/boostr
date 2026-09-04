//! CPU reference for the materialized biased-attention backward pass.
//!
//! Plain loops, written for clarity: this is the parity reference the CUDA
//! `alibi_bwd` kernels are checked against, not a fast path.
//!
//! F32 only. Any other dtype is refused rather than mis-computed.

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

fn check(t: &Tensor<CpuRuntime>, arg: &'static str, expected: &[usize]) -> Result<()> {
    if t.dtype() != DType::F32 {
        return Err(Error::InvalidArgument {
            arg,
            reason: format!(
                "alibi_attention_bwd on CPU is F32 only, got {:?}",
                t.dtype()
            ),
        });
    }
    if t.shape() != expected {
        return Err(Error::InvalidArgument {
            arg,
            reason: format!("expected shape {:?}, got {:?}", expected, t.shape()),
        });
    }
    if !t.is_contiguous() {
        return Err(Error::InvalidArgument {
            arg,
            reason: "alibi_attention_bwd requires contiguous inputs".into(),
        });
    }
    Ok(())
}

/// Backward for the materialized biased-attention path. Returns (dQ, dK, dV).
///
/// - `grad_probs  = grad_output @ V^T`
/// - `grad_scores = probs * (grad_probs - rowsum(grad_probs * probs))`
/// - `grad_q = (grad_scores @ K) * scale`
/// - `grad_k = (grad_scores^T @ Q) * scale`
/// - `grad_v = probs^T @ grad_output`
#[allow(clippy::too_many_arguments)]
pub(super) fn alibi_attention_bwd_impl(
    _client: &CpuClient,
    grad_output: &Tensor<CpuRuntime>,
    probs: &Tensor<CpuRuntime>,
    q: &Tensor<CpuRuntime>,
    k: &Tensor<CpuRuntime>,
    v: &Tensor<CpuRuntime>,
    batch_size: usize,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
    let probs_shape = probs.shape();
    if probs_shape.len() != 4 {
        return Err(Error::InvalidArgument {
            arg: "probs",
            reason: format!("expected 4D [B, H, S_q, S_k], got {}D", probs_shape.len()),
        });
    }
    let seq_len_q = probs_shape[2];
    let seq_len_k = probs_shape[3];

    check(
        probs,
        "probs",
        &[batch_size, num_heads, seq_len_q, seq_len_k],
    )?;
    check(
        grad_output,
        "grad_output",
        &[batch_size, num_heads, seq_len_q, head_dim],
    )?;
    check(q, "q", &[batch_size, num_heads, seq_len_q, head_dim])?;
    check(k, "k", &[batch_size, num_heads, seq_len_k, head_dim])?;
    check(v, "v", &[batch_size, num_heads, seq_len_k, head_dim])?;

    let go = grad_output.to_vec::<f32>();
    let p = probs.to_vec::<f32>();
    let qv = q.to_vec::<f32>();
    let kv = k.to_vec::<f32>();
    let vv = v.to_vec::<f32>();

    let bh_count = batch_size * num_heads;
    let score_plane = seq_len_q * seq_len_k;
    let q_plane = seq_len_q * head_dim;
    let k_plane = seq_len_k * head_dim;

    let mut grad_scores = vec![0.0f32; bh_count * score_plane];
    let mut grad_q = vec![0.0f32; bh_count * q_plane];
    let mut grad_k = vec![0.0f32; bh_count * k_plane];
    let mut grad_v = vec![0.0f32; bh_count * k_plane];

    for bh in 0..bh_count {
        let s_base = bh * score_plane;
        let q_base = bh * q_plane;
        let k_base = bh * k_plane;

        // Steps 1 and 2, fused per row: grad_probs = grad_output @ V^T, then
        // grad_scores = probs * (grad_probs - rowsum(grad_probs * probs)).
        for i in 0..seq_len_q {
            let row = s_base + i * seq_len_k;

            let mut grad_probs_row = vec![0.0f32; seq_len_k];
            for (kk, gp) in grad_probs_row.iter_mut().enumerate() {
                let mut acc = 0.0f32;
                for d in 0..head_dim {
                    acc += go[q_base + i * head_dim + d] * vv[k_base + kk * head_dim + d];
                }
                *gp = acc;
            }

            let mut row_dot = 0.0f32;
            for (kk, gp) in grad_probs_row.iter().enumerate() {
                row_dot += gp * p[row + kk];
            }

            for (kk, gp) in grad_probs_row.iter().enumerate() {
                grad_scores[row + kk] = p[row + kk] * (gp - row_dot);
            }
        }

        // Step 3: grad_q = (grad_scores @ K) * scale.
        for i in 0..seq_len_q {
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for kk in 0..seq_len_k {
                    acc +=
                        grad_scores[s_base + i * seq_len_k + kk] * kv[k_base + kk * head_dim + d];
                }
                grad_q[q_base + i * head_dim + d] = acc * scale;
            }
        }

        // Step 4: grad_k = (grad_scores^T @ Q) * scale.
        // Step 5: grad_v = probs^T @ grad_output. No scale: the forward applies
        // `scale` before the softmax, so it reaches dV through `probs` alone.
        for kk in 0..seq_len_k {
            for d in 0..head_dim {
                let mut acc_k = 0.0f32;
                let mut acc_v = 0.0f32;
                for i in 0..seq_len_q {
                    let s = s_base + i * seq_len_k + kk;
                    acc_k += grad_scores[s] * qv[q_base + i * head_dim + d];
                    acc_v += p[s] * go[q_base + i * head_dim + d];
                }
                grad_k[k_base + kk * head_dim + d] = acc_k * scale;
                grad_v[k_base + kk * head_dim + d] = acc_v;
            }
        }
    }

    let device = q.device();
    let dq = Tensor::<CpuRuntime>::from_slice(
        &grad_q,
        &[batch_size, num_heads, seq_len_q, head_dim],
        device,
    )?;
    let dk = Tensor::<CpuRuntime>::from_slice(
        &grad_k,
        &[batch_size, num_heads, seq_len_k, head_dim],
        device,
    )?;
    let dv = Tensor::<CpuRuntime>::from_slice(
        &grad_v,
        &[batch_size, num_heads, seq_len_k, head_dim],
        device,
    )?;

    Ok((dq, dk, dv))
}
