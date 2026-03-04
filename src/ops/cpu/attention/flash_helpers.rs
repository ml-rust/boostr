//! Helper functions for CPU flash/standard attention (forward and backward passes).

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CumulativeOps, MatmulOps, ReduceOps, ScalarOps, ShapeOps,
};
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

/// Standard attention forward on Tensors (not Vars).
/// Returns (output, logsumexp) matching FlashAttention's interface.
///
/// Computes: output = softmax(Q @ K^T / sqrt(d) + mask) @ V
///           lse = logsumexp(Q @ K^T / sqrt(d) + mask, dim=-1)
#[allow(clippy::too_many_arguments)]
pub(super) fn standard_attention_fwd(
    client: &CpuClient,
    q: &Tensor<CpuRuntime>,
    k: &Tensor<CpuRuntime>,
    v: &Tensor<CpuRuntime>,
    causal: bool,
    num_heads: usize,
    num_kv_heads: usize,
    window_size: usize,
) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
    let q_shape = q.shape();
    let head_dim = q_shape[3];
    let seq_len_q = q_shape[2];
    let seq_len_k = k.shape()[2];
    let _batch_size = q_shape[0];
    let scale = (head_dim as f64).sqrt().recip();

    // GQA: expand KV heads to match query heads
    let (k_expanded, v_expanded) = if num_kv_heads < num_heads {
        let repeats = num_heads / num_kv_heads;
        let k_exp = client
            .repeat_interleave(k, repeats, Some(1))
            .map_err(Error::Numr)?;
        let v_exp = client
            .repeat_interleave(v, repeats, Some(1))
            .map_err(Error::Numr)?;
        (k_exp, v_exp)
    } else {
        (k.clone(), v.clone())
    };

    // Q @ K^T → [B, H, S_q, S_k]
    let k_t = k_expanded.transpose(-2, -1).map_err(Error::Numr)?;
    let k_t = k_t.contiguous();
    let scores = client.matmul(q, &k_t).map_err(Error::Numr)?;

    // Scale
    let scores = client.mul_scalar(&scores, scale).map_err(Error::Numr)?;

    // Attention mask (causal + sliding window)
    let scores = if causal || window_size > 0 {
        let mask = build_attention_mask(seq_len_q, seq_len_k, causal, window_size, q.device())?;
        // Cast mask to match scores dtype (e.g. BF16 activations)
        let mask = if mask.dtype() != scores.dtype() {
            use numr::ops::TypeConversionOps;
            client.cast(&mask, scores.dtype()).map_err(Error::Numr)?
        } else {
            mask
        };
        client.add(&scores, &mask).map_err(Error::Numr)?
    } else {
        scores
    };

    // Logsumexp for backward pass: [B, H, S_q]
    let lse = client
        .logsumexp(&scores, &[3], false)
        .map_err(Error::Numr)?;

    // Softmax → [B, H, S_q, S_k]
    let weights = client.softmax(&scores, -1).map_err(Error::Numr)?;

    // Weights @ V → [B, H, S_q, D]
    let output = client.matmul(&weights, &v_expanded).map_err(Error::Numr)?;

    // Cast LSE to F32 if not already
    let lse = if lse.dtype() != DType::F32 {
        use numr::ops::TypeConversionOps;
        client.cast(&lse, DType::F32).map_err(Error::Numr)?
    } else {
        lse
    };

    Ok((output, lse))
}

/// Standard attention backward on Tensors.
/// Given dO, recomputes attention weights and computes dQ, dK, dV.
#[allow(clippy::too_many_arguments)]
pub(super) fn standard_attention_bwd(
    client: &CpuClient,
    dout: &Tensor<CpuRuntime>,
    q: &Tensor<CpuRuntime>,
    k: &Tensor<CpuRuntime>,
    v: &Tensor<CpuRuntime>,
    output: &Tensor<CpuRuntime>,
    _lse: &Tensor<CpuRuntime>,
    causal: bool,
    num_heads: usize,
    num_kv_heads: usize,
    window_size: usize,
) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
    let q_shape = q.shape();
    let head_dim = q_shape[3];
    let seq_len_q = q_shape[2];
    let seq_len_k = k.shape()[2];
    let scale = (head_dim as f64).sqrt().recip();

    // GQA: expand KV heads
    let (k_expanded, v_expanded) = if num_kv_heads < num_heads {
        let repeats = num_heads / num_kv_heads;
        let k_exp = client
            .repeat_interleave(k, repeats, Some(1))
            .map_err(Error::Numr)?;
        let v_exp = client
            .repeat_interleave(v, repeats, Some(1))
            .map_err(Error::Numr)?;
        (k_exp, v_exp)
    } else {
        (k.clone(), v.clone())
    };

    // Recompute scores and attention weights
    let k_t = k_expanded.transpose(-2, -1).map_err(Error::Numr)?;
    let k_t = k_t.contiguous();
    let scores = client.matmul(q, &k_t).map_err(Error::Numr)?;
    let scores = client.mul_scalar(&scores, scale).map_err(Error::Numr)?;

    let scores = if causal || window_size > 0 {
        let mask = build_attention_mask(seq_len_q, seq_len_k, causal, window_size, q.device())?;
        let mask = if mask.dtype() != scores.dtype() {
            use numr::ops::TypeConversionOps;
            client.cast(&mask, scores.dtype()).map_err(Error::Numr)?
        } else {
            mask
        };
        client.add(&scores, &mask).map_err(Error::Numr)?
    } else {
        scores
    };

    let weights = client.softmax(&scores, -1).map_err(Error::Numr)?;

    // dV = P^T @ dO  → [B, H, S_k, D]
    let weights_t = weights.transpose(-2, -1).map_err(Error::Numr)?;
    let weights_t = weights_t.contiguous();
    let dv_expanded = client.matmul(&weights_t, dout).map_err(Error::Numr)?;

    // dP = dO @ V^T  → [B, H, S_q, S_k]
    let v_t = v_expanded.transpose(-2, -1).map_err(Error::Numr)?;
    let v_t = v_t.contiguous();
    let dp = client.matmul(dout, &v_t).map_err(Error::Numr)?;

    // D = rowsum(dO ⊙ O)  → [B, H, S_q]
    let do_times_o = client.mul(dout, output).map_err(Error::Numr)?;
    let d_buf = client.sum(&do_times_o, &[3], false).map_err(Error::Numr)?;

    // dS = P * (dP - D)  → [B, H, S_q, S_k]
    // Broadcast D from [B,H,Sq] to [B,H,Sq,Sk]
    let d_expanded = d_buf.unsqueeze(-1).map_err(Error::Numr)?;
    let d_expanded = d_expanded.broadcast_to(dp.shape()).map_err(Error::Numr)?;
    let ds = client.sub(&dp, &d_expanded).map_err(Error::Numr)?;
    let ds = client.mul(&weights, &ds).map_err(Error::Numr)?;

    // Scale dS
    let ds = client.mul_scalar(&ds, scale).map_err(Error::Numr)?;

    // dQ = dS @ K  → [B, H, S_q, D]
    let dq = client.matmul(&ds, &k_expanded).map_err(Error::Numr)?;

    // dK = dS^T @ Q  → [B, H, S_k, D]
    let ds_t = ds.transpose(-2, -1).map_err(Error::Numr)?;
    let ds_t = ds_t.contiguous();
    let dk_expanded = client.matmul(&ds_t, q).map_err(Error::Numr)?;

    // If GQA, sum gradients across repeated heads back to num_kv_heads
    let (dk, dv) = if num_kv_heads < num_heads {
        let repeats = num_heads / num_kv_heads;
        let dk = sum_gqa_grads(client, &dk_expanded, num_kv_heads, repeats)?;
        let dv = sum_gqa_grads(client, &dv_expanded, num_kv_heads, repeats)?;
        (dk, dv)
    } else {
        (dk_expanded, dv_expanded)
    };

    Ok((dq, dk, dv))
}

/// Sum gradients from expanded heads back to num_kv_heads for GQA.
/// Input: [B, num_heads, S, D] → Output: [B, num_kv_heads, S, D]
pub(super) fn sum_gqa_grads(
    client: &CpuClient,
    grad: &Tensor<CpuRuntime>,
    num_kv_heads: usize,
    repeats: usize,
) -> Result<Tensor<CpuRuntime>> {
    let shape = grad.shape();
    let (b, _nh, s, d) = (shape[0], shape[1], shape[2], shape[3]);
    // Reshape [B, num_kv_heads * repeats, S, D] → [B, num_kv_heads, repeats, S, D]
    let reshaped = grad
        .reshape(&[b, num_kv_heads, repeats, s, d])
        .map_err(Error::Numr)?;
    // Sum over the repeats dim (dim=2)
    client.sum(&reshaped, &[2], false).map_err(Error::Numr)
}

/// Build additive causal mask: 0 for allowed, -inf for masked.
/// Build attention mask combining causal and sliding window constraints.
/// - causal: position j > i is masked
/// - window_size > 0: position j < i - window_size + 1 is masked
pub(super) fn build_attention_mask(
    seq_len_q: usize,
    seq_len_k: usize,
    causal: bool,
    window_size: usize,
    device: &<CpuRuntime as numr::runtime::Runtime>::Device,
) -> Result<Tensor<CpuRuntime>> {
    let mut mask_data = vec![0.0f32; seq_len_q * seq_len_k];
    for i in 0..seq_len_q {
        for j in 0..seq_len_k {
            let masked = (causal && j > i) || (window_size > 0 && (j + window_size) <= i);
            if masked {
                mask_data[i * seq_len_k + j] = f32::NEG_INFINITY;
            }
        }
    }
    Ok(Tensor::<CpuRuntime>::from_slice(
        &mask_data,
        &[1, 1, seq_len_q, seq_len_k],
        device,
    ))
}
