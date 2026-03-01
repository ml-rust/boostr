//! CPU implementation of AttentionOps and FlashAttentionOps
//!
//! AttentionOps delegates to impl_generic (Var-based autograd).
//! FlashAttentionOps uses standard O(N²) attention via numr Tensor ops
//! (no fused kernel — CPU fallback).

use crate::error::{Error, Result};
use crate::ops::impl_generic::attention::multi_head_attention_impl;
use crate::ops::traits::{AttentionOps, FlashAttentionOps};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CumulativeOps, MatmulOps, ReduceOps, ScalarOps, ShapeOps,
};
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl AttentionOps<CpuRuntime> for CpuClient {
    fn multi_head_attention(
        &self,
        q: &Var<CpuRuntime>,
        k: &Var<CpuRuntime>,
        v: &Var<CpuRuntime>,
        mask: Option<&Var<CpuRuntime>>,
        num_heads: usize,
    ) -> Result<Var<CpuRuntime>> {
        multi_head_attention_impl(self, q, k, v, mask, num_heads)
    }
}

/// Standard attention forward on Tensors (not Vars).
/// Returns (output, logsumexp) matching FlashAttention's interface.
///
/// Computes: output = softmax(Q @ K^T / sqrt(d) + mask) @ V
///           lse = logsumexp(Q @ K^T / sqrt(d) + mask, dim=-1)
#[allow(clippy::too_many_arguments)]
fn standard_attention_fwd(
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
fn standard_attention_bwd(
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
fn sum_gqa_grads(
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
fn build_attention_mask(
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

impl FlashAttentionOps<CpuRuntime> for CpuClient {
    fn flash_attention_fwd(
        &self,
        q: &Tensor<CpuRuntime>,
        k: &Tensor<CpuRuntime>,
        v: &Tensor<CpuRuntime>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        causal: bool,
        window_size: usize,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        // Fast path: fused decode attention for S_q=1 (single token generation)
        // Avoids all intermediate tensor allocations and GQA expansion
        let seq_len_q = q.shape()[2];
        if seq_len_q == 1 && !causal && window_size == 0 && q.dtype() == DType::F32 {
            return super::decode_attention::fused_decode_attention(
                q,
                k,
                v,
                num_heads,
                num_kv_heads,
                head_dim,
            );
        }

        let _ = head_dim; // validated by shape
        standard_attention_fwd(self, q, k, v, causal, num_heads, num_kv_heads, window_size)
    }

    fn flash_attention_fwd_fp8(
        &self,
        _q: &Tensor<CpuRuntime>,
        _k: &Tensor<CpuRuntime>,
        _v: &Tensor<CpuRuntime>,
        _num_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _causal: bool,
        _q_scale: f32,
        _k_scale: f32,
        _v_scale: f32,
        _o_scale: f32,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        Err(Error::InvalidArgument {
            arg: "dtype",
            reason: "FP8 Flash Attention is not supported on CPU".into(),
        })
    }

    fn flash_attention_bwd(
        &self,
        dout: &Tensor<CpuRuntime>,
        q: &Tensor<CpuRuntime>,
        k: &Tensor<CpuRuntime>,
        v: &Tensor<CpuRuntime>,
        output: &Tensor<CpuRuntime>,
        lse: &Tensor<CpuRuntime>,
        num_heads: usize,
        num_kv_heads: usize,
        _head_dim: usize,
        causal: bool,
        window_size: usize,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        standard_attention_bwd(
            self,
            dout,
            q,
            k,
            v,
            output,
            lse,
            causal,
            num_heads,
            num_kv_heads,
            window_size,
        )
    }

    fn flash_attention_bwd_fp8(
        &self,
        _dout: &Tensor<CpuRuntime>,
        _q: &Tensor<CpuRuntime>,
        _k: &Tensor<CpuRuntime>,
        _v: &Tensor<CpuRuntime>,
        _output: &Tensor<CpuRuntime>,
        _lse: &Tensor<CpuRuntime>,
        _num_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _causal: bool,
        _q_scale: f32,
        _k_scale: f32,
        _v_scale: f32,
        _do_scale: f32,
        _o_scale: f32,
        _dq_scale: f32,
        _dk_scale: f32,
        _dv_scale: f32,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        Err(Error::InvalidArgument {
            arg: "dtype",
            reason: "FP8 Flash Attention backward is not supported on CPU".into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::ops::UnaryOps;

    fn rand_tensor(
        shape: &[usize],
        _client: &CpuClient,
        device: &<CpuRuntime as numr::runtime::Runtime>::Device,
    ) -> Tensor<CpuRuntime> {
        // Simple deterministic pseudo-random data
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
        Tensor::<CpuRuntime>::from_slice(&data, shape, device)
    }

    #[test]
    fn test_flash_fwd_output_shape() {
        let (client, device) = cpu_setup();
        let (b, h, s, d) = (2, 4, 8, 16);
        let q = rand_tensor(&[b, h, s, d], &client, &device);
        let k = rand_tensor(&[b, h, s, d], &client, &device);
        let v = rand_tensor(&[b, h, s, d], &client, &device);

        let (out, lse) = client
            .flash_attention_fwd(&q, &k, &v, h, h, d, false, 0)
            .unwrap();
        assert_eq!(out.shape(), &[b, h, s, d]);
        assert_eq!(lse.shape(), &[b, h, s]);
    }

    #[test]
    fn test_flash_fwd_causal() {
        let (client, device) = cpu_setup();
        let (b, h, s, d) = (1, 2, 6, 8);
        let q = rand_tensor(&[b, h, s, d], &client, &device);
        let k = rand_tensor(&[b, h, s, d], &client, &device);
        let v = rand_tensor(&[b, h, s, d], &client, &device);

        let (out_causal, _) = client
            .flash_attention_fwd(&q, &k, &v, h, h, d, true, 0)
            .unwrap();
        let (out_full, _) = client
            .flash_attention_fwd(&q, &k, &v, h, h, d, false, 0)
            .unwrap();

        // Causal and full should differ (unless trivial inputs)
        let diff = client.sub(&out_causal, &out_full).unwrap();
        let abs_diff = client.abs(&diff).unwrap();
        let max_diff = client.max(&abs_diff, &[], false).unwrap();
        let max_val = max_diff.to_vec::<f32>()[0];
        assert!(
            max_val > 1e-6,
            "Causal and non-causal outputs should differ"
        );
    }

    #[test]
    fn test_flash_fwd_sliding_window() {
        let (client, device) = cpu_setup();
        let (b, h, s, d) = (1, 2, 12, 8);
        let q = rand_tensor(&[b, h, s, d], &client, &device);
        let k = rand_tensor(&[b, h, s, d], &client, &device);
        let v = rand_tensor(&[b, h, s, d], &client, &device);

        let (out_window, _) = client
            .flash_attention_fwd(&q, &k, &v, h, h, d, false, 4)
            .unwrap();
        let (out_full, _) = client
            .flash_attention_fwd(&q, &k, &v, h, h, d, false, 0)
            .unwrap();

        let ow = out_window.to_vec::<f32>();
        let of = out_full.to_vec::<f32>();
        let max_diff = ow
            .iter()
            .zip(of.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1e-6,
            "Sliding window should differ from full attention"
        );
    }

    #[test]
    fn test_flash_fwd_gqa() {
        let (client, device) = cpu_setup();
        let (b, h, nkv, s, d) = (1, 8, 2, 4, 16);
        let q = rand_tensor(&[b, h, s, d], &client, &device);
        let k = rand_tensor(&[b, nkv, s, d], &client, &device);
        let v = rand_tensor(&[b, nkv, s, d], &client, &device);

        let (out, lse) = client
            .flash_attention_fwd(&q, &k, &v, h, nkv, d, false, 0)
            .unwrap();
        assert_eq!(out.shape(), &[b, h, s, d]);
        assert_eq!(lse.shape(), &[b, h, s]);
    }

    #[test]
    fn test_flash_bwd_produces_gradients() {
        let (client, device) = cpu_setup();
        let (b, h, s, d) = (1, 2, 4, 8);
        let q = rand_tensor(&[b, h, s, d], &client, &device);
        let k = rand_tensor(&[b, h, s, d], &client, &device);
        let v = rand_tensor(&[b, h, s, d], &client, &device);

        let (out, lse) = client
            .flash_attention_fwd(&q, &k, &v, h, h, d, false, 0)
            .unwrap();
        let dout = rand_tensor(&[b, h, s, d], &client, &device);

        let (dq, dk, dv) = client
            .flash_attention_bwd(&dout, &q, &k, &v, &out, &lse, h, h, d, false, 0)
            .unwrap();
        assert_eq!(dq.shape(), &[b, h, s, d]);
        assert_eq!(dk.shape(), &[b, h, s, d]);
        assert_eq!(dv.shape(), &[b, h, s, d]);

        // Gradients should be non-zero
        let dq_abs = client.abs(&dq).unwrap();
        let dq_sum = client.sum(&dq_abs, &[], false).unwrap();
        assert!(dq_sum.to_vec::<f32>()[0] > 1e-6);
    }

    #[test]
    fn test_flash_bwd_causal_gradients() {
        let (client, device) = cpu_setup();
        let (b, h, s, d) = (1, 2, 4, 8);
        let q = rand_tensor(&[b, h, s, d], &client, &device);
        let k = rand_tensor(&[b, h, s, d], &client, &device);
        let v = rand_tensor(&[b, h, s, d], &client, &device);

        let (out, lse) = client
            .flash_attention_fwd(&q, &k, &v, h, h, d, true, 0)
            .unwrap();
        let dout = rand_tensor(&[b, h, s, d], &client, &device);

        let (dq, dk, dv) = client
            .flash_attention_bwd(&dout, &q, &k, &v, &out, &lse, h, h, d, true, 0)
            .unwrap();
        assert_eq!(dq.shape(), &[b, h, s, d]);
        assert_eq!(dk.shape(), &[b, h, s, d]);
        assert_eq!(dv.shape(), &[b, h, s, d]);
    }

    #[test]
    fn test_flash_bwd_gqa_gradient_shapes() {
        let (client, device) = cpu_setup();
        let (b, h, nkv, s, d) = (1, 8, 2, 4, 16);
        let q = rand_tensor(&[b, h, s, d], &client, &device);
        let k = rand_tensor(&[b, nkv, s, d], &client, &device);
        let v = rand_tensor(&[b, nkv, s, d], &client, &device);

        let (out, lse) = client
            .flash_attention_fwd(&q, &k, &v, h, nkv, d, false, 0)
            .unwrap();
        let dout = rand_tensor(&[b, h, s, d], &client, &device);

        let (dq, dk, dv) = client
            .flash_attention_bwd(&dout, &q, &k, &v, &out, &lse, h, nkv, d, false, 0)
            .unwrap();
        assert_eq!(dq.shape(), &[b, h, s, d]);
        assert_eq!(dk.shape(), &[b, nkv, s, d]);
        assert_eq!(dv.shape(), &[b, nkv, s, d]);
    }

    #[test]
    fn test_var_flash_attention_autograd() {
        use crate::ops::autograd_attention::var_flash_attention;
        use numr::autograd::Var;

        let (_client, device) = cpu_setup();
        let (b, h, s, d) = (1, 2, 4, 8);

        let q_t = rand_tensor(&[b, h, s, d], &_client, &device);
        let k_t = rand_tensor(&[b, h, s, d], &_client, &device);
        let v_t = rand_tensor(&[b, h, s, d], &_client, &device);

        let q = Var::new(q_t, true);
        let k = Var::new(k_t, true);
        let v = Var::new(v_t, true);

        let out = var_flash_attention::<CpuRuntime>(&q, &k, &v, h, h, d, false, 0).unwrap();
        assert_eq!(out.tensor().shape(), &[b, h, s, d]);
        assert!(
            out.grad_fn().is_some(),
            "Output should have grad_fn when inputs require grad"
        );
    }
}
