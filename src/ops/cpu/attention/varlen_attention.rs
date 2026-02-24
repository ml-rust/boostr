//! CPU fallback for variable-length (ragged) attention
//!
//! Unpacks sequences from the packed buffer, runs standard attention per-sequence,
//! and writes results back to the packed output buffer.

use crate::error::Result;
use crate::ops::traits::VarLenAttentionOps;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl VarLenAttentionOps<CpuRuntime> for CpuClient {
    fn varlen_attention_fwd(
        &self,
        q: &Tensor<CpuRuntime>,
        k: &Tensor<CpuRuntime>,
        v: &Tensor<CpuRuntime>,
        cu_seqlens_q: &Tensor<CpuRuntime>,
        cu_seqlens_k: &Tensor<CpuRuntime>,
        batch_size: usize,
        num_heads: usize,
        _max_seqlen_q: usize,
        _max_seqlen_k: usize,
        head_dim: usize,
        causal: bool,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        let total_tokens_q = q.shape()[0];
        let device = q.device();

        let q_data = q.to_vec::<f32>();
        let k_data = k.to_vec::<f32>();
        let v_data = v.to_vec::<f32>();
        let cu_q = cu_seqlens_q.to_vec::<i32>();
        let cu_k = cu_seqlens_k.to_vec::<i32>();

        let mut out = vec![0.0f32; total_tokens_q * num_heads * head_dim];
        let mut lse = vec![0.0f32; total_tokens_q * num_heads];

        let scale = (head_dim as f32).sqrt().recip();

        for b in 0..batch_size {
            let sq_start = cu_q[b] as usize;
            let sq_end = cu_q[b + 1] as usize;
            let sk_start = cu_k[b] as usize;
            let sk_end = cu_k[b + 1] as usize;
            let seq_len_q = sq_end - sq_start;
            let seq_len_k = sk_end - sk_start;

            for h in 0..num_heads {
                for qi in 0..seq_len_q {
                    let q_offset = ((sq_start + qi) * num_heads + h) * head_dim;

                    let mut max_score = f32::NEG_INFINITY;
                    let mut scores = Vec::with_capacity(seq_len_k);

                    for ki in 0..seq_len_k {
                        if causal && qi < ki {
                            scores.push(f32::NEG_INFINITY);
                            continue;
                        }
                        let k_offset = ((sk_start + ki) * num_heads + h) * head_dim;
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q_data[q_offset + d] * k_data[k_offset + d];
                        }
                        let s = dot * scale;
                        max_score = max_score.max(s);
                        scores.push(s);
                    }

                    let mut sum_exp = 0.0f32;
                    let mut exp_scores = Vec::with_capacity(seq_len_k);
                    for &s in &scores {
                        let e = (s - max_score).exp();
                        sum_exp += e;
                        exp_scores.push(e);
                    }

                    let o_offset = ((sq_start + qi) * num_heads + h) * head_dim;
                    for (ki, &exp_s) in exp_scores.iter().enumerate() {
                        let weight = exp_s / sum_exp;
                        let v_offset = ((sk_start + ki) * num_heads + h) * head_dim;
                        for d in 0..head_dim {
                            out[o_offset + d] += weight * v_data[v_offset + d];
                        }
                    }

                    lse[(sq_start + qi) * num_heads + h] = max_score + sum_exp.ln();
                }
            }
        }

        let output =
            Tensor::<CpuRuntime>::from_slice(&out, &[total_tokens_q, num_heads, head_dim], device);
        let lse_tensor =
            Tensor::<CpuRuntime>::from_slice(&lse, &[total_tokens_q, num_heads], device);
        Ok((output, lse_tensor))
    }

    fn varlen_attention_bwd(
        &self,
        dout: &Tensor<CpuRuntime>,
        q: &Tensor<CpuRuntime>,
        k: &Tensor<CpuRuntime>,
        v: &Tensor<CpuRuntime>,
        output: &Tensor<CpuRuntime>,
        lse: &Tensor<CpuRuntime>,
        cu_seqlens_q: &Tensor<CpuRuntime>,
        cu_seqlens_k: &Tensor<CpuRuntime>,
        batch_size: usize,
        num_heads: usize,
        _max_seqlen_q: usize,
        _max_seqlen_k: usize,
        head_dim: usize,
        causal: bool,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        let total_tokens_q = q.shape()[0];
        let total_tokens_k = k.shape()[0];
        let device = q.device();

        let q_data = q.to_vec::<f32>();
        let k_data = k.to_vec::<f32>();
        let v_data = v.to_vec::<f32>();
        let o_data = output.to_vec::<f32>();
        let lse_data = lse.to_vec::<f32>();
        let do_data = dout.to_vec::<f32>();
        let cu_q = cu_seqlens_q.to_vec::<i32>();
        let cu_k = cu_seqlens_k.to_vec::<i32>();

        let mut dq = vec![0.0f32; total_tokens_q * num_heads * head_dim];
        let mut dk = vec![0.0f32; total_tokens_k * num_heads * head_dim];
        let mut dv = vec![0.0f32; total_tokens_k * num_heads * head_dim];

        let scale = (head_dim as f32).sqrt().recip();

        for b in 0..batch_size {
            let sq_start = cu_q[b] as usize;
            let sq_end = cu_q[b + 1] as usize;
            let sk_start = cu_k[b] as usize;
            let sk_end = cu_k[b + 1] as usize;
            let seq_len_q = sq_end - sq_start;
            let seq_len_k = sk_end - sk_start;

            for h in 0..num_heads {
                for qi in 0..seq_len_q {
                    let q_off = ((sq_start + qi) * num_heads + h) * head_dim;
                    let l = lse_data[(sq_start + qi) * num_heads + h];

                    // D = sum(dO * O)
                    let o_off = ((sq_start + qi) * num_heads + h) * head_dim;
                    let mut d_val = 0.0f32;
                    for d in 0..head_dim {
                        d_val += do_data[o_off + d] * o_data[o_off + d];
                    }

                    for ki in 0..seq_len_k {
                        if causal && qi < ki {
                            continue;
                        }
                        let k_off = ((sk_start + ki) * num_heads + h) * head_dim;

                        // Recompute score and prob
                        let mut score = 0.0f32;
                        for d in 0..head_dim {
                            score += q_data[q_off + d] * k_data[k_off + d];
                        }
                        score *= scale;
                        let prob = (score - l).exp();

                        // grad_prob = V @ dO
                        let v_off = ((sk_start + ki) * num_heads + h) * head_dim;
                        let mut grad_prob = 0.0f32;
                        for d in 0..head_dim {
                            grad_prob += v_data[v_off + d] * do_data[o_off + d];
                        }

                        let grad_score = prob * (grad_prob - d_val);

                        // Accumulate gradients
                        for d in 0..head_dim {
                            dq[q_off + d] += scale * grad_score * k_data[k_off + d];
                            dk[k_off + d] += scale * grad_score * q_data[q_off + d];
                            dv[v_off + d] += prob * do_data[o_off + d];
                        }
                    }
                }
            }
        }

        let dq_t =
            Tensor::<CpuRuntime>::from_slice(&dq, &[total_tokens_q, num_heads, head_dim], device);
        let dk_t =
            Tensor::<CpuRuntime>::from_slice(&dk, &[total_tokens_k, num_heads, head_dim], device);
        let dv_t =
            Tensor::<CpuRuntime>::from_slice(&dv, &[total_tokens_k, num_heads, head_dim], device);
        Ok((dq_t, dk_t, dv_t))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;

    #[test]
    fn test_varlen_fwd_shape() {
        let (client, dev) = cpu_setup();

        // 2 sequences: [3, 2] tokens, 2 heads, head_dim=4
        let total_q = 5;
        let num_heads = 2;
        let head_dim = 4;

        let q_data = vec![0.1f32; total_q * num_heads * head_dim];
        let k_data = vec![0.1f32; total_q * num_heads * head_dim];
        let v_data = vec![0.2f32; total_q * num_heads * head_dim];

        let q = Tensor::<CpuRuntime>::from_slice(&q_data, &[total_q, num_heads, head_dim], &dev);
        let k = Tensor::<CpuRuntime>::from_slice(&k_data, &[total_q, num_heads, head_dim], &dev);
        let v = Tensor::<CpuRuntime>::from_slice(&v_data, &[total_q, num_heads, head_dim], &dev);

        let cu_seqlens = vec![0i32, 3, 5];
        let cu = Tensor::<CpuRuntime>::from_slice(&cu_seqlens, &[3], &dev);

        let (out, lse) = client
            .varlen_attention_fwd(&q, &k, &v, &cu, &cu, 2, num_heads, 3, 3, head_dim, false)
            .unwrap();

        assert_eq!(out.shape(), &[total_q, num_heads, head_dim]);
        assert_eq!(lse.shape(), &[total_q, num_heads]);
    }

    #[test]
    fn test_varlen_fwd_causal() {
        let (client, dev) = cpu_setup();

        // Single sequence of 4 tokens, 1 head, head_dim=2
        let total_q = 4;
        let num_heads = 1;
        let head_dim = 2;

        let q_data: Vec<f32> = (0..total_q * num_heads * head_dim)
            .map(|i| (i as f32) * 0.1 + 0.1)
            .collect();
        let k_data = q_data.clone();
        let v_data: Vec<f32> = (0..total_q * num_heads * head_dim)
            .map(|i| (i as f32) * 0.05)
            .collect();

        let q = Tensor::<CpuRuntime>::from_slice(&q_data, &[total_q, num_heads, head_dim], &dev);
        let k = Tensor::<CpuRuntime>::from_slice(&k_data, &[total_q, num_heads, head_dim], &dev);
        let v = Tensor::<CpuRuntime>::from_slice(&v_data, &[total_q, num_heads, head_dim], &dev);

        let cu_seqlens = vec![0i32, 4];
        let cu = Tensor::<CpuRuntime>::from_slice(&cu_seqlens, &[2], &dev);

        let (out_causal, _) = client
            .varlen_attention_fwd(&q, &k, &v, &cu, &cu, 1, num_heads, 4, 4, head_dim, true)
            .unwrap();
        let (out_full, _) = client
            .varlen_attention_fwd(&q, &k, &v, &cu, &cu, 1, num_heads, 4, 4, head_dim, false)
            .unwrap();

        let causal_data = out_causal.to_vec::<f32>();
        let full_data = out_full.to_vec::<f32>();

        // Last token: causal sees all 4, non-causal sees all 4 → same
        let last_off = (total_q - 1) * num_heads * head_dim;
        for d in 0..head_dim {
            assert!(
                (causal_data[last_off + d] - full_data[last_off + d]).abs() < 1e-5,
                "Last token should match between causal and non-causal"
            );
        }

        // Second token (idx=1): causal sees [0,1], non-causal sees [0,1,2,3] → different
        let second_off = num_heads * head_dim;
        let differs = (0..head_dim)
            .any(|d| (causal_data[second_off + d] - full_data[second_off + d]).abs() > 1e-6);
        assert!(
            differs,
            "Middle tokens should differ between causal and non-causal"
        );
    }

    #[test]
    fn test_varlen_bwd_shapes() {
        let (client, dev) = cpu_setup();

        let total_q = 5;
        let num_heads = 2;
        let head_dim = 4;

        let n = total_q * num_heads * head_dim;
        let q_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).sin()).collect();
        let k_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.7).cos()).collect();
        let v_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.5 + 1.0).sin()).collect();

        let q = Tensor::<CpuRuntime>::from_slice(&q_data, &[total_q, num_heads, head_dim], &dev);
        let k = Tensor::<CpuRuntime>::from_slice(&k_data, &[total_q, num_heads, head_dim], &dev);
        let v = Tensor::<CpuRuntime>::from_slice(&v_data, &[total_q, num_heads, head_dim], &dev);

        let cu_seqlens = vec![0i32, 3, 5];
        let cu = Tensor::<CpuRuntime>::from_slice(&cu_seqlens, &[3], &dev);

        let (out, lse) = client
            .varlen_attention_fwd(&q, &k, &v, &cu, &cu, 2, num_heads, 3, 3, head_dim, false)
            .unwrap();

        let do_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.2).cos() * 0.1).collect();
        let dout =
            Tensor::<CpuRuntime>::from_slice(&do_data, &[total_q, num_heads, head_dim], &dev);

        let (dq, dk, dv) = client
            .varlen_attention_bwd(
                &dout, &q, &k, &v, &out, &lse, &cu, &cu, 2, num_heads, 3, 3, head_dim, false,
            )
            .unwrap();

        assert_eq!(dq.shape(), &[total_q, num_heads, head_dim]);
        assert_eq!(dk.shape(), &[total_q, num_heads, head_dim]);
        assert_eq!(dv.shape(), &[total_q, num_heads, head_dim]);

        // Gradients should be non-zero
        let dq_data = dq.to_vec::<f32>();
        let has_nonzero = dq_data.iter().any(|&x: &f32| x.abs() > 1e-10);
        assert!(has_nonzero, "dQ should have non-zero gradients");
    }
}
