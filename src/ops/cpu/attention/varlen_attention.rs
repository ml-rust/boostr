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
        num_kv_heads: usize,
        _max_seqlen_q: usize,
        _max_seqlen_k: usize,
        head_dim: usize,
        causal: bool,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        // GQA ratio: each kv head serves (num_heads / num_kv_heads) query heads.
        // For MHA num_kv_heads == num_heads so the ratio is 1 and the mapping is
        // the identity: kv_h = q_h / 1 = q_h.
        let gqa_ratio = num_heads / num_kv_heads;

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
                // GQA: map query head h to the corresponding kv head.
                let kv_h = h / gqa_ratio;

                for qi in 0..seq_len_q {
                    let q_offset = ((sq_start + qi) * num_heads + h) * head_dim;

                    let mut max_score = f32::NEG_INFINITY;
                    let mut scores = Vec::with_capacity(seq_len_k);

                    for ki in 0..seq_len_k {
                        if causal && qi < ki {
                            scores.push(f32::NEG_INFINITY);
                            continue;
                        }
                        // K/V row stride uses num_kv_heads (GQA layout).
                        let k_offset = ((sk_start + ki) * num_kv_heads + kv_h) * head_dim;
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
                        // V row stride also uses num_kv_heads.
                        let v_offset = ((sk_start + ki) * num_kv_heads + kv_h) * head_dim;
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
        num_kv_heads: usize,
        _max_seqlen_q: usize,
        _max_seqlen_k: usize,
        head_dim: usize,
        causal: bool,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        // GQA backward: multiple Q heads share one KV head.
        // dK and dV accumulate contributions from all Q heads mapping to the same
        // KV head — the serial += across the h-loop is the scalar analog of atomicAdd.
        let gqa_ratio = num_heads / num_kv_heads;

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
        let mut dk = vec![0.0f32; total_tokens_k * num_kv_heads * head_dim];
        let mut dv = vec![0.0f32; total_tokens_k * num_kv_heads * head_dim];

        let scale = (head_dim as f32).sqrt().recip();

        for b in 0..batch_size {
            let sq_start = cu_q[b] as usize;
            let sq_end = cu_q[b + 1] as usize;
            let sk_start = cu_k[b] as usize;
            let sk_end = cu_k[b + 1] as usize;
            let seq_len_q = sq_end - sq_start;
            let seq_len_k = sk_end - sk_start;

            for h in 0..num_heads {
                // GQA: map query head h to the corresponding kv head.
                let kv_h = h / gqa_ratio;

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
                        // K/V row stride uses num_kv_heads (GQA layout)
                        let k_off = ((sk_start + ki) * num_kv_heads + kv_h) * head_dim;

                        // Recompute score and prob
                        let mut score = 0.0f32;
                        for d in 0..head_dim {
                            score += q_data[q_off + d] * k_data[k_off + d];
                        }
                        score *= scale;
                        let prob = (score - l).exp();

                        // grad_prob = V @ dO
                        let v_off = ((sk_start + ki) * num_kv_heads + kv_h) * head_dim;
                        let mut grad_prob = 0.0f32;
                        for d in 0..head_dim {
                            grad_prob += v_data[v_off + d] * do_data[o_off + d];
                        }

                        let grad_score = prob * (grad_prob - d_val);

                        // Accumulate gradients — dk/dv indexed by kv_h (GQA scatter)
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
        let dk_t = Tensor::<CpuRuntime>::from_slice(
            &dk,
            &[total_tokens_k, num_kv_heads, head_dim],
            device,
        );
        let dv_t = Tensor::<CpuRuntime>::from_slice(
            &dv,
            &[total_tokens_k, num_kv_heads, head_dim],
            device,
        );
        Ok((dq_t, dk_t, dv_t))
    }
}

#[cfg(test)]
#[path = "varlen_attention_tests.rs"]
mod tests;
