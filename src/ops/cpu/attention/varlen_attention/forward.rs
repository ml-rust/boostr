//! Forward kernel: per-sequence softmax attention over the packed buffer,
//! returning the output and the per-row log-sum-exp.

use crate::error::Result;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

#[allow(clippy::too_many_arguments)]
pub(super) fn varlen_attention_fwd_cpu(
    q: &Tensor<CpuRuntime>,
    k: &Tensor<CpuRuntime>,
    v: &Tensor<CpuRuntime>,
    cu_seqlens_q: &Tensor<CpuRuntime>,
    cu_seqlens_k: &Tensor<CpuRuntime>,
    batch_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
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
        // Bottom-right (ABSOLUTE) causal alignment, per sequence: this
        // sequence's `seq_len_q` query rows are the LAST positions of its
        // `seq_len_k` keys, so row `qi` sits at absolute position
        // `key_offset + qi`. A full prefill (`seq_len_q == seq_len_k`) gives
        // `key_offset == 0`, leaving the rule unchanged. Same convention as
        // `ops/impl_generic/attention/flash_standard.rs::build_attention_mask`.
        let key_offset = seq_len_k.saturating_sub(seq_len_q);

        for h in 0..num_heads {
            // GQA: map query head h to the corresponding kv head.
            let kv_h = h / gqa_ratio;

            for qi in 0..seq_len_q {
                let q_offset = ((sq_start + qi) * num_heads + h) * head_dim;

                let mut max_score = f32::NEG_INFINITY;
                let mut scores = Vec::with_capacity(seq_len_k);

                for ki in 0..seq_len_k {
                    if causal && key_offset + qi < ki {
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
        Tensor::<CpuRuntime>::from_slice(&out, &[total_tokens_q, num_heads, head_dim], device)?;
    let lse_tensor = Tensor::<CpuRuntime>::from_slice(&lse, &[total_tokens_q, num_heads], device)?;
    Ok((output, lse_tensor))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::traits::VarLenAttentionOps;
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

        let q = Tensor::<CpuRuntime>::from_slice(&q_data, &[total_q, num_heads, head_dim], &dev)
            .unwrap();
        let k = Tensor::<CpuRuntime>::from_slice(&k_data, &[total_q, num_heads, head_dim], &dev)
            .unwrap();
        let v = Tensor::<CpuRuntime>::from_slice(&v_data, &[total_q, num_heads, head_dim], &dev)
            .unwrap();

        let cu_seqlens = vec![0i32, 3, 5];
        let cu = Tensor::<CpuRuntime>::from_slice(&cu_seqlens, &[3], &dev).unwrap();

        let (out, lse) = client
            .varlen_attention_fwd(
                &q, &k, &v, &cu, &cu, 2, num_heads, num_heads, 3, 3, head_dim, false,
            )
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

        let q = Tensor::<CpuRuntime>::from_slice(&q_data, &[total_q, num_heads, head_dim], &dev)
            .unwrap();
        let k = Tensor::<CpuRuntime>::from_slice(&k_data, &[total_q, num_heads, head_dim], &dev)
            .unwrap();
        let v = Tensor::<CpuRuntime>::from_slice(&v_data, &[total_q, num_heads, head_dim], &dev)
            .unwrap();

        let cu_seqlens = vec![0i32, 4];
        let cu = Tensor::<CpuRuntime>::from_slice(&cu_seqlens, &[2], &dev).unwrap();

        let (out_causal, _) = client
            .varlen_attention_fwd(
                &q, &k, &v, &cu, &cu, 1, num_heads, num_heads, 4, 4, head_dim, true,
            )
            .unwrap();
        let (out_full, _) = client
            .varlen_attention_fwd(
                &q, &k, &v, &cu, &cu, 1, num_heads, num_heads, 4, 4, head_dim, false,
            )
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

    /// GQA equivalence: varlen fwd with GQA (num_kv_heads=2, num_heads=8) must
    /// produce the same output as MHA with K/V expanded by repeating each kv head
    /// (num_heads / num_kv_heads) = 4 times along the head axis.
    #[test]
    fn test_varlen_gqa_equals_expanded_mha() {
        let (client, dev) = cpu_setup();

        let num_heads = 8usize;
        let num_kv_heads = 2usize;
        let gqa_ratio = num_heads / num_kv_heads; // 4
        let head_dim = 64usize;

        // Two sequences: lengths 3 and 5
        let total_tokens = 8usize;
        let batch_size = 2usize;

        let n_q = total_tokens * num_heads * head_dim;
        let n_kv = total_tokens * num_kv_heads * head_dim;

        // Deterministic inputs
        let q_data: Vec<f32> = (0..n_q).map(|i| ((i as f32) * 0.13).sin() * 0.3).collect();
        let k_data: Vec<f32> = (0..n_kv).map(|i| ((i as f32) * 0.07).cos() * 0.2).collect();
        let v_data: Vec<f32> = (0..n_kv)
            .map(|i| ((i as f32) * 0.17).sin() * 0.25)
            .collect();

        // Expand K and V: [total_tokens, num_kv_heads, head_dim] →
        //                 [total_tokens, num_heads, head_dim]
        // Each kv head is repeated gqa_ratio times consecutively.
        let mut k_expanded = vec![0.0f32; total_tokens * num_heads * head_dim];
        let mut v_expanded = vec![0.0f32; total_tokens * num_heads * head_dim];
        for tok in 0..total_tokens {
            for kv_h in 0..num_kv_heads {
                for rep in 0..gqa_ratio {
                    let q_h = kv_h * gqa_ratio + rep;
                    let src_base = (tok * num_kv_heads + kv_h) * head_dim;
                    let dst_base = (tok * num_heads + q_h) * head_dim;
                    k_expanded[dst_base..dst_base + head_dim]
                        .copy_from_slice(&k_data[src_base..src_base + head_dim]);
                    v_expanded[dst_base..dst_base + head_dim]
                        .copy_from_slice(&v_data[src_base..src_base + head_dim]);
                }
            }
        }

        let cu_seqlens = vec![0i32, 3, 8];
        let max_seqlen = 5usize;

        let q =
            Tensor::<CpuRuntime>::from_slice(&q_data, &[total_tokens, num_heads, head_dim], &dev)
                .unwrap();
        let k_gqa = Tensor::<CpuRuntime>::from_slice(
            &k_data,
            &[total_tokens, num_kv_heads, head_dim],
            &dev,
        )
        .unwrap();
        let v_gqa = Tensor::<CpuRuntime>::from_slice(
            &v_data,
            &[total_tokens, num_kv_heads, head_dim],
            &dev,
        )
        .unwrap();
        let k_exp = Tensor::<CpuRuntime>::from_slice(
            &k_expanded,
            &[total_tokens, num_heads, head_dim],
            &dev,
        )
        .unwrap();
        let v_exp = Tensor::<CpuRuntime>::from_slice(
            &v_expanded,
            &[total_tokens, num_heads, head_dim],
            &dev,
        )
        .unwrap();
        let cu = Tensor::<CpuRuntime>::from_slice(&cu_seqlens, &[batch_size + 1], &dev).unwrap();

        // Reference: MHA with expanded K/V (num_kv_heads == num_heads)
        let (out_ref, _) = client
            .varlen_attention_fwd(
                &q, &k_exp, &v_exp, &cu, &cu, batch_size, num_heads, num_heads, max_seqlen,
                max_seqlen, head_dim, false,
            )
            .unwrap();

        // Under test: GQA with packed K/V (num_kv_heads < num_heads)
        let (out_gqa, _) = client
            .varlen_attention_fwd(
                &q,
                &k_gqa,
                &v_gqa,
                &cu,
                &cu,
                batch_size,
                num_heads,
                num_kv_heads,
                max_seqlen,
                max_seqlen,
                head_dim,
                false,
            )
            .unwrap();

        let ref_vec = out_ref.to_vec::<f32>();
        let gqa_vec = out_gqa.to_vec::<f32>();

        assert_eq!(ref_vec.len(), gqa_vec.len(), "output length mismatch");
        for (i, (&r, &g)) in ref_vec.iter().zip(gqa_vec.iter()).enumerate() {
            assert!(
                (r - g).abs() < 1e-4,
                "GQA vs expanded-MHA mismatch at index {i}: ref={r}, gqa={g}, diff={}",
                (r - g).abs()
            );
        }
    }
}
