//! Backward kernel: recomputes probabilities from the saved log-sum-exp and
//! accumulates dQ, dK, dV with GQA scatter into the shared kv heads.

use crate::error::Result;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

#[allow(clippy::too_many_arguments)]
pub(super) fn varlen_attention_bwd_cpu(
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
                let q_off = ((sq_start + qi) * num_heads + h) * head_dim;
                let l = lse_data[(sq_start + qi) * num_heads + h];

                // D = sum(dO * O)
                let o_off = ((sq_start + qi) * num_heads + h) * head_dim;
                let mut d_val = 0.0f32;
                for d in 0..head_dim {
                    d_val += do_data[o_off + d] * o_data[o_off + d];
                }

                for ki in 0..seq_len_k {
                    if causal && key_offset + qi < ki {
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
        Tensor::<CpuRuntime>::from_slice(&dq, &[total_tokens_q, num_heads, head_dim], device)?;
    let dk_t =
        Tensor::<CpuRuntime>::from_slice(&dk, &[total_tokens_k, num_kv_heads, head_dim], device)?;
    let dv_t =
        Tensor::<CpuRuntime>::from_slice(&dv, &[total_tokens_k, num_kv_heads, head_dim], device)?;
    Ok((dq_t, dk_t, dv_t))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::traits::VarLenAttentionOps;
    use crate::test_utils::cpu_setup;

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

        let do_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.2).cos() * 0.1).collect();
        let dout =
            Tensor::<CpuRuntime>::from_slice(&do_data, &[total_q, num_heads, head_dim], &dev)
                .unwrap();

        let (dq, dk, dv) = client
            .varlen_attention_bwd(
                &dout, &q, &k, &v, &out, &lse, &cu, &cu, 2, num_heads, num_heads, 3, 3, head_dim,
                false,
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

    /// GQA backward equivalence:
    ///   - Run varlen_attention_bwd with GQA (num_kv_heads=2, num_heads=8, hd=64).
    ///   - Run varlen_attention_bwd with K/V expanded to 8 heads (each kv head
    ///     repeated 4×) → MHA reference.
    ///   - Assert dq == dq_exp (within 1e-4).
    ///   - Assert dk[:,kv_h,:] == sum over 4 q-heads mapping to kv_h of dk_exp[:,q_h,:]
    ///     (within 1e-4).  Same for dv.
    #[test]
    fn test_varlen_bwd_gqa_equals_expanded_mha() {
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

        // Expand K and V to full num_heads
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

        let do_data: Vec<f32> = (0..n_q).map(|i| ((i as f32) * 0.11).cos() * 0.1).collect();

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
        let dout =
            Tensor::<CpuRuntime>::from_slice(&do_data, &[total_tokens, num_heads, head_dim], &dev)
                .unwrap();
        let cu = Tensor::<CpuRuntime>::from_slice(&cu_seqlens, &[batch_size + 1], &dev).unwrap();

        // --- GQA fwd + bwd ---
        let (out_gqa, lse_gqa) = client
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
        let (dq_gqa, dk_gqa, dv_gqa) = client
            .varlen_attention_bwd(
                &dout,
                &q,
                &k_gqa,
                &v_gqa,
                &out_gqa,
                &lse_gqa,
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

        // --- Expanded MHA fwd + bwd ---
        let (out_exp, lse_exp) = client
            .varlen_attention_fwd(
                &q, &k_exp, &v_exp, &cu, &cu, batch_size, num_heads, num_heads, max_seqlen,
                max_seqlen, head_dim, false,
            )
            .unwrap();
        let (dq_exp, dk_exp, dv_exp) = client
            .varlen_attention_bwd(
                &dout, &q, &k_exp, &v_exp, &out_exp, &lse_exp, &cu, &cu, batch_size, num_heads,
                num_heads, max_seqlen, max_seqlen, head_dim, false,
            )
            .unwrap();

        let dq_g = dq_gqa.to_vec::<f32>();
        let dq_e = dq_exp.to_vec::<f32>();
        let dk_g = dk_gqa.to_vec::<f32>(); // [total_tokens, num_kv_heads, head_dim]
        let dk_e = dk_exp.to_vec::<f32>(); // [total_tokens, num_heads,    head_dim]
        let dv_g = dv_gqa.to_vec::<f32>();
        let dv_e = dv_exp.to_vec::<f32>();

        // 1. dQ must match exactly
        assert_eq!(dq_g.len(), dq_e.len(), "dq length mismatch");
        for (i, (&a, &b)) in dq_g.iter().zip(dq_e.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "dQ mismatch at index {i}: gqa={a}, exp={b}, diff={}",
                (a - b).abs()
            );
        }

        // 2. dk_gqa[:,kv_h,:] == sum over the gqa_ratio Q-heads that map to kv_h
        //    of dk_exp[:,q_h,:].
        for tok in 0..total_tokens {
            for kv_h in 0..num_kv_heads {
                for d in 0..head_dim {
                    let gqa_val = dk_g[(tok * num_kv_heads + kv_h) * head_dim + d];
                    let mut exp_sum = 0.0f32;
                    for rep in 0..gqa_ratio {
                        let q_h = kv_h * gqa_ratio + rep;
                        exp_sum += dk_e[(tok * num_heads + q_h) * head_dim + d];
                    }
                    assert!(
                        (gqa_val - exp_sum).abs() < 1e-4,
                        "dK scatter mismatch tok={tok} kv_h={kv_h} d={d}: gqa={gqa_val}, exp_sum={exp_sum}"
                    );
                }
            }
        }

        // 3. Same check for dV
        for tok in 0..total_tokens {
            for kv_h in 0..num_kv_heads {
                for d in 0..head_dim {
                    let gqa_val = dv_g[(tok * num_kv_heads + kv_h) * head_dim + d];
                    let mut exp_sum = 0.0f32;
                    for rep in 0..gqa_ratio {
                        let q_h = kv_h * gqa_ratio + rep;
                        exp_sum += dv_e[(tok * num_heads + q_h) * head_dim + d];
                    }
                    assert!(
                        (gqa_val - exp_sum).abs() < 1e-4,
                        "dV scatter mismatch tok={tok} kv_h={kv_h} d={d}: gqa={gqa_val}, exp_sum={exp_sum}"
                    );
                }
            }
        }
    }
}
