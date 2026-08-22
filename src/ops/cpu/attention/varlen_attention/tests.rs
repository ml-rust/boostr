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

    let q = Tensor::<CpuRuntime>::from_slice(&q_data, &[total_q, num_heads, head_dim], &dev);
    let k = Tensor::<CpuRuntime>::from_slice(&k_data, &[total_q, num_heads, head_dim], &dev);
    let v = Tensor::<CpuRuntime>::from_slice(&v_data, &[total_q, num_heads, head_dim], &dev);

    let cu_seqlens = vec![0i32, 4];
    let cu = Tensor::<CpuRuntime>::from_slice(&cu_seqlens, &[2], &dev);

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
        .varlen_attention_fwd(
            &q, &k, &v, &cu, &cu, 2, num_heads, num_heads, 3, 3, head_dim, false,
        )
        .unwrap();

    let do_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.2).cos() * 0.1).collect();
    let dout = Tensor::<CpuRuntime>::from_slice(&do_data, &[total_q, num_heads, head_dim], &dev);

    let (dq, dk, dv) = client
        .varlen_attention_bwd(
            &dout, &q, &k, &v, &out, &lse, &cu, &cu, 2, num_heads, num_heads, 3, 3, head_dim, false,
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

    let q = Tensor::<CpuRuntime>::from_slice(&q_data, &[total_tokens, num_heads, head_dim], &dev);
    let k_gqa =
        Tensor::<CpuRuntime>::from_slice(&k_data, &[total_tokens, num_kv_heads, head_dim], &dev);
    let v_gqa =
        Tensor::<CpuRuntime>::from_slice(&v_data, &[total_tokens, num_kv_heads, head_dim], &dev);
    let k_exp =
        Tensor::<CpuRuntime>::from_slice(&k_expanded, &[total_tokens, num_heads, head_dim], &dev);
    let v_exp =
        Tensor::<CpuRuntime>::from_slice(&v_expanded, &[total_tokens, num_heads, head_dim], &dev);
    let dout =
        Tensor::<CpuRuntime>::from_slice(&do_data, &[total_tokens, num_heads, head_dim], &dev);
    let cu = Tensor::<CpuRuntime>::from_slice(&cu_seqlens, &[batch_size + 1], &dev);

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
            &q, &k_exp, &v_exp, &cu, &cu, batch_size, num_heads, num_heads, max_seqlen, max_seqlen,
            head_dim, false,
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

    let q = Tensor::<CpuRuntime>::from_slice(&q_data, &[total_tokens, num_heads, head_dim], &dev);
    let k_gqa =
        Tensor::<CpuRuntime>::from_slice(&k_data, &[total_tokens, num_kv_heads, head_dim], &dev);
    let v_gqa =
        Tensor::<CpuRuntime>::from_slice(&v_data, &[total_tokens, num_kv_heads, head_dim], &dev);
    let k_exp =
        Tensor::<CpuRuntime>::from_slice(&k_expanded, &[total_tokens, num_heads, head_dim], &dev);
    let v_exp =
        Tensor::<CpuRuntime>::from_slice(&v_expanded, &[total_tokens, num_heads, head_dim], &dev);
    let cu = Tensor::<CpuRuntime>::from_slice(&cu_seqlens, &[batch_size + 1], &dev);

    // Reference: MHA with expanded K/V (num_kv_heads == num_heads)
    let (out_ref, _) = client
        .varlen_attention_fwd(
            &q, &k_exp, &v_exp, &cu, &cu, batch_size, num_heads, num_heads, max_seqlen, max_seqlen,
            head_dim, false,
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
