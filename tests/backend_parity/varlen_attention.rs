//! Backend parity tests for VarLenAttentionOps.

use super::helpers::*;
use boostr::ops::traits::attention::varlen_attention::VarLenAttentionOps;

#[test]
fn test_varlen_attention_fwd_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let batch_size = 2;
    let num_heads = 2;
    let head_dim = 64;
    // Two sequences: lengths 4 and 6
    let total_tokens = 10;
    let max_seqlen = 6;

    let q = det_tensor(&[total_tokens, num_heads, head_dim], &cpu_device);
    let k = det_tensor(&[total_tokens, num_heads, head_dim], &cpu_device);
    let v = det_tensor(&[total_tokens, num_heads, head_dim], &cpu_device);
    let cu_data: Vec<i32> = vec![0, 4, 10];
    let cu_seqlens_q = det_i32_tensor(&cu_data, &[batch_size + 1], &cpu_device);
    let cu_seqlens_k = det_i32_tensor(&cu_data, &[batch_size + 1], &cpu_device);

    let (cpu_out, _) = cpu_client
        .varlen_attention_fwd(
            &q,
            &k,
            &v,
            &cu_seqlens_q,
            &cu_seqlens_k,
            batch_size,
            num_heads,
            num_heads, // MHA: num_kv_heads == num_heads
            max_seqlen,
            max_seqlen,
            head_dim,
            false,
        )
        .unwrap();
    let cpu_out_vec = cpu_out.to_vec::<f32>();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::attention::varlen_attention::VarLenAttentionOps as _;
        use numr::tensor::Tensor;
        let q_c = Tensor::try_from_slice(
            &q.to_vec::<f32>(),
            &[total_tokens, num_heads, head_dim],
            &cuda_device,
        )
        .unwrap();
        let k_c = Tensor::try_from_slice(
            &k.to_vec::<f32>(),
            &[total_tokens, num_heads, head_dim],
            &cuda_device,
        )
        .unwrap();
        let v_c = Tensor::try_from_slice(
            &v.to_vec::<f32>(),
            &[total_tokens, num_heads, head_dim],
            &cuda_device,
        )
        .unwrap();
        let csq = Tensor::try_from_slice(&cu_data, &[batch_size + 1], &cuda_device).unwrap();
        let csk = Tensor::try_from_slice(&cu_data, &[batch_size + 1], &cuda_device).unwrap();
        let (out, _) = cuda_client
            .varlen_attention_fwd(
                &q_c, &k_c, &v_c, &csq, &csk, batch_size, num_heads, num_heads, max_seqlen,
                max_seqlen, head_dim, false,
            )
            .unwrap();
        assert_parity_f32(&out.to_vec::<f32>(), &cpu_out_vec, "varlen_fwd CUDA vs CPU");
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::attention::varlen_attention::VarLenAttentionOps as _;
        use numr::tensor::Tensor;
        let q_w = Tensor::try_from_slice(
            &q.to_vec::<f32>(),
            &[total_tokens, num_heads, head_dim],
            &wgpu_device,
        )
        .unwrap();
        let k_w = Tensor::try_from_slice(
            &k.to_vec::<f32>(),
            &[total_tokens, num_heads, head_dim],
            &wgpu_device,
        )
        .unwrap();
        let v_w = Tensor::try_from_slice(
            &v.to_vec::<f32>(),
            &[total_tokens, num_heads, head_dim],
            &wgpu_device,
        )
        .unwrap();
        let csq = Tensor::try_from_slice(&cu_data, &[batch_size + 1], &wgpu_device).unwrap();
        let csk = Tensor::try_from_slice(&cu_data, &[batch_size + 1], &wgpu_device).unwrap();
        let (out, _) = wgpu_client
            .varlen_attention_fwd(
                &q_w, &k_w, &v_w, &csq, &csk, batch_size, num_heads, num_heads, max_seqlen,
                max_seqlen, head_dim, false,
            )
            .unwrap();
        assert_parity_f32(&out.to_vec::<f32>(), &cpu_out_vec, "varlen_fwd WGPU vs CPU");
    });
}

#[test]
fn test_varlen_attention_bwd_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let batch_size = 1;
    let num_heads = 2;
    let head_dim = 64;
    let total_tokens = 6;
    let max_seqlen = 6;

    let q = det_tensor(&[total_tokens, num_heads, head_dim], &cpu_device);
    let k = det_tensor(&[total_tokens, num_heads, head_dim], &cpu_device);
    let v = det_tensor(&[total_tokens, num_heads, head_dim], &cpu_device);
    let cu_data: Vec<i32> = vec![0, 6];
    let cu_seqlens_q = det_i32_tensor(&cu_data, &[batch_size + 1], &cpu_device);
    let cu_seqlens_k = det_i32_tensor(&cu_data, &[batch_size + 1], &cpu_device);

    let (out, lse) = cpu_client
        .varlen_attention_fwd(
            &q,
            &k,
            &v,
            &cu_seqlens_q,
            &cu_seqlens_k,
            batch_size,
            num_heads,
            num_heads, // MHA: num_kv_heads == num_heads
            max_seqlen,
            max_seqlen,
            head_dim,
            false,
        )
        .unwrap();
    let dout = det_tensor(&[total_tokens, num_heads, head_dim], &cpu_device);
    let (cpu_dq, cpu_dk, cpu_dv) = cpu_client
        .varlen_attention_bwd(
            &dout,
            &q,
            &k,
            &v,
            &out,
            &lse,
            &cu_seqlens_q,
            &cu_seqlens_k,
            batch_size,
            num_heads,
            num_heads, // MHA: num_kv_heads == num_heads
            max_seqlen,
            max_seqlen,
            head_dim,
            false,
        )
        .unwrap();
    let _cpu_dq_vec = cpu_dq.to_vec::<f32>();
    let _cpu_dk_vec = cpu_dk.to_vec::<f32>();
    let _cpu_dv_vec = cpu_dv.to_vec::<f32>();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::attention::varlen_attention::VarLenAttentionOps as _;
        use numr::tensor::Tensor;
        let q_c = Tensor::try_from_slice(
            &q.to_vec::<f32>(),
            &[total_tokens, num_heads, head_dim],
            &cuda_device,
        )
        .unwrap();
        let k_c = Tensor::try_from_slice(
            &k.to_vec::<f32>(),
            &[total_tokens, num_heads, head_dim],
            &cuda_device,
        )
        .unwrap();
        let v_c = Tensor::try_from_slice(
            &v.to_vec::<f32>(),
            &[total_tokens, num_heads, head_dim],
            &cuda_device,
        )
        .unwrap();
        let csq = Tensor::try_from_slice(&cu_data, &[batch_size + 1], &cuda_device).unwrap();
        let csk = Tensor::try_from_slice(&cu_data, &[batch_size + 1], &cuda_device).unwrap();
        let (out_c, lse_c) = cuda_client
            .varlen_attention_fwd(
                &q_c, &k_c, &v_c, &csq, &csk, batch_size, num_heads, num_heads, max_seqlen,
                max_seqlen, head_dim, false,
            )
            .unwrap();
        let dout_c = Tensor::try_from_slice(
            &dout.to_vec::<f32>(),
            &[total_tokens, num_heads, head_dim],
            &cuda_device,
        )
        .unwrap();
        let (dq, dk, dv) = cuda_client
            .varlen_attention_bwd(
                &dout_c, &q_c, &k_c, &v_c, &out_c, &lse_c, &csq, &csk, batch_size, num_heads,
                num_heads, max_seqlen, max_seqlen, head_dim, false,
            )
            .unwrap();
        assert_parity_f32_relaxed(
            &dq.to_vec::<f32>(),
            &_cpu_dq_vec,
            "varlen_bwd dQ CUDA vs CPU",
        );
        assert_parity_f32_relaxed(
            &dk.to_vec::<f32>(),
            &_cpu_dk_vec,
            "varlen_bwd dK CUDA vs CPU",
        );
        assert_parity_f32_relaxed(
            &dv.to_vec::<f32>(),
            &_cpu_dv_vec,
            "varlen_bwd dV CUDA vs CPU",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::attention::varlen_attention::VarLenAttentionOps as _;
        use numr::tensor::Tensor;
        let q_w = Tensor::try_from_slice(
            &q.to_vec::<f32>(),
            &[total_tokens, num_heads, head_dim],
            &wgpu_device,
        )
        .unwrap();
        let k_w = Tensor::try_from_slice(
            &k.to_vec::<f32>(),
            &[total_tokens, num_heads, head_dim],
            &wgpu_device,
        )
        .unwrap();
        let v_w = Tensor::try_from_slice(
            &v.to_vec::<f32>(),
            &[total_tokens, num_heads, head_dim],
            &wgpu_device,
        )
        .unwrap();
        let csq = Tensor::try_from_slice(&cu_data, &[batch_size + 1], &wgpu_device).unwrap();
        let csk = Tensor::try_from_slice(&cu_data, &[batch_size + 1], &wgpu_device).unwrap();
        let (out_w, lse_w) = wgpu_client
            .varlen_attention_fwd(
                &q_w, &k_w, &v_w, &csq, &csk, batch_size, num_heads, num_heads, max_seqlen,
                max_seqlen, head_dim, false,
            )
            .unwrap();
        let dout_w = Tensor::try_from_slice(
            &dout.to_vec::<f32>(),
            &[total_tokens, num_heads, head_dim],
            &wgpu_device,
        )
        .unwrap();
        let (dq, dk, dv) = wgpu_client
            .varlen_attention_bwd(
                &dout_w, &q_w, &k_w, &v_w, &out_w, &lse_w, &csq, &csk, batch_size, num_heads,
                num_heads, max_seqlen, max_seqlen, head_dim, false,
            )
            .unwrap();
        assert_parity_f32_relaxed(
            &dq.to_vec::<f32>(),
            &_cpu_dq_vec,
            "varlen_bwd dQ WGPU vs CPU",
        );
        assert_parity_f32_relaxed(
            &dk.to_vec::<f32>(),
            &_cpu_dk_vec,
            "varlen_bwd dK WGPU vs CPU",
        );
        assert_parity_f32_relaxed(
            &dv.to_vec::<f32>(),
            &_cpu_dv_vec,
            "varlen_bwd dV WGPU vs CPU",
        );
    });
}

/// Naive packed-attention reference that mirrors the CPU varlen kernel's exact
/// arithmetic order, so its output is bitwise comparable.
///
/// `absolute` selects the masking convention:
/// - `true`  — bottom-right: key `ki` is masked when `key_offset + qi < ki`,
///   with the PER-SEQUENCE `key_offset = seq_len_k - seq_len_q`.
/// - `false` — the legacy top-left rule: key `ki` is masked when `qi < ki`.
#[allow(clippy::too_many_arguments)]
fn varlen_reference_fwd(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    cu_q: &[i32],
    cu_k: &[i32],
    batch_size: usize,
    num_heads: usize,
    head_dim: usize,
    total_tokens_q: usize,
    causal: bool,
    absolute: bool,
) -> Vec<f32> {
    let mut out = vec![0.0f32; total_tokens_q * num_heads * head_dim];
    let scale = (head_dim as f32).sqrt().recip();

    for b in 0..batch_size {
        let sq_start = cu_q[b] as usize;
        let sk_start = cu_k[b] as usize;
        let seq_len_q = cu_q[b + 1] as usize - sq_start;
        let seq_len_k = cu_k[b + 1] as usize - sk_start;
        let key_offset = if absolute {
            seq_len_k.saturating_sub(seq_len_q)
        } else {
            0
        };

        for h in 0..num_heads {
            for qi in 0..seq_len_q {
                let q_offset = ((sq_start + qi) * num_heads + h) * head_dim;

                let mut max_score = f32::NEG_INFINITY;
                let mut scores = Vec::with_capacity(seq_len_k);
                for ki in 0..seq_len_k {
                    if causal && key_offset + qi < ki {
                        scores.push(f32::NEG_INFINITY);
                        continue;
                    }
                    let k_offset = ((sk_start + ki) * num_heads + h) * head_dim;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[q_offset + d] * k[k_offset + d];
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
                        out[o_offset + d] += weight * v[v_offset + d];
                    }
                }
            }
        }
    }
    out
}

/// Causal varlen with `seq_len_q != seq_len_k` per sequence must mask by the
/// ABSOLUTE (bottom-right) position: within sequence `s` the queries are the
/// LAST `seq_len_q` positions of that sequence's `seq_len_k` keys, so query row
/// `qi` sits at `key_offset + qi` with a PER-SEQUENCE
/// `key_offset = seq_len_k - seq_len_q`.
///
/// The two sequences here have DIFFERENT offsets (3 and 1), so a single global
/// offset fails this as surely as the old top-left rule does.
#[test]
fn test_varlen_causal_unequal_seqlens_uses_absolute_positions() {
    let (cpu_client, cpu_device) = setup_cpu();
    let batch_size = 2;
    let num_heads = 2;
    let head_dim = 64;
    // seq 0: 2 queries against 5 keys (key_offset 3)
    // seq 1: 3 queries against 4 keys (key_offset 1)
    let cu_q_data: Vec<i32> = vec![0, 2, 5];
    let cu_k_data: Vec<i32> = vec![0, 5, 9];
    let total_tokens_q = 5;
    let total_tokens_k = 9;
    let max_seqlen_q = 3;
    let max_seqlen_k = 5;

    let q = det_tensor(&[total_tokens_q, num_heads, head_dim], &cpu_device);
    let k = det_tensor(&[total_tokens_k, num_heads, head_dim], &cpu_device);
    let v = det_tensor(&[total_tokens_k, num_heads, head_dim], &cpu_device);
    let cu_seqlens_q = det_i32_tensor(&cu_q_data, &[batch_size + 1], &cpu_device);
    let cu_seqlens_k = det_i32_tensor(&cu_k_data, &[batch_size + 1], &cpu_device);

    let q_vec = q.to_vec::<f32>();
    let k_vec = k.to_vec::<f32>();
    let v_vec = v.to_vec::<f32>();

    let absolute_ref = varlen_reference_fwd(
        &q_vec,
        &k_vec,
        &v_vec,
        &cu_q_data,
        &cu_k_data,
        batch_size,
        num_heads,
        head_dim,
        total_tokens_q,
        true,
        true,
    );
    let legacy_ref = varlen_reference_fwd(
        &q_vec,
        &k_vec,
        &v_vec,
        &cu_q_data,
        &cu_k_data,
        batch_size,
        num_heads,
        head_dim,
        total_tokens_q,
        true,
        false,
    );
    // Guard the guard: the two conventions must actually disagree here, or the
    // assertion below would pass under either one.
    assert!(
        absolute_ref
            .iter()
            .zip(legacy_ref.iter())
            .any(|(a, b)| (a - b).abs() > 1e-4),
        "test setup is degenerate: absolute and top-left masking agree"
    );

    let (cpu_out, _) = cpu_client
        .varlen_attention_fwd(
            &q,
            &k,
            &v,
            &cu_seqlens_q,
            &cu_seqlens_k,
            batch_size,
            num_heads,
            num_heads,
            max_seqlen_q,
            max_seqlen_k,
            head_dim,
            true,
        )
        .unwrap();
    let cpu_out_vec = cpu_out.to_vec::<f32>();
    assert_eq!(
        cpu_out_vec, absolute_ref,
        "CPU varlen causal must mask by absolute per-sequence position"
    );

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::attention::varlen_attention::VarLenAttentionOps as _;
        use numr::tensor::Tensor;
        let q_c =
            Tensor::try_from_slice(&q_vec, &[total_tokens_q, num_heads, head_dim], &cuda_device)
                .unwrap();
        let k_c =
            Tensor::try_from_slice(&k_vec, &[total_tokens_k, num_heads, head_dim], &cuda_device)
                .unwrap();
        let v_c =
            Tensor::try_from_slice(&v_vec, &[total_tokens_k, num_heads, head_dim], &cuda_device)
                .unwrap();
        let csq = Tensor::try_from_slice(&cu_q_data, &[batch_size + 1], &cuda_device).unwrap();
        let csk = Tensor::try_from_slice(&cu_k_data, &[batch_size + 1], &cuda_device).unwrap();
        let (out, _) = cuda_client
            .varlen_attention_fwd(
                &q_c,
                &k_c,
                &v_c,
                &csq,
                &csk,
                batch_size,
                num_heads,
                num_heads,
                max_seqlen_q,
                max_seqlen_k,
                head_dim,
                true,
            )
            .unwrap();
        assert_parity_f32(
            &out.to_vec::<f32>(),
            &cpu_out_vec,
            "varlen causal unequal CUDA vs CPU",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::attention::varlen_attention::VarLenAttentionOps as _;
        use numr::tensor::Tensor;
        let q_w =
            Tensor::try_from_slice(&q_vec, &[total_tokens_q, num_heads, head_dim], &wgpu_device)
                .unwrap();
        let k_w =
            Tensor::try_from_slice(&k_vec, &[total_tokens_k, num_heads, head_dim], &wgpu_device)
                .unwrap();
        let v_w =
            Tensor::try_from_slice(&v_vec, &[total_tokens_k, num_heads, head_dim], &wgpu_device)
                .unwrap();
        let csq = Tensor::try_from_slice(&cu_q_data, &[batch_size + 1], &wgpu_device).unwrap();
        let csk = Tensor::try_from_slice(&cu_k_data, &[batch_size + 1], &wgpu_device).unwrap();
        let (out, _) = wgpu_client
            .varlen_attention_fwd(
                &q_w,
                &k_w,
                &v_w,
                &csq,
                &csk,
                batch_size,
                num_heads,
                num_heads,
                max_seqlen_q,
                max_seqlen_k,
                head_dim,
                true,
            )
            .unwrap();
        assert_parity_f32(
            &out.to_vec::<f32>(),
            &cpu_out_vec,
            "varlen causal unequal WGPU vs CPU",
        );
    });
}

/// Regression guard: with `seq_len_q == seq_len_k` the per-sequence
/// `key_offset` is 0, so the absolute rule reduces to the legacy top-left rule.
/// The reference here uses the LEGACY rule and the comparison is EXACT, so any
/// drift in the equal-length path is caught bitwise.
#[test]
fn test_varlen_causal_equal_seqlens_bit_identical_to_legacy() {
    let (cpu_client, cpu_device) = setup_cpu();
    let batch_size = 2;
    let num_heads = 2;
    let head_dim = 64;
    let cu_data: Vec<i32> = vec![0, 4, 10];
    let total_tokens = 10;
    let max_seqlen = 6;

    let q = det_tensor(&[total_tokens, num_heads, head_dim], &cpu_device);
    let k = det_tensor(&[total_tokens, num_heads, head_dim], &cpu_device);
    let v = det_tensor(&[total_tokens, num_heads, head_dim], &cpu_device);
    let cu_seqlens_q = det_i32_tensor(&cu_data, &[batch_size + 1], &cpu_device);
    let cu_seqlens_k = det_i32_tensor(&cu_data, &[batch_size + 1], &cpu_device);

    let legacy_ref = varlen_reference_fwd(
        &q.to_vec::<f32>(),
        &k.to_vec::<f32>(),
        &v.to_vec::<f32>(),
        &cu_data,
        &cu_data,
        batch_size,
        num_heads,
        head_dim,
        total_tokens,
        true,
        false,
    );

    let (cpu_out, _) = cpu_client
        .varlen_attention_fwd(
            &q,
            &k,
            &v,
            &cu_seqlens_q,
            &cu_seqlens_k,
            batch_size,
            num_heads,
            num_heads,
            max_seqlen,
            max_seqlen,
            head_dim,
            true,
        )
        .unwrap();
    let cpu_out_vec = cpu_out.to_vec::<f32>();
    assert_eq!(
        cpu_out_vec, legacy_ref,
        "equal-length causal varlen changed: must stay bit-identical to top-left"
    );

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::attention::varlen_attention::VarLenAttentionOps as _;
        use numr::tensor::Tensor;
        let shape = [total_tokens, num_heads, head_dim];
        let q_c = Tensor::try_from_slice(&q.to_vec::<f32>(), &shape, &cuda_device).unwrap();
        let k_c = Tensor::try_from_slice(&k.to_vec::<f32>(), &shape, &cuda_device).unwrap();
        let v_c = Tensor::try_from_slice(&v.to_vec::<f32>(), &shape, &cuda_device).unwrap();
        let csq = Tensor::try_from_slice(&cu_data, &[batch_size + 1], &cuda_device).unwrap();
        let csk = Tensor::try_from_slice(&cu_data, &[batch_size + 1], &cuda_device).unwrap();
        let (out, _) = cuda_client
            .varlen_attention_fwd(
                &q_c, &k_c, &v_c, &csq, &csk, batch_size, num_heads, num_heads, max_seqlen,
                max_seqlen, head_dim, true,
            )
            .unwrap();
        assert_parity_f32(
            &out.to_vec::<f32>(),
            &cpu_out_vec,
            "varlen causal equal CUDA vs CPU",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::attention::varlen_attention::VarLenAttentionOps as _;
        use numr::tensor::Tensor;
        let shape = [total_tokens, num_heads, head_dim];
        let q_w = Tensor::try_from_slice(&q.to_vec::<f32>(), &shape, &wgpu_device).unwrap();
        let k_w = Tensor::try_from_slice(&k.to_vec::<f32>(), &shape, &wgpu_device).unwrap();
        let v_w = Tensor::try_from_slice(&v.to_vec::<f32>(), &shape, &wgpu_device).unwrap();
        let csq = Tensor::try_from_slice(&cu_data, &[batch_size + 1], &wgpu_device).unwrap();
        let csk = Tensor::try_from_slice(&cu_data, &[batch_size + 1], &wgpu_device).unwrap();
        let (out, _) = wgpu_client
            .varlen_attention_fwd(
                &q_w, &k_w, &v_w, &csq, &csk, batch_size, num_heads, num_heads, max_seqlen,
                max_seqlen, head_dim, true,
            )
            .unwrap();
        assert_parity_f32(
            &out.to_vec::<f32>(),
            &cpu_out_vec,
            "varlen causal equal WGPU vs CPU",
        );
    });
}

/// Backward with `seq_len_q < seq_len_k` and `causal`. Under the absolute
/// convention every key of the sequence is visible to at least one query (the
/// last query sits at `seq_len_k - 1`), so EVERY key row receives gradient.
/// Under the old top-left rule keys `ki >= seq_len_q` were masked for all
/// queries and their `dK`/`dV` rows came back exactly zero — which is what this
/// asserts against.
#[test]
fn test_varlen_causal_unequal_seqlens_bwd_reaches_all_keys() {
    let (cpu_client, cpu_device) = setup_cpu();
    let batch_size = 1;
    let num_heads = 2;
    let head_dim = 64;
    // 2 queries against 6 keys → key_offset 4, query positions 4 and 5.
    let cu_q_data: Vec<i32> = vec![0, 2];
    let cu_k_data: Vec<i32> = vec![0, 6];
    let total_tokens_q = 2;
    let total_tokens_k = 6;
    let max_seqlen_q = 2;
    let max_seqlen_k = 6;

    let q = det_tensor(&[total_tokens_q, num_heads, head_dim], &cpu_device);
    let k = det_tensor(&[total_tokens_k, num_heads, head_dim], &cpu_device);
    let v = det_tensor(&[total_tokens_k, num_heads, head_dim], &cpu_device);
    let dout = det_tensor(&[total_tokens_q, num_heads, head_dim], &cpu_device);
    let cu_seqlens_q = det_i32_tensor(&cu_q_data, &[batch_size + 1], &cpu_device);
    let cu_seqlens_k = det_i32_tensor(&cu_k_data, &[batch_size + 1], &cpu_device);

    let (out, lse) = cpu_client
        .varlen_attention_fwd(
            &q,
            &k,
            &v,
            &cu_seqlens_q,
            &cu_seqlens_k,
            batch_size,
            num_heads,
            num_heads,
            max_seqlen_q,
            max_seqlen_k,
            head_dim,
            true,
        )
        .unwrap();
    let (cpu_dq, cpu_dk, cpu_dv) = cpu_client
        .varlen_attention_bwd(
            &dout,
            &q,
            &k,
            &v,
            &out,
            &lse,
            &cu_seqlens_q,
            &cu_seqlens_k,
            batch_size,
            num_heads,
            num_heads,
            max_seqlen_q,
            max_seqlen_k,
            head_dim,
            true,
        )
        .unwrap();
    let _cpu_dq_vec = cpu_dq.to_vec::<f32>();
    let _cpu_dk_vec = cpu_dk.to_vec::<f32>();
    let cpu_dv_vec = cpu_dv.to_vec::<f32>();

    // Keys 2..6 are exactly the rows the old top-left rule left untouched.
    for ki in 0..total_tokens_k {
        for h in 0..num_heads {
            let base = (ki * num_heads + h) * head_dim;
            let dv_row: f32 = cpu_dv_vec[base..base + head_dim]
                .iter()
                .map(|x| x.abs())
                .sum();
            assert!(
                dv_row > 1e-6,
                "dV row for key {ki} head {h} is zero: causal masking is still top-left"
            );
        }
    }

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::attention::varlen_attention::VarLenAttentionOps as _;
        use numr::tensor::Tensor;
        let q_shape = [total_tokens_q, num_heads, head_dim];
        let k_shape = [total_tokens_k, num_heads, head_dim];
        let q_c = Tensor::try_from_slice(&q.to_vec::<f32>(), &q_shape, &cuda_device).unwrap();
        let k_c = Tensor::try_from_slice(&k.to_vec::<f32>(), &k_shape, &cuda_device).unwrap();
        let v_c = Tensor::try_from_slice(&v.to_vec::<f32>(), &k_shape, &cuda_device).unwrap();
        let dout_c = Tensor::try_from_slice(&dout.to_vec::<f32>(), &q_shape, &cuda_device).unwrap();
        let csq = Tensor::try_from_slice(&cu_q_data, &[batch_size + 1], &cuda_device).unwrap();
        let csk = Tensor::try_from_slice(&cu_k_data, &[batch_size + 1], &cuda_device).unwrap();
        let (out_c, lse_c) = cuda_client
            .varlen_attention_fwd(
                &q_c,
                &k_c,
                &v_c,
                &csq,
                &csk,
                batch_size,
                num_heads,
                num_heads,
                max_seqlen_q,
                max_seqlen_k,
                head_dim,
                true,
            )
            .unwrap();
        let (dq, dk, dv) = cuda_client
            .varlen_attention_bwd(
                &dout_c,
                &q_c,
                &k_c,
                &v_c,
                &out_c,
                &lse_c,
                &csq,
                &csk,
                batch_size,
                num_heads,
                num_heads,
                max_seqlen_q,
                max_seqlen_k,
                head_dim,
                true,
            )
            .unwrap();
        assert_parity_f32_relaxed(
            &dq.to_vec::<f32>(),
            &_cpu_dq_vec,
            "varlen bwd unequal dQ CUDA vs CPU",
        );
        assert_parity_f32_relaxed(
            &dk.to_vec::<f32>(),
            &_cpu_dk_vec,
            "varlen bwd unequal dK CUDA vs CPU",
        );
        assert_parity_f32_relaxed(
            &dv.to_vec::<f32>(),
            &cpu_dv_vec,
            "varlen bwd unequal dV CUDA vs CPU",
        );
    });
}
