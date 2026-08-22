//! Backend parity tests for PagedAttentionOps.

use super::helpers::*;
use boostr::ops::traits::attention::paged_attention::PagedAttentionOps;

#[test]
fn test_paged_attention_fwd_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (b, h, s, d) = (1, 2, 4, 64);
    let block_size = 4;
    let num_blocks = 1;

    let q = det_tensor(&[b, h, s, d], &cpu_device);
    let k_blocks = det_tensor(&[num_blocks, block_size, 1, d], &cpu_device);
    let v_blocks = det_tensor(&[num_blocks, block_size, 1, d], &cpu_device);
    let bt_data: Vec<i32> = vec![0];
    let block_table = det_i32_tensor(&bt_data, &[b, 1], &cpu_device);

    let (cpu_out, _) = cpu_client
        .paged_attention_fwd(
            &q,
            &k_blocks,
            &v_blocks,
            &block_table,
            h,
            1, // num_kv_heads
            s,
            s,
            d,
            block_size,
            false,
        )
        .unwrap();
    let cpu_out_vec = cpu_out.to_vec::<f32>();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::attention::paged_attention::PagedAttentionOps as _;
        use numr::tensor::Tensor;
        let q_c = Tensor::from_slice(&q.to_vec::<f32>(), &[b, h, s, d], &cuda_device);
        let kb = Tensor::from_slice(
            &k_blocks.to_vec::<f32>(),
            &[num_blocks, block_size, 1, d],
            &cuda_device,
        );
        let vb = Tensor::from_slice(
            &v_blocks.to_vec::<f32>(),
            &[num_blocks, block_size, 1, d],
            &cuda_device,
        );
        let bt = Tensor::from_slice(&bt_data, &[b, 1], &cuda_device);
        let (out, _) = cuda_client
            .paged_attention_fwd(&q_c, &kb, &vb, &bt, h, 1, s, s, d, block_size, false)
            .unwrap();
        assert_parity_f32(&out.to_vec::<f32>(), &cpu_out_vec, "paged_fwd CUDA vs CPU");
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::attention::paged_attention::PagedAttentionOps as _;
        use numr::tensor::Tensor;
        let q_w = Tensor::from_slice(&q.to_vec::<f32>(), &[b, h, s, d], &wgpu_device);
        let kb = Tensor::from_slice(
            &k_blocks.to_vec::<f32>(),
            &[num_blocks, block_size, 1, d],
            &wgpu_device,
        );
        let vb = Tensor::from_slice(
            &v_blocks.to_vec::<f32>(),
            &[num_blocks, block_size, 1, d],
            &wgpu_device,
        );
        let bt = Tensor::from_slice(&bt_data, &[b, 1], &wgpu_device);
        let (out, _) = wgpu_client
            .paged_attention_fwd(&q_w, &kb, &vb, &bt, h, 1, s, s, d, block_size, false)
            .unwrap();
        assert_parity_f32(&out.to_vec::<f32>(), &cpu_out_vec, "paged_fwd WGPU vs CPU");
    });
}

#[test]
fn test_paged_attention_bwd_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    // Sequence spans two blocks (block_size 2, seq 4) to exercise the block-table
    // gather/scatter index map.
    let (b, h, s, d) = (1, 2, 4, 32);
    let block_size = 2;
    let num_blocks = 2;
    let bt_data: Vec<i32> = vec![0, 1];

    let q = det_tensor(&[b, h, s, d], &cpu_device);
    let k_blocks = det_tensor(&[num_blocks, block_size, 1, d], &cpu_device);
    let v_blocks = det_tensor(&[num_blocks, block_size, 1, d], &cpu_device);
    let block_table = det_i32_tensor(&bt_data, &[b, num_blocks], &cpu_device);

    let (cpu_out, cpu_lse) = cpu_client
        .paged_attention_fwd(
            &q,
            &k_blocks,
            &v_blocks,
            &block_table,
            h,
            1,
            s,
            s,
            d,
            block_size,
            true,
        )
        .unwrap();
    let dout = det_tensor(&[b, h, s, d], &cpu_device);
    let (cpu_dq, cpu_dk, cpu_dv) = cpu_client
        .paged_attention_bwd(
            &dout,
            &q,
            &k_blocks,
            &v_blocks,
            &cpu_out,
            &cpu_lse,
            &block_table,
            h,
            1,
            s,
            s,
            d,
            block_size,
            true,
        )
        .unwrap();
    let cpu_dq_vec = cpu_dq.to_vec::<f32>();
    let cpu_dk_vec = cpu_dk.to_vec::<f32>();
    let cpu_dv_vec = cpu_dv.to_vec::<f32>();

    // The CPU backward is the reference for the comparison below, but only the
    // wgpu block consumes it — paged_attention_bwd has no CUDA parity coverage
    // yet. Assert the reference is sane unconditionally so this is a real test
    // under every feature set rather than a bare smoke call.
    assert_eq!(cpu_dq.shape(), &[b, h, s, d], "paged_bwd dQ shape");
    assert_eq!(cpu_dk.shape(), k_blocks.shape(), "paged_bwd dK shape");
    assert_eq!(cpu_dv.shape(), v_blocks.shape(), "paged_bwd dV shape");
    for (name, v) in [
        ("dQ", &cpu_dq_vec),
        ("dK", &cpu_dk_vec),
        ("dV", &cpu_dv_vec),
    ] {
        assert!(
            v.iter().all(|x| x.is_finite()),
            "paged_bwd {name} CPU reference has non-finite values"
        );
    }

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::attention::paged_attention::PagedAttentionOps as _;
        use numr::tensor::Tensor;
        let q_w = Tensor::from_slice(&q.to_vec::<f32>(), &[b, h, s, d], &wgpu_device);
        let kb = Tensor::from_slice(
            &k_blocks.to_vec::<f32>(),
            &[num_blocks, block_size, 1, d],
            &wgpu_device,
        );
        let vb = Tensor::from_slice(
            &v_blocks.to_vec::<f32>(),
            &[num_blocks, block_size, 1, d],
            &wgpu_device,
        );
        let bt = Tensor::from_slice(&bt_data, &[b, num_blocks], &wgpu_device);
        let (out_w, lse_w) = wgpu_client
            .paged_attention_fwd(&q_w, &kb, &vb, &bt, h, 1, s, s, d, block_size, true)
            .unwrap();
        let dout_w = Tensor::from_slice(&dout.to_vec::<f32>(), &[b, h, s, d], &wgpu_device);
        let (dq, dk, dv) = wgpu_client
            .paged_attention_bwd(
                &dout_w, &q_w, &kb, &vb, &out_w, &lse_w, &bt, h, 1, s, s, d, block_size, true,
            )
            .unwrap();
        assert_parity_f32_relaxed(&dq.to_vec::<f32>(), &cpu_dq_vec, "paged_bwd dQ WGPU vs CPU");
        assert_parity_f32_relaxed(&dk.to_vec::<f32>(), &cpu_dk_vec, "paged_bwd dK WGPU vs CPU");
        assert_parity_f32_relaxed(&dv.to_vec::<f32>(), &cpu_dv_vec, "paged_bwd dV WGPU vs CPU");
    });
}

/// Naive paged-attention reference.
///
/// The block table indexes keys by their ABSOLUTE position in the sequence, so
/// with `absolute = true` query row `i` sits at `key_offset + i` where
/// `key_offset = seq_len_k - seq_len_q` (the `seq_len_q` queries are the LAST
/// positions of the `seq_len_k` context). `absolute = false` reproduces the
/// legacy top-left rule, where `i` was taken as the position itself.
///
/// Assumes `num_kv_heads == 1`, matching the CPU paged path.
#[allow(clippy::too_many_arguments)]
fn paged_reference_fwd(
    q: &[f32],
    k_blocks: &[f32],
    v_blocks: &[f32],
    block_table: &[i32],
    batch_size: usize,
    num_heads: usize,
    seq_len_q: usize,
    seq_len_k: usize,
    head_dim: usize,
    block_size: usize,
    max_num_blocks: usize,
    causal: bool,
    absolute: bool,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch_size * num_heads * seq_len_q * head_dim];
    let scale = (head_dim as f32).sqrt().recip();
    let key_offset = if absolute {
        seq_len_k.saturating_sub(seq_len_q)
    } else {
        0
    };

    for b in 0..batch_size {
        for h in 0..num_heads {
            for i in 0..seq_len_q {
                let q_off = ((b * num_heads + h) * seq_len_q + i) * head_dim;

                let mut scores = vec![f32::NEG_INFINITY; seq_len_k];
                let mut max_score = f32::NEG_INFINITY;
                for (j, score) in scores.iter_mut().enumerate() {
                    if causal && key_offset + i < j {
                        continue;
                    }
                    let physical = block_table[b * max_num_blocks + j / block_size] as usize;
                    let kv_off = (physical * block_size + j % block_size) * head_dim;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[q_off + d] * k_blocks[kv_off + d];
                    }
                    *score = dot * scale;
                    max_score = max_score.max(*score);
                }

                let mut sum_exp = 0.0f32;
                let exps: Vec<f32> = scores
                    .iter()
                    .map(|s| {
                        let e = (s - max_score).exp();
                        sum_exp += e;
                        e
                    })
                    .collect();

                for (j, e) in exps.iter().enumerate() {
                    let weight = e / sum_exp;
                    if weight == 0.0 {
                        continue;
                    }
                    let physical = block_table[b * max_num_blocks + j / block_size] as usize;
                    let kv_off = (physical * block_size + j % block_size) * head_dim;
                    for d in 0..head_dim {
                        out[q_off + d] += weight * v_blocks[kv_off + d];
                    }
                }
            }
        }
    }
    out
}

/// Causal paged attention with `seq_len_q < seq_len_k` must mask by ABSOLUTE
/// position: the block table addresses keys by absolute position, and the
/// `seq_len_q` query rows are the LAST positions of the `seq_len_k` context, so
/// query row `i` is at `key_offset + i` with `key_offset = seq_len_k - seq_len_q`.
#[test]
fn test_paged_causal_unequal_seqlens_uses_absolute_positions() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (b, h, d) = (1usize, 2usize, 64usize);
    let block_size = 4;
    let num_blocks = 2;
    let max_num_blocks = 2;
    let seq_len_k = 8;
    let seq_len_q = 3; // key_offset = 5
    let bt_data: Vec<i32> = vec![0, 1];

    let q = det_tensor(&[b, h, seq_len_q, d], &cpu_device);
    let k_blocks = det_tensor(&[num_blocks, block_size, 1, d], &cpu_device);
    let v_blocks = det_tensor(&[num_blocks, block_size, 1, d], &cpu_device);
    let block_table = det_i32_tensor(&bt_data, &[b, max_num_blocks], &cpu_device);

    let q_vec = q.to_vec::<f32>();
    let kb_vec = k_blocks.to_vec::<f32>();
    let vb_vec = v_blocks.to_vec::<f32>();

    let absolute_ref = paged_reference_fwd(
        &q_vec,
        &kb_vec,
        &vb_vec,
        &bt_data,
        b,
        h,
        seq_len_q,
        seq_len_k,
        d,
        block_size,
        max_num_blocks,
        true,
        true,
    );
    let legacy_ref = paged_reference_fwd(
        &q_vec,
        &kb_vec,
        &vb_vec,
        &bt_data,
        b,
        h,
        seq_len_q,
        seq_len_k,
        d,
        block_size,
        max_num_blocks,
        true,
        false,
    );
    assert!(
        absolute_ref
            .iter()
            .zip(legacy_ref.iter())
            .any(|(x, y)| (x - y).abs() > 1e-4),
        "test setup is degenerate: absolute and top-left masking agree"
    );

    let (cpu_out, _) = cpu_client
        .paged_attention_fwd(
            &q,
            &k_blocks,
            &v_blocks,
            &block_table,
            h,
            1,
            seq_len_q,
            seq_len_k,
            d,
            block_size,
            true,
        )
        .unwrap();
    let cpu_out_vec = cpu_out.to_vec::<f32>();
    assert_parity_f32(
        &cpu_out_vec,
        &absolute_ref,
        "paged causal unequal CPU vs absolute reference",
    );

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::attention::paged_attention::PagedAttentionOps as _;
        use numr::tensor::Tensor;
        let q_c = Tensor::from_slice(&q_vec, &[b, h, seq_len_q, d], &cuda_device);
        let kb = Tensor::from_slice(&kb_vec, &[num_blocks, block_size, 1, d], &cuda_device);
        let vb = Tensor::from_slice(&vb_vec, &[num_blocks, block_size, 1, d], &cuda_device);
        let bt = Tensor::from_slice(&bt_data, &[b, max_num_blocks], &cuda_device);
        let (out, _) = cuda_client
            .paged_attention_fwd(
                &q_c, &kb, &vb, &bt, h, 1, seq_len_q, seq_len_k, d, block_size, true,
            )
            .unwrap();
        assert_parity_f32(
            &out.to_vec::<f32>(),
            &cpu_out_vec,
            "paged causal unequal CUDA vs CPU",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::attention::paged_attention::PagedAttentionOps as _;
        use numr::tensor::Tensor;
        let q_w = Tensor::from_slice(&q_vec, &[b, h, seq_len_q, d], &wgpu_device);
        let kb = Tensor::from_slice(&kb_vec, &[num_blocks, block_size, 1, d], &wgpu_device);
        let vb = Tensor::from_slice(&vb_vec, &[num_blocks, block_size, 1, d], &wgpu_device);
        let bt = Tensor::from_slice(&bt_data, &[b, max_num_blocks], &wgpu_device);
        let (out, _) = wgpu_client
            .paged_attention_fwd(
                &q_w, &kb, &vb, &bt, h, 1, seq_len_q, seq_len_k, d, block_size, true,
            )
            .unwrap();
        assert_parity_f32(
            &out.to_vec::<f32>(),
            &cpu_out_vec,
            "paged causal unequal WGPU vs CPU",
        );
    });
}

/// Regression guard: `seq_len_q == seq_len_k` gives `key_offset == 0`, so the
/// absolute rule reduces to the legacy top-left rule. The reference here uses
/// the LEGACY rule, so any drift in the equal-length path is caught.
#[test]
fn test_paged_causal_equal_seqlens_matches_legacy_rule() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (b, h, d) = (1usize, 2usize, 64usize);
    let block_size = 4;
    let num_blocks = 2;
    let max_num_blocks = 2;
    let s = 8;
    let bt_data: Vec<i32> = vec![0, 1];

    let q = det_tensor(&[b, h, s, d], &cpu_device);
    let k_blocks = det_tensor(&[num_blocks, block_size, 1, d], &cpu_device);
    let v_blocks = det_tensor(&[num_blocks, block_size, 1, d], &cpu_device);
    let block_table = det_i32_tensor(&bt_data, &[b, max_num_blocks], &cpu_device);

    let legacy_ref = paged_reference_fwd(
        &q.to_vec::<f32>(),
        &k_blocks.to_vec::<f32>(),
        &v_blocks.to_vec::<f32>(),
        &bt_data,
        b,
        h,
        s,
        s,
        d,
        block_size,
        max_num_blocks,
        true,
        false,
    );

    let (cpu_out, _) = cpu_client
        .paged_attention_fwd(
            &q,
            &k_blocks,
            &v_blocks,
            &block_table,
            h,
            1,
            s,
            s,
            d,
            block_size,
            true,
        )
        .unwrap();
    let cpu_out_vec = cpu_out.to_vec::<f32>();
    assert_parity_f32(
        &cpu_out_vec,
        &legacy_ref,
        "equal-length causal paged changed: must stay top-left equivalent",
    );

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::attention::paged_attention::PagedAttentionOps as _;
        use numr::tensor::Tensor;
        let q_c = Tensor::from_slice(&q.to_vec::<f32>(), &[b, h, s, d], &cuda_device);
        let kb = Tensor::from_slice(
            &k_blocks.to_vec::<f32>(),
            &[num_blocks, block_size, 1, d],
            &cuda_device,
        );
        let vb = Tensor::from_slice(
            &v_blocks.to_vec::<f32>(),
            &[num_blocks, block_size, 1, d],
            &cuda_device,
        );
        let bt = Tensor::from_slice(&bt_data, &[b, max_num_blocks], &cuda_device);
        let (out, _) = cuda_client
            .paged_attention_fwd(&q_c, &kb, &vb, &bt, h, 1, s, s, d, block_size, true)
            .unwrap();
        assert_parity_f32(
            &out.to_vec::<f32>(),
            &cpu_out_vec,
            "paged causal equal CUDA vs CPU",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::attention::paged_attention::PagedAttentionOps as _;
        use numr::tensor::Tensor;
        let q_w = Tensor::from_slice(&q.to_vec::<f32>(), &[b, h, s, d], &wgpu_device);
        let kb = Tensor::from_slice(
            &k_blocks.to_vec::<f32>(),
            &[num_blocks, block_size, 1, d],
            &wgpu_device,
        );
        let vb = Tensor::from_slice(
            &v_blocks.to_vec::<f32>(),
            &[num_blocks, block_size, 1, d],
            &wgpu_device,
        );
        let bt = Tensor::from_slice(&bt_data, &[b, max_num_blocks], &wgpu_device);
        let (out, _) = wgpu_client
            .paged_attention_fwd(&q_w, &kb, &vb, &bt, h, 1, s, s, d, block_size, true)
            .unwrap();
        assert_parity_f32(
            &out.to_vec::<f32>(),
            &cpu_out_vec,
            "paged causal equal WGPU vs CPU",
        );
    });
}

/// Rearrange `[num_kv_heads, seq_len, head_dim]` into the paged block layout
/// `[num_blocks, block_size, num_kv_heads, head_dim]`.
///
/// The block table is the identity here, so logical token `t` lives at block
/// `t / block_size`, offset `t % block_size`.
#[cfg(feature = "cuda")]
fn to_paged_layout(src: &[f32], num_kv_heads: usize, seq_len: usize, head_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; num_kv_heads * seq_len * head_dim];
    for h in 0..num_kv_heads {
        for t in 0..seq_len {
            for e in 0..head_dim {
                out[(t * num_kv_heads + h) * head_dim + e] = src[(h * seq_len + t) * head_dim + e];
            }
        }
    }
    out
}

/// Inverse of `to_paged_layout`.
#[cfg(feature = "cuda")]
fn from_paged_layout(
    src: &[f32],
    num_kv_heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; num_kv_heads * seq_len * head_dim];
    for h in 0..num_kv_heads {
        for t in 0..seq_len {
            for e in 0..head_dim {
                out[(h * seq_len + t) * head_dim + e] = src[(t * num_kv_heads + h) * head_dim + e];
            }
        }
    }
    out
}

/// CPU/CUDA parity for the paged backward with `num_kv_heads` KV heads.
///
/// `head_dim` is 64: paged backward has no head_dim-32 kernel, and 64 is the
/// smallest supported size — its small-block config needs 40KB of shared memory,
/// which fits every card that runs these tests.
///
/// The reference is the CPU standard-attention backward over the equivalent
/// CONTIGUOUS K/V, which supports GQA directly. The CPU paged path cannot serve
/// as the reference: it gathers blocks as if `num_kv_heads == 1`.
#[cfg(feature = "cuda")]
fn assert_paged_bwd_kv_parity(num_heads: usize, num_kv_heads: usize, label: &str) {
    use boostr::ops::traits::attention::flash::FlashAttentionOps as _;
    use numr::tensor::Tensor;

    let (cpu_client, cpu_device) = setup_cpu();
    let (b, s, d) = (1usize, 8usize, 64usize);
    let block_size = 4usize;
    let num_blocks = s / block_size;
    let bt_data: Vec<i32> = (0..num_blocks as i32).collect();

    let q = det_tensor(&[b, num_heads, s, d], &cpu_device);
    let k = det_tensor(&[b, num_kv_heads, s, d], &cpu_device);
    let v = det_tensor(&[b, num_kv_heads, s, d], &cpu_device);
    let dout = det_tensor(&[b, num_heads, s, d], &cpu_device);

    let (out, lse) = cpu_client
        .flash_attention_fwd(&q, &k, &v, num_heads, num_kv_heads, d, true, 0, None)
        .unwrap();
    let (cpu_dq, cpu_dk, cpu_dv) = cpu_client
        .flash_attention_bwd(
            &dout,
            &q,
            &k,
            &v,
            &out,
            &lse,
            num_heads,
            num_kv_heads,
            d,
            true,
            0,
        )
        .unwrap();
    let cpu_dq_vec = cpu_dq.to_vec::<f32>();
    let cpu_dk_vec = cpu_dk.to_vec::<f32>();
    let cpu_dv_vec = cpu_dv.to_vec::<f32>();

    let k_paged = to_paged_layout(&k.to_vec::<f32>(), num_kv_heads, s, d);
    let v_paged = to_paged_layout(&v.to_vec::<f32>(), num_kv_heads, s, d);

    with_cuda_backend(|cuda_client, cuda_device| {
        let q_c = Tensor::from_slice(&q.to_vec::<f32>(), &[b, num_heads, s, d], &cuda_device);
        let dout_c = Tensor::from_slice(&dout.to_vec::<f32>(), &[b, num_heads, s, d], &cuda_device);
        let kb = Tensor::from_slice(
            &k_paged,
            &[num_blocks, block_size, num_kv_heads, d],
            &cuda_device,
        );
        let vb = Tensor::from_slice(
            &v_paged,
            &[num_blocks, block_size, num_kv_heads, d],
            &cuda_device,
        );
        let bt = Tensor::from_slice(&bt_data, &[b, num_blocks], &cuda_device);

        let (out_c, lse_c) = cuda_client
            .paged_attention_fwd(
                &q_c,
                &kb,
                &vb,
                &bt,
                num_heads,
                num_kv_heads,
                s,
                s,
                d,
                block_size,
                true,
            )
            .unwrap();
        let (dq_c, dk_c, dv_c) = cuda_client
            .paged_attention_bwd(
                &dout_c,
                &q_c,
                &kb,
                &vb,
                &out_c,
                &lse_c,
                &bt,
                num_heads,
                num_kv_heads,
                s,
                s,
                d,
                block_size,
                true,
            )
            .unwrap();

        assert_eq!(
            dk_c.shape(),
            [num_blocks, block_size, num_kv_heads, d],
            "{}: CUDA dK keeps the paged KV layout",
            label
        );
        assert_eq!(
            dv_c.shape(),
            [num_blocks, block_size, num_kv_heads, d],
            "{}: CUDA dV keeps the paged KV layout",
            label
        );

        let dk_contig = from_paged_layout(&dk_c.to_vec::<f32>(), num_kv_heads, s, d);
        let dv_contig = from_paged_layout(&dv_c.to_vec::<f32>(), num_kv_heads, s, d);
        assert_parity_f32_relaxed(
            &dq_c.to_vec::<f32>(),
            &cpu_dq_vec,
            &format!("{} dQ CUDA vs CPU", label),
        );
        assert_parity_f32_relaxed(
            &dk_contig,
            &cpu_dk_vec,
            &format!("{} dK CUDA vs CPU", label),
        );
        assert_parity_f32_relaxed(
            &dv_contig,
            &cpu_dv_vec,
            &format!("{} dV CUDA vs CPU", label),
        );
    });
}

/// GQA paged backward: 4 query heads over 2 KV heads.
///
/// Regression: the CUDA paged backward kernel indexed the KV blocks as a
/// single-head layout `[num_blocks, block_size, head_dim]` while the forward
/// kernel and the launcher use `[num_blocks, block_size, num_kv_heads, head_dim]`,
/// and the launcher passed a `num_kv_heads` argument the kernel did not declare —
/// so every argument after it was read from the wrong slot.
#[cfg(feature = "cuda")]
#[test]
fn test_paged_attention_bwd_gqa_parity() {
    assert_paged_bwd_kv_parity(4, 2, "paged_bwd gqa 4h/2kv");
}

/// MQA paged backward: 4 query heads share a single KV head.
#[cfg(feature = "cuda")]
#[test]
fn test_paged_attention_bwd_mqa_parity() {
    assert_paged_bwd_kv_parity(4, 1, "paged_bwd mqa 4h/1kv");
}

/// Non-GQA paged backward stays correct: one KV head per query head.
#[cfg(feature = "cuda")]
#[test]
fn test_paged_attention_bwd_no_gqa_parity() {
    assert_paged_bwd_kv_parity(4, 4, "paged_bwd 4h/4kv");
}

/// `num_kv_heads == 0` is rejected: the kernel's head mapping divides by
/// `num_heads / num_kv_heads`.
#[cfg(feature = "cuda")]
#[test]
fn test_paged_attention_bwd_zero_kv_heads_rejected() {
    use numr::tensor::Tensor;

    let (cpu_client, cpu_device) = setup_cpu();
    let (b, h, s, d) = (1usize, 4usize, 8usize, 64usize);
    let block_size = 4usize;
    let num_blocks = s / block_size;
    let bt_data: Vec<i32> = (0..num_blocks as i32).collect();

    let q = det_tensor(&[b, h, s, d], &cpu_device);
    let k = det_tensor(&[b, 1, s, d], &cpu_device);
    let v = det_tensor(&[b, 1, s, d], &cpu_device);
    let dout = det_tensor(&[b, h, s, d], &cpu_device);
    let (out, lse) = cpu_client
        .paged_attention_fwd(
            &q,
            &det_tensor(&[num_blocks, block_size, 1, d], &cpu_device),
            &det_tensor(&[num_blocks, block_size, 1, d], &cpu_device),
            &det_i32_tensor(&bt_data, &[b, num_blocks], &cpu_device),
            h,
            1,
            s,
            s,
            d,
            block_size,
            true,
        )
        .unwrap();

    with_cuda_backend(|cuda_client, cuda_device| {
        let q_c = Tensor::from_slice(&q.to_vec::<f32>(), &[b, h, s, d], &cuda_device);
        let dout_c = Tensor::from_slice(&dout.to_vec::<f32>(), &[b, h, s, d], &cuda_device);
        let out_c = Tensor::from_slice(&out.to_vec::<f32>(), &[b, h, s, d], &cuda_device);
        let lse_c = Tensor::from_slice(&lse.to_vec::<f32>(), &[b, h, s], &cuda_device);
        let kb = Tensor::from_slice(
            &to_paged_layout(&k.to_vec::<f32>(), 1, s, d),
            &[num_blocks, block_size, 1, d],
            &cuda_device,
        );
        let vb = Tensor::from_slice(
            &to_paged_layout(&v.to_vec::<f32>(), 1, s, d),
            &[num_blocks, block_size, 1, d],
            &cuda_device,
        );
        let bt = Tensor::from_slice(&bt_data, &[b, num_blocks], &cuda_device);

        let err = match cuda_client.paged_attention_bwd(
            &dout_c, &q_c, &kb, &vb, &out_c, &lse_c, &bt, h, 0, s, s, d, block_size, true,
        ) {
            Ok(_) => panic!("num_kv_heads == 0 must be rejected"),
            Err(e) => e,
        };
        let msg = err.to_string();
        assert!(
            msg.contains("num_kv_heads") && msg.contains("num_heads"),
            "error must name num_heads and num_kv_heads, got: {msg}"
        );
        assert!(
            msg.contains("(4)") && msg.contains("(0)"),
            "error must report both values, got: {msg}"
        );
    });
}
