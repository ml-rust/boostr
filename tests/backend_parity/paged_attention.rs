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
        let q_c = Tensor::from_slice(&q.to_vec::<f32>(), &[b, h, s, d], &cuda_device).unwrap();
        let kb = Tensor::from_slice(
            &k_blocks.to_vec::<f32>(),
            &[num_blocks, block_size, 1, d],
            &cuda_device,
        )
        .unwrap();
        let vb = Tensor::from_slice(
            &v_blocks.to_vec::<f32>(),
            &[num_blocks, block_size, 1, d],
            &cuda_device,
        )
        .unwrap();
        let bt = Tensor::from_slice(&bt_data, &[b, 1], &cuda_device).unwrap();
        let (out, _) = cuda_client
            .paged_attention_fwd(&q_c, &kb, &vb, &bt, h, 1, s, s, d, block_size, false)
            .unwrap();
        assert_parity_f32(&out.to_vec::<f32>(), &cpu_out_vec, "paged_fwd CUDA vs CPU");
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::attention::paged_attention::PagedAttentionOps as _;
        use numr::tensor::Tensor;
        let q_w = Tensor::from_slice(&q.to_vec::<f32>(), &[b, h, s, d], &wgpu_device).unwrap();
        let kb = Tensor::from_slice(
            &k_blocks.to_vec::<f32>(),
            &[num_blocks, block_size, 1, d],
            &wgpu_device,
        )
        .unwrap();
        let vb = Tensor::from_slice(
            &v_blocks.to_vec::<f32>(),
            &[num_blocks, block_size, 1, d],
            &wgpu_device,
        )
        .unwrap();
        let bt = Tensor::from_slice(&bt_data, &[b, 1], &wgpu_device).unwrap();
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
        let q_w = Tensor::from_slice(&q.to_vec::<f32>(), &[b, h, s, d], &wgpu_device).unwrap();
        let kb = Tensor::from_slice(
            &k_blocks.to_vec::<f32>(),
            &[num_blocks, block_size, 1, d],
            &wgpu_device,
        )
        .unwrap();
        let vb = Tensor::from_slice(
            &v_blocks.to_vec::<f32>(),
            &[num_blocks, block_size, 1, d],
            &wgpu_device,
        )
        .unwrap();
        let bt = Tensor::from_slice(&bt_data, &[b, num_blocks], &wgpu_device).unwrap();
        let (out_w, lse_w) = wgpu_client
            .paged_attention_fwd(&q_w, &kb, &vb, &bt, h, 1, s, s, d, block_size, true)
            .unwrap();
        let dout_w =
            Tensor::from_slice(&dout.to_vec::<f32>(), &[b, h, s, d], &wgpu_device).unwrap();
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
/// Assumes `num_kv_heads == 1`: the key/value lookup below has no `kv_head`
/// term in `kv_off`, so it only ever reads the first (and only) KV head's
/// slot in the cache. This is a property of THIS naive reference, not of the
/// production CPU paged path — `gather_paged_kv`/`expand_kv_heads` in
/// `src/ops/cpu/attention/paged_kv_layout.rs` handle arbitrary `num_kv_heads`
/// correctly (that gather/expand/reduce/scatter grouping is exactly what a
/// prior fix corrected). Every call site below passes `num_kv_heads = 1`, so
/// the mismatch is latent, not exercised.
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
        let q_c = Tensor::from_slice(&q_vec, &[b, h, seq_len_q, d], &cuda_device).unwrap();
        let kb =
            Tensor::from_slice(&kb_vec, &[num_blocks, block_size, 1, d], &cuda_device).unwrap();
        let vb =
            Tensor::from_slice(&vb_vec, &[num_blocks, block_size, 1, d], &cuda_device).unwrap();
        let bt = Tensor::from_slice(&bt_data, &[b, max_num_blocks], &cuda_device).unwrap();
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
        let q_w = Tensor::from_slice(&q_vec, &[b, h, seq_len_q, d], &wgpu_device).unwrap();
        let kb =
            Tensor::from_slice(&kb_vec, &[num_blocks, block_size, 1, d], &wgpu_device).unwrap();
        let vb =
            Tensor::from_slice(&vb_vec, &[num_blocks, block_size, 1, d], &wgpu_device).unwrap();
        let bt = Tensor::from_slice(&bt_data, &[b, max_num_blocks], &wgpu_device).unwrap();
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
        let q_c = Tensor::from_slice(&q.to_vec::<f32>(), &[b, h, s, d], &cuda_device).unwrap();
        let kb = Tensor::from_slice(
            &k_blocks.to_vec::<f32>(),
            &[num_blocks, block_size, 1, d],
            &cuda_device,
        )
        .unwrap();
        let vb = Tensor::from_slice(
            &v_blocks.to_vec::<f32>(),
            &[num_blocks, block_size, 1, d],
            &cuda_device,
        )
        .unwrap();
        let bt = Tensor::from_slice(&bt_data, &[b, max_num_blocks], &cuda_device).unwrap();
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
        let q_w = Tensor::from_slice(&q.to_vec::<f32>(), &[b, h, s, d], &wgpu_device).unwrap();
        let kb = Tensor::from_slice(
            &k_blocks.to_vec::<f32>(),
            &[num_blocks, block_size, 1, d],
            &wgpu_device,
        )
        .unwrap();
        let vb = Tensor::from_slice(
            &v_blocks.to_vec::<f32>(),
            &[num_blocks, block_size, 1, d],
            &wgpu_device,
        )
        .unwrap();
        let bt = Tensor::from_slice(&bt_data, &[b, max_num_blocks], &wgpu_device).unwrap();
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
/// `head_dim` must be 64 or 128: paged backward has no head_dim-32 kernel, and
/// those are the two sizes its shared-memory block configs are compiled for.
///
/// The reference is the CPU standard-attention backward over the equivalent
/// CONTIGUOUS K/V, which supports GQA directly.
///
/// The CPU paged path (`gather_paged_kv`/`expand_kv_heads`/`reduce_kv_heads`/
/// `scatter_to_paged` in `src/ops/cpu/attention/paged_kv_layout.rs`) now
/// handles arbitrary `num_kv_heads` correctly — a prior fix corrected the bug
/// where it gathered blocks as if `num_kv_heads == 1`. It is still not reused
/// as the reference here: it shares the same gather/scatter contract the CUDA
/// paged kernel is being checked against, so a bug in that shared contract
/// would cancel out. The flash-attention reference is independent of that
/// contract, which is a stronger check.
///
/// `dtype` casts Q/K/V/dO to `F16`/`BF16` before the CUDA call and casts the
/// results back to `F32` for comparison (`BF16`/`F16` require boostr's `f16`
/// feature; the case is skipped and reported if it is off). `seq_len` is
/// exposed because the backward atomicAdd defect (see `paged_bwd_tol` below)
/// scales with how many query rows share a KV head over how many keys — a
/// longer sequence makes it visible; the three existing `F32` call sites keep
/// `seq_len = 8` unchanged so this stays a pure regression guard for them.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn assert_paged_bwd_kv_parity(
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    dtype: numr::dtype::DType,
    label: &str,
) {
    use boostr::ops::traits::attention::flash::FlashAttentionOps as _;
    use numr::dtype::DType;
    use numr::tensor::Tensor;

    if dtype != DType::F32 && !cfg!(feature = "f16") {
        eprintln!(
            "SKIPPED: {label} [{:?}] — boostr built without the `f16` feature, so \
             {:?} tensors cannot be constructed",
            dtype, dtype
        );
        return;
    }

    let (cpu_client, cpu_device) = setup_cpu();
    let (b, s, d) = (1usize, seq_len, head_dim);
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

    // n_contrib: retained at `seq_len * heads_per_kv` — see `paged_bwd_tol`'s
    // doc for why this deliberately overstates the kernel's actual per-tile
    // atomic count rather than being tightened to match it.
    let heads_per_kv = num_heads / num_kv_heads;
    let n_contrib = s * heads_per_kv;
    let (dq_atol, dq_rtol) = paged_bwd_tol(dtype, None);
    let (dkv_atol, dkv_rtol) = paged_bwd_tol(dtype, Some(n_contrib));

    with_cuda_backend(|cuda_client, cuda_device| {
        let q_c = cast_to_dtype(
            &q.to_vec::<f32>(),
            &[b, num_heads, s, d],
            &cuda_device,
            dtype,
        );
        let dout_c = cast_to_dtype(
            &dout.to_vec::<f32>(),
            &[b, num_heads, s, d],
            &cuda_device,
            dtype,
        );
        let kb = cast_to_dtype(
            &k_paged,
            &[num_blocks, block_size, num_kv_heads, d],
            &cuda_device,
            dtype,
        );
        let vb = cast_to_dtype(
            &v_paged,
            &[num_blocks, block_size, num_kv_heads, d],
            &cuda_device,
            dtype,
        );
        let bt = Tensor::from_slice(&bt_data, &[b, num_blocks], &cuda_device).unwrap();

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

        let dk_contig = from_paged_layout(&read_back_f32(&dk_c), num_kv_heads, s, d);
        let dv_contig = from_paged_layout(&read_back_f32(&dv_c), num_kv_heads, s, d);
        // dQ accumulates in FP32 registers inside the kernel (only the final
        // store rounds to `dtype`) — it should track the reference far more
        // tightly than dK/dV, which round on every atomicAdd. That gap is
        // the signature of the accumulation defect, not the input rounding.
        let dq_norm = assert_paged_bwd_diff(
            &read_back_f32(&dq_c),
            &cpu_dq_vec,
            dq_atol,
            dq_rtol,
            &format!("{} dQ CUDA vs CPU [{:?}]", label, dtype),
            "dQ",
            dtype,
            num_heads,
            num_kv_heads,
            d,
            s,
        );
        let dk_norm = assert_paged_bwd_diff(
            &dk_contig,
            &cpu_dk_vec,
            dkv_atol,
            dkv_rtol,
            &format!("{} dK CUDA vs CPU [{:?}]", label, dtype),
            "dK",
            dtype,
            num_heads,
            num_kv_heads,
            d,
            s,
        );
        let dv_norm = assert_paged_bwd_diff(
            &dv_contig,
            &cpu_dv_vec,
            dkv_atol,
            dkv_rtol,
            &format!("{} dV CUDA vs CPU [{:?}]", label, dtype),
            "dV",
            dtype,
            num_heads,
            num_kv_heads,
            d,
            s,
        );

        // ADDITIONAL guard, on top of the absolute tolerances above: dQ is
        // the FP32-accumulated control (never touched by any half-storage
        // accumulation, defect or fix), so its normalized error tracks pure
        // input-quantization error only. dK/dV normalized error should sit
        // within a fixed multiple of dQ's at the same shape and dtype — that
        // RATIO, not the absolute tolerance (which stays wide enough to
        // tolerate real cancellation, see `paged_bwd_tol`), is what actually
        // detects a return to per-(q_row, k_idx, d) half accumulation:
        // measured, that regression put dK/dV up to ~780x less accurate than
        // dQ at the same shape; the current one-atomic-per-Q-block kernel
        // measures 2.8x (F16) to 6.7x (BF16). 25x is chosen to give the
        // current numbers real headroom (>3x above the worst measured 6.7x)
        // while staying more than an order of magnitude below the pre-fix
        // 780x, so ordinary kernel/compiler variation will not trip it but a
        // real regression to per-element rounding will.
        const DKV_TO_DQ_RATIO_LIMIT: f32 = 25.0;
        for (name, norm) in [("dK", dk_norm), ("dV", dv_norm)] {
            let ratio = norm / dq_norm;
            assert!(
                ratio <= DKV_TO_DQ_RATIO_LIMIT,
                "{label} {name} CUDA vs CPU [{dtype:?}]: normalized error (max_abs_diff / \
                 ref_rms) is {norm:.4e}, {ratio:.1}x dQ's {dq_norm:.4e} (limit \
                 {DKV_TO_DQ_RATIO_LIMIT}x) — this most likely means {name}'s atomicAdd \
                 accumulation regressed from one atomic per (k_idx, d) per Q-block back to \
                 one atomic per (q_row, k_idx, d), rounding the running sum to `{dtype:?}` on \
                 every contribution instead of once per tile"
            );
        }
    });
}

/// Casts a fixture built as `F32` (via `Tensor::from_slice`, since `half::f16`
/// is not a numr `Element`) to the dtype under test.
#[cfg(feature = "cuda")]
fn cast_to_dtype(
    data: &[f32],
    shape: &[usize],
    device: &numr::runtime::cuda::CudaDevice,
    dtype: numr::dtype::DType,
) -> numr::tensor::Tensor<numr::runtime::cuda::CudaRuntime> {
    use numr::dtype::DType;
    use numr::tensor::Tensor;
    let t = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(data, shape, device).unwrap();
    if dtype == DType::F32 {
        t
    } else {
        t.to_dtype(dtype).unwrap_or_else(|e| {
            panic!("cast fixture to {dtype:?} failed: {e}");
        })
    }
}

/// Reads a CUDA result tensor back to `Vec<f32>`, casting through `F32` first
/// when it is stored as `F16`/`BF16`.
#[cfg(feature = "cuda")]
fn read_back_f32(t: &numr::tensor::Tensor<numr::runtime::cuda::CudaRuntime>) -> Vec<f32> {
    use numr::dtype::DType;
    if t.dtype() == DType::F32 {
        t.to_vec::<f32>()
    } else {
        t.to_dtype(DType::F32)
            .expect("cast kernel result back to F32 for comparison")
            .to_vec::<f32>()
    }
}

/// Backward tolerance for the paged kernel, derived from first principles —
/// NOT tuned to make any particular case pass.
///
/// `(atol, rtol)` where `rtol` scales the reference RMS, matching the
/// `report_and_assert` convention in `mqa_gqa_attention.rs`. The base pair is
/// the quantization-only backward error: Q/K/V/dO are rounded to `dtype`
/// before the kernel runs, then that rounding propagates through the
/// score/softmax/dS chain. `mqa_gqa_attention.rs` measured this same
/// mechanism for its own backward pass, so its numbers are reused here
/// (`f16`: atol 6e-3, rtol 3e-2; `bf16`: atol 4e-2, rtol 1e-1); `f32` keeps
/// this file's original 1e-5/1e-4.
///
/// `dQ` accumulates in FP32 registers inside the kernel and is cast to
/// `dtype` exactly once, at the very end — the base pair is its whole
/// tolerance. Pass `n_contrib = None`.
///
/// `dK`/`dV` accumulate every Q row of a tile into an FP32 register per
/// K-row, then issue ONE `atomicAdd` into `dtype` storage per `(k_idx, d)`
/// per Q-block — not once per `(q_row, k_idx, d)` as an earlier version of
/// this kernel did. Each such add still rounds the running sum to `dtype`'s
/// mantissa, so for `n` sequential rounded additions Higham's classical
/// recursive-summation bound still applies: extra relative error `(n-1) * u`
/// (`u` = unit roundoff: `2^-24` f32, `2^-11` f16, `2^-8` bf16), added to the
/// base `rtol`.
///
/// The STRICT `n` for the current kernel is `num_q_blocks * heads_per_kv`,
/// where `num_q_blocks = ceil(seq_len_q / BLOCK_M)` — for this file's fixtures
/// (`seq_len_q = 32`, `BLOCK_M` in `{32, 64}` for the head_dim/dtype configs
/// tested) that is 1, not `seq_len_q`. The caller instead passes
/// `n = seq_len * (num_heads / num_kv_heads)`, ~16x larger — DELIBERATELY,
/// not by oversight. Measured (max abs deviation / reference RMS): the
/// strict, ~16x-tighter bound sits BELOW the measured BF16 dK/dV deviation
/// and would fail. That is not the kernel being wrong — Higham's bound is
/// normalized by `sum(|x_i|)`, not `|sum(x_i)|`, so it does not model the
/// real partial cancellation across a tile's contributions, and the strict
/// bound under-predicts the true error for that reason. The looser, retained
/// `n` is a deliberately conservative upper bound chosen to stay robust to
/// that modeling gap. Do NOT tighten it to the strict value: that would
/// convert this into a test that flakes on ordinary kernel/compiler
/// variation rather than on an actual regression. (The dK/dV-to-dQ ratio
/// guard in `assert_paged_bwd_kv_parity` is what actually re-detects a
/// regression to per-`(q_row, k_idx, d)` accumulation — this tolerance is a
/// coarse sanity floor, not the precise instrument.)
#[cfg(feature = "cuda")]
fn paged_bwd_tol(dtype: numr::dtype::DType, n_contrib: Option<usize>) -> (f32, f32) {
    use numr::dtype::DType;
    let (atol, rtol_base) = match dtype {
        DType::F32 => (1e-5, 1e-4),
        DType::F16 => (6e-3, 3e-2),
        DType::BF16 => (4e-2, 1e-1),
        other => unimplemented!("paged_bwd_tol: unsupported dtype {other:?}"),
    };
    let u: f32 = match dtype {
        DType::F32 => 2f32.powi(-24),
        DType::F16 => 2f32.powi(-11),
        DType::BF16 => 2f32.powi(-8),
        other => unimplemented!("paged_bwd_tol: unsupported dtype {other:?}"),
    };
    let rtol = match n_contrib {
        None => rtol_base,
        Some(n) => rtol_base + (n.saturating_sub(1) as f32) * u,
    };
    (atol, rtol)
}

/// Compares against the reference, reporting the max absolute deviation, the
/// max relative deviation, and the index of each — so a failure shows how bad
/// it is, not just that it failed. Returns the normalized error
/// (`max_abs_diff / ref_rms`) so callers can compare it across tensors, e.g.
/// the dK/dV-to-dQ ratio guard in `assert_paged_bwd_kv_parity`.
///
/// Also prints a `PAGED_BWD_DIAG` line UNCONDITIONALLY (pass or fail) via
/// `println!`, so the measured deviation is visible even when the run is
/// green — a wide `rtol` (BF16 MQA is 0.596) passing proves only that the
/// kernel isn't catastrophically broken, not that the deviation is small.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn assert_paged_bwd_diff(
    actual: &[f32],
    expected: &[f32],
    atol: f32,
    rtol: f32,
    label: &str,
    tensor: &str,
    dtype: numr::dtype::DType,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> f32 {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: element count mismatch: kernel {} vs reference {}",
        actual.len(),
        expected.len()
    );

    let mut max_abs = 0.0f32;
    let mut max_abs_idx = 0usize;
    let mut max_rel = 0.0f32;
    let mut max_rel_idx = 0usize;
    let mut sq_sum = 0.0f64;
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            a.is_finite(),
            "{label}: kernel produced non-finite value {a} at index {i} (reference {e})"
        );
        let diff = (a - e).abs();
        if diff > max_abs {
            max_abs = diff;
            max_abs_idx = i;
        }
        let rel = diff / (e.abs() + 1e-12);
        if rel > max_rel {
            max_rel = rel;
            max_rel_idx = i;
        }
        sq_sum += (*e as f64) * (*e as f64);
    }
    let rms = (sq_sum / expected.len() as f64).sqrt() as f32;
    let tol = atol + rtol * rms;

    println!(
        "PAGED_BWD_DIAG tensor={tensor} dtype={:?} num_heads={num_heads} \
         num_kv_heads={num_kv_heads} head_dim={head_dim} seq_len={seq_len} \
         max_abs={max_abs:.6e} max_abs_idx={max_abs_idx} max_rel={max_rel:.6e} \
         max_rel_idx={max_rel_idx} ref_rms={rms:.6e} atol={atol:.6e} rtol={rtol:.6e} \
         tol={tol:.6e} label=\"{label}\"",
        dtype
    );

    eprintln!(
        "{label}: n={}, max_abs_diff={max_abs:.4e} (index {max_abs_idx}), \
         max_rel_diff={max_rel:.4e} (index {max_rel_idx}), ref_rms={rms:.4e}, tol={tol:.4e}",
        expected.len()
    );

    assert!(
        rms > 1e-6,
        "{label}: reference RMS is {rms:.4e} — the fixture is degenerate, so agreement \
         would prove nothing. Fix the fixture, not the tolerance."
    );
    assert!(
        max_abs <= tol,
        "{label}: max_abs_diff {max_abs:.4e} at index {max_abs_idx} (max_rel_diff \
         {max_rel:.4e} at index {max_rel_idx}) exceeds tol {tol:.4e} (ref_rms {rms:.4e}); \
         kernel={} reference={}",
        actual[max_abs_idx],
        expected[max_abs_idx]
    );

    max_abs / rms
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
    assert_paged_bwd_kv_parity(4, 2, 64, 8, numr::dtype::DType::F32, "paged_bwd gqa 4h/2kv");
}

/// MQA paged backward: 4 query heads share a single KV head.
#[cfg(feature = "cuda")]
#[test]
fn test_paged_attention_bwd_mqa_parity() {
    assert_paged_bwd_kv_parity(4, 1, 64, 8, numr::dtype::DType::F32, "paged_bwd mqa 4h/1kv");
}

/// Non-GQA paged backward stays correct: one KV head per query head.
#[cfg(feature = "cuda")]
#[test]
fn test_paged_attention_bwd_no_gqa_parity() {
    assert_paged_bwd_kv_parity(4, 4, 64, 8, numr::dtype::DType::F32, "paged_bwd 4h/4kv");
}

// ============================================================================
// F16 / BF16 paged backward parity
//
// `seq_len = 32` (vs 8 for the F32 regression tests above) so the atomicAdd
// accumulation defect described in `paged_bwd_tol` has enough contributions
// per element to be visible: under causal masking, key 0 in each of these
// configs is written `seq_len * (num_heads / num_kv_heads)` times.
// ============================================================================

/// GQA, head_dim 64, F16: 4 query heads over 2 KV heads (16 contributions/key).
#[cfg(feature = "cuda")]
#[test]
fn test_paged_attention_bwd_gqa_hd64_f16_parity() {
    assert_paged_bwd_kv_parity(
        4,
        2,
        64,
        32,
        numr::dtype::DType::F16,
        "paged_bwd gqa 4h/2kv hd64",
    );
}

/// GQA, head_dim 64, BF16.
#[cfg(feature = "cuda")]
#[test]
fn test_paged_attention_bwd_gqa_hd64_bf16_parity() {
    assert_paged_bwd_kv_parity(
        4,
        2,
        64,
        32,
        numr::dtype::DType::BF16,
        "paged_bwd gqa 4h/2kv hd64",
    );
}

/// GQA, head_dim 128, F16.
#[cfg(feature = "cuda")]
#[test]
fn test_paged_attention_bwd_gqa_hd128_f16_parity() {
    assert_paged_bwd_kv_parity(
        4,
        2,
        128,
        32,
        numr::dtype::DType::F16,
        "paged_bwd gqa 4h/2kv hd128",
    );
}

/// GQA, head_dim 128, BF16.
#[cfg(feature = "cuda")]
#[test]
fn test_paged_attention_bwd_gqa_hd128_bf16_parity() {
    assert_paged_bwd_kv_parity(
        4,
        2,
        128,
        32,
        numr::dtype::DType::BF16,
        "paged_bwd gqa 4h/2kv hd128",
    );
}

/// MQA, head_dim 64, F16: 4 query heads share 1 KV head (32 contributions/key
/// — the worst case among these three head configs).
#[cfg(feature = "cuda")]
#[test]
fn test_paged_attention_bwd_mqa_hd64_f16_parity() {
    assert_paged_bwd_kv_parity(
        4,
        1,
        64,
        32,
        numr::dtype::DType::F16,
        "paged_bwd mqa 4h/1kv hd64",
    );
}

/// MQA, head_dim 64, BF16.
#[cfg(feature = "cuda")]
#[test]
fn test_paged_attention_bwd_mqa_hd64_bf16_parity() {
    assert_paged_bwd_kv_parity(
        4,
        1,
        64,
        32,
        numr::dtype::DType::BF16,
        "paged_bwd mqa 4h/1kv hd64",
    );
}

/// MQA, head_dim 128, F16.
#[cfg(feature = "cuda")]
#[test]
fn test_paged_attention_bwd_mqa_hd128_f16_parity() {
    assert_paged_bwd_kv_parity(
        4,
        1,
        128,
        32,
        numr::dtype::DType::F16,
        "paged_bwd mqa 4h/1kv hd128",
    );
}

/// MQA, head_dim 128, BF16.
#[cfg(feature = "cuda")]
#[test]
fn test_paged_attention_bwd_mqa_hd128_bf16_parity() {
    assert_paged_bwd_kv_parity(
        4,
        1,
        128,
        32,
        numr::dtype::DType::BF16,
        "paged_bwd mqa 4h/1kv hd128",
    );
}

/// No-GQA, head_dim 64, F16: one KV head per query head (8 contributions/key
/// — the best case among these three head configs).
#[cfg(feature = "cuda")]
#[test]
fn test_paged_attention_bwd_no_gqa_hd64_f16_parity() {
    assert_paged_bwd_kv_parity(
        4,
        4,
        64,
        32,
        numr::dtype::DType::F16,
        "paged_bwd 4h/4kv hd64",
    );
}

/// No-GQA, head_dim 64, BF16.
#[cfg(feature = "cuda")]
#[test]
fn test_paged_attention_bwd_no_gqa_hd64_bf16_parity() {
    assert_paged_bwd_kv_parity(
        4,
        4,
        64,
        32,
        numr::dtype::DType::BF16,
        "paged_bwd 4h/4kv hd64",
    );
}

/// No-GQA, head_dim 128, F16.
#[cfg(feature = "cuda")]
#[test]
fn test_paged_attention_bwd_no_gqa_hd128_f16_parity() {
    assert_paged_bwd_kv_parity(
        4,
        4,
        128,
        32,
        numr::dtype::DType::F16,
        "paged_bwd 4h/4kv hd128",
    );
}

/// No-GQA, head_dim 128, BF16.
#[cfg(feature = "cuda")]
#[test]
fn test_paged_attention_bwd_no_gqa_hd128_bf16_parity() {
    assert_paged_bwd_kv_parity(
        4,
        4,
        128,
        32,
        numr::dtype::DType::BF16,
        "paged_bwd 4h/4kv hd128",
    );
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
        let q_c = Tensor::from_slice(&q.to_vec::<f32>(), &[b, h, s, d], &cuda_device).unwrap();
        let dout_c =
            Tensor::from_slice(&dout.to_vec::<f32>(), &[b, h, s, d], &cuda_device).unwrap();
        let out_c = Tensor::from_slice(&out.to_vec::<f32>(), &[b, h, s, d], &cuda_device).unwrap();
        let lse_c = Tensor::from_slice(&lse.to_vec::<f32>(), &[b, h, s], &cuda_device).unwrap();
        let kb = Tensor::from_slice(
            &to_paged_layout(&k.to_vec::<f32>(), 1, s, d),
            &[num_blocks, block_size, 1, d],
            &cuda_device,
        )
        .unwrap();
        let vb = Tensor::from_slice(
            &to_paged_layout(&v.to_vec::<f32>(), 1, s, d),
            &[num_blocks, block_size, 1, d],
            &cuda_device,
        )
        .unwrap();
        let bt = Tensor::from_slice(&bt_data, &[b, num_blocks], &cuda_device).unwrap();

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
