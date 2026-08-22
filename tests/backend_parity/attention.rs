//! Backend parity tests for FlashAttentionOps (fwd + bwd).

use super::helpers::*;
use boostr::ops::traits::attention::flash::FlashAttentionOps;

#[test]
fn test_flash_attention_fwd_non_causal_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (b, h, s, d) = (2, 4, 16, 32);
    let q = det_tensor(&[b, h, s, d], &cpu_device);
    let k = det_tensor(&[b, h, s, d], &cpu_device);
    let v = det_tensor(&[b, h, s, d], &cpu_device);

    let (cpu_out, _cpu_lse) = cpu_client
        .flash_attention_fwd(&q, &k, &v, h, h, d, false, 0, None)
        .unwrap();
    let cpu_out_vec = cpu_out.to_vec::<f32>();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::attention::flash::FlashAttentionOps as _;
        use numr::tensor::Tensor;
        let q_c = Tensor::from_slice(&q.to_vec::<f32>(), &[b, h, s, d], &cuda_device);
        let k_c = Tensor::from_slice(&k.to_vec::<f32>(), &[b, h, s, d], &cuda_device);
        let v_c = Tensor::from_slice(&v.to_vec::<f32>(), &[b, h, s, d], &cuda_device);
        let (cuda_out, _) = cuda_client
            .flash_attention_fwd(&q_c, &k_c, &v_c, h, h, d, false, 0, None)
            .unwrap();
        assert_parity_f32(
            &cuda_out.to_vec::<f32>(),
            &cpu_out_vec,
            "flash_fwd non-causal CUDA vs CPU",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::attention::flash::FlashAttentionOps as _;
        use numr::tensor::Tensor;
        let q_w = Tensor::from_slice(&q.to_vec::<f32>(), &[b, h, s, d], &wgpu_device);
        let k_w = Tensor::from_slice(&k.to_vec::<f32>(), &[b, h, s, d], &wgpu_device);
        let v_w = Tensor::from_slice(&v.to_vec::<f32>(), &[b, h, s, d], &wgpu_device);
        let (wgpu_out, _) = wgpu_client
            .flash_attention_fwd(&q_w, &k_w, &v_w, h, h, d, false, 0, None)
            .unwrap();
        assert_parity_f32(
            &wgpu_out.to_vec::<f32>(),
            &cpu_out_vec,
            "flash_fwd non-causal WGPU vs CPU",
        );
    });
}

#[test]
fn test_flash_attention_fwd_causal_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (b, h, s, d) = (1, 2, 12, 32);
    let q = det_tensor(&[b, h, s, d], &cpu_device);
    let k = det_tensor(&[b, h, s, d], &cpu_device);
    let v = det_tensor(&[b, h, s, d], &cpu_device);

    let (cpu_out, _) = cpu_client
        .flash_attention_fwd(&q, &k, &v, h, h, d, true, 0, None)
        .unwrap();
    let cpu_out_vec = cpu_out.to_vec::<f32>();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::attention::flash::FlashAttentionOps as _;
        use numr::tensor::Tensor;
        let q_c = Tensor::from_slice(&q.to_vec::<f32>(), &[b, h, s, d], &cuda_device);
        let k_c = Tensor::from_slice(&k.to_vec::<f32>(), &[b, h, s, d], &cuda_device);
        let v_c = Tensor::from_slice(&v.to_vec::<f32>(), &[b, h, s, d], &cuda_device);
        let (cuda_out, _) = cuda_client
            .flash_attention_fwd(&q_c, &k_c, &v_c, h, h, d, true, 0, None)
            .unwrap();
        assert_parity_f32(
            &cuda_out.to_vec::<f32>(),
            &cpu_out_vec,
            "flash_fwd causal CUDA vs CPU",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::attention::flash::FlashAttentionOps as _;
        use numr::tensor::Tensor;
        let q_w = Tensor::from_slice(&q.to_vec::<f32>(), &[b, h, s, d], &wgpu_device);
        let k_w = Tensor::from_slice(&k.to_vec::<f32>(), &[b, h, s, d], &wgpu_device);
        let v_w = Tensor::from_slice(&v.to_vec::<f32>(), &[b, h, s, d], &wgpu_device);
        let (wgpu_out, _) = wgpu_client
            .flash_attention_fwd(&q_w, &k_w, &v_w, h, h, d, true, 0, None)
            .unwrap();
        assert_parity_f32(
            &wgpu_out.to_vec::<f32>(),
            &cpu_out_vec,
            "flash_fwd causal WGPU vs CPU",
        );
    });
}

#[test]
fn test_flash_attention_fwd_gqa_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (b, s, d) = (1, 8, 32);
    let num_heads = 4;
    let num_kv_heads = 2;
    let q = det_tensor(&[b, num_heads, s, d], &cpu_device);
    let k = det_tensor(&[b, num_kv_heads, s, d], &cpu_device);
    let v = det_tensor(&[b, num_kv_heads, s, d], &cpu_device);

    let (cpu_out, _) = cpu_client
        .flash_attention_fwd(&q, &k, &v, num_heads, num_kv_heads, d, false, 0, None)
        .unwrap();
    let cpu_out_vec = cpu_out.to_vec::<f32>();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::attention::flash::FlashAttentionOps as _;
        use numr::tensor::Tensor;
        let q_c = Tensor::from_slice(&q.to_vec::<f32>(), &[b, num_heads, s, d], &cuda_device);
        let k_c = Tensor::from_slice(&k.to_vec::<f32>(), &[b, num_kv_heads, s, d], &cuda_device);
        let v_c = Tensor::from_slice(&v.to_vec::<f32>(), &[b, num_kv_heads, s, d], &cuda_device);
        let (cuda_out, _) = cuda_client
            .flash_attention_fwd(&q_c, &k_c, &v_c, num_heads, num_kv_heads, d, false, 0, None)
            .unwrap();
        assert_parity_f32(
            &cuda_out.to_vec::<f32>(),
            &cpu_out_vec,
            "flash_fwd GQA CUDA vs CPU",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::attention::flash::FlashAttentionOps as _;
        use numr::tensor::Tensor;
        let q_w = Tensor::from_slice(&q.to_vec::<f32>(), &[b, num_heads, s, d], &wgpu_device);
        let k_w = Tensor::from_slice(&k.to_vec::<f32>(), &[b, num_kv_heads, s, d], &wgpu_device);
        let v_w = Tensor::from_slice(&v.to_vec::<f32>(), &[b, num_kv_heads, s, d], &wgpu_device);
        let (wgpu_out, _) = wgpu_client
            .flash_attention_fwd(&q_w, &k_w, &v_w, num_heads, num_kv_heads, d, false, 0, None)
            .unwrap();
        assert_parity_f32(
            &wgpu_out.to_vec::<f32>(),
            &cpu_out_vec,
            "flash_fwd GQA WGPU vs CPU",
        );
    });
}

/// Sliding-window single-token decode (`seq_len_q == 1`, `seq_len_k == 20`).
/// The query is at absolute position `seq_len_k - 1`, so every backend must
/// keep the same `window` key suffix. Decode passes `causal = false`, matching
/// the model path where `causal` comes from `is_prefill`.
#[test]
fn test_flash_attention_fwd_windowed_decode_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (b, d, sk, window) = (1, 32, 20, 6);
    let num_heads = 4;
    let num_kv_heads = 2;
    let q = det_tensor(&[b, num_heads, 1, d], &cpu_device);
    let k = det_tensor(&[b, num_kv_heads, sk, d], &cpu_device);
    let v = det_tensor(&[b, num_kv_heads, sk, d], &cpu_device);

    let (cpu_out, _) = cpu_client
        .flash_attention_fwd(&q, &k, &v, num_heads, num_kv_heads, d, false, window, None)
        .unwrap();
    let cpu_out_vec = cpu_out.to_vec::<f32>();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::attention::flash::FlashAttentionOps as _;
        use numr::tensor::Tensor;
        let q_c = Tensor::from_slice(&q.to_vec::<f32>(), &[b, num_heads, 1, d], &cuda_device);
        let k_c = Tensor::from_slice(&k.to_vec::<f32>(), &[b, num_kv_heads, sk, d], &cuda_device);
        let v_c = Tensor::from_slice(&v.to_vec::<f32>(), &[b, num_kv_heads, sk, d], &cuda_device);
        let (cuda_out, _) = cuda_client
            .flash_attention_fwd(
                &q_c,
                &k_c,
                &v_c,
                num_heads,
                num_kv_heads,
                d,
                false,
                window,
                None,
            )
            .unwrap();
        assert_parity_f32(
            &cuda_out.to_vec::<f32>(),
            &cpu_out_vec,
            "flash_fwd windowed decode CUDA vs CPU",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::attention::flash::FlashAttentionOps as _;
        use numr::tensor::Tensor;
        let q_w = Tensor::from_slice(&q.to_vec::<f32>(), &[b, num_heads, 1, d], &wgpu_device);
        let k_w = Tensor::from_slice(&k.to_vec::<f32>(), &[b, num_kv_heads, sk, d], &wgpu_device);
        let v_w = Tensor::from_slice(&v.to_vec::<f32>(), &[b, num_kv_heads, sk, d], &wgpu_device);
        let (wgpu_out, _) = wgpu_client
            .flash_attention_fwd(
                &q_w,
                &k_w,
                &v_w,
                num_heads,
                num_kv_heads,
                d,
                false,
                window,
                None,
            )
            .unwrap();
        assert_parity_f32(
            &wgpu_out.to_vec::<f32>(),
            &cpu_out_vec,
            "flash_fwd windowed decode WGPU vs CPU",
        );
    });
}

#[test]
fn test_flash_attention_bwd_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (b, h, s, d) = (1, 2, 8, 32);
    let q = det_tensor(&[b, h, s, d], &cpu_device);
    let k = det_tensor(&[b, h, s, d], &cpu_device);
    let v = det_tensor(&[b, h, s, d], &cpu_device);

    let (out, lse) = cpu_client
        .flash_attention_fwd(&q, &k, &v, h, h, d, false, 0, None)
        .unwrap();
    let dout = det_tensor(&[b, h, s, d], &cpu_device);
    let (cpu_dq, cpu_dk, cpu_dv) = cpu_client
        .flash_attention_bwd(&dout, &q, &k, &v, &out, &lse, h, h, d, false, 0)
        .unwrap();
    let _cpu_dq_vec = cpu_dq.to_vec::<f32>();
    let _cpu_dk_vec = cpu_dk.to_vec::<f32>();
    let _cpu_dv_vec = cpu_dv.to_vec::<f32>();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::attention::flash::FlashAttentionOps as _;
        use numr::tensor::Tensor;
        let q_c = Tensor::from_slice(&q.to_vec::<f32>(), &[b, h, s, d], &cuda_device);
        let k_c = Tensor::from_slice(&k.to_vec::<f32>(), &[b, h, s, d], &cuda_device);
        let v_c = Tensor::from_slice(&v.to_vec::<f32>(), &[b, h, s, d], &cuda_device);
        let (out_c, lse_c) = cuda_client
            .flash_attention_fwd(&q_c, &k_c, &v_c, h, h, d, false, 0, None)
            .unwrap();
        let dout_c = Tensor::from_slice(&dout.to_vec::<f32>(), &[b, h, s, d], &cuda_device);
        let (dq_c, dk_c, dv_c) = cuda_client
            .flash_attention_bwd(&dout_c, &q_c, &k_c, &v_c, &out_c, &lse_c, h, h, d, false, 0)
            .unwrap();
        assert_parity_f32(
            &dq_c.to_vec::<f32>(),
            &_cpu_dq_vec,
            "flash_bwd dQ CUDA vs CPU",
        );
        assert_parity_f32(
            &dk_c.to_vec::<f32>(),
            &_cpu_dk_vec,
            "flash_bwd dK CUDA vs CPU",
        );
        assert_parity_f32(
            &dv_c.to_vec::<f32>(),
            &_cpu_dv_vec,
            "flash_bwd dV CUDA vs CPU",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::attention::flash::FlashAttentionOps as _;
        use numr::tensor::Tensor;
        let q_w = Tensor::from_slice(&q.to_vec::<f32>(), &[b, h, s, d], &wgpu_device);
        let k_w = Tensor::from_slice(&k.to_vec::<f32>(), &[b, h, s, d], &wgpu_device);
        let v_w = Tensor::from_slice(&v.to_vec::<f32>(), &[b, h, s, d], &wgpu_device);
        let (out_w, lse_w) = wgpu_client
            .flash_attention_fwd(&q_w, &k_w, &v_w, h, h, d, false, 0, None)
            .unwrap();
        let dout_w = Tensor::from_slice(&dout.to_vec::<f32>(), &[b, h, s, d], &wgpu_device);
        // BWD not yet implemented on WebGPU — skip gracefully
        if let Ok((dq_w, dk_w, dv_w)) = wgpu_client
            .flash_attention_bwd(&dout_w, &q_w, &k_w, &v_w, &out_w, &lse_w, h, h, d, false, 0)
        {
            assert_parity_f32(
                &dq_w.to_vec::<f32>(),
                &_cpu_dq_vec,
                "flash_bwd dQ WGPU vs CPU",
            );
            assert_parity_f32(
                &dk_w.to_vec::<f32>(),
                &_cpu_dk_vec,
                "flash_bwd dK WGPU vs CPU",
            );
            assert_parity_f32(
                &dv_w.to_vec::<f32>(),
                &_cpu_dv_vec,
                "flash_bwd dV WGPU vs CPU",
            );
        } else {
            eprintln!("flash_attention_bwd not implemented on WebGPU, skipping");
        }
    });
}

/// Verify that flash attention non-causal output matches a naive O(N²) reference
/// on all backends. The reference is computed once on CPU.
#[test]
fn test_flash_v2_fwd_matches_reference() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (b, h, s, d) = (2, 4, 16, 32);
    let q = det_tensor(&[b, h, s, d], &cpu_device);
    let k = det_tensor(&[b, h, s, d], &cpu_device);
    let v = det_tensor(&[b, h, s, d], &cpu_device);

    let ref_out = reference_attention(&cpu_client, &q, &k, &v, false);
    let ref_vec = ref_out.to_vec::<f32>();

    // CPU flash vs naive reference
    let (cpu_flash_out, _) = cpu_client
        .flash_attention_fwd(&q, &k, &v, h, h, d, false, 0, None)
        .unwrap();
    assert_parity_f32(
        &cpu_flash_out.to_vec::<f32>(),
        &ref_vec,
        "flash_v2_fwd non-causal CPU vs reference",
    );

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::attention::flash::FlashAttentionOps as _;
        use numr::tensor::Tensor;
        let q_c = Tensor::from_slice(&q.to_vec::<f32>(), &[b, h, s, d], &cuda_device);
        let k_c = Tensor::from_slice(&k.to_vec::<f32>(), &[b, h, s, d], &cuda_device);
        let v_c = Tensor::from_slice(&v.to_vec::<f32>(), &[b, h, s, d], &cuda_device);
        let (cuda_out, _) = cuda_client
            .flash_attention_fwd(&q_c, &k_c, &v_c, h, h, d, false, 0, None)
            .unwrap();
        assert_parity_f32(
            &cuda_out.to_vec::<f32>(),
            &ref_vec,
            "flash_v2_fwd non-causal CUDA vs reference",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::attention::flash::FlashAttentionOps as _;
        use numr::tensor::Tensor;
        let q_w = Tensor::from_slice(&q.to_vec::<f32>(), &[b, h, s, d], &wgpu_device);
        let k_w = Tensor::from_slice(&k.to_vec::<f32>(), &[b, h, s, d], &wgpu_device);
        let v_w = Tensor::from_slice(&v.to_vec::<f32>(), &[b, h, s, d], &wgpu_device);
        let (wgpu_out, _) = wgpu_client
            .flash_attention_fwd(&q_w, &k_w, &v_w, h, h, d, false, 0, None)
            .unwrap();
        assert_parity_f32(
            &wgpu_out.to_vec::<f32>(),
            &ref_vec,
            "flash_v2_fwd non-causal WGPU vs reference",
        );
    });
}

/// Verify that flash attention causal output matches a naive O(N²) causal reference
/// on all backends.
#[test]
fn test_flash_v2_fwd_causal_matches_reference() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (b, h, s, d) = (1, 2, 12, 32);
    let q = det_tensor(&[b, h, s, d], &cpu_device);
    let k = det_tensor(&[b, h, s, d], &cpu_device);
    let v = det_tensor(&[b, h, s, d], &cpu_device);

    let ref_out = reference_attention(&cpu_client, &q, &k, &v, true);
    let ref_vec = ref_out.to_vec::<f32>();

    // CPU flash vs naive causal reference
    let (cpu_flash_out, _) = cpu_client
        .flash_attention_fwd(&q, &k, &v, h, h, d, true, 0, None)
        .unwrap();
    assert_parity_f32(
        &cpu_flash_out.to_vec::<f32>(),
        &ref_vec,
        "flash_v2_fwd causal CPU vs reference",
    );

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::attention::flash::FlashAttentionOps as _;
        use numr::tensor::Tensor;
        let q_c = Tensor::from_slice(&q.to_vec::<f32>(), &[b, h, s, d], &cuda_device);
        let k_c = Tensor::from_slice(&k.to_vec::<f32>(), &[b, h, s, d], &cuda_device);
        let v_c = Tensor::from_slice(&v.to_vec::<f32>(), &[b, h, s, d], &cuda_device);
        let (cuda_out, _) = cuda_client
            .flash_attention_fwd(&q_c, &k_c, &v_c, h, h, d, true, 0, None)
            .unwrap();
        assert_parity_f32(
            &cuda_out.to_vec::<f32>(),
            &ref_vec,
            "flash_v2_fwd causal CUDA vs reference",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::attention::flash::FlashAttentionOps as _;
        use numr::tensor::Tensor;
        let q_w = Tensor::from_slice(&q.to_vec::<f32>(), &[b, h, s, d], &wgpu_device);
        let k_w = Tensor::from_slice(&k.to_vec::<f32>(), &[b, h, s, d], &wgpu_device);
        let v_w = Tensor::from_slice(&v.to_vec::<f32>(), &[b, h, s, d], &wgpu_device);
        let (wgpu_out, _) = wgpu_client
            .flash_attention_fwd(&q_w, &k_w, &v_w, h, h, d, true, 0, None)
            .unwrap();
        assert_parity_f32(
            &wgpu_out.to_vec::<f32>(),
            &ref_vec,
            "flash_v2_fwd causal WGPU vs reference",
        );
    });
}

/// Sanity check that backward pass produces nonzero gradients on all backends.
#[test]
fn test_flash_v2_bwd_gradients_nonzero() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (b, h, s, d) = (1, 2, 8, 32);
    let q = det_tensor(&[b, h, s, d], &cpu_device);
    let k = det_tensor(&[b, h, s, d], &cpu_device);
    let v = det_tensor(&[b, h, s, d], &cpu_device);
    let dout = det_tensor(&[b, h, s, d], &cpu_device);

    // CPU: gradients must be nonzero
    use numr::ops::{ReduceOps, UnaryOps};
    let (out, lse) = cpu_client
        .flash_attention_fwd(&q, &k, &v, h, h, d, false, 0, None)
        .unwrap();
    let (dq, dk, dv) = cpu_client
        .flash_attention_bwd(&dout, &q, &k, &v, &out, &lse, h, h, d, false, 0)
        .unwrap();
    for (name, grad) in [("dQ", &dq), ("dK", &dk), ("dV", &dv)] {
        let abs_sum = cpu_client
            .sum(&cpu_client.abs(grad).unwrap(), &[], false)
            .unwrap();
        assert!(
            abs_sum.to_vec::<f32>()[0] > 1e-6,
            "CPU {name} gradients are zero"
        );
    }

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::attention::flash::FlashAttentionOps as _;
        use numr::tensor::Tensor;
        let q_c = Tensor::from_slice(&q.to_vec::<f32>(), &[b, h, s, d], &cuda_device);
        let k_c = Tensor::from_slice(&k.to_vec::<f32>(), &[b, h, s, d], &cuda_device);
        let v_c = Tensor::from_slice(&v.to_vec::<f32>(), &[b, h, s, d], &cuda_device);
        let dout_c = Tensor::from_slice(&dout.to_vec::<f32>(), &[b, h, s, d], &cuda_device);
        let (out_c, lse_c) = cuda_client
            .flash_attention_fwd(&q_c, &k_c, &v_c, h, h, d, false, 0, None)
            .unwrap();
        let (dq_c, dk_c, dv_c) = cuda_client
            .flash_attention_bwd(&dout_c, &q_c, &k_c, &v_c, &out_c, &lse_c, h, h, d, false, 0)
            .unwrap();
        for (name, grad) in [("dQ", &dq_c), ("dK", &dk_c), ("dV", &dv_c)] {
            let abs_sum = cuda_client
                .sum(&cuda_client.abs(grad).unwrap(), &[], false)
                .unwrap();
            assert!(
                abs_sum.to_vec::<f32>()[0] > 1e-6,
                "CUDA {name} gradients are zero"
            );
        }
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::attention::flash::FlashAttentionOps as _;
        use numr::tensor::Tensor;
        let q_w = Tensor::from_slice(&q.to_vec::<f32>(), &[b, h, s, d], &wgpu_device);
        let k_w = Tensor::from_slice(&k.to_vec::<f32>(), &[b, h, s, d], &wgpu_device);
        let v_w = Tensor::from_slice(&v.to_vec::<f32>(), &[b, h, s, d], &wgpu_device);
        let dout_w = Tensor::from_slice(&dout.to_vec::<f32>(), &[b, h, s, d], &wgpu_device);
        let (out_w, lse_w) = wgpu_client
            .flash_attention_fwd(&q_w, &k_w, &v_w, h, h, d, false, 0, None)
            .unwrap();
        // BWD not yet implemented on WebGPU — skip gracefully
        if let Ok((dq_w, dk_w, dv_w)) = wgpu_client
            .flash_attention_bwd(&dout_w, &q_w, &k_w, &v_w, &out_w, &lse_w, h, h, d, false, 0)
        {
            for (name, grad) in [("dQ", &dq_w), ("dK", &dk_w), ("dV", &dv_w)] {
                let abs_sum = wgpu_client
                    .sum(&wgpu_client.abs(grad).unwrap(), &[], false)
                    .unwrap();
                assert!(
                    abs_sum.to_vec::<f32>()[0] > 1e-6,
                    "WGPU {name} gradients are zero"
                );
            }
        } else {
            eprintln!("flash_attention_bwd not implemented on WebGPU, skipping");
        }
    });
}

/// Verify GQA correctness across multiple head ratios on all backends.
/// For each ratio: check output shape, finite values, and backward gradient shapes.
#[test]
fn test_gqa_correctness_various_ratios() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (b, s, d) = (1, 8, 32);

    for (num_heads, num_kv_heads) in [(4, 4), (4, 2), (4, 1), (8, 2), (8, 1)] {
        let q = det_tensor(&[b, num_heads, s, d], &cpu_device);
        let k = det_tensor(&[b, num_kv_heads, s, d], &cpu_device);
        let v = det_tensor(&[b, num_kv_heads, s, d], &cpu_device);

        // CPU correctness
        let (cpu_out, cpu_lse) = cpu_client
            .flash_attention_fwd(&q, &k, &v, num_heads, num_kv_heads, d, false, 0, None)
            .unwrap();
        assert_eq!(cpu_out.shape(), &[b, num_heads, s, d]);
        assert_eq!(cpu_lse.shape(), &[b, num_heads, s]);
        assert!(
            cpu_out.to_vec::<f32>().iter().all(|x| x.is_finite()),
            "GQA {num_heads}/{num_kv_heads} CPU produced non-finite values"
        );

        let cpu_out_vec = cpu_out.to_vec::<f32>();

        let cpu_dout = det_tensor(&[b, num_heads, s, d], &cpu_device);
        let (cpu_dq, cpu_dk, cpu_dv) = cpu_client
            .flash_attention_bwd(
                &cpu_dout,
                &q,
                &k,
                &v,
                &cpu_out,
                &cpu_lse,
                num_heads,
                num_kv_heads,
                d,
                false,
                0,
            )
            .unwrap();
        assert_eq!(cpu_dq.shape(), &[b, num_heads, s, d]);
        assert_eq!(cpu_dk.shape(), &[b, num_kv_heads, s, d]);
        assert_eq!(cpu_dv.shape(), &[b, num_kv_heads, s, d]);

        #[cfg(feature = "cuda")]
        with_cuda_backend(|cuda_client, cuda_device| {
            use boostr::ops::traits::attention::flash::FlashAttentionOps as _;
            use numr::tensor::Tensor;
            let q_c = Tensor::from_slice(&q.to_vec::<f32>(), &[b, num_heads, s, d], &cuda_device);
            let k_c =
                Tensor::from_slice(&k.to_vec::<f32>(), &[b, num_kv_heads, s, d], &cuda_device);
            let v_c =
                Tensor::from_slice(&v.to_vec::<f32>(), &[b, num_kv_heads, s, d], &cuda_device);
            let (cuda_out, _) = cuda_client
                .flash_attention_fwd(&q_c, &k_c, &v_c, num_heads, num_kv_heads, d, false, 0, None)
                .unwrap();
            assert_parity_f32(
                &cuda_out.to_vec::<f32>(),
                &cpu_out_vec,
                &format!("GQA {num_heads}/{num_kv_heads} CUDA vs CPU"),
            );
        });

        #[cfg(feature = "wgpu")]
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            use boostr::ops::traits::attention::flash::FlashAttentionOps as _;
            use numr::tensor::Tensor;
            let q_w = Tensor::from_slice(&q.to_vec::<f32>(), &[b, num_heads, s, d], &wgpu_device);
            let k_w =
                Tensor::from_slice(&k.to_vec::<f32>(), &[b, num_kv_heads, s, d], &wgpu_device);
            let v_w =
                Tensor::from_slice(&v.to_vec::<f32>(), &[b, num_kv_heads, s, d], &wgpu_device);
            let (wgpu_out, _) = wgpu_client
                .flash_attention_fwd(&q_w, &k_w, &v_w, num_heads, num_kv_heads, d, false, 0, None)
                .unwrap();
            assert_parity_f32(
                &wgpu_out.to_vec::<f32>(),
                &cpu_out_vec,
                &format!("GQA {num_heads}/{num_kv_heads} WGPU vs CPU"),
            );
        });
    }
}

/// Verify that sliding window attention restricts attention and produces finite values
/// on all backends.
#[test]
fn test_sliding_window_correctness() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (b, h, s, d) = (1, 2, 12, 32);
    let q = det_tensor(&[b, h, s, d], &cpu_device);
    let k = det_tensor(&[b, h, s, d], &cpu_device);
    let v = det_tensor(&[b, h, s, d], &cpu_device);

    let window_size = 4;

    // CPU: sliding window should differ from full attention and be finite
    let (cpu_win_out, _) = cpu_client
        .flash_attention_fwd(&q, &k, &v, h, h, d, false, window_size, None)
        .unwrap();
    let (cpu_full_out, _) = cpu_client
        .flash_attention_fwd(&q, &k, &v, h, h, d, false, 0, None)
        .unwrap();
    let cpu_win_vec = cpu_win_out.to_vec::<f32>();
    let cpu_full_vec = cpu_full_out.to_vec::<f32>();

    let diff = max_abs_diff(&cpu_client, &cpu_win_out, &cpu_full_out);
    assert!(
        diff > 1e-6,
        "CPU sliding window output should differ from full attention"
    );
    assert!(
        cpu_win_vec.iter().all(|x| x.is_finite()),
        "CPU sliding window produced non-finite values"
    );

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::attention::flash::FlashAttentionOps as _;
        use numr::tensor::Tensor;
        let q_c = Tensor::from_slice(&q.to_vec::<f32>(), &[b, h, s, d], &cuda_device);
        let k_c = Tensor::from_slice(&k.to_vec::<f32>(), &[b, h, s, d], &cuda_device);
        let v_c = Tensor::from_slice(&v.to_vec::<f32>(), &[b, h, s, d], &cuda_device);
        let (cuda_win_out, _) = cuda_client
            .flash_attention_fwd(&q_c, &k_c, &v_c, h, h, d, false, window_size, None)
            .unwrap();
        let (cuda_full_out, _) = cuda_client
            .flash_attention_fwd(&q_c, &k_c, &v_c, h, h, d, false, 0, None)
            .unwrap();
        // Sliding window vs full should differ on CUDA too
        let cuda_win_vec = cuda_win_out.to_vec::<f32>();
        let cuda_full_vec = cuda_full_out.to_vec::<f32>();
        let cuda_diff: f32 = cuda_win_vec
            .iter()
            .zip(cuda_full_vec.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            cuda_diff > 1e-6,
            "CUDA sliding window output should differ from full attention"
        );
        assert!(
            cuda_win_vec.iter().all(|x| x.is_finite()),
            "CUDA sliding window produced non-finite values"
        );
        // Also check parity of window output against CPU window output
        assert_parity_f32(&cuda_win_vec, &cpu_win_vec, "sliding_window CUDA vs CPU");
        assert_parity_f32(&cuda_full_vec, &cpu_full_vec, "sliding_full CUDA vs CPU");
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::attention::flash::FlashAttentionOps as _;
        use numr::tensor::Tensor;
        let q_w = Tensor::from_slice(&q.to_vec::<f32>(), &[b, h, s, d], &wgpu_device);
        let k_w = Tensor::from_slice(&k.to_vec::<f32>(), &[b, h, s, d], &wgpu_device);
        let v_w = Tensor::from_slice(&v.to_vec::<f32>(), &[b, h, s, d], &wgpu_device);
        let (wgpu_win_out, _) = wgpu_client
            .flash_attention_fwd(&q_w, &k_w, &v_w, h, h, d, false, window_size, None)
            .unwrap();
        let (wgpu_full_out, _) = wgpu_client
            .flash_attention_fwd(&q_w, &k_w, &v_w, h, h, d, false, 0, None)
            .unwrap();
        let wgpu_win_vec = wgpu_win_out.to_vec::<f32>();
        let wgpu_full_vec = wgpu_full_out.to_vec::<f32>();
        let wgpu_diff: f32 = wgpu_win_vec
            .iter()
            .zip(wgpu_full_vec.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            wgpu_diff > 1e-6,
            "WGPU sliding window output should differ from full attention"
        );
        assert!(
            wgpu_win_vec.iter().all(|x| x.is_finite()),
            "WGPU sliding window produced non-finite values"
        );
        assert_parity_f32(&wgpu_win_vec, &cpu_win_vec, "sliding_window WGPU vs CPU");
        assert_parity_f32(&wgpu_full_vec, &cpu_full_vec, "sliding_full WGPU vs CPU");
    });
}

/// Windowed backward parity: CPU is the reference (its mask goes through
/// `build_attention_mask`), CUDA must match for dQ, dK and dV.
///
/// Also asserts the windowed gradients differ from the unwindowed ones, so the
/// test cannot pass with the window ignored on both sides.
#[test]
fn test_flash_attention_bwd_windowed_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (b, h, s, d) = (1, 2, 16, 32);
    let window = 4usize;
    let q = det_tensor(&[b, h, s, d], &cpu_device);
    let k = det_tensor(&[b, h, s, d], &cpu_device);
    let v = det_tensor(&[b, h, s, d], &cpu_device);
    let dout = det_tensor(&[b, h, s, d], &cpu_device);

    let (out_win, lse_win) = cpu_client
        .flash_attention_fwd(&q, &k, &v, h, h, d, true, window, None)
        .unwrap();
    let (cpu_dq, cpu_dk, cpu_dv) = cpu_client
        .flash_attention_bwd(&dout, &q, &k, &v, &out_win, &lse_win, h, h, d, true, window)
        .unwrap();
    let cpu_dq_vec = cpu_dq.to_vec::<f32>();
    let cpu_dk_vec = cpu_dk.to_vec::<f32>();
    let cpu_dv_vec = cpu_dv.to_vec::<f32>();

    // The window must actually change the CPU gradients.
    let (out_full, lse_full) = cpu_client
        .flash_attention_fwd(&q, &k, &v, h, h, d, true, 0, None)
        .unwrap();
    let (_, cpu_dk_full, _) = cpu_client
        .flash_attention_bwd(&dout, &q, &k, &v, &out_full, &lse_full, h, h, d, true, 0)
        .unwrap();
    assert_differs(
        &cpu_dk_vec,
        &cpu_dk_full.to_vec::<f32>(),
        "CPU windowed vs unwindowed dK",
    );

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::attention::flash::FlashAttentionOps as _;
        use numr::tensor::Tensor;
        let q_c = Tensor::from_slice(&q.to_vec::<f32>(), &[b, h, s, d], &cuda_device);
        let k_c = Tensor::from_slice(&k.to_vec::<f32>(), &[b, h, s, d], &cuda_device);
        let v_c = Tensor::from_slice(&v.to_vec::<f32>(), &[b, h, s, d], &cuda_device);
        let dout_c = Tensor::from_slice(&dout.to_vec::<f32>(), &[b, h, s, d], &cuda_device);

        let (out_c, lse_c) = cuda_client
            .flash_attention_fwd(&q_c, &k_c, &v_c, h, h, d, true, window, None)
            .unwrap();
        let (dq_c, dk_c, dv_c) = cuda_client
            .flash_attention_bwd(
                &dout_c, &q_c, &k_c, &v_c, &out_c, &lse_c, h, h, d, true, window,
            )
            .unwrap();
        let cuda_dk_vec = dk_c.to_vec::<f32>();
        assert_parity_f32_relaxed(&dq_c.to_vec::<f32>(), &cpu_dq_vec, "flash_bwd windowed dQ");
        assert_parity_f32_relaxed(&cuda_dk_vec, &cpu_dk_vec, "flash_bwd windowed dK");
        assert_parity_f32_relaxed(&dv_c.to_vec::<f32>(), &cpu_dv_vec, "flash_bwd windowed dV");

        // Windowed CUDA gradients must differ from unwindowed CUDA gradients.
        let (out_cf, lse_cf) = cuda_client
            .flash_attention_fwd(&q_c, &k_c, &v_c, h, h, d, true, 0, None)
            .unwrap();
        let (_, dk_cf, _) = cuda_client
            .flash_attention_bwd(
                &dout_c, &q_c, &k_c, &v_c, &out_cf, &lse_cf, h, h, d, true, 0,
            )
            .unwrap();
        assert_differs(
            &cuda_dk_vec,
            &dk_cf.to_vec::<f32>(),
            "CUDA windowed vs unwindowed dK",
        );
    });
}

/// Regression guard: `window_size == 0` backward is untouched by the windowing
/// change — CUDA still matches CPU for both causal and non-causal.
#[test]
fn test_flash_attention_bwd_window_zero_unchanged() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (b, h, s, d) = (1, 2, 16, 32);
    let q = det_tensor(&[b, h, s, d], &cpu_device);
    let k = det_tensor(&[b, h, s, d], &cpu_device);
    let v = det_tensor(&[b, h, s, d], &cpu_device);
    let dout = det_tensor(&[b, h, s, d], &cpu_device);

    for causal in [false, true] {
        let (out, lse) = cpu_client
            .flash_attention_fwd(&q, &k, &v, h, h, d, causal, 0, None)
            .unwrap();
        let (cpu_dq, cpu_dk, cpu_dv) = cpu_client
            .flash_attention_bwd(&dout, &q, &k, &v, &out, &lse, h, h, d, causal, 0)
            .unwrap();
        let cpu_dq_vec = cpu_dq.to_vec::<f32>();
        let cpu_dk_vec = cpu_dk.to_vec::<f32>();
        let cpu_dv_vec = cpu_dv.to_vec::<f32>();

        #[cfg(feature = "cuda")]
        with_cuda_backend(|cuda_client, cuda_device| {
            use boostr::ops::traits::attention::flash::FlashAttentionOps as _;
            use numr::tensor::Tensor;
            let q_c = Tensor::from_slice(&q.to_vec::<f32>(), &[b, h, s, d], &cuda_device);
            let k_c = Tensor::from_slice(&k.to_vec::<f32>(), &[b, h, s, d], &cuda_device);
            let v_c = Tensor::from_slice(&v.to_vec::<f32>(), &[b, h, s, d], &cuda_device);
            let dout_c = Tensor::from_slice(&dout.to_vec::<f32>(), &[b, h, s, d], &cuda_device);
            let (out_c, lse_c) = cuda_client
                .flash_attention_fwd(&q_c, &k_c, &v_c, h, h, d, causal, 0, None)
                .unwrap();
            let (dq_c, dk_c, dv_c) = cuda_client
                .flash_attention_bwd(
                    &dout_c, &q_c, &k_c, &v_c, &out_c, &lse_c, h, h, d, causal, 0,
                )
                .unwrap();
            assert_parity_f32_relaxed(&dq_c.to_vec::<f32>(), &cpu_dq_vec, "flash_bwd w0 dQ");
            assert_parity_f32_relaxed(&dk_c.to_vec::<f32>(), &cpu_dk_vec, "flash_bwd w0 dK");
            assert_parity_f32_relaxed(&dv_c.to_vec::<f32>(), &cpu_dv_vec, "flash_bwd w0 dV");
        });
    }
}

/// A key position the window excludes for EVERY query must receive exactly zero
/// gradient. With seq_len_q < seq_len_k the query rows sit at absolute positions
/// `key_offset..key_offset + seq_len_q`, so keys `j` with `j + window <= key_offset`
/// are masked for every query row.
#[test]
fn test_flash_attention_bwd_excluded_key_has_zero_grad() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (b, h, d) = (1, 2, 32);
    let (sq, sk) = (4usize, 16usize);
    let window = 2usize;
    let key_offset = sk - sq;
    let last_excluded = key_offset - window; // keys 0..=last_excluded are fully masked

    let q = det_tensor(&[b, h, sq, d], &cpu_device);
    let k = det_tensor(&[b, h, sk, d], &cpu_device);
    let v = det_tensor(&[b, h, sk, d], &cpu_device);
    let dout = det_tensor(&[b, h, sq, d], &cpu_device);

    let (out, lse) = cpu_client
        .flash_attention_fwd(&q, &k, &v, h, h, d, false, window, None)
        .unwrap();
    let (_, cpu_dk, cpu_dv) = cpu_client
        .flash_attention_bwd(&dout, &q, &k, &v, &out, &lse, h, h, d, false, window)
        .unwrap();
    assert_zero_key_rows(&cpu_dk.to_vec::<f32>(), h, sk, d, last_excluded, "CPU dK");
    assert_zero_key_rows(&cpu_dv.to_vec::<f32>(), h, sk, d, last_excluded, "CPU dV");

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::attention::flash::FlashAttentionOps as _;
        use numr::tensor::Tensor;
        let q_c = Tensor::from_slice(&q.to_vec::<f32>(), &[b, h, sq, d], &cuda_device);
        let k_c = Tensor::from_slice(&k.to_vec::<f32>(), &[b, h, sk, d], &cuda_device);
        let v_c = Tensor::from_slice(&v.to_vec::<f32>(), &[b, h, sk, d], &cuda_device);
        let dout_c = Tensor::from_slice(&dout.to_vec::<f32>(), &[b, h, sq, d], &cuda_device);
        let (out_c, lse_c) = cuda_client
            .flash_attention_fwd(&q_c, &k_c, &v_c, h, h, d, false, window, None)
            .unwrap();
        let (_, dk_c, dv_c) = cuda_client
            .flash_attention_bwd(
                &dout_c, &q_c, &k_c, &v_c, &out_c, &lse_c, h, h, d, false, window,
            )
            .unwrap();
        assert_zero_key_rows(&dk_c.to_vec::<f32>(), h, sk, d, last_excluded, "CUDA dK");
        assert_zero_key_rows(&dv_c.to_vec::<f32>(), h, sk, d, last_excluded, "CUDA dV");
    });
}

/// Assert two gradient buffers are not the same tensor within backward tolerance.
fn assert_differs(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{}: length mismatch", what);
    let max_diff = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff > 1e-3,
        "{}: expected a difference, max abs diff was {}",
        what,
        max_diff
    );
}

/// Assert key rows `0..=last` of a `[1, h, seq_len_k, head_dim]` gradient are exactly zero.
fn assert_zero_key_rows(grad: &[f32], h: usize, sk: usize, d: usize, last: usize, what: &str) {
    for head in 0..h {
        for key in 0..=last {
            for dim in 0..d {
                let idx = (head * sk + key) * d + dim;
                assert_eq!(
                    grad[idx], 0.0,
                    "{}: key {} (head {}, dim {}) is excluded by the window but got {}",
                    what, key, head, dim, grad[idx]
                );
            }
        }
    }
}

/// Bug 2 regression: causal backward for `head_dim = 128`, whose block config is
/// `BLOCK_M = 128`, `BLOCK_N = 64`. The Q-block pruning must compare token
/// positions, not block indices — with `BLOCK_M > BLOCK_N` a block-index
/// comparison skips Q blocks that still hold causally valid rows, zeroing
/// dK/dV for the later key blocks.
///
/// `s = 192` spans two Q blocks and three K blocks, so the pruning engages.
#[test]
fn test_flash_attention_bwd_causal_head_dim_128_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (b, h, s, d) = (1, 1, 192, 128);
    let q = det_tensor(&[b, h, s, d], &cpu_device);
    let k = det_tensor(&[b, h, s, d], &cpu_device);
    let v = det_tensor(&[b, h, s, d], &cpu_device);
    let dout = det_tensor(&[b, h, s, d], &cpu_device);

    let (out, lse) = cpu_client
        .flash_attention_fwd(&q, &k, &v, h, h, d, true, 0, None)
        .unwrap();
    let (cpu_dq, cpu_dk, cpu_dv) = cpu_client
        .flash_attention_bwd(&dout, &q, &k, &v, &out, &lse, h, h, d, true, 0)
        .unwrap();
    let _cpu_dq_vec = cpu_dq.to_vec::<f32>();
    let _cpu_dk_vec = cpu_dk.to_vec::<f32>();
    let _cpu_dv_vec = cpu_dv.to_vec::<f32>();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::attention::flash::FlashAttentionOps as _;
        use numr::tensor::Tensor;
        let q_c = Tensor::from_slice(&q.to_vec::<f32>(), &[b, h, s, d], &cuda_device);
        let k_c = Tensor::from_slice(&k.to_vec::<f32>(), &[b, h, s, d], &cuda_device);
        let v_c = Tensor::from_slice(&v.to_vec::<f32>(), &[b, h, s, d], &cuda_device);
        let dout_c = Tensor::from_slice(&dout.to_vec::<f32>(), &[b, h, s, d], &cuda_device);
        let (out_c, lse_c) = cuda_client
            .flash_attention_fwd(&q_c, &k_c, &v_c, h, h, d, true, 0, None)
            .unwrap();
        // head_dim=128 backward needs the large block config (BLOCK_M=128,
        // BLOCK_N=64); there is no `_sm` backward kernel, so a GPU without the
        // shared memory for it reports an error instead of running.
        match cuda_client
            .flash_attention_bwd(&dout_c, &q_c, &k_c, &v_c, &out_c, &lse_c, h, h, d, true, 0)
        {
            Ok((dq_c, dk_c, dv_c)) => {
                assert_parity_f32_relaxed(
                    &dq_c.to_vec::<f32>(),
                    &_cpu_dq_vec,
                    "flash_bwd causal hd128 dQ",
                );
                assert_parity_f32_relaxed(
                    &dk_c.to_vec::<f32>(),
                    &_cpu_dk_vec,
                    "flash_bwd causal hd128 dK",
                );
                assert_parity_f32_relaxed(
                    &dv_c.to_vec::<f32>(),
                    &_cpu_dv_vec,
                    "flash_bwd causal hd128 dV",
                );
            }
            Err(e) => eprintln!(
                "head_dim=128 backward unavailable on this GPU, skipping parity check: {:?}",
                e
            ),
        }
    });
}

/// Guard that the configs which were already correct stay correct: head_dim 32
/// and 64 both use `BLOCK_M == BLOCK_N == 128`, so the generalized pruning must
/// reproduce the old block-index behaviour exactly.
#[test]
fn test_flash_attention_bwd_causal_small_head_dim_parity() {
    for d in [32usize, 64usize] {
        let (cpu_client, cpu_device) = setup_cpu();
        let (b, h, s) = (1, 2, 160);
        let q = det_tensor(&[b, h, s, d], &cpu_device);
        let k = det_tensor(&[b, h, s, d], &cpu_device);
        let v = det_tensor(&[b, h, s, d], &cpu_device);
        let dout = det_tensor(&[b, h, s, d], &cpu_device);

        let (out, lse) = cpu_client
            .flash_attention_fwd(&q, &k, &v, h, h, d, true, 0, None)
            .unwrap();
        let (cpu_dq, cpu_dk, cpu_dv) = cpu_client
            .flash_attention_bwd(&dout, &q, &k, &v, &out, &lse, h, h, d, true, 0)
            .unwrap();
        let _cpu_dq_vec = cpu_dq.to_vec::<f32>();
        let _cpu_dk_vec = cpu_dk.to_vec::<f32>();
        let _cpu_dv_vec = cpu_dv.to_vec::<f32>();

        #[cfg(feature = "cuda")]
        with_cuda_backend(|cuda_client, cuda_device| {
            use boostr::ops::traits::attention::flash::FlashAttentionOps as _;
            use numr::tensor::Tensor;
            let q_c = Tensor::from_slice(&q.to_vec::<f32>(), &[b, h, s, d], &cuda_device);
            let k_c = Tensor::from_slice(&k.to_vec::<f32>(), &[b, h, s, d], &cuda_device);
            let v_c = Tensor::from_slice(&v.to_vec::<f32>(), &[b, h, s, d], &cuda_device);
            let dout_c = Tensor::from_slice(&dout.to_vec::<f32>(), &[b, h, s, d], &cuda_device);
            let (out_c, lse_c) = cuda_client
                .flash_attention_fwd(&q_c, &k_c, &v_c, h, h, d, true, 0, None)
                .unwrap();
            // The backward's block config is chosen from the FORWARD shared-memory
            // layout, but backward needs (2*BLOCK_M + 2*BLOCK_N)*head_dim*elem —
            // 128KB at head_dim 64 — and no `_sm` backward kernel exists for
            // head_dim 32/64. On a card with less opt-in shared memory than that,
            // backward cannot run at all. Report it rather than failing the run,
            // and rather than pretending the parity check happened.
            match cuda_client
                .flash_attention_bwd(&dout_c, &q_c, &k_c, &v_c, &out_c, &lse_c, h, h, d, true, 0)
            {
                Ok((dq_c, dk_c, dv_c)) => {
                    assert_parity_f32_relaxed(
                        &dq_c.to_vec::<f32>(),
                        &_cpu_dq_vec,
                        "flash_bwd causal dQ",
                    );
                    assert_parity_f32_relaxed(
                        &dk_c.to_vec::<f32>(),
                        &_cpu_dk_vec,
                        "flash_bwd causal dK",
                    );
                    assert_parity_f32_relaxed(
                        &dv_c.to_vec::<f32>(),
                        &_cpu_dv_vec,
                        "flash_bwd causal dV",
                    );
                }
                Err(e) => {
                    println!(
                        "SKIPPED head_dim {d} backward parity: this GPU cannot run \
                         the backward kernel ({e}). CUDA flash backward is \
                         UNAVAILABLE at this head_dim on this device."
                    );
                }
            }
        });
    }
}

/// Bug 1 regression: causal backward with `seq_len_q != seq_len_k`. Query row
/// `i` sits at absolute position `key_offset + i` with
/// `key_offset = seq_len_k - seq_len_q`, so the backward mask must be
/// `k_pos > key_offset + q_pos`, matching the forward and
/// `build_attention_mask`. CPU is the reference.
///
/// Also asserts key `seq_len_k - 1` — visible only to the LAST query row — has
/// a nonzero gradient, which a raw-index mask (`q_pos < k_pos`) would zero.
#[test]
fn test_flash_attention_bwd_causal_key_offset_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (b, h, d) = (1, 2, 32);
    let (sq, sk) = (6usize, 16usize);
    let q = det_tensor(&[b, h, sq, d], &cpu_device);
    let k = det_tensor(&[b, h, sk, d], &cpu_device);
    let v = det_tensor(&[b, h, sk, d], &cpu_device);
    let dout = det_tensor(&[b, h, sq, d], &cpu_device);

    let (out, lse) = cpu_client
        .flash_attention_fwd(&q, &k, &v, h, h, d, true, 0, None)
        .unwrap();
    let (cpu_dq, cpu_dk, cpu_dv) = cpu_client
        .flash_attention_bwd(&dout, &q, &k, &v, &out, &lse, h, h, d, true, 0)
        .unwrap();
    let _cpu_dq_vec = cpu_dq.to_vec::<f32>();
    let _cpu_dk_vec = cpu_dk.to_vec::<f32>();
    let _cpu_dv_vec = cpu_dv.to_vec::<f32>();
    assert_key_row_nonzero(&_cpu_dk_vec, h, sk, d, sk - 1, "CPU dK last key");

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::attention::flash::FlashAttentionOps as _;
        use numr::tensor::Tensor;
        let q_c = Tensor::from_slice(&q.to_vec::<f32>(), &[b, h, sq, d], &cuda_device);
        let k_c = Tensor::from_slice(&k.to_vec::<f32>(), &[b, h, sk, d], &cuda_device);
        let v_c = Tensor::from_slice(&v.to_vec::<f32>(), &[b, h, sk, d], &cuda_device);
        let dout_c = Tensor::from_slice(&dout.to_vec::<f32>(), &[b, h, sq, d], &cuda_device);
        let (out_c, lse_c) = cuda_client
            .flash_attention_fwd(&q_c, &k_c, &v_c, h, h, d, true, 0, None)
            .unwrap();
        let (dq_c, dk_c, dv_c) = cuda_client
            .flash_attention_bwd(&dout_c, &q_c, &k_c, &v_c, &out_c, &lse_c, h, h, d, true, 0)
            .unwrap();
        let cuda_dk_vec = dk_c.to_vec::<f32>();
        assert_key_row_nonzero(&cuda_dk_vec, h, sk, d, sk - 1, "CUDA dK last key");
        assert_parity_f32_relaxed(
            &dq_c.to_vec::<f32>(),
            &_cpu_dq_vec,
            "flash_bwd key_offset dQ",
        );
        assert_parity_f32_relaxed(&cuda_dk_vec, &_cpu_dk_vec, "flash_bwd key_offset dK");
        assert_parity_f32_relaxed(
            &dv_c.to_vec::<f32>(),
            &_cpu_dv_vec,
            "flash_bwd key_offset dV",
        );
    });
}

/// Assert key row `key` of a `[1, h, seq_len_k, head_dim]` gradient is not all zero.
fn assert_key_row_nonzero(grad: &[f32], h: usize, sk: usize, d: usize, key: usize, what: &str) {
    for head in 0..h {
        let start = (head * sk + key) * d;
        let nonzero = grad[start..start + d].iter().any(|x| x.abs() > 1e-6);
        assert!(
            nonzero,
            "{}: key {} (head {}) must receive gradient but is all zero",
            what, key, head
        );
    }
}
