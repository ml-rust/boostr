//! Backend parity tests for FusedQkvOps (fwd + bwd).

use super::helpers::*;
use boostr::ops::traits::attention::fused_qkv::FusedQkvOps;

#[test]
fn test_fused_qkv_projection_mha_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (b, s, h) = (2, 8, 256);
    let (num_heads, num_kv_heads, head_dim) = (4, 4, 64);
    let total_proj = num_heads * head_dim + 2 * num_kv_heads * head_dim;

    let input = det_tensor(&[b, s, h], &cpu_device);
    let weight = det_tensor(&[total_proj, h], &cpu_device);

    let (cpu_q, cpu_k, cpu_v) = cpu_client
        .fused_qkv_projection(&input, &weight, None, num_heads, num_kv_heads, head_dim)
        .unwrap();
    let cpu_q_vec = cpu_q.to_vec::<f32>();
    let cpu_k_vec = cpu_k.to_vec::<f32>();
    let cpu_v_vec = cpu_v.to_vec::<f32>();

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::attention::fused_qkv::FusedQkvOps as _;
        use numr::tensor::Tensor;
        let input_w = Tensor::from_slice(&input.to_vec::<f32>(), &[b, s, h], &wgpu_device).unwrap();
        let weight_w =
            Tensor::from_slice(&weight.to_vec::<f32>(), &[total_proj, h], &wgpu_device).unwrap();
        let (wgpu_q, wgpu_k, wgpu_v) = wgpu_client
            .fused_qkv_projection(&input_w, &weight_w, None, num_heads, num_kv_heads, head_dim)
            .unwrap();
        assert_parity_f32_relaxed(
            &wgpu_q.to_vec::<f32>(),
            &cpu_q_vec,
            "fused_qkv_proj MHA Q: WGPU vs CPU",
        );
        assert_parity_f32_relaxed(
            &wgpu_k.to_vec::<f32>(),
            &cpu_k_vec,
            "fused_qkv_proj MHA K: WGPU vs CPU",
        );
        assert_parity_f32_relaxed(
            &wgpu_v.to_vec::<f32>(),
            &cpu_v_vec,
            "fused_qkv_proj MHA V: WGPU vs CPU",
        );
    });

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::attention::fused_qkv::FusedQkvOps as _;
        use numr::tensor::Tensor;
        let input_c = Tensor::from_slice(&input.to_vec::<f32>(), &[b, s, h], &cuda_device).unwrap();
        let weight_c =
            Tensor::from_slice(&weight.to_vec::<f32>(), &[total_proj, h], &cuda_device).unwrap();
        let (cuda_q, cuda_k, cuda_v) = cuda_client
            .fused_qkv_projection(&input_c, &weight_c, None, num_heads, num_kv_heads, head_dim)
            .unwrap();
        assert_parity_f32_relaxed(
            &cuda_q.to_vec::<f32>(),
            &cpu_q_vec,
            "fused_qkv_proj MHA Q: CUDA vs CPU",
        );
        assert_parity_f32_relaxed(
            &cuda_k.to_vec::<f32>(),
            &cpu_k_vec,
            "fused_qkv_proj MHA K: CUDA vs CPU",
        );
        assert_parity_f32_relaxed(
            &cuda_v.to_vec::<f32>(),
            &cpu_v_vec,
            "fused_qkv_proj MHA V: CUDA vs CPU",
        );
    });
}

#[test]
fn test_fused_qkv_projection_gqa_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (b, s, h) = (2, 8, 256);
    let (num_heads, num_kv_heads, head_dim) = (8, 2, 32);
    let total_proj = num_heads * head_dim + 2 * num_kv_heads * head_dim;

    let input = det_tensor(&[b, s, h], &cpu_device);
    let weight = det_tensor(&[total_proj, h], &cpu_device);

    let (cpu_q, cpu_k, cpu_v) = cpu_client
        .fused_qkv_projection(&input, &weight, None, num_heads, num_kv_heads, head_dim)
        .unwrap();
    let cpu_q_vec = cpu_q.to_vec::<f32>();
    let cpu_k_vec = cpu_k.to_vec::<f32>();
    let cpu_v_vec = cpu_v.to_vec::<f32>();

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::attention::fused_qkv::FusedQkvOps as _;
        use numr::tensor::Tensor;
        let input_w = Tensor::from_slice(&input.to_vec::<f32>(), &[b, s, h], &wgpu_device).unwrap();
        let weight_w =
            Tensor::from_slice(&weight.to_vec::<f32>(), &[total_proj, h], &wgpu_device).unwrap();
        let (wgpu_q, wgpu_k, wgpu_v) = wgpu_client
            .fused_qkv_projection(&input_w, &weight_w, None, num_heads, num_kv_heads, head_dim)
            .unwrap();
        assert_parity_f32_relaxed(
            &wgpu_q.to_vec::<f32>(),
            &cpu_q_vec,
            "fused_qkv_proj GQA Q: WGPU vs CPU",
        );
        assert_parity_f32_relaxed(
            &wgpu_k.to_vec::<f32>(),
            &cpu_k_vec,
            "fused_qkv_proj GQA K: WGPU vs CPU",
        );
        assert_parity_f32_relaxed(
            &wgpu_v.to_vec::<f32>(),
            &cpu_v_vec,
            "fused_qkv_proj GQA V: WGPU vs CPU",
        );
    });

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::attention::fused_qkv::FusedQkvOps as _;
        use numr::tensor::Tensor;
        let input_c = Tensor::from_slice(&input.to_vec::<f32>(), &[b, s, h], &cuda_device).unwrap();
        let weight_c =
            Tensor::from_slice(&weight.to_vec::<f32>(), &[total_proj, h], &cuda_device).unwrap();
        let (cuda_q, cuda_k, cuda_v) = cuda_client
            .fused_qkv_projection(&input_c, &weight_c, None, num_heads, num_kv_heads, head_dim)
            .unwrap();
        assert_parity_f32_relaxed(
            &cuda_q.to_vec::<f32>(),
            &cpu_q_vec,
            "fused_qkv_proj GQA Q: CUDA vs CPU",
        );
        assert_parity_f32_relaxed(
            &cuda_k.to_vec::<f32>(),
            &cpu_k_vec,
            "fused_qkv_proj GQA K: CUDA vs CPU",
        );
        assert_parity_f32_relaxed(
            &cuda_v.to_vec::<f32>(),
            &cpu_v_vec,
            "fused_qkv_proj GQA V: CUDA vs CPU",
        );
    });
}

#[test]
fn test_fused_qkv_projection_with_bias_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (b, s, h) = (1, 4, 128);
    let (num_heads, num_kv_heads, head_dim) = (4, 2, 32);
    let total_proj = num_heads * head_dim + 2 * num_kv_heads * head_dim;

    let input = det_tensor(&[b, s, h], &cpu_device);
    let weight = det_tensor(&[total_proj, h], &cpu_device);
    let bias = det_tensor(&[total_proj], &cpu_device);

    let (cpu_q, cpu_k, cpu_v) = cpu_client
        .fused_qkv_projection(
            &input,
            &weight,
            Some(&bias),
            num_heads,
            num_kv_heads,
            head_dim,
        )
        .unwrap();
    let cpu_q_vec = cpu_q.to_vec::<f32>();
    let cpu_k_vec = cpu_k.to_vec::<f32>();
    let cpu_v_vec = cpu_v.to_vec::<f32>();

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::attention::fused_qkv::FusedQkvOps as _;
        use numr::tensor::Tensor;
        let input_w = Tensor::from_slice(&input.to_vec::<f32>(), &[b, s, h], &wgpu_device).unwrap();
        let weight_w =
            Tensor::from_slice(&weight.to_vec::<f32>(), &[total_proj, h], &wgpu_device).unwrap();
        let bias_w =
            Tensor::from_slice(&bias.to_vec::<f32>(), &[total_proj], &wgpu_device).unwrap();
        let (wgpu_q, wgpu_k, wgpu_v) = wgpu_client
            .fused_qkv_projection(
                &input_w,
                &weight_w,
                Some(&bias_w),
                num_heads,
                num_kv_heads,
                head_dim,
            )
            .unwrap();
        assert_parity_f32_relaxed(
            &wgpu_q.to_vec::<f32>(),
            &cpu_q_vec,
            "fused_qkv_proj bias Q: WGPU vs CPU",
        );
        assert_parity_f32_relaxed(
            &wgpu_k.to_vec::<f32>(),
            &cpu_k_vec,
            "fused_qkv_proj bias K: WGPU vs CPU",
        );
        assert_parity_f32_relaxed(
            &wgpu_v.to_vec::<f32>(),
            &cpu_v_vec,
            "fused_qkv_proj bias V: WGPU vs CPU",
        );
    });

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::attention::fused_qkv::FusedQkvOps as _;
        use numr::tensor::Tensor;
        let input_c = Tensor::from_slice(&input.to_vec::<f32>(), &[b, s, h], &cuda_device).unwrap();
        let weight_c =
            Tensor::from_slice(&weight.to_vec::<f32>(), &[total_proj, h], &cuda_device).unwrap();
        let bias_c =
            Tensor::from_slice(&bias.to_vec::<f32>(), &[total_proj], &cuda_device).unwrap();
        let (cuda_q, cuda_k, cuda_v) = cuda_client
            .fused_qkv_projection(
                &input_c,
                &weight_c,
                Some(&bias_c),
                num_heads,
                num_kv_heads,
                head_dim,
            )
            .unwrap();
        assert_parity_f32_relaxed(
            &cuda_q.to_vec::<f32>(),
            &cpu_q_vec,
            "fused_qkv_proj bias Q: CUDA vs CPU",
        );
        assert_parity_f32_relaxed(
            &cuda_k.to_vec::<f32>(),
            &cpu_k_vec,
            "fused_qkv_proj bias K: CUDA vs CPU",
        );
        assert_parity_f32_relaxed(
            &cuda_v.to_vec::<f32>(),
            &cpu_v_vec,
            "fused_qkv_proj bias V: CUDA vs CPU",
        );
    });
}

#[test]
fn test_fused_output_projection_residual_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (b, s, h, proj_dim) = (2, 8, 256, 256);

    let attn_out = det_tensor(&[b, s, proj_dim], &cpu_device);
    let weight = det_tensor(&[h, proj_dim], &cpu_device);
    let residual = det_tensor(&[b, s, h], &cpu_device);

    let cpu_out = cpu_client
        .fused_output_projection_residual(&attn_out, &weight, None, &residual)
        .unwrap();
    let cpu_out_vec = cpu_out.to_vec::<f32>();

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::attention::fused_qkv::FusedQkvOps as _;
        use numr::tensor::Tensor;
        let attn_w =
            Tensor::from_slice(&attn_out.to_vec::<f32>(), &[b, s, proj_dim], &wgpu_device).unwrap();
        let weight_w =
            Tensor::from_slice(&weight.to_vec::<f32>(), &[h, proj_dim], &wgpu_device).unwrap();
        let res_w =
            Tensor::from_slice(&residual.to_vec::<f32>(), &[b, s, h], &wgpu_device).unwrap();
        let wgpu_out = wgpu_client
            .fused_output_projection_residual(&attn_w, &weight_w, None, &res_w)
            .unwrap();
        assert_parity_f32_relaxed(
            &wgpu_out.to_vec::<f32>(),
            &cpu_out_vec,
            "fused_output_proj_residual: WGPU vs CPU",
        );
    });

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::attention::fused_qkv::FusedQkvOps as _;
        use numr::tensor::Tensor;
        let attn_c =
            Tensor::from_slice(&attn_out.to_vec::<f32>(), &[b, s, proj_dim], &cuda_device).unwrap();
        let weight_c =
            Tensor::from_slice(&weight.to_vec::<f32>(), &[h, proj_dim], &cuda_device).unwrap();
        let res_c =
            Tensor::from_slice(&residual.to_vec::<f32>(), &[b, s, h], &cuda_device).unwrap();
        let cuda_out = cuda_client
            .fused_output_projection_residual(&attn_c, &weight_c, None, &res_c)
            .unwrap();
        assert_parity_f32_relaxed(
            &cuda_out.to_vec::<f32>(),
            &cpu_out_vec,
            "fused_output_proj_residual: CUDA vs CPU",
        );
    });
}

#[test]
fn test_fused_qkv_projection_bwd_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (b, s, h) = (1, 4, 128);
    let (num_heads, num_kv_heads, head_dim) = (4, 2, 32);
    let total_proj = num_heads * head_dim + 2 * num_kv_heads * head_dim;

    let dq = det_tensor(&[b, num_heads, s, head_dim], &cpu_device);
    let dk = det_tensor(&[b, num_kv_heads, s, head_dim], &cpu_device);
    let dv = det_tensor(&[b, num_kv_heads, s, head_dim], &cpu_device);
    let input = det_tensor(&[b, s, h], &cpu_device);
    let weight = det_tensor(&[total_proj, h], &cpu_device);

    let (cpu_di, cpu_dw, cpu_db) = cpu_client
        .fused_qkv_projection_bwd(
            &dq,
            &dk,
            &dv,
            &input,
            &weight,
            true,
            num_heads,
            num_kv_heads,
            head_dim,
        )
        .unwrap();
    let cpu_di_vec = cpu_di.to_vec::<f32>();
    let cpu_dw_vec = cpu_dw.to_vec::<f32>();
    let cpu_db_vec = cpu_db.as_ref().unwrap().to_vec::<f32>();

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::attention::fused_qkv::FusedQkvOps as _;
        use numr::tensor::Tensor;
        let dq_w = Tensor::from_slice(
            &dq.to_vec::<f32>(),
            &[b, num_heads, s, head_dim],
            &wgpu_device,
        )
        .unwrap();
        let dk_w = Tensor::from_slice(
            &dk.to_vec::<f32>(),
            &[b, num_kv_heads, s, head_dim],
            &wgpu_device,
        )
        .unwrap();
        let dv_w = Tensor::from_slice(
            &dv.to_vec::<f32>(),
            &[b, num_kv_heads, s, head_dim],
            &wgpu_device,
        )
        .unwrap();
        let input_w = Tensor::from_slice(&input.to_vec::<f32>(), &[b, s, h], &wgpu_device).unwrap();
        let weight_w =
            Tensor::from_slice(&weight.to_vec::<f32>(), &[total_proj, h], &wgpu_device).unwrap();
        let (wgpu_di, wgpu_dw, wgpu_db) = wgpu_client
            .fused_qkv_projection_bwd(
                &dq_w,
                &dk_w,
                &dv_w,
                &input_w,
                &weight_w,
                true,
                num_heads,
                num_kv_heads,
                head_dim,
            )
            .unwrap();
        assert_parity_f32_relaxed(
            &wgpu_di.to_vec::<f32>(),
            &cpu_di_vec,
            "fused_qkv_bwd d_input: WGPU vs CPU",
        );
        assert_parity_f32_relaxed(
            &wgpu_dw.to_vec::<f32>(),
            &cpu_dw_vec,
            "fused_qkv_bwd d_weight: WGPU vs CPU",
        );
        assert_parity_f32_relaxed(
            &wgpu_db.as_ref().unwrap().to_vec::<f32>(),
            &cpu_db_vec,
            "fused_qkv_bwd d_bias: WGPU vs CPU",
        );
    });

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::attention::fused_qkv::FusedQkvOps as _;
        use numr::tensor::Tensor;
        let dq_c = Tensor::from_slice(
            &dq.to_vec::<f32>(),
            &[b, num_heads, s, head_dim],
            &cuda_device,
        )
        .unwrap();
        let dk_c = Tensor::from_slice(
            &dk.to_vec::<f32>(),
            &[b, num_kv_heads, s, head_dim],
            &cuda_device,
        )
        .unwrap();
        let dv_c = Tensor::from_slice(
            &dv.to_vec::<f32>(),
            &[b, num_kv_heads, s, head_dim],
            &cuda_device,
        )
        .unwrap();
        let input_c = Tensor::from_slice(&input.to_vec::<f32>(), &[b, s, h], &cuda_device).unwrap();
        let weight_c =
            Tensor::from_slice(&weight.to_vec::<f32>(), &[total_proj, h], &cuda_device).unwrap();
        let (cuda_di, cuda_dw, cuda_db) = cuda_client
            .fused_qkv_projection_bwd(
                &dq_c,
                &dk_c,
                &dv_c,
                &input_c,
                &weight_c,
                true,
                num_heads,
                num_kv_heads,
                head_dim,
            )
            .unwrap();
        assert_parity_f32_relaxed(
            &cuda_di.to_vec::<f32>(),
            &cpu_di_vec,
            "fused_qkv_bwd d_input: CUDA vs CPU",
        );
        assert_parity_f32_relaxed(
            &cuda_dw.to_vec::<f32>(),
            &cpu_dw_vec,
            "fused_qkv_bwd d_weight: CUDA vs CPU",
        );
        assert_parity_f32_relaxed(
            &cuda_db.as_ref().unwrap().to_vec::<f32>(),
            &cpu_db_vec,
            "fused_qkv_bwd d_bias: CUDA vs CPU",
        );
    });
}

/// Regression test for the fused CUDA kernel's `h_idx` bug: it computed
/// `h_idx = idx % total` (i.e. `h_idx == idx`), which reads `bias`
/// out of bounds whenever `B*S > 1`. `B*S = 16` here forces that path.
#[test]
fn test_fused_output_projection_residual_with_bias_batched_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (b, s, h, proj_dim) = (2, 8, 256, 256);

    let attn_out = det_tensor(&[b, s, proj_dim], &cpu_device);
    let weight = det_tensor(&[h, proj_dim], &cpu_device);
    let bias = det_tensor(&[h], &cpu_device);
    let residual = det_tensor(&[b, s, h], &cpu_device);

    let cpu_out = cpu_client
        .fused_output_projection_residual(&attn_out, &weight, Some(&bias), &residual)
        .unwrap();
    let cpu_out_vec = cpu_out.to_vec::<f32>();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::attention::fused_qkv::FusedQkvOps as _;
        use numr::tensor::Tensor;
        let attn_c =
            Tensor::from_slice(&attn_out.to_vec::<f32>(), &[b, s, proj_dim], &cuda_device).unwrap();
        let weight_c =
            Tensor::from_slice(&weight.to_vec::<f32>(), &[h, proj_dim], &cuda_device).unwrap();
        let bias_c = Tensor::from_slice(&bias.to_vec::<f32>(), &[h], &cuda_device).unwrap();
        let res_c =
            Tensor::from_slice(&residual.to_vec::<f32>(), &[b, s, h], &cuda_device).unwrap();
        let cuda_out = cuda_client
            .fused_output_projection_residual(&attn_c, &weight_c, Some(&bias_c), &res_c)
            .unwrap();
        assert_parity_f32_relaxed(
            &cuda_out.to_vec::<f32>(),
            &cpu_out_vec,
            "fused_output_proj_residual with bias, B*S>1: CUDA vs CPU",
        );
    });
}

/// F32/F64 are the only dtypes the fused CUDA kernel supports; F16 must
/// fall back to the generic elementwise path. Pins that fallback so a
/// future edit cannot silently route F16 into the f32-only kernel.
#[cfg(all(feature = "cuda", feature = "f16"))]
#[test]
fn test_fused_qkv_projection_f16_fallback_parity() {
    use numr::dtype::DType;
    use numr::tensor::Tensor;

    let (cpu_client, cpu_device) = setup_cpu();
    let (b, s, h) = (2, 4, 64);
    let (num_heads, num_kv_heads, head_dim) = (4, 2, 16);
    let total_proj = num_heads * head_dim + 2 * num_kv_heads * head_dim;

    let input = det_tensor(&[b, s, h], &cpu_device);
    let weight = det_tensor(&[total_proj, h], &cpu_device);

    let (cpu_q, cpu_k, cpu_v) = cpu_client
        .fused_qkv_projection(&input, &weight, None, num_heads, num_kv_heads, head_dim)
        .unwrap();
    let cpu_q_vec = cpu_q.to_vec::<f32>();
    let cpu_k_vec = cpu_k.to_vec::<f32>();
    let cpu_v_vec = cpu_v.to_vec::<f32>();

    with_cuda_backend(|cuda_client, cuda_device| {
        // `to_dtype` narrows the F32 fixtures to F16 — `from_slice` cannot
        // build `half::f16` tensors directly since it is not a numr `Element`.
        let input_c = Tensor::from_slice(&input.to_vec::<f32>(), &[b, s, h], &cuda_device)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let weight_c = Tensor::from_slice(&weight.to_vec::<f32>(), &[total_proj, h], &cuda_device)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let (cuda_q, cuda_k, cuda_v) = cuda_client
            .fused_qkv_projection(&input_c, &weight_c, None, num_heads, num_kv_heads, head_dim)
            .unwrap();
        assert_parity_f32_tol(
            &cuda_q.to_dtype(DType::F32).unwrap().to_vec::<f32>(),
            &cpu_q_vec,
            "fused_qkv_proj F16 fallback Q: CUDA vs CPU",
            0.05,
            0.02,
        );
        assert_parity_f32_tol(
            &cuda_k.to_dtype(DType::F32).unwrap().to_vec::<f32>(),
            &cpu_k_vec,
            "fused_qkv_proj F16 fallback K: CUDA vs CPU",
            0.05,
            0.02,
        );
        assert_parity_f32_tol(
            &cuda_v.to_dtype(DType::F32).unwrap().to_vec::<f32>(),
            &cpu_v_vec,
            "fused_qkv_proj F16 fallback V: CUDA vs CPU",
            0.05,
            0.02,
        );
    });
}
