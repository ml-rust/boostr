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
        let q_c = Tensor::from_slice(
            &q.to_vec::<f32>(),
            &[total_tokens, num_heads, head_dim],
            &cuda_device,
        );
        let k_c = Tensor::from_slice(
            &k.to_vec::<f32>(),
            &[total_tokens, num_heads, head_dim],
            &cuda_device,
        );
        let v_c = Tensor::from_slice(
            &v.to_vec::<f32>(),
            &[total_tokens, num_heads, head_dim],
            &cuda_device,
        );
        let csq = Tensor::from_slice(&cu_data, &[batch_size + 1], &cuda_device);
        let csk = Tensor::from_slice(&cu_data, &[batch_size + 1], &cuda_device);
        let (out, _) = cuda_client
            .varlen_attention_fwd(
                &q_c, &k_c, &v_c, &csq, &csk, batch_size, num_heads, max_seqlen, max_seqlen,
                head_dim, false,
            )
            .unwrap();
        assert_parity_f32(&out.to_vec::<f32>(), &cpu_out_vec, "varlen_fwd CUDA vs CPU");
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::attention::varlen_attention::VarLenAttentionOps as _;
        use numr::tensor::Tensor;
        let q_w = Tensor::from_slice(
            &q.to_vec::<f32>(),
            &[total_tokens, num_heads, head_dim],
            &wgpu_device,
        );
        let k_w = Tensor::from_slice(
            &k.to_vec::<f32>(),
            &[total_tokens, num_heads, head_dim],
            &wgpu_device,
        );
        let v_w = Tensor::from_slice(
            &v.to_vec::<f32>(),
            &[total_tokens, num_heads, head_dim],
            &wgpu_device,
        );
        let csq = Tensor::from_slice(&cu_data, &[batch_size + 1], &wgpu_device);
        let csk = Tensor::from_slice(&cu_data, &[batch_size + 1], &wgpu_device);
        let (out, _) = wgpu_client
            .varlen_attention_fwd(
                &q_w, &k_w, &v_w, &csq, &csk, batch_size, num_heads, max_seqlen, max_seqlen,
                head_dim, false,
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
            max_seqlen,
            max_seqlen,
            head_dim,
            false,
        )
        .unwrap();
    let cpu_dq_vec = cpu_dq.to_vec::<f32>();
    let cpu_dk_vec = cpu_dk.to_vec::<f32>();
    let cpu_dv_vec = cpu_dv.to_vec::<f32>();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::attention::varlen_attention::VarLenAttentionOps as _;
        use numr::tensor::Tensor;
        let q_c = Tensor::from_slice(
            &q.to_vec::<f32>(),
            &[total_tokens, num_heads, head_dim],
            &cuda_device,
        );
        let k_c = Tensor::from_slice(
            &k.to_vec::<f32>(),
            &[total_tokens, num_heads, head_dim],
            &cuda_device,
        );
        let v_c = Tensor::from_slice(
            &v.to_vec::<f32>(),
            &[total_tokens, num_heads, head_dim],
            &cuda_device,
        );
        let csq = Tensor::from_slice(&cu_data, &[batch_size + 1], &cuda_device);
        let csk = Tensor::from_slice(&cu_data, &[batch_size + 1], &cuda_device);
        let (out_c, lse_c) = cuda_client
            .varlen_attention_fwd(
                &q_c, &k_c, &v_c, &csq, &csk, batch_size, num_heads, max_seqlen, max_seqlen,
                head_dim, false,
            )
            .unwrap();
        let dout_c = Tensor::from_slice(
            &dout.to_vec::<f32>(),
            &[total_tokens, num_heads, head_dim],
            &cuda_device,
        );
        let (dq, dk, dv) = cuda_client
            .varlen_attention_bwd(
                &dout_c, &q_c, &k_c, &v_c, &out_c, &lse_c, &csq, &csk, batch_size, num_heads,
                max_seqlen, max_seqlen, head_dim, false,
            )
            .unwrap();
        assert_parity_f32_relaxed(
            &dq.to_vec::<f32>(),
            &cpu_dq_vec,
            "varlen_bwd dQ CUDA vs CPU",
        );
        assert_parity_f32_relaxed(
            &dk.to_vec::<f32>(),
            &cpu_dk_vec,
            "varlen_bwd dK CUDA vs CPU",
        );
        assert_parity_f32_relaxed(
            &dv.to_vec::<f32>(),
            &cpu_dv_vec,
            "varlen_bwd dV CUDA vs CPU",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::attention::varlen_attention::VarLenAttentionOps as _;
        use numr::tensor::Tensor;
        let q_w = Tensor::from_slice(
            &q.to_vec::<f32>(),
            &[total_tokens, num_heads, head_dim],
            &wgpu_device,
        );
        let k_w = Tensor::from_slice(
            &k.to_vec::<f32>(),
            &[total_tokens, num_heads, head_dim],
            &wgpu_device,
        );
        let v_w = Tensor::from_slice(
            &v.to_vec::<f32>(),
            &[total_tokens, num_heads, head_dim],
            &wgpu_device,
        );
        let csq = Tensor::from_slice(&cu_data, &[batch_size + 1], &wgpu_device);
        let csk = Tensor::from_slice(&cu_data, &[batch_size + 1], &wgpu_device);
        let (_, _) = wgpu_client
            .varlen_attention_fwd(
                &q_w, &k_w, &v_w, &csq, &csk, batch_size, num_heads, max_seqlen, max_seqlen,
                head_dim, false,
            )
            .unwrap();
        // BWD not yet implemented on WebGPU — skip gracefully
        eprintln!("varlen_attention_bwd not implemented on WebGPU, skipping");
    });
}
