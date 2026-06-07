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
