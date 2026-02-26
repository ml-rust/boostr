//! Backend parity tests for fused INT4 operations (SwiGLU, QKV)

use super::helpers::*;
use boostr::{FusedQuantOps, QuantMatmulOps};
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

/// Create AWQ INT4 weight data for testing
fn create_awq_weights(
    k: usize,
    n: usize,
    seed: usize,
    device: &CpuDevice,
) -> (Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>) {
    let awq_shifts = [0u32, 16, 4, 20, 8, 24, 12, 28];
    let n_packed = n / 8;
    let num_groups = 1;

    let mut qweight_data = vec![0u32; k * n_packed];
    for ki in 0..k {
        for pj in 0..n_packed {
            let mut packed = 0u32;
            for (sub, &shift) in awq_shifts.iter().enumerate() {
                let val = ((seed + ki * n_packed + pj + sub) % 16) as u32;
                packed |= val << shift;
            }
            qweight_data[ki * n_packed + pj] = packed;
        }
    }
    let qweight = Tensor::<CpuRuntime>::from_slice(
        bytemuck::cast_slice::<u32, f32>(&qweight_data),
        &[k, n_packed],
        device,
    );

    let scales_data: Vec<f32> = (0..num_groups * n)
        .map(|i| 0.01 + ((seed + i) as f32 * 0.001).sin().abs() * 0.1)
        .collect();
    let zeros_data: Vec<f32> = (0..num_groups * n)
        .map(|i| 7.0 + ((seed + i) as f32 * 0.003).cos() * 0.5)
        .collect();
    let scales = Tensor::<CpuRuntime>::from_slice(&scales_data, &[num_groups, n], device);
    let zeros = Tensor::<CpuRuntime>::from_slice(&zeros_data, &[num_groups, n], device);

    (qweight, scales, zeros)
}

#[test]
fn test_fused_int4_swiglu_parity() {
    let (cpu_client, cpu_device) = setup_cpu();

    let m = 2;
    let k = 32;
    let n = 16;
    let group_size = 32;

    let input = det_tensor(&[m, k], &cpu_device);
    let (gate_qw, gate_sc, gate_zr) = create_awq_weights(k, n, 0, &cpu_device);
    let (up_qw, up_sc, up_zr) = create_awq_weights(k, n, 42, &cpu_device);

    // Method 1: Fused SwiGLU
    let fused_result = cpu_client
        .fused_int4_swiglu(
            &input, &gate_qw, &gate_sc, &gate_zr, &up_qw, &up_sc, &up_zr, group_size,
        )
        .unwrap();
    assert_eq!(fused_result.shape(), &[m, n]);

    // Method 2: Separate gate + up + SwiGLU
    let gate_result = cpu_client
        .int4_gemm(&input, &gate_qw, &gate_sc, &gate_zr, group_size)
        .unwrap();
    let up_result = cpu_client
        .int4_gemm(&input, &up_qw, &up_sc, &up_zr, group_size)
        .unwrap();

    // Manual SwiGLU: silu(gate) * up
    let gate_vec = gate_result.to_vec::<f32>();
    let up_vec = up_result.to_vec::<f32>();
    let expected: Vec<f32> = gate_vec
        .iter()
        .zip(up_vec.iter())
        .map(|(&g, &u)| {
            let silu_g = g / (1.0 + (-g).exp());
            silu_g * u
        })
        .collect();

    let cpu_fused_vec = fused_result.to_vec::<f32>();
    assert_parity_f32_relaxed(&cpu_fused_vec, &expected, "fused_int4_swiglu vs separate");

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::FusedQuantOps as _;
        use numr::tensor::Tensor;

        let to_cuda = |t: &Tensor<CpuRuntime>| -> Tensor<numr::runtime::cuda::CudaRuntime> {
            Tensor::from_slice(&t.to_vec::<f32>(), t.shape(), &cuda_device)
        };
        let input_c = to_cuda(&input);
        let gqw_c = to_cuda(&gate_qw);
        let gsc_c = to_cuda(&gate_sc);
        let gzr_c = to_cuda(&gate_zr);
        let uqw_c = to_cuda(&up_qw);
        let usc_c = to_cuda(&up_sc);
        let uzr_c = to_cuda(&up_zr);

        let cuda_result = cuda_client
            .fused_int4_swiglu(
                &input_c, &gqw_c, &gsc_c, &gzr_c, &uqw_c, &usc_c, &uzr_c, group_size,
            )
            .unwrap();
        assert_parity_f32_relaxed(
            &cuda_result.to_vec::<f32>(),
            &cpu_fused_vec,
            "fused_int4_swiglu CUDA vs CPU",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::FusedQuantOps as _;
        use numr::tensor::Tensor;

        let to_wgpu = |t: &Tensor<CpuRuntime>| -> Tensor<numr::runtime::wgpu::WgpuRuntime> {
            Tensor::from_slice(&t.to_vec::<f32>(), t.shape(), &wgpu_device)
        };
        let input_w = to_wgpu(&input);
        let gqw_w = to_wgpu(&gate_qw);
        let gsc_w = to_wgpu(&gate_sc);
        let gzr_w = to_wgpu(&gate_zr);
        let uqw_w = to_wgpu(&up_qw);
        let usc_w = to_wgpu(&up_sc);
        let uzr_w = to_wgpu(&up_zr);

        let wgpu_result = wgpu_client
            .fused_int4_swiglu(
                &input_w, &gqw_w, &gsc_w, &gzr_w, &uqw_w, &usc_w, &uzr_w, group_size,
            )
            .unwrap();
        assert_parity_f32_relaxed(
            &wgpu_result.to_vec::<f32>(),
            &cpu_fused_vec,
            "fused_int4_swiglu WebGPU vs CPU",
        );
    });
}

#[test]
fn test_fused_int4_qkv_parity() {
    let (cpu_client, cpu_device) = setup_cpu();

    let m = 2;
    let k = 32;
    let nq = 16;
    let nkv = 8;
    let group_size = 32;

    let input = det_tensor(&[m, k], &cpu_device);
    let (qw_q, sc_q, zr_q) = create_awq_weights(k, nq, 0, &cpu_device);
    let (qw_k, sc_k, zr_k) = create_awq_weights(k, nkv, 10, &cpu_device);
    let (qw_v, sc_v, zr_v) = create_awq_weights(k, nkv, 20, &cpu_device);

    // Method 1: Fused QKV
    let (q_fused, k_fused, v_fused) = cpu_client
        .fused_int4_qkv(
            &input, &qw_q, &sc_q, &zr_q, &qw_k, &sc_k, &zr_k, &qw_v, &sc_v, &zr_v, group_size,
        )
        .unwrap();
    assert_eq!(q_fused.shape(), &[m, nq]);
    assert_eq!(k_fused.shape(), &[m, nkv]);
    assert_eq!(v_fused.shape(), &[m, nkv]);

    // Method 2: Separate Q, K, V projections
    let q_sep = cpu_client
        .int4_gemm(&input, &qw_q, &sc_q, &zr_q, group_size)
        .unwrap();
    let k_sep = cpu_client
        .int4_gemm(&input, &qw_k, &sc_k, &zr_k, group_size)
        .unwrap();
    let v_sep = cpu_client
        .int4_gemm(&input, &qw_v, &sc_v, &zr_v, group_size)
        .unwrap();

    let cpu_q_vec = q_fused.to_vec::<f32>();
    let cpu_k_vec = k_fused.to_vec::<f32>();
    let cpu_v_vec = v_fused.to_vec::<f32>();

    assert_parity_f32_relaxed(
        &cpu_q_vec,
        &q_sep.to_vec::<f32>(),
        "fused_int4_qkv Q vs separate",
    );
    assert_parity_f32_relaxed(
        &cpu_k_vec,
        &k_sep.to_vec::<f32>(),
        "fused_int4_qkv K vs separate",
    );
    assert_parity_f32_relaxed(
        &cpu_v_vec,
        &v_sep.to_vec::<f32>(),
        "fused_int4_qkv V vs separate",
    );

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::FusedQuantOps as _;
        use numr::tensor::Tensor;

        let to_cuda = |t: &Tensor<CpuRuntime>| -> Tensor<numr::runtime::cuda::CudaRuntime> {
            Tensor::from_slice(&t.to_vec::<f32>(), t.shape(), &cuda_device)
        };

        let (cuda_q, cuda_k, cuda_v) = cuda_client
            .fused_int4_qkv(
                &to_cuda(&input),
                &to_cuda(&qw_q),
                &to_cuda(&sc_q),
                &to_cuda(&zr_q),
                &to_cuda(&qw_k),
                &to_cuda(&sc_k),
                &to_cuda(&zr_k),
                &to_cuda(&qw_v),
                &to_cuda(&sc_v),
                &to_cuda(&zr_v),
                group_size,
            )
            .unwrap();

        assert_parity_f32_relaxed(&cuda_q.to_vec::<f32>(), &cpu_q_vec, "QKV Q CUDA vs CPU");
        assert_parity_f32_relaxed(&cuda_k.to_vec::<f32>(), &cpu_k_vec, "QKV K CUDA vs CPU");
        assert_parity_f32_relaxed(&cuda_v.to_vec::<f32>(), &cpu_v_vec, "QKV V CUDA vs CPU");
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::FusedQuantOps as _;
        use numr::tensor::Tensor;

        let to_wgpu = |t: &Tensor<CpuRuntime>| -> Tensor<numr::runtime::wgpu::WgpuRuntime> {
            Tensor::from_slice(&t.to_vec::<f32>(), t.shape(), &wgpu_device)
        };

        let (wgpu_q, wgpu_k, wgpu_v) = wgpu_client
            .fused_int4_qkv(
                &to_wgpu(&input),
                &to_wgpu(&qw_q),
                &to_wgpu(&sc_q),
                &to_wgpu(&zr_q),
                &to_wgpu(&qw_k),
                &to_wgpu(&sc_k),
                &to_wgpu(&zr_k),
                &to_wgpu(&qw_v),
                &to_wgpu(&sc_v),
                &to_wgpu(&zr_v),
                group_size,
            )
            .unwrap();

        assert_parity_f32_relaxed(&wgpu_q.to_vec::<f32>(), &cpu_q_vec, "QKV Q WebGPU vs CPU");
        assert_parity_f32_relaxed(&wgpu_k.to_vec::<f32>(), &cpu_k_vec, "QKV K WebGPU vs CPU");
        assert_parity_f32_relaxed(&wgpu_v.to_vec::<f32>(), &cpu_v_vec, "QKV V WebGPU vs CPU");
    });
}
