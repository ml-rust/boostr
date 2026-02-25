//! Backend parity tests for SsmKernelOps.

use super::helpers::*;
use boostr::ops::traits::architecture::ssm_kernels::SsmKernelOps;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

/// Small problem dimensions for testing.
const BATCH: usize = 2;
const SEQLEN: usize = 16;
const NHEADS: usize = 4;
const HEADDIM: usize = 8;
const NGROUPS: usize = 2;
const DSTATE: usize = 4;
const CHUNK_SIZE: usize = 8;

fn make_test_tensors(
    device: &numr::runtime::cpu::CpuDevice,
) -> (
    Tensor<CpuRuntime>,
    Tensor<CpuRuntime>,
    Tensor<CpuRuntime>,
    Tensor<CpuRuntime>,
    Tensor<CpuRuntime>,
) {
    let dt = det_tensor(&[BATCH, SEQLEN, NHEADS], device);
    let a_data: Vec<f32> = (0..NHEADS).map(|i| -0.5 - 0.1 * i as f32).collect();
    let a = Tensor::<CpuRuntime>::from_slice(&a_data, &[NHEADS], device);
    let x = det_tensor(&[BATCH, SEQLEN, NHEADS, HEADDIM], device);
    let b = det_tensor(&[BATCH, SEQLEN, NGROUPS, DSTATE], device);
    let c = det_tensor(&[BATCH, SEQLEN, NGROUPS, DSTATE], device);
    (dt, a, x, b, c)
}

#[test]
fn test_ssd_chunk_cumsum_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (dt, a, _x, _b, _c) = make_test_tensors(&cpu_device);

    let (cpu_dt_out, cpu_da_cumsum) = cpu_client
        .ssd_chunk_cumsum(&dt, &a, None, CHUNK_SIZE, true)
        .unwrap();

    let cpu_dt_vec = cpu_dt_out.to_vec::<f32>();
    let cpu_da_vec = cpu_da_cumsum.to_vec::<f32>();

    // Verify shapes
    let nchunks = SEQLEN.div_ceil(CHUNK_SIZE);
    assert_eq!(cpu_dt_out.shape(), &[BATCH, NHEADS, nchunks, CHUNK_SIZE]);
    assert_eq!(cpu_da_cumsum.shape(), &[BATCH, NHEADS, nchunks, CHUNK_SIZE]);

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::architecture::ssm_kernels::SsmKernelOps as _;
        let dt_c = Tensor::from_slice(&dt.to_vec::<f32>(), dt.shape(), &cuda_device);
        let a_c = Tensor::from_slice(&a.to_vec::<f32>(), a.shape(), &cuda_device);
        let (dt_out_c, da_c) = cuda_client
            .ssd_chunk_cumsum(&dt_c, &a_c, None, CHUNK_SIZE, true)
            .unwrap();
        assert_parity_f32(
            &dt_out_c.to_vec::<f32>(),
            &cpu_dt_vec,
            "cumsum dt CUDA vs CPU",
        );
        assert_parity_f32(&da_c.to_vec::<f32>(), &cpu_da_vec, "cumsum dA CUDA vs CPU");
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::architecture::ssm_kernels::SsmKernelOps as _;
        let dt_w = Tensor::from_slice(&dt.to_vec::<f32>(), dt.shape(), &wgpu_device);
        let a_w = Tensor::from_slice(&a.to_vec::<f32>(), a.shape(), &wgpu_device);
        let (dt_out_w, da_w) = wgpu_client
            .ssd_chunk_cumsum(&dt_w, &a_w, None, CHUNK_SIZE, true)
            .unwrap();
        assert_parity_f32(
            &dt_out_w.to_vec::<f32>(),
            &cpu_dt_vec,
            "cumsum dt WGPU vs CPU",
        );
        assert_parity_f32(&da_w.to_vec::<f32>(), &cpu_da_vec, "cumsum dA WGPU vs CPU");
    });
}

#[test]
fn test_ssd_chunk_cumsum_with_bias_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (dt, a, _x, _b, _c) = make_test_tensors(&cpu_device);
    let bias_data: Vec<f32> = (0..NHEADS).map(|i| 0.1 * i as f32).collect();
    let dt_bias = Tensor::<CpuRuntime>::from_slice(&bias_data, &[NHEADS], &cpu_device);

    let (cpu_dt_out, cpu_da_cumsum) = cpu_client
        .ssd_chunk_cumsum(&dt, &a, Some(&dt_bias), CHUNK_SIZE, true)
        .unwrap();

    let cpu_dt_vec = cpu_dt_out.to_vec::<f32>();
    let cpu_da_vec = cpu_da_cumsum.to_vec::<f32>();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::architecture::ssm_kernels::SsmKernelOps as _;
        let dt_c = Tensor::from_slice(&dt.to_vec::<f32>(), dt.shape(), &cuda_device);
        let a_c = Tensor::from_slice(&a.to_vec::<f32>(), a.shape(), &cuda_device);
        let bias_c = Tensor::from_slice(&dt_bias.to_vec::<f32>(), dt_bias.shape(), &cuda_device);
        let (dt_out_c, da_c) = cuda_client
            .ssd_chunk_cumsum(&dt_c, &a_c, Some(&bias_c), CHUNK_SIZE, true)
            .unwrap();
        assert_parity_f32(
            &dt_out_c.to_vec::<f32>(),
            &cpu_dt_vec,
            "cumsum+bias dt CUDA",
        );
        assert_parity_f32(&da_c.to_vec::<f32>(), &cpu_da_vec, "cumsum+bias dA CUDA");
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::architecture::ssm_kernels::SsmKernelOps as _;
        let dt_w = Tensor::from_slice(&dt.to_vec::<f32>(), dt.shape(), &wgpu_device);
        let a_w = Tensor::from_slice(&a.to_vec::<f32>(), a.shape(), &wgpu_device);
        let bias_w = Tensor::from_slice(&dt_bias.to_vec::<f32>(), dt_bias.shape(), &wgpu_device);
        let (dt_out_w, da_w) = wgpu_client
            .ssd_chunk_cumsum(&dt_w, &a_w, Some(&bias_w), CHUNK_SIZE, true)
            .unwrap();
        assert_parity_f32(
            &dt_out_w.to_vec::<f32>(),
            &cpu_dt_vec,
            "cumsum+bias dt WGPU",
        );
        assert_parity_f32(&da_w.to_vec::<f32>(), &cpu_da_vec, "cumsum+bias dA WGPU");
    });
}

#[test]
fn test_ssd_chunk_state_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (dt, a, x, b, _c) = make_test_tensors(&cpu_device);

    let (cpu_dt_out, cpu_da_cumsum) = cpu_client
        .ssd_chunk_cumsum(&dt, &a, None, CHUNK_SIZE, true)
        .unwrap();

    let cpu_states = cpu_client
        .ssd_chunk_state(&x, &b, &cpu_dt_out, &cpu_da_cumsum)
        .unwrap();
    let cpu_states_vec = cpu_states.to_vec::<f32>();

    let nchunks = SEQLEN.div_ceil(CHUNK_SIZE);
    assert_eq!(
        cpu_states.shape(),
        &[BATCH, nchunks, NHEADS, HEADDIM, DSTATE]
    );

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::architecture::ssm_kernels::SsmKernelOps as _;
        let dt_c = Tensor::from_slice(&dt.to_vec::<f32>(), dt.shape(), &cuda_device);
        let a_c = Tensor::from_slice(&a.to_vec::<f32>(), a.shape(), &cuda_device);
        let x_c = Tensor::from_slice(&x.to_vec::<f32>(), x.shape(), &cuda_device);
        let b_c = Tensor::from_slice(&b.to_vec::<f32>(), b.shape(), &cuda_device);
        let (dt_out_c, da_c) = cuda_client
            .ssd_chunk_cumsum(&dt_c, &a_c, None, CHUNK_SIZE, true)
            .unwrap();
        let states_c = cuda_client
            .ssd_chunk_state(&x_c, &b_c, &dt_out_c, &da_c)
            .unwrap();
        assert_parity_f32_relaxed(
            &states_c.to_vec::<f32>(),
            &cpu_states_vec,
            "chunk_state CUDA vs CPU",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::architecture::ssm_kernels::SsmKernelOps as _;
        let dt_w = Tensor::from_slice(&dt.to_vec::<f32>(), dt.shape(), &wgpu_device);
        let a_w = Tensor::from_slice(&a.to_vec::<f32>(), a.shape(), &wgpu_device);
        let x_w = Tensor::from_slice(&x.to_vec::<f32>(), x.shape(), &wgpu_device);
        let b_w = Tensor::from_slice(&b.to_vec::<f32>(), b.shape(), &wgpu_device);
        let (dt_out_w, da_w) = wgpu_client
            .ssd_chunk_cumsum(&dt_w, &a_w, None, CHUNK_SIZE, true)
            .unwrap();
        let states_w = wgpu_client
            .ssd_chunk_state(&x_w, &b_w, &dt_out_w, &da_w)
            .unwrap();
        assert_parity_f32_relaxed(
            &states_w.to_vec::<f32>(),
            &cpu_states_vec,
            "chunk_state WGPU vs CPU",
        );
    });
}

#[test]
fn test_ssd_state_passing_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (dt, a, x, b, _c) = make_test_tensors(&cpu_device);

    let (cpu_dt_out, cpu_da_cumsum) = cpu_client
        .ssd_chunk_cumsum(&dt, &a, None, CHUNK_SIZE, true)
        .unwrap();
    let cpu_states = cpu_client
        .ssd_chunk_state(&x, &b, &cpu_dt_out, &cpu_da_cumsum)
        .unwrap();
    let cpu_propagated = cpu_client
        .ssd_state_passing(&cpu_states, &cpu_da_cumsum)
        .unwrap();
    let cpu_prop_vec = cpu_propagated.to_vec::<f32>();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::architecture::ssm_kernels::SsmKernelOps as _;
        let dt_c = Tensor::from_slice(&dt.to_vec::<f32>(), dt.shape(), &cuda_device);
        let a_c = Tensor::from_slice(&a.to_vec::<f32>(), a.shape(), &cuda_device);
        let x_c = Tensor::from_slice(&x.to_vec::<f32>(), x.shape(), &cuda_device);
        let b_c = Tensor::from_slice(&b.to_vec::<f32>(), b.shape(), &cuda_device);
        let (dt_out_c, da_c) = cuda_client
            .ssd_chunk_cumsum(&dt_c, &a_c, None, CHUNK_SIZE, true)
            .unwrap();
        let states_c = cuda_client
            .ssd_chunk_state(&x_c, &b_c, &dt_out_c, &da_c)
            .unwrap();
        let prop_c = cuda_client.ssd_state_passing(&states_c, &da_c).unwrap();
        assert_parity_f32_relaxed(
            &prop_c.to_vec::<f32>(),
            &cpu_prop_vec,
            "state_passing CUDA vs CPU",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::architecture::ssm_kernels::SsmKernelOps as _;
        let dt_w = Tensor::from_slice(&dt.to_vec::<f32>(), dt.shape(), &wgpu_device);
        let a_w = Tensor::from_slice(&a.to_vec::<f32>(), a.shape(), &wgpu_device);
        let x_w = Tensor::from_slice(&x.to_vec::<f32>(), x.shape(), &wgpu_device);
        let b_w = Tensor::from_slice(&b.to_vec::<f32>(), b.shape(), &wgpu_device);
        let (dt_out_w, da_w) = wgpu_client
            .ssd_chunk_cumsum(&dt_w, &a_w, None, CHUNK_SIZE, true)
            .unwrap();
        let states_w = wgpu_client
            .ssd_chunk_state(&x_w, &b_w, &dt_out_w, &da_w)
            .unwrap();
        let prop_w = wgpu_client.ssd_state_passing(&states_w, &da_w).unwrap();
        assert_parity_f32_relaxed(
            &prop_w.to_vec::<f32>(),
            &cpu_prop_vec,
            "state_passing WGPU vs CPU",
        );
    });
}

#[test]
fn test_ssd_chunk_scan_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (dt, a, x, b, c) = make_test_tensors(&cpu_device);

    let (cpu_dt_out, cpu_da_cumsum) = cpu_client
        .ssd_chunk_cumsum(&dt, &a, None, CHUNK_SIZE, true)
        .unwrap();
    let cpu_states = cpu_client
        .ssd_chunk_state(&x, &b, &cpu_dt_out, &cpu_da_cumsum)
        .unwrap();
    let cpu_propagated = cpu_client
        .ssd_state_passing(&cpu_states, &cpu_da_cumsum)
        .unwrap();
    let cpu_output = cpu_client
        .ssd_chunk_scan(&x, &cpu_propagated, &c, &cpu_da_cumsum, None)
        .unwrap();
    let cpu_out_vec = cpu_output.to_vec::<f32>();

    assert_eq!(cpu_output.shape(), &[BATCH, SEQLEN, NHEADS, HEADDIM]);

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::architecture::ssm_kernels::SsmKernelOps as _;
        let dt_c = Tensor::from_slice(&dt.to_vec::<f32>(), dt.shape(), &cuda_device);
        let a_c = Tensor::from_slice(&a.to_vec::<f32>(), a.shape(), &cuda_device);
        let x_c = Tensor::from_slice(&x.to_vec::<f32>(), x.shape(), &cuda_device);
        let b_c = Tensor::from_slice(&b.to_vec::<f32>(), b.shape(), &cuda_device);
        let c_c = Tensor::from_slice(&c.to_vec::<f32>(), c.shape(), &cuda_device);
        let (dt_out_c, da_c) = cuda_client
            .ssd_chunk_cumsum(&dt_c, &a_c, None, CHUNK_SIZE, true)
            .unwrap();
        let states_c = cuda_client
            .ssd_chunk_state(&x_c, &b_c, &dt_out_c, &da_c)
            .unwrap();
        let prop_c = cuda_client.ssd_state_passing(&states_c, &da_c).unwrap();
        let out_c = cuda_client
            .ssd_chunk_scan(&x_c, &prop_c, &c_c, &da_c, None)
            .unwrap();
        assert_parity_f32_relaxed(
            &out_c.to_vec::<f32>(),
            &cpu_out_vec,
            "chunk_scan CUDA vs CPU",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::architecture::ssm_kernels::SsmKernelOps as _;
        let dt_w = Tensor::from_slice(&dt.to_vec::<f32>(), dt.shape(), &wgpu_device);
        let a_w = Tensor::from_slice(&a.to_vec::<f32>(), a.shape(), &wgpu_device);
        let x_w = Tensor::from_slice(&x.to_vec::<f32>(), x.shape(), &wgpu_device);
        let b_w = Tensor::from_slice(&b.to_vec::<f32>(), b.shape(), &wgpu_device);
        let c_w = Tensor::from_slice(&c.to_vec::<f32>(), c.shape(), &wgpu_device);
        let (dt_out_w, da_w) = wgpu_client
            .ssd_chunk_cumsum(&dt_w, &a_w, None, CHUNK_SIZE, true)
            .unwrap();
        let states_w = wgpu_client
            .ssd_chunk_state(&x_w, &b_w, &dt_out_w, &da_w)
            .unwrap();
        let prop_w = wgpu_client.ssd_state_passing(&states_w, &da_w).unwrap();
        let out_w = wgpu_client
            .ssd_chunk_scan(&x_w, &prop_w, &c_w, &da_w, None)
            .unwrap();
        assert_parity_f32_relaxed(
            &out_w.to_vec::<f32>(),
            &cpu_out_vec,
            "chunk_scan WGPU vs CPU",
        );
    });
}

#[test]
fn test_ssd_chunk_scan_with_d_skip_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (dt, a, x, b, c) = make_test_tensors(&cpu_device);
    let d_data: Vec<f32> = (0..NHEADS).map(|i| 0.5 + 0.1 * i as f32).collect();
    let d = Tensor::<CpuRuntime>::from_slice(&d_data, &[NHEADS], &cpu_device);

    let (cpu_dt_out, cpu_da_cumsum) = cpu_client
        .ssd_chunk_cumsum(&dt, &a, None, CHUNK_SIZE, true)
        .unwrap();
    let cpu_states = cpu_client
        .ssd_chunk_state(&x, &b, &cpu_dt_out, &cpu_da_cumsum)
        .unwrap();
    let cpu_propagated = cpu_client
        .ssd_state_passing(&cpu_states, &cpu_da_cumsum)
        .unwrap();
    let cpu_output = cpu_client
        .ssd_chunk_scan(&x, &cpu_propagated, &c, &cpu_da_cumsum, Some(&d))
        .unwrap();
    let cpu_out_vec = cpu_output.to_vec::<f32>();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::architecture::ssm_kernels::SsmKernelOps as _;
        let dt_c = Tensor::from_slice(&dt.to_vec::<f32>(), dt.shape(), &cuda_device);
        let a_c = Tensor::from_slice(&a.to_vec::<f32>(), a.shape(), &cuda_device);
        let x_c = Tensor::from_slice(&x.to_vec::<f32>(), x.shape(), &cuda_device);
        let b_c = Tensor::from_slice(&b.to_vec::<f32>(), b.shape(), &cuda_device);
        let c_c = Tensor::from_slice(&c.to_vec::<f32>(), c.shape(), &cuda_device);
        let d_c = Tensor::from_slice(&d.to_vec::<f32>(), d.shape(), &cuda_device);
        let (dt_out_c, da_c) = cuda_client
            .ssd_chunk_cumsum(&dt_c, &a_c, None, CHUNK_SIZE, true)
            .unwrap();
        let states_c = cuda_client
            .ssd_chunk_state(&x_c, &b_c, &dt_out_c, &da_c)
            .unwrap();
        let prop_c = cuda_client.ssd_state_passing(&states_c, &da_c).unwrap();
        let out_c = cuda_client
            .ssd_chunk_scan(&x_c, &prop_c, &c_c, &da_c, Some(&d_c))
            .unwrap();
        assert_parity_f32_relaxed(
            &out_c.to_vec::<f32>(),
            &cpu_out_vec,
            "chunk_scan+D CUDA vs CPU",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::architecture::ssm_kernels::SsmKernelOps as _;
        let dt_w = Tensor::from_slice(&dt.to_vec::<f32>(), dt.shape(), &wgpu_device);
        let a_w = Tensor::from_slice(&a.to_vec::<f32>(), a.shape(), &wgpu_device);
        let x_w = Tensor::from_slice(&x.to_vec::<f32>(), x.shape(), &wgpu_device);
        let b_w = Tensor::from_slice(&b.to_vec::<f32>(), b.shape(), &wgpu_device);
        let c_w = Tensor::from_slice(&c.to_vec::<f32>(), c.shape(), &wgpu_device);
        let d_w = Tensor::from_slice(&d.to_vec::<f32>(), d.shape(), &wgpu_device);
        let (dt_out_w, da_w) = wgpu_client
            .ssd_chunk_cumsum(&dt_w, &a_w, None, CHUNK_SIZE, true)
            .unwrap();
        let states_w = wgpu_client
            .ssd_chunk_state(&x_w, &b_w, &dt_out_w, &da_w)
            .unwrap();
        let prop_w = wgpu_client.ssd_state_passing(&states_w, &da_w).unwrap();
        let out_w = wgpu_client
            .ssd_chunk_scan(&x_w, &prop_w, &c_w, &da_w, Some(&d_w))
            .unwrap();
        assert_parity_f32_relaxed(
            &out_w.to_vec::<f32>(),
            &cpu_out_vec,
            "chunk_scan+D WGPU vs CPU",
        );
    });
}

#[test]
fn test_ssd_end_to_end_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (dt, a, x, b, c) = make_test_tensors(&cpu_device);
    let d_data: Vec<f32> = (0..NHEADS).map(|i| 0.3 + 0.05 * i as f32).collect();
    let d = Tensor::<CpuRuntime>::from_slice(&d_data, &[NHEADS], &cpu_device);
    let bias_data: Vec<f32> = (0..NHEADS).map(|i| 0.1 * i as f32).collect();
    let dt_bias = Tensor::<CpuRuntime>::from_slice(&bias_data, &[NHEADS], &cpu_device);

    // Full pipeline: cumsum → chunk_state → state_passing → chunk_scan
    let (cpu_dt_out, cpu_da) = cpu_client
        .ssd_chunk_cumsum(&dt, &a, Some(&dt_bias), CHUNK_SIZE, true)
        .unwrap();
    let cpu_states = cpu_client
        .ssd_chunk_state(&x, &b, &cpu_dt_out, &cpu_da)
        .unwrap();
    let cpu_propagated = cpu_client.ssd_state_passing(&cpu_states, &cpu_da).unwrap();
    let cpu_output = cpu_client
        .ssd_chunk_scan(&x, &cpu_propagated, &c, &cpu_da, Some(&d))
        .unwrap();
    let cpu_out_vec = cpu_output.to_vec::<f32>();

    // Verify output is non-trivial
    let any_nonzero = cpu_out_vec.iter().any(|&v| v.abs() > 1e-10);
    assert!(any_nonzero, "End-to-end output is all zeros — likely a bug");

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::architecture::ssm_kernels::SsmKernelOps as _;
        let dt_c = Tensor::from_slice(&dt.to_vec::<f32>(), dt.shape(), &cuda_device);
        let a_c = Tensor::from_slice(&a.to_vec::<f32>(), a.shape(), &cuda_device);
        let x_c = Tensor::from_slice(&x.to_vec::<f32>(), x.shape(), &cuda_device);
        let b_c = Tensor::from_slice(&b.to_vec::<f32>(), b.shape(), &cuda_device);
        let c_c = Tensor::from_slice(&c.to_vec::<f32>(), c.shape(), &cuda_device);
        let d_c = Tensor::from_slice(&d.to_vec::<f32>(), d.shape(), &cuda_device);
        let bias_c = Tensor::from_slice(&dt_bias.to_vec::<f32>(), dt_bias.shape(), &cuda_device);
        let (dt_out_c, da_c) = cuda_client
            .ssd_chunk_cumsum(&dt_c, &a_c, Some(&bias_c), CHUNK_SIZE, true)
            .unwrap();
        let states_c = cuda_client
            .ssd_chunk_state(&x_c, &b_c, &dt_out_c, &da_c)
            .unwrap();
        let prop_c = cuda_client.ssd_state_passing(&states_c, &da_c).unwrap();
        let out_c = cuda_client
            .ssd_chunk_scan(&x_c, &prop_c, &c_c, &da_c, Some(&d_c))
            .unwrap();
        assert_parity_f32_relaxed(
            &out_c.to_vec::<f32>(),
            &cpu_out_vec,
            "end-to-end CUDA vs CPU",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::architecture::ssm_kernels::SsmKernelOps as _;
        let dt_w = Tensor::from_slice(&dt.to_vec::<f32>(), dt.shape(), &wgpu_device);
        let a_w = Tensor::from_slice(&a.to_vec::<f32>(), a.shape(), &wgpu_device);
        let x_w = Tensor::from_slice(&x.to_vec::<f32>(), x.shape(), &wgpu_device);
        let b_w = Tensor::from_slice(&b.to_vec::<f32>(), b.shape(), &wgpu_device);
        let c_w = Tensor::from_slice(&c.to_vec::<f32>(), c.shape(), &wgpu_device);
        let d_w = Tensor::from_slice(&d.to_vec::<f32>(), d.shape(), &wgpu_device);
        let bias_w = Tensor::from_slice(&dt_bias.to_vec::<f32>(), dt_bias.shape(), &wgpu_device);
        let (dt_out_w, da_w) = wgpu_client
            .ssd_chunk_cumsum(&dt_w, &a_w, Some(&bias_w), CHUNK_SIZE, true)
            .unwrap();
        let states_w = wgpu_client
            .ssd_chunk_state(&x_w, &b_w, &dt_out_w, &da_w)
            .unwrap();
        let prop_w = wgpu_client.ssd_state_passing(&states_w, &da_w).unwrap();
        let out_w = wgpu_client
            .ssd_chunk_scan(&x_w, &prop_w, &c_w, &da_w, Some(&d_w))
            .unwrap();
        assert_parity_f32_relaxed(
            &out_w.to_vec::<f32>(),
            &cpu_out_vec,
            "end-to-end WGPU vs CPU",
        );
    });
}
