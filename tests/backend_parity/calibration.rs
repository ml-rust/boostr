//! Backend parity tests for CalibrationOps.

use super::helpers::*;
use boostr::ops::traits::CalibrationOps;
use numr::dtype::DType;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

// ============================================================================
// AWQ channel scores
// ============================================================================

#[test]
fn test_awq_channel_scores_cpu() {
    let (client, device) = setup_cpu();
    let act = det_tensor(&[4, 8], &device);
    let w = det_tensor(&[6, 8], &device);

    let result = client.awq_channel_scores(&act, &w).unwrap();
    assert_eq!(result.shape(), &[8]);

    let data = result.to_vec::<f32>();
    for &v in &data {
        assert!(v >= 0.0, "scores should be non-negative, got {}", v);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_awq_channel_scores_cuda_parity() {
    let (_cpu_client, cpu_device) = setup_cpu();
    let act_data = det_tensor(&[4, 8], &cpu_device);
    let w_data = det_tensor(&[6, 8], &cpu_device);

    let cpu_result = _cpu_client.awq_channel_scores(&act_data, &w_data).unwrap();
    let cpu_vec = cpu_result.to_vec::<f32>();

    with_cuda_backend(|cuda_client, cuda_device| {
        use numr::runtime::cuda::CudaRuntime;
        let act_c =
            Tensor::<CudaRuntime>::from_slice(&act_data.to_vec::<f32>(), &[4, 8], &cuda_device)
                .unwrap();
        let w_c = Tensor::<CudaRuntime>::from_slice(&w_data.to_vec::<f32>(), &[6, 8], &cuda_device)
            .unwrap();

        let result = cuda_client.awq_channel_scores(&act_c, &w_c).unwrap();
        assert_parity_f32(
            &result.to_vec::<f32>(),
            &cpu_vec,
            "awq_channel_scores CUDA vs CPU",
        );
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_awq_channel_scores_wgpu_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let act_data = det_tensor(&[4, 8], &cpu_device);
    let w_data = det_tensor(&[6, 8], &cpu_device);

    let cpu_result = cpu_client.awq_channel_scores(&act_data, &w_data).unwrap();
    let cpu_vec = cpu_result.to_vec::<f32>();

    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use numr::runtime::wgpu::WgpuRuntime;
        let act_w =
            Tensor::<WgpuRuntime>::from_slice(&act_data.to_vec::<f32>(), &[4, 8], &wgpu_device)
                .unwrap();
        let w_w = Tensor::<WgpuRuntime>::from_slice(&w_data.to_vec::<f32>(), &[6, 8], &wgpu_device)
            .unwrap();

        let result = wgpu_client.awq_channel_scores(&act_w, &w_w).unwrap();
        assert_parity_f32(
            &result.to_vec::<f32>(),
            &cpu_vec,
            "awq_channel_scores WGPU vs CPU",
        );
    });
}

// ============================================================================
// Fisher information
// ============================================================================

#[test]
fn test_fisher_information_cpu() {
    let (client, device) = setup_cpu();
    let grads = det_tensor(&[16, 32], &device);

    let result = client.fisher_information(&grads).unwrap();
    assert_eq!(result.shape(), &[32]);

    let data = result.to_vec::<f32>();
    for &v in &data {
        assert!(v >= 0.0, "Fisher values should be non-negative, got {}", v);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_fisher_information_cuda_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let grad_data = det_tensor(&[16, 32], &cpu_device);

    let cpu_result = cpu_client.fisher_information(&grad_data).unwrap();
    let cpu_vec = cpu_result.to_vec::<f32>();

    with_cuda_backend(|cuda_client, cuda_device| {
        use numr::runtime::cuda::CudaRuntime;
        let grad_c =
            Tensor::<CudaRuntime>::from_slice(&grad_data.to_vec::<f32>(), &[16, 32], &cuda_device)
                .unwrap();

        let result = cuda_client.fisher_information(&grad_c).unwrap();
        assert_parity_f32(
            &result.to_vec::<f32>(),
            &cpu_vec,
            "fisher_information CUDA vs CPU",
        );
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_fisher_information_wgpu_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let grad_data = det_tensor(&[16, 32], &cpu_device);

    let cpu_result = cpu_client.fisher_information(&grad_data).unwrap();
    let cpu_vec = cpu_result.to_vec::<f32>();

    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use numr::runtime::wgpu::WgpuRuntime;
        let grad_w =
            Tensor::<WgpuRuntime>::from_slice(&grad_data.to_vec::<f32>(), &[16, 32], &wgpu_device)
                .unwrap();

        let result = wgpu_client.fisher_information(&grad_w).unwrap();
        assert_parity_f32(
            &result.to_vec::<f32>(),
            &cpu_vec,
            "fisher_information WGPU vs CPU",
        );
    });
}

// ============================================================================
// GPTQ Hessian update
// ============================================================================

#[test]
fn test_gptq_hessian_update_cpu() {
    let (client, device) = setup_cpu();
    let h = Tensor::<CpuRuntime>::zeros(&[8, 8], DType::F32, &device).unwrap();
    let x = det_tensor(&[4, 8], &device);

    let result = client.gptq_hessian_update(&h, &x).unwrap();
    assert_eq!(result.shape(), &[8, 8]);

    // Verify symmetry: H = X^T X should be symmetric
    let data = result.to_vec::<f32>();
    for i in 0..8 {
        for j in 0..8 {
            let diff = (data[i * 8 + j] - data[j * 8 + i]).abs();
            assert!(
                diff < 1e-5,
                "not symmetric at [{},{}]: {} vs {}",
                i,
                j,
                data[i * 8 + j],
                data[j * 8 + i]
            );
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_gptq_hessian_update_cuda_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let h_data = Tensor::<CpuRuntime>::zeros(&[8, 8], DType::F32, &cpu_device).unwrap();
    let x_data = det_tensor(&[4, 8], &cpu_device);

    let cpu_result = cpu_client.gptq_hessian_update(&h_data, &x_data).unwrap();
    let cpu_vec = cpu_result.to_vec::<f32>();

    with_cuda_backend(|cuda_client, cuda_device| {
        use numr::runtime::cuda::CudaRuntime;
        let h_c = Tensor::<CudaRuntime>::from_slice(&h_data.to_vec::<f32>(), &[8, 8], &cuda_device)
            .unwrap();
        let x_c = Tensor::<CudaRuntime>::from_slice(&x_data.to_vec::<f32>(), &[4, 8], &cuda_device)
            .unwrap();

        let result = cuda_client.gptq_hessian_update(&h_c, &x_c).unwrap();
        assert_parity_f32(
            &result.to_vec::<f32>(),
            &cpu_vec,
            "gptq_hessian_update CUDA vs CPU",
        );
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_gptq_hessian_update_wgpu_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let h_data = Tensor::<CpuRuntime>::zeros(&[8, 8], DType::F32, &cpu_device).unwrap();
    let x_data = det_tensor(&[4, 8], &cpu_device);

    let cpu_result = cpu_client.gptq_hessian_update(&h_data, &x_data).unwrap();
    let cpu_vec = cpu_result.to_vec::<f32>();

    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use numr::runtime::wgpu::WgpuRuntime;
        let h_w = Tensor::<WgpuRuntime>::from_slice(&h_data.to_vec::<f32>(), &[8, 8], &wgpu_device)
            .unwrap();
        let x_w = Tensor::<WgpuRuntime>::from_slice(&x_data.to_vec::<f32>(), &[4, 8], &wgpu_device)
            .unwrap();

        let result = wgpu_client.gptq_hessian_update(&h_w, &x_w).unwrap();
        assert_parity_f32(
            &result.to_vec::<f32>(),
            &cpu_vec,
            "gptq_hessian_update WGPU vs CPU",
        );
    });
}

// ============================================================================
// GPTQ column quantization
// ============================================================================

#[test]
fn test_gptq_quantize_column_cpu() {
    let (client, device) = setup_cpu();

    let w_data: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.01).collect();
    let w = Tensor::<CpuRuntime>::from_slice(&w_data, &[8, 16], &device).unwrap();

    // Identity H_inv
    let mut h_inv_data = vec![0.0f32; 16 * 16];
    for i in 0..16 {
        h_inv_data[i * 16 + i] = 1.0;
    }
    let h_inv = Tensor::<CpuRuntime>::from_slice(&h_inv_data, &[16, 16], &device).unwrap();

    let (q, scales, zeros) = client
        .gptq_quantize_column(&w, &h_inv, 4, 4, false)
        .unwrap();

    assert_eq!(q.shape(), &[8, 16]);
    assert_eq!(scales.shape(), &[8, 4]);
    assert_eq!(zeros.shape(), &[8, 4]);

    // Verify shapes and that scales are positive
    let q_data = q.to_vec::<f32>();
    let s_data = scales.to_vec::<f32>();
    assert_eq!(q_data.len(), 128);
    for &s in &s_data {
        assert!(s > 0.0, "scale should be positive, got {}", s);
    }
    // Quantized values should be finite
    for (i, &v) in q_data.iter().enumerate() {
        assert!(v.is_finite(), "non-finite quantized value at {}: {}", i, v);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_gptq_quantize_column_cuda_parity() {
    let (cpu_client, cpu_device) = setup_cpu();

    let w_data: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.01).collect();
    let w = Tensor::<CpuRuntime>::from_slice(&w_data, &[8, 16], &cpu_device).unwrap();

    let mut h_inv_data = vec![0.0f32; 16 * 16];
    for i in 0..16 {
        h_inv_data[i * 16 + i] = 1.0;
    }
    let h_inv = Tensor::<CpuRuntime>::from_slice(&h_inv_data, &[16, 16], &cpu_device).unwrap();

    let (cpu_q, cpu_s, cpu_z) = cpu_client
        .gptq_quantize_column(&w, &h_inv, 4, 4, false)
        .unwrap();
    let cpu_q_vec = cpu_q.to_vec::<f32>();
    let cpu_s_vec = cpu_s.to_vec::<f32>();
    let cpu_z_vec = cpu_z.to_vec::<f32>();

    with_cuda_backend(|cuda_client, cuda_device| {
        use numr::runtime::cuda::CudaRuntime;
        let w_c = Tensor::<CudaRuntime>::from_slice(&w_data, &[8, 16], &cuda_device).unwrap();
        let h_c = Tensor::<CudaRuntime>::from_slice(&h_inv_data, &[16, 16], &cuda_device).unwrap();

        let (q, s, z) = cuda_client
            .gptq_quantize_column(&w_c, &h_c, 4, 4, false)
            .unwrap();
        // Relaxed tolerance — accumulated quantization error across columns
        assert_parity_f32_relaxed(
            &q.to_vec::<f32>(),
            &cpu_q_vec,
            "gptq_quantize_column CUDA vs CPU (q)",
        );
        assert_parity_f32_relaxed(
            &s.to_vec::<f32>(),
            &cpu_s_vec,
            "gptq_quantize_column CUDA vs CPU (scales)",
        );
        assert_parity_f32_relaxed(
            &z.to_vec::<f32>(),
            &cpu_z_vec,
            "gptq_quantize_column CUDA vs CPU (zeros)",
        );
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_gptq_quantize_column_wgpu_parity() {
    let (cpu_client, cpu_device) = setup_cpu();

    let w_data: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.01).collect();
    let w = Tensor::<CpuRuntime>::from_slice(&w_data, &[8, 16], &cpu_device).unwrap();

    let mut h_inv_data = vec![0.0f32; 16 * 16];
    for i in 0..16 {
        h_inv_data[i * 16 + i] = 1.0;
    }
    let h_inv = Tensor::<CpuRuntime>::from_slice(&h_inv_data, &[16, 16], &cpu_device).unwrap();

    let (cpu_q, cpu_s, cpu_z) = cpu_client
        .gptq_quantize_column(&w, &h_inv, 4, 4, false)
        .unwrap();
    let cpu_q_vec = cpu_q.to_vec::<f32>();
    let cpu_s_vec = cpu_s.to_vec::<f32>();
    let cpu_z_vec = cpu_z.to_vec::<f32>();

    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use numr::runtime::wgpu::WgpuRuntime;
        let w_w = Tensor::<WgpuRuntime>::from_slice(&w_data, &[8, 16], &wgpu_device).unwrap();
        let h_w = Tensor::<WgpuRuntime>::from_slice(&h_inv_data, &[16, 16], &wgpu_device).unwrap();

        let (q, s, z) = wgpu_client
            .gptq_quantize_column(&w_w, &h_w, 4, 4, false)
            .unwrap();
        assert_parity_f32_relaxed(
            &q.to_vec::<f32>(),
            &cpu_q_vec,
            "gptq_quantize_column WGPU vs CPU (q)",
        );
        assert_parity_f32_relaxed(
            &s.to_vec::<f32>(),
            &cpu_s_vec,
            "gptq_quantize_column WGPU vs CPU (scales)",
        );
        assert_parity_f32_relaxed(
            &z.to_vec::<f32>(),
            &cpu_z_vec,
            "gptq_quantize_column WGPU vs CPU (zeros)",
        );
    });
}

// ============================================================================
// F16/BF16 accumulation-precision defect (CUDA)
//
// `fisher_accumulate_*` and `awq_score_reduce_*` issue one atomic per matrix
// element straight into F16/BF16 output storage (a CAS loop that rounds the
// running sum to the storage mantissa on every add), with no block-level
// reduction. Squared-gradient / abs-weight terms are strictly positive, so
// this is a systematic UNDERCOUNT that grows with the accumulation count,
// not noise — once the running sum outgrows a term by more than the mantissa
// resolves, further terms round to a no-op.
//
// The comparison below isolates this from ordinary input-rounding error: the
// SAME already-rounded (F16/BF16 -> F32) values are run once through the
// CUDA half-precision kernel and once through the CPU F32 reference path, so
// any divergence is accumulation loss, not quantization of the inputs. `N` /
// `M` (the swept dimension) is exactly the reduction count: `N` for
// `fisher_information` (gradient samples), `M` for `awq_channel_scores`
// (weight rows in `awq_score_reduce_*`).
//
// Tolerance derivation: a CORRECT kernel accumulates in fp32 and rounds to
// half storage exactly ONCE (the final write), so the only expected
// deviation from the F32 reference is that single rounding — bounded by the
// dtype's relative machine epsilon (F16 mantissa 10 bits -> 2^-11, BF16
// mantissa 7 bits -> 2^-8, per the existing `mqa_gqa_attention.rs` table).
// `rtol`/`atol` below use 3x that eps as headroom for the extra ops (square,
// mean-divide) — anything beyond that is accumulation loss, not rounding.
//
// Prediction (stall point where relative error crosses the tolerance):
// BF16 around N/M ~ 256 (7-bit mantissa exhausts fastest), F16 around
// N/M ~ 2000-4000 (10-bit mantissa). Error sign should be consistently
// NEGATIVE (undercount) and its magnitude should GROW monotonically with
// N/M — that trend, not a single large-N failure, is what proves the
// accumulation mechanism (vs. e.g. a one-off rounding artifact).
//
// This test failed against the pre-fix kernel (the atomic-into-half-storage
// accumulation described above) and passes against the fixed one below — it
// exists to give before/after evidence for the fix. Do not loosen the
// tolerance to make it pass.
// ============================================================================

#[cfg(feature = "cuda")]
fn fisher_accum_defect_case(dtype: DType, eps: f32, dtype_name: &str) {
    if !cfg!(feature = "f16") {
        eprintln!(
            "SKIPPED: fisher_information/{dtype_name} accumulation defect — boostr built \
             without the `f16` feature, so {:?} tensors cannot be constructed",
            dtype
        );
        return;
    }
    with_cuda_backend(|cuda_client, cuda_device| {
        use numr::runtime::cuda::CudaRuntime;
        let (cpu_client, cpu_device) = setup_cpu();
        const P: usize = 4;
        let ns = [64usize, 256, 1024, 4096];
        let mut failures: Vec<String> = Vec::new();

        for &n in &ns {
            let raw: Vec<f32> = (0..n * P)
                .map(|i| 0.4 + 0.35 * ((i as f32) * 0.083).sin())
                .collect();
            let grad_f32 = Tensor::<CpuRuntime>::from_slice(&raw, &[n, P], &cpu_device).unwrap();
            let grad_rounded = grad_f32
                .to_dtype(dtype)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap();
            let rounded_vec = grad_rounded.to_vec::<f32>();

            // F32 reference accumulation over the SAME already-rounded values.
            let reference = cpu_client.fisher_information(&grad_rounded).unwrap();
            let ref_vec = reference.to_vec::<f32>();

            // CUDA half-precision accumulation path (the kernel under test).
            let grad_c = Tensor::<CudaRuntime>::from_slice(&rounded_vec, &[n, P], &cuda_device)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let actual = cuda_client.fisher_information(&grad_c).unwrap();
            let actual_vec = actual.to_dtype(DType::F32).unwrap().to_vec::<f32>();

            let max_abs = actual_vec
                .iter()
                .zip(ref_vec.iter())
                .map(|(a, r)| (a - r).abs())
                .fold(0.0f32, f32::max);
            let mean_signed_rel_err: f32 = actual_vec
                .iter()
                .zip(ref_vec.iter())
                .map(|(a, r)| (a - r) / r)
                .sum::<f32>()
                / P as f32;
            let ref_mag: f32 = ref_vec.iter().map(|v| v.abs()).sum::<f32>() / P as f32;
            let rtol = 3.0 * eps;
            let atol = 3.0 * eps * ref_mag.max(1e-6);

            println!(
                "CALIB_DIAG op=fisher_information kernel=fisher_accumulate_{dtype_name} \
                 dtype={dtype_name} n={n} p={P} max_abs={max_abs:.6e} \
                 mean_signed_rel_err={mean_signed_rel_err:.6e} ref_mag={ref_mag:.6e} \
                 rtol={rtol:.6e} atol={atol:.6e}"
            );

            let mut case_ok = true;
            for (i, (a, r)) in actual_vec.iter().zip(ref_vec.iter()).enumerate() {
                let diff = (a - r).abs();
                let tol = atol + rtol * r.abs();
                if diff > tol {
                    case_ok = false;
                    if failures.len() < 4 {
                        failures.push(format!(
                            "n={n} idx={i}: actual={a} ref={r} diff={diff:.6e} tol={tol:.6e}"
                        ));
                    }
                }
            }
            if !case_ok {
                failures.push(format!("n={n}: FAILED (see CALIB_DIAG line above)"));
            }
        }

        assert!(
            failures.is_empty(),
            "fisher_information/{dtype_name}: CUDA accumulation diverges from the F32 \
             reference beyond a single-rounding tolerance — atomic accumulation into half \
             storage undercounts as N grows:\n{}",
            failures.join("\n")
        );
    });
}

#[cfg(feature = "cuda")]
#[test]
fn test_fisher_information_cuda_f16_accum_defect() {
    fisher_accum_defect_case(DType::F16, 2f32.powi(-11), "f16");
}

#[cfg(feature = "cuda")]
#[test]
fn test_fisher_information_cuda_bf16_accum_defect() {
    fisher_accum_defect_case(DType::BF16, 2f32.powi(-8), "bf16");
}

#[cfg(feature = "cuda")]
fn awq_score_reduce_accum_defect_case(dtype: DType, eps: f32, dtype_name: &str) {
    if !cfg!(feature = "f16") {
        eprintln!(
            "SKIPPED: awq_channel_scores/{dtype_name} accumulation defect — boostr built \
             without the `f16` feature, so {:?} tensors cannot be constructed",
            dtype
        );
        return;
    }
    with_cuda_backend(|cuda_client, cuda_device| {
        use numr::runtime::cuda::CudaRuntime;
        let (cpu_client, cpu_device) = setup_cpu();
        const K: usize = 4;
        const N_ACT: usize = 4; // activation rows — fixed; NOT the swept accumulation dim
        let ms = [64usize, 256, 1024, 4096]; // M = weight rows = awq_score_reduce's accumulation count
        let mut failures: Vec<String> = Vec::new();

        // Activations (and their act_scale) are shared across every M — build once.
        let act_raw: Vec<f32> = (0..N_ACT * K)
            .map(|i| 0.2 + 0.6 * ((i as f32) * 0.29).sin().abs())
            .collect();
        let act_f32 = Tensor::<CpuRuntime>::from_slice(&act_raw, &[N_ACT, K], &cpu_device).unwrap();
        let act_rounded = act_f32
            .to_dtype(dtype)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let act_rounded_vec = act_rounded.to_vec::<f32>();

        for &m in &ms {
            let w_raw: Vec<f32> = (0..m * K)
                .map(|i| 0.3 + 0.25 * ((i as f32) * 0.071).sin())
                .collect();
            let w_f32 = Tensor::<CpuRuntime>::from_slice(&w_raw, &[m, K], &cpu_device).unwrap();
            let w_rounded = w_f32.to_dtype(dtype).unwrap().to_dtype(DType::F32).unwrap();
            let w_rounded_vec = w_rounded.to_vec::<f32>();

            // F32 reference accumulation over the SAME already-rounded values.
            let reference = cpu_client
                .awq_channel_scores(&act_rounded, &w_rounded)
                .unwrap();
            let ref_vec = reference.to_vec::<f32>();

            // CUDA half-precision accumulation path (the kernel under test).
            let act_c =
                Tensor::<CudaRuntime>::from_slice(&act_rounded_vec, &[N_ACT, K], &cuda_device)
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap();
            let w_c = Tensor::<CudaRuntime>::from_slice(&w_rounded_vec, &[m, K], &cuda_device)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let actual = cuda_client.awq_channel_scores(&act_c, &w_c).unwrap();
            let actual_vec = actual.to_dtype(DType::F32).unwrap().to_vec::<f32>();

            let max_abs = actual_vec
                .iter()
                .zip(ref_vec.iter())
                .map(|(a, r)| (a - r).abs())
                .fold(0.0f32, f32::max);
            let mean_signed_rel_err: f32 = actual_vec
                .iter()
                .zip(ref_vec.iter())
                .map(|(a, r)| (a - r) / r)
                .sum::<f32>()
                / K as f32;
            let ref_mag: f32 = ref_vec.iter().map(|v| v.abs()).sum::<f32>() / K as f32;
            let rtol = 3.0 * eps;
            let atol = 3.0 * eps * ref_mag.max(1e-6);

            println!(
                "CALIB_DIAG op=awq_channel_scores kernel=awq_score_reduce_{dtype_name} \
                 dtype={dtype_name} m={m} k={K} max_abs={max_abs:.6e} \
                 mean_signed_rel_err={mean_signed_rel_err:.6e} ref_mag={ref_mag:.6e} \
                 rtol={rtol:.6e} atol={atol:.6e}"
            );

            let mut case_ok = true;
            for (i, (a, r)) in actual_vec.iter().zip(ref_vec.iter()).enumerate() {
                let diff = (a - r).abs();
                let tol = atol + rtol * r.abs();
                if diff > tol {
                    case_ok = false;
                    if failures.len() < 4 {
                        failures.push(format!(
                            "m={m} idx={i}: actual={a} ref={r} diff={diff:.6e} tol={tol:.6e}"
                        ));
                    }
                }
            }
            if !case_ok {
                failures.push(format!("m={m}: FAILED (see CALIB_DIAG line above)"));
            }
        }

        assert!(
            failures.is_empty(),
            "awq_channel_scores/{dtype_name}: CUDA awq_score_reduce accumulation diverges \
             from the F32 reference beyond a single-rounding tolerance — atomic accumulation \
             into half storage undercounts as M grows:\n{}",
            failures.join("\n")
        );
    });
}

#[cfg(feature = "cuda")]
#[test]
fn test_awq_channel_scores_cuda_f16_accum_defect() {
    awq_score_reduce_accum_defect_case(DType::F16, 2f32.powi(-11), "f16");
}

#[cfg(feature = "cuda")]
#[test]
fn test_awq_channel_scores_cuda_bf16_accum_defect() {
    awq_score_reduce_accum_defect_case(DType::BF16, 2f32.powi(-8), "bf16");
}
