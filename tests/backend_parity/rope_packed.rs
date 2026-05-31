//! Backend parity tests for RoPEPackedOps.

use super::helpers::*;
use boostr::ops::traits::position::rope::RoPEOps;
use boostr::ops::traits::position::rope_packed::RoPEPackedOps;
use numr::autograd::Var;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

/// Build sequential position_ids [0, 1, ..., n-1] as I32.
fn seq_pids_cpu(n: usize, device: &numr::runtime::cpu::CpuDevice) -> Tensor<CpuRuntime> {
    let ids: Vec<i32> = (0..n as i32).collect();
    Tensor::<CpuRuntime>::from_slice(&ids, &[n], device)
}

#[test]
fn test_apply_rope_packed_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (total_tokens, num_heads, head_dim) = (8, 2, 32);
    let half_d = head_dim / 2;
    let max_seq = 16;

    let x_data = det_tensor(&[total_tokens, num_heads, head_dim], &cpu_device);
    let cos_data = det_tensor(&[max_seq, half_d], &cpu_device);
    let sin_data = det_tensor(&[max_seq, half_d], &cpu_device);
    let pids = seq_pids_cpu(total_tokens, &cpu_device);

    let x = Var::<CpuRuntime>::new(x_data.clone(), false);
    let cos = Var::<CpuRuntime>::new(cos_data.clone(), false);
    let sin = Var::<CpuRuntime>::new(sin_data.clone(), false);

    let cpu_result = cpu_client.apply_rope_packed(&x, &cos, &sin, &pids).unwrap();
    let cpu_result_vec = cpu_result.tensor().to_vec::<f32>();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::position::rope_packed::RoPEPackedOps as _;
        use numr::autograd::Var;
        use numr::runtime::cuda::CudaRuntime;
        use numr::tensor::Tensor;

        let x_c = Var::<CudaRuntime>::new(
            Tensor::from_slice(
                &x_data.to_vec::<f32>(),
                &[total_tokens, num_heads, head_dim],
                &cuda_device,
            ),
            false,
        );
        let cos_c = Var::<CudaRuntime>::new(
            Tensor::from_slice(&cos_data.to_vec::<f32>(), &[max_seq, half_d], &cuda_device),
            false,
        );
        let sin_c = Var::<CudaRuntime>::new(
            Tensor::from_slice(&sin_data.to_vec::<f32>(), &[max_seq, half_d], &cuda_device),
            false,
        );
        let pids_c =
            Tensor::<CudaRuntime>::from_slice(&pids.to_vec::<i32>(), &[total_tokens], &cuda_device);

        let result = cuda_client
            .apply_rope_packed(&x_c, &cos_c, &sin_c, &pids_c)
            .unwrap();
        assert_parity_f32(
            &result.tensor().to_vec::<f32>(),
            &cpu_result_vec,
            "apply_rope_packed CUDA vs CPU",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::position::rope_packed::RoPEPackedOps as _;
        use numr::autograd::Var;
        use numr::runtime::wgpu::WgpuRuntime;
        use numr::tensor::Tensor;

        let x_w = Var::<WgpuRuntime>::new(
            Tensor::from_slice(
                &x_data.to_vec::<f32>(),
                &[total_tokens, num_heads, head_dim],
                &wgpu_device,
            ),
            false,
        );
        let cos_w = Var::<WgpuRuntime>::new(
            Tensor::from_slice(&cos_data.to_vec::<f32>(), &[max_seq, half_d], &wgpu_device),
            false,
        );
        let sin_w = Var::<WgpuRuntime>::new(
            Tensor::from_slice(&sin_data.to_vec::<f32>(), &[max_seq, half_d], &wgpu_device),
            false,
        );
        let pids_w =
            Tensor::<WgpuRuntime>::from_slice(&pids.to_vec::<i32>(), &[total_tokens], &wgpu_device);

        let result = wgpu_client
            .apply_rope_packed(&x_w, &cos_w, &sin_w, &pids_w)
            .unwrap();
        assert_parity_f32(
            &result.tensor().to_vec::<f32>(),
            &cpu_result_vec,
            "apply_rope_packed WGPU vs CPU",
        );
    });
}

/// Verify that apply_rope_packed with position_ids=[0..S-1] (single sequence)
/// produces numerically equivalent output to apply_rope on x reshaped to [B,H,S,D] (B=1).
///
/// This proves the two ops share identical rotation numerics.
#[test]
fn test_packed_matches_standard_single_sequence() {
    let (cpu_client, cpu_device) = setup_cpu();
    let (s, h, d) = (4usize, 2usize, 16usize);
    let half_d = d / 2;

    let x_data = det_tensor(&[s, h, d], &cpu_device);
    let cos_data = det_tensor(&[s, half_d], &cpu_device);
    let sin_data = det_tensor(&[s, half_d], &cpu_device);

    // --- Packed path: x=[S, H, D], position_ids=[0,1,...,S-1] ---
    let x_packed = Var::<CpuRuntime>::new(x_data.clone(), false);
    let cos = Var::<CpuRuntime>::new(cos_data.clone(), false);
    let sin = Var::<CpuRuntime>::new(sin_data.clone(), false);
    let pids = seq_pids_cpu(s, &cpu_device);

    let out_packed = cpu_client
        .apply_rope_packed(&x_packed, &cos, &sin, &pids)
        .unwrap();
    let packed_vec = out_packed.tensor().to_vec::<f32>();

    // --- Standard path: x=[1, H, S, D] ---
    // Permute [S, H, D] to [1, H, S, D]:
    // packed layout: x_data[s_idx * H * D + h_idx * D + d_idx]
    // standard layout: x_4d[0, h_idx, s_idx, d_idx] → linear: h_idx*S*D + s_idx*D + d_idx
    let x_data_vec = x_data.to_vec::<f32>();
    let mut x_4d_data = vec![0.0f32; s * h * d];
    for sv in 0..s {
        for hv in 0..h {
            for dv in 0..d {
                let src = sv * h * d + hv * d + dv;
                let dst = hv * s * d + sv * d + dv;
                x_4d_data[dst] = x_data_vec[src];
            }
        }
    }
    let x_standard = Var::<CpuRuntime>::new(
        Tensor::<CpuRuntime>::from_slice(&x_4d_data, &[1, h, s, d], &cpu_device),
        false,
    );
    let out_standard = cpu_client.apply_rope(&x_standard, &cos, &sin).unwrap();
    let standard_4d = out_standard.tensor().to_vec::<f32>();

    // Convert standard output back from [1,H,S,D] to [S,H,D] order:
    let mut standard_vec = vec![0.0f32; s * h * d];
    for sv in 0..s {
        for hv in 0..h {
            for dv in 0..d {
                let src = hv * s * d + sv * d + dv;
                let dst = sv * h * d + hv * d + dv;
                standard_vec[dst] = standard_4d[src];
            }
        }
    }

    assert_parity_f32(
        &packed_vec,
        &standard_vec,
        "packed(B=1 seq) vs standard apply_rope",
    );
}

#[test]
fn test_packed_position_reset_parity() {
    // Two sequences packed: [seq0_tok0, seq0_tok1, seq1_tok0, seq1_tok1]
    // position_ids = [0, 1, 0, 1]
    // seq0 and seq1 get the same positions — their outputs should be equal if their inputs are.
    let (cpu_client, cpu_device) = setup_cpu();
    let h = 1usize;
    let d = 8usize;
    let half_d = d / 2;
    let max_seq = 8usize;

    let cos_data: Vec<f32> = (0..max_seq * half_d)
        .map(|i| (i as f32 * 0.4).cos())
        .collect();
    let sin_data: Vec<f32> = (0..max_seq * half_d)
        .map(|i| (i as f32 * 0.4).sin())
        .collect();
    let cos = Var::<CpuRuntime>::new(
        Tensor::<CpuRuntime>::from_slice(&cos_data, &[max_seq, half_d], &cpu_device),
        false,
    );
    let sin = Var::<CpuRuntime>::new(
        Tensor::<CpuRuntime>::from_slice(&sin_data, &[max_seq, half_d], &cpu_device),
        false,
    );

    // Same x values for tok0 and tok2, same for tok1 and tok3
    let tok0_vals: Vec<f32> = (0..h * d).map(|i| i as f32 + 0.1).collect();
    let tok1_vals: Vec<f32> = (0..h * d).map(|i| i as f32 + 10.1).collect();
    let x_data: Vec<f32> = tok0_vals
        .iter()
        .chain(tok1_vals.iter())
        .chain(tok0_vals.iter())
        .chain(tok1_vals.iter())
        .copied()
        .collect();

    let x = Var::<CpuRuntime>::new(
        Tensor::<CpuRuntime>::from_slice(&x_data, &[4, h, d], &cpu_device),
        false,
    );
    let pids = Tensor::<CpuRuntime>::from_slice(&[0i32, 1, 0, 1], &[4], &cpu_device);

    let out = cpu_client.apply_rope_packed(&x, &cos, &sin, &pids).unwrap();
    let out_vec = out.tensor().to_vec::<f32>();

    let stride = h * d;
    // tok0 and tok2 must match (same x, same position)
    for i in 0..stride {
        assert!(
            (out_vec[i] - out_vec[2 * stride + i]).abs() < 1e-5,
            "tok0 vs tok2 mismatch at {i}"
        );
    }
    // tok1 and tok3 must match
    for i in 0..stride {
        assert!(
            (out_vec[stride + i] - out_vec[3 * stride + i]).abs() < 1e-5,
            "tok1 vs tok3 mismatch at {i}"
        );
    }
}
