//! Backend parity tests for SpeculativeOps.

use super::helpers::*;
use boostr::ops::traits::inference::speculative::SpeculativeOps;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

/// Create test probability distributions that are valid (sum to ~1 per position).
fn make_test_probs(
    batch: usize,
    positions: usize,
    vocab: usize,
    device: &numr::runtime::cpu::CpuDevice,
    offset: f32,
) -> Tensor<CpuRuntime> {
    let n = batch * positions * vocab;
    let mut data: Vec<f32> = (0..n)
        .map(|i| ((i as f32 * 0.1 + offset).sin() * 0.5 + 1.0).max(0.01))
        .collect();

    // Normalize each [vocab_size] slice to sum to 1
    for b in 0..batch {
        for p in 0..positions {
            let start = (b * positions + p) * vocab;
            let end = start + vocab;
            let sum: f32 = data[start..end].iter().sum();
            for val in &mut data[start..end] {
                *val /= sum;
            }
        }
    }

    Tensor::<CpuRuntime>::from_slice(&data, &[batch, positions, vocab], device)
}

#[test]
fn test_verify_speculative_tokens_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let batch = 2;
    let k = 4;
    let vocab = 32;
    let seed = 42u64;

    let draft_probs = make_test_probs(batch, k, vocab, &cpu_device, 0.0);
    let target_probs = make_test_probs(batch, k + 1, vocab, &cpu_device, 0.3);

    // Create draft tokens (pick token with highest draft prob for high acceptance)
    let dp_data = draft_probs.to_vec::<f32>();
    let mut tokens = Vec::new();
    for b in 0..batch {
        for p in 0..k {
            let off = (b * k + p) * vocab;
            let best = dp_data[off..off + vocab]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            tokens.push(best as i32);
        }
    }
    let draft_tokens = Tensor::<CpuRuntime>::from_slice(&tokens, &[batch, k], &cpu_device);

    let cpu_results = cpu_client
        .verify_speculative_tokens(&draft_probs, &target_probs, &draft_tokens, seed)
        .unwrap();

    assert_eq!(cpu_results.len(), batch);
    for (b, result) in cpu_results.iter().enumerate() {
        assert!(
            result.num_accepted <= k,
            "batch {}: accepted {} > K={}",
            b,
            result.num_accepted,
            k
        );
        assert_eq!(result.accepted_tokens.len(), result.num_accepted);
        assert!(
            (result.bonus_token as usize) < vocab,
            "batch {}: bonus_token {} >= vocab {}",
            b,
            result.bonus_token,
            vocab
        );
    }

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::inference::speculative::SpeculativeOps as _;

        let dp_c = Tensor::from_slice(
            &draft_probs.to_vec::<f32>(),
            &[batch, k, vocab],
            &cuda_device,
        );
        let tp_c = Tensor::from_slice(
            &target_probs.to_vec::<f32>(),
            &[batch, k + 1, vocab],
            &cuda_device,
        );
        let dt_c = Tensor::from_slice(&tokens, &[batch, k], &cuda_device);

        let cuda_results = cuda_client
            .verify_speculative_tokens(&dp_c, &tp_c, &dt_c, seed)
            .unwrap();

        assert_eq!(cpu_results.len(), cuda_results.len());
        // All backends use philox_uniform — exact parity expected
        for (b, (cpu_r, cuda_r)) in cpu_results.iter().zip(cuda_results.iter()).enumerate() {
            assert_eq!(
                cpu_r.num_accepted, cuda_r.num_accepted,
                "CUDA batch {}: num_accepted mismatch",
                b
            );
            assert_eq!(
                cpu_r.accepted_tokens, cuda_r.accepted_tokens,
                "CUDA batch {}: accepted_tokens mismatch",
                b
            );
            assert_eq!(
                cpu_r.bonus_token, cuda_r.bonus_token,
                "CUDA batch {}: bonus_token mismatch",
                b
            );
        }
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::inference::speculative::SpeculativeOps as _;

        let dp_w = Tensor::from_slice(
            &draft_probs.to_vec::<f32>(),
            &[batch, k, vocab],
            &wgpu_device,
        );
        let tp_w = Tensor::from_slice(
            &target_probs.to_vec::<f32>(),
            &[batch, k + 1, vocab],
            &wgpu_device,
        );
        let dt_w = Tensor::from_slice(&tokens, &[batch, k], &wgpu_device);

        let wgpu_results = wgpu_client
            .verify_speculative_tokens(&dp_w, &tp_w, &dt_w, seed)
            .unwrap();

        // All backends use philox_uniform — exact parity expected
        for (b, (cpu_r, wgpu_r)) in cpu_results.iter().zip(wgpu_results.iter()).enumerate() {
            assert_eq!(
                cpu_r.num_accepted, wgpu_r.num_accepted,
                "WGPU batch {}: num_accepted mismatch",
                b
            );
            assert_eq!(
                cpu_r.accepted_tokens, wgpu_r.accepted_tokens,
                "WGPU batch {}: accepted_tokens mismatch",
                b
            );
            assert_eq!(
                cpu_r.bonus_token, wgpu_r.bonus_token,
                "WGPU batch {}: bonus_token mismatch",
                b
            );
        }
    });
}

#[test]
fn test_compute_acceptance_probs_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let batch = 2;
    let k = 3;
    let vocab = 16;

    let draft = make_test_probs(batch, k, vocab, &cpu_device, 0.0);
    let target = make_test_probs(batch, k, vocab, &cpu_device, 0.5);

    let (acceptance, residual) = cpu_client
        .compute_acceptance_probs(&draft, &target)
        .unwrap();

    let cpu_acc_vec = acceptance.to_vec::<f32>();
    let cpu_res_vec = residual.to_vec::<f32>();

    for (i, &a) in cpu_acc_vec.iter().enumerate() {
        assert!(
            (0.0..=1.0 + 1e-6).contains(&a),
            "acceptance[{}]={} not in [0,1]",
            i,
            a
        );
    }
    for (i, &r) in cpu_res_vec.iter().enumerate() {
        assert!(r >= -1e-7, "residual[{}]={} is negative", i, r);
    }

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::inference::speculative::SpeculativeOps as _;

        let dp_c = Tensor::from_slice(&draft.to_vec::<f32>(), &[batch, k, vocab], &cuda_device);
        let tp_c = Tensor::from_slice(&target.to_vec::<f32>(), &[batch, k, vocab], &cuda_device);

        let (cuda_acc, cuda_res) = cuda_client.compute_acceptance_probs(&dp_c, &tp_c).unwrap();

        assert_parity_f32(
            &cuda_acc.to_vec::<f32>(),
            &cpu_acc_vec,
            "compute_acceptance_probs CUDA vs CPU",
        );
        assert_parity_f32(
            &cuda_res.to_vec::<f32>(),
            &cpu_res_vec,
            "compute_residual_probs CUDA vs CPU",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::inference::speculative::SpeculativeOps as _;

        let dp_w = Tensor::from_slice(&draft.to_vec::<f32>(), &[batch, k, vocab], &wgpu_device);
        let tp_w = Tensor::from_slice(&target.to_vec::<f32>(), &[batch, k, vocab], &wgpu_device);

        let (wgpu_acc, wgpu_res) = wgpu_client.compute_acceptance_probs(&dp_w, &tp_w).unwrap();

        assert_parity_f32(
            &wgpu_acc.to_vec::<f32>(),
            &cpu_acc_vec,
            "compute_acceptance_probs WGPU vs CPU",
        );
        assert_parity_f32(
            &wgpu_res.to_vec::<f32>(),
            &cpu_res_vec,
            "compute_residual_probs WGPU vs CPU",
        );
    });
}

#[test]
fn test_compute_expected_tokens_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let batch = 4;
    let k = 6;

    let rates: Vec<f32> = (0..batch * k)
        .map(|i| 0.5 + 0.4 * (i as f32 * 0.3).sin())
        .collect();
    let rates_tensor = Tensor::<CpuRuntime>::from_slice(&rates, &[batch, k], &cpu_device);

    let cpu_expected = cpu_client.compute_expected_tokens(&rates_tensor).unwrap();
    let cpu_vec = cpu_expected.to_vec::<f32>();

    assert_eq!(cpu_vec.len(), batch);

    // Analytical check for uniform rate=0.8 case
    let uniform_rates: Vec<f32> = vec![0.8; batch * k];
    let uniform_tensor = Tensor::<CpuRuntime>::from_slice(&uniform_rates, &[batch, k], &cpu_device);
    let uniform_expected = cpu_client.compute_expected_tokens(&uniform_tensor).unwrap();
    let uniform_vec = uniform_expected.to_vec::<f32>();
    let analytical = (1..=k).fold(0.0f32, |acc, i| acc + 0.8f32.powi(i as i32)) + 1.0;
    for (b, &val) in uniform_vec.iter().enumerate() {
        assert!(
            (val - analytical).abs() < 1e-4,
            "batch {}: expected {} got {}",
            b,
            analytical,
            val
        );
    }

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::ops::traits::inference::speculative::SpeculativeOps as _;

        let rates_c = Tensor::from_slice(&rates, &[batch, k], &cuda_device);
        let cuda_expected = cuda_client.compute_expected_tokens(&rates_c).unwrap();

        assert_parity_f32(
            &cuda_expected.to_vec::<f32>(),
            &cpu_vec,
            "compute_expected_tokens CUDA vs CPU",
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        use boostr::ops::traits::inference::speculative::SpeculativeOps as _;

        let rates_w = Tensor::from_slice(&rates, &[batch, k], &wgpu_device);
        let wgpu_expected = wgpu_client.compute_expected_tokens(&rates_w).unwrap();

        assert_parity_f32(
            &wgpu_expected.to_vec::<f32>(),
            &cpu_vec,
            "compute_expected_tokens WGPU vs CPU",
        );
    });
}

#[test]
fn test_verify_deterministic() {
    let (cpu_client, cpu_device) = setup_cpu();
    let batch = 2;
    let k = 4;
    let vocab = 32;
    let seed = 12345u64;

    let draft_probs = make_test_probs(batch, k, vocab, &cpu_device, 0.0);
    let target_probs = make_test_probs(batch, k + 1, vocab, &cpu_device, 0.3);
    let tokens: Vec<i32> = (0..batch * k).map(|i| (i % vocab) as i32).collect();
    let draft_tokens = Tensor::<CpuRuntime>::from_slice(&tokens, &[batch, k], &cpu_device);

    let r1 = cpu_client
        .verify_speculative_tokens(&draft_probs, &target_probs, &draft_tokens, seed)
        .unwrap();
    let r2 = cpu_client
        .verify_speculative_tokens(&draft_probs, &target_probs, &draft_tokens, seed)
        .unwrap();

    for (a, b) in r1.iter().zip(r2.iter()) {
        assert_eq!(
            a.num_accepted, b.num_accepted,
            "determinism: num_accepted differs"
        );
        assert_eq!(
            a.accepted_tokens, b.accepted_tokens,
            "determinism: accepted_tokens differ"
        );
        assert_eq!(
            a.bonus_token, b.bonus_token,
            "determinism: bonus_token differs"
        );
    }
}
