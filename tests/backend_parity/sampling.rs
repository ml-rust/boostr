//! Backend parity tests for SamplingOps and GrammarDfaOps.
//!
//! The CPU result is the reference. The stochastic chain draws a random
//! value per backend, so those cases pin the draw with `top_k = 1` or a
//! distribution with one dominant token.

#![cfg_attr(
    not(any(feature = "wgpu", feature = "cuda")),
    allow(dead_code, unused_imports)
)]

use super::helpers::*;
use boostr::ops::traits::inference::grammar::{DeviceGrammarDfa, GrammarDfaOps, INVALID_STATE};
use boostr::ops::traits::inference::sampling::SamplingOps;
use numr::runtime::Runtime;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

/// The Llama 3 vocabulary size: exercises the strided loops of the kernel.
const LARGE_VOCAB: usize = 128_256;

/// Deterministic logits with distinct values.
fn logits_data(n: usize, phase: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 * 0.37 + phase).sin() * 4.0) + (i % 7) as f32 * 0.01)
        .collect()
}

/// Penalty inputs: the current argmax and a few others.
fn penalty_inputs(last_row: &[f32]) -> (Vec<i64>, Vec<i32>) {
    let argmax = last_row
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |best, (i, &v)| {
            if v > best.1 { (i, v) } else { best }
        })
        .0;
    let ids = vec![argmax as i64, 3, 11, (last_row.len() - 1) as i64];
    let counts = vec![4, 1, 2, 3];
    (ids, counts)
}

/// CPU and backend `logits_to_token` on the same inputs.
#[allow(clippy::too_many_arguments)]
fn logits_to_token_on<R: Runtime<DType = numr::dtype::DType>>(
    client: &R::Client,
    device: &R::Device,
    data: &[f32],
    shape: &[usize],
    ids: &[i64],
    counts: &[i32],
    temperature: f32,
    top_k: usize,
    seed: Option<u64>,
) -> i64
where
    R::Client: SamplingOps<R>,
{
    let logits = Tensor::<R>::from_slice(data, shape, device).unwrap();
    let ids_t = Tensor::<R>::from_slice(ids, &[ids.len()], device).unwrap();
    let counts_t = Tensor::<R>::from_slice(counts, &[counts.len()], device).unwrap();
    let out = client
        .logits_to_token(
            &logits,
            &ids_t,
            &counts_t,
            ids.len(),
            1.3,
            0.4,
            0.2,
            temperature,
            top_k,
            1.0,
            0.0,
            seed,
        )
        .unwrap();
    out.to_vec::<i64>()[0]
}

/// One grammar case: 2 states, `a` leads to the accepting state 1.
fn grammar_case<R: Runtime<DType = numr::dtype::DType>>(
    device: &R::Device,
    current_state: u32,
) -> DeviceGrammarDfa<R> {
    let num_states = 2;
    let mut table = vec![INVALID_STATE as f32; num_states * 256];
    table[b'a' as usize] = 1.0;
    table[256 + b'b' as usize] = 0.0;
    let accepting = [0.0f32, 1.0];
    // Tokens: "a", "b", "ab", "", "ba", "aa".
    let tokens: [&[u8]; 6] = [b"a", b"b", b"ab", b"", b"ba", b"aa"];
    let mut bytes = Vec::new();
    let mut offsets = Vec::new();
    for t in tokens {
        offsets.push(bytes.len() as f32);
        bytes.extend(t.iter().map(|&b| b as f32));
    }
    offsets.push(bytes.len() as f32);
    DeviceGrammarDfa {
        transition_table: Tensor::<R>::from_slice(&table, &[table.len()], device).unwrap(),
        accepting_mask: Tensor::<R>::from_slice(&accepting, &[2], device).unwrap(),
        vocab_bytes: Tensor::<R>::from_slice(&bytes, &[bytes.len()], device).unwrap(),
        vocab_offsets: Tensor::<R>::from_slice(&offsets, &[offsets.len()], device).unwrap(),
        current_state,
        num_states,
        vocab_size: tokens.len(),
    }
}

#[test]
fn test_logits_to_token_greedy_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    for (seq_len, vocab) in [(1usize, 17usize), (3, 1000), (2, LARGE_VOCAB)] {
        let data = logits_data(seq_len * vocab, 0.3);
        let (ids, counts) = penalty_inputs(&data[(seq_len - 1) * vocab..]);
        let shape = [1, seq_len, vocab];
        let cpu = logits_to_token_on::<CpuRuntime>(
            &cpu_client,
            &cpu_device,
            &data,
            &shape,
            &ids,
            &counts,
            0.0,
            0,
            None,
        );

        #[cfg(feature = "wgpu")]
        with_wgpu_backend(|client, device| {
            let gpu = logits_to_token_on::<numr::runtime::wgpu::WgpuRuntime>(
                &client, &device, &data, &shape, &ids, &counts, 0.0, 0, None,
            );
            assert_eq!(
                gpu, cpu,
                "wgpu greedy token, seq_len {seq_len}, vocab {vocab}"
            );
        });
        let _ = cpu;
    }
}

#[test]
fn test_logits_to_token_greedy_tie_takes_lowest_index() {
    let (cpu_client, cpu_device) = setup_cpu();
    let mut data = vec![0.0f32; 600];
    data[123] = 5.0;
    data[77] = 5.0;
    data[599] = 5.0;
    let cpu = logits_to_token_on::<CpuRuntime>(
        &cpu_client,
        &cpu_device,
        &data,
        &[1, 1, 600],
        &[],
        &[],
        0.0,
        0,
        None,
    );
    assert_eq!(cpu, 77);

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|client, device| {
        let gpu = logits_to_token_on::<numr::runtime::wgpu::WgpuRuntime>(
            &client,
            &device,
            &data,
            &[1, 1, 600],
            &[],
            &[],
            0.0,
            0,
            None,
        );
        assert_eq!(gpu, cpu, "wgpu greedy tie");
    });
}

#[test]
fn test_logits_to_token_top_k_one_matches_greedy() {
    let (cpu_client, cpu_device) = setup_cpu();
    let vocab = 5000;
    let data = logits_data(vocab, 1.1);
    let (ids, counts) = penalty_inputs(&data);
    let greedy = logits_to_token_on::<CpuRuntime>(
        &cpu_client,
        &cpu_device,
        &data,
        &[1, 1, vocab],
        &ids,
        &counts,
        0.0,
        0,
        None,
    );
    let cpu = logits_to_token_on::<CpuRuntime>(
        &cpu_client,
        &cpu_device,
        &data,
        &[1, 1, vocab],
        &ids,
        &counts,
        0.8,
        1,
        Some(7),
    );
    assert_eq!(cpu, greedy);

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|client, device| {
        let gpu = logits_to_token_on::<numr::runtime::wgpu::WgpuRuntime>(
            &client,
            &device,
            &data,
            &[1, 1, vocab],
            &ids,
            &counts,
            0.8,
            1,
            Some(7),
        );
        assert_eq!(gpu, greedy, "wgpu top_k = 1 sample");
    });
}

#[test]
fn test_sample_token_dominant_token() {
    let (cpu_client, cpu_device) = setup_cpu();
    let vocab = 3000;
    let mut data = vec![0.0f32; vocab];
    data[2048] = 60.0;
    let cpu_logits = Tensor::<CpuRuntime>::from_slice(&data, &[vocab], &cpu_device).unwrap();
    let cpu = cpu_client
        .sample_token(&cpu_logits, 1.0, 0, 0.9, 0.05)
        .unwrap();
    assert_eq!(cpu, 2048);

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|client, device| {
        let logits = Tensor::from_slice(&data, &[vocab], &device).unwrap();
        for _ in 0..4 {
            let gpu = client.sample_token(&logits, 1.0, 0, 0.9, 0.05).unwrap();
            assert_eq!(gpu, cpu, "wgpu sample_token");
        }
    });
}

#[test]
fn test_sample_token_stays_inside_top_k() {
    let vocab = 2000;
    let data = logits_data(vocab, 0.7);
    let mut ranked: Vec<usize> = (0..vocab).collect();
    ranked.sort_by(|&a, &b| data[b].partial_cmp(&data[a]).unwrap());
    let allowed = &ranked[..5];
    let _ = allowed;

    #[cfg(feature = "cuda")]
    with_cuda_backend(|client, device| {
        let logits = Tensor::from_slice(&data, &[vocab], &device).unwrap();
        for _ in 0..16 {
            let gpu = client.sample_token(&logits, 0.7, 5, 1.0, 0.0).unwrap() as usize;
            assert!(
                allowed.contains(&gpu),
                "cuda token {gpu} is outside the top 5"
            );
        }
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|client, device| {
        let logits = Tensor::from_slice(&data, &[vocab], &device).unwrap();
        for _ in 0..16 {
            let gpu = client.sample_token(&logits, 0.7, 5, 1.0, 0.0).unwrap() as usize;
            assert!(
                allowed.contains(&gpu),
                "wgpu token {gpu} is outside the top 5"
            );
        }
    });
}

#[test]
fn test_apply_sampling_penalties_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    let vocab = 777;
    let data = logits_data(vocab, 0.0);
    let (ids, counts) = penalty_inputs(&data);
    let cpu_logits = Tensor::<CpuRuntime>::from_slice(&data, &[vocab], &cpu_device).unwrap();
    let cpu_ids = Tensor::<CpuRuntime>::from_slice(&ids, &[ids.len()], &cpu_device).unwrap();
    let cpu_counts =
        Tensor::<CpuRuntime>::from_slice(&counts, &[counts.len()], &cpu_device).unwrap();
    cpu_client
        .apply_sampling_penalties(&cpu_logits, &cpu_ids, &cpu_counts, 1.3, 0.4, 0.2)
        .unwrap();
    let cpu = cpu_logits.to_vec::<f32>();
    assert_ne!(cpu, data, "the penalties change the CPU logits");

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|client, device| {
        let logits = Tensor::from_slice(&data, &[vocab], &device).unwrap();
        let ids_t = Tensor::from_slice(&ids, &[ids.len()], &device).unwrap();
        let counts_t = Tensor::from_slice(&counts, &[counts.len()], &device).unwrap();
        client
            .apply_sampling_penalties(&logits, &ids_t, &counts_t, 1.3, 0.4, 0.2)
            .unwrap();
        assert_parity_f32(&logits.to_vec::<f32>(), &cpu, "apply_sampling_penalties");
    });
}

#[test]
fn test_grammar_dfa_mask_parity() {
    let (cpu_client, cpu_device) = setup_cpu();
    for state in [0u32, 1] {
        let data = logits_data(2 * 6, 0.5);
        let cpu_logits = Tensor::<CpuRuntime>::from_slice(&data, &[1, 2, 6], &cpu_device).unwrap();
        let cpu = cpu_client
            .grammar_dfa_mask_logits(&cpu_logits, &grammar_case::<CpuRuntime>(&cpu_device, state))
            .unwrap()
            .to_vec::<f32>();

        #[cfg(feature = "cuda")]
        with_cuda_backend(|client, device| {
            let logits = Tensor::from_slice(&data, &[1, 2, 6], &device).unwrap();
            let grammar = grammar_case::<numr::runtime::cuda::CudaRuntime>(&device, state);
            let gpu = client
                .grammar_dfa_mask_logits(&logits, &grammar)
                .unwrap()
                .to_vec::<f32>();
            assert_eq!(gpu, cpu, "cuda grammar mask from state {state}");
        });

        #[cfg(feature = "wgpu")]
        with_wgpu_backend(|client, device| {
            let logits = Tensor::from_slice(&data, &[1, 2, 6], &device).unwrap();
            let grammar = grammar_case::<numr::runtime::wgpu::WgpuRuntime>(&device, state);
            let gpu = client
                .grammar_dfa_mask_logits(&logits, &grammar)
                .unwrap()
                .to_vec::<f32>();
            assert_eq!(gpu, cpu, "wgpu grammar mask from state {state}");
            assert_eq!(
                logits.to_vec::<f32>(),
                data,
                "the input logits stay unchanged"
            );
        });
        let _ = cpu;
    }
}
