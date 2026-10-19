//! Backend parity tests for SamplingOps.
//!
//! The CPU result is the reference. A seeded draw reads the backend's own
//! uniform, then checks the token against the CPU probabilities under it.
//! An unseeded draw checks the token stays inside the CPU kept set.

#![cfg_attr(
    not(any(feature = "wgpu", feature = "cuda")),
    allow(dead_code, unused_imports)
)]

use super::helpers::*;
use boostr::ops::traits::inference::sample_seed::SampleSeed;
use boostr::ops::traits::inference::sampling::SamplingOps;
use boostr::ops::traits::inference::sampling_probs::{FilterRule, SamplingProbsOps};
use numr::dtype::DType;
use numr::ops::AdvancedRandomOps;
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
    seed: Option<SampleSeed>,
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
        Some(SampleSeed::new(7, 0)),
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
            Some(SampleSeed::new(7, 0)),
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

/// A draw point closer than this to a CDF step is too close to call across backends.
const DRAW_MARGIN: f32 = 1e-5;

/// Logits whose softmax is `p` up to rounding. Equal entries of `p` give bitwise ties.
fn log_probs(p: &[f32]) -> Vec<f32> {
    p.iter().map(|x| x.ln()).collect()
}

fn rule(temperature: f32, top_k: usize, top_p: f32, min_p: f32) -> FilterRule {
    FilterRule {
        temperature,
        top_k,
        top_p,
        min_p,
    }
}

/// Rows with ties at the top-k or top-p threshold, each with the rule that hits them.
fn tie_cases() -> Vec<(&'static str, Vec<f32>, FilterRule)> {
    let vocab = 1000;
    let mut wide = logits_data(vocab, 0.6);
    let mut ranked: Vec<usize> = (0..vocab).collect();
    ranked.sort_by(|&a, &b| wide[b].total_cmp(&wide[a]));
    let kth = wide[ranked[4]];
    for &i in &ranked[5..8] {
        wide[i] = kth;
    }
    vec![
        (
            "top-k tie",
            log_probs(&[0.4, 0.3, 0.3]),
            rule(1.0, 2, 1.0, 0.0),
        ),
        ("large-row top-k tie", wide, rule(1.0, 5, 1.0, 0.0)),
        (
            "top-p tie at tau",
            log_probs(&[0.5, 0.25, 0.25]),
            rule(1.0, 0, 0.6, 0.0),
        ),
        (
            "top-p before min-p",
            log_probs(&[0.5, 0.2, 0.15, 0.15]),
            rule(1.0, 0, 0.65, 0.35),
        ),
        (
            "combined tie",
            log_probs(&[0.4, 0.3, 0.3, 0.0001]),
            rule(1.0, 2, 0.9, 0.1),
        ),
    ]
}

/// The CPU canonical probabilities of one logits row.
fn cpu_probs(data: &[f32], r: FilterRule) -> Vec<f32> {
    let (client, device) = setup_cpu();
    let logits = Tensor::<CpuRuntime>::from_slice(data, &[data.len()], &device).unwrap();
    client.sampling_probs(&logits, &r).unwrap().to_vec::<f32>()
}

/// The id-order draw over `probs` at uniform `u`, or `None` when `u` lies near a CDF step.
///
/// It skips every removed token. Rounding past the end returns the last kept token.
fn reference_draw(probs: &[f32], u: f32) -> Option<usize> {
    let total: f32 = probs.iter().sum();
    let target = u * total;
    let mut cumsum = 0.0f32;
    let mut last_kept = 0;
    for (i, &p) in probs.iter().enumerate() {
        if p <= 0.0 {
            continue;
        }
        cumsum += p;
        last_kept = i;
        if (cumsum - target).abs() < DRAW_MARGIN {
            return None;
        }
        if cumsum > target {
            return Some(i);
        }
    }
    Some(last_kept)
}

/// Checks seeded `logits_to_token` on one backend against the CPU probabilities.
///
/// Each seed reads the backend's own uniform, so the check holds even when
/// the backends draw different uniforms for one seed.
fn check_seeded_tokens<R: Runtime<DType = DType>>(
    client: &R::Client,
    device: &R::Device,
    backend: &str,
) where
    R::Client: SamplingOps<R> + AdvancedRandomOps<R>,
{
    for (what, data, r) in tie_cases() {
        let probs = cpu_probs(&data, r);
        let vocab = data.len();
        let logits = Tensor::<R>::from_slice(&data, &[1, 1, vocab], device).unwrap();
        let ids = Tensor::<R>::from_slice(&[0i64], &[1], device).unwrap();
        let counts = Tensor::<R>::from_slice(&[1i32], &[1], device).unwrap();
        let mut checked = 0;
        for counter in 0..64u64 {
            let seed = SampleSeed::new(11, counter);
            let u = client
                .philox_uniform(&[1], seed.key, seed.counter, DType::F32)
                .unwrap()
                .to_vec::<f32>()[0];
            let Some(want) = reference_draw(&probs, u) else {
                continue;
            };
            let token = client
                .logits_to_token(
                    &logits,
                    &ids,
                    &counts,
                    0,
                    1.0,
                    0.0,
                    0.0,
                    r.temperature,
                    r.top_k,
                    r.top_p,
                    r.min_p,
                    Some(seed),
                )
                .unwrap()
                .to_vec::<i64>()[0];
            assert_eq!(
                token, want as i64,
                "{backend} {what}: seed counter {counter}, u {u}"
            );
            checked += 1;
        }
        assert!(
            checked > 32,
            "{backend} {what}: only {checked} draws were clear of a step"
        );
    }
}

/// Checks unseeded `sample_token` on one backend stays inside the CPU kept set.
///
/// A kept set of at most 4 tokens must also see every token in 64 draws.
fn check_unseeded_tokens<R: Runtime<DType = DType>>(
    client: &R::Client,
    device: &R::Device,
    backend: &str,
) where
    R::Client: SamplingOps<R>,
{
    for (what, data, r) in tie_cases() {
        let probs = cpu_probs(&data, r);
        let kept: Vec<usize> = (0..probs.len()).filter(|&i| probs[i] > 0.0).collect();
        let logits = Tensor::<R>::from_slice(&data, &[data.len()], device).unwrap();
        let mut seen = vec![false; probs.len()];
        for _ in 0..64 {
            let token = client
                .sample_token(&logits, r.temperature, r.top_k, r.top_p, r.min_p)
                .unwrap() as usize;
            assert!(
                kept.contains(&token),
                "{backend} {what}: token {token} is outside the kept set {kept:?}"
            );
            seen[token] = true;
        }
        if kept.len() <= 4 {
            for &i in &kept {
                assert!(seen[i], "{backend} {what}: kept token {i} never drawn");
            }
        }
    }
}

#[test]
fn test_reference_draw_matches_the_tie_rule() {
    let probs = cpu_probs(&log_probs(&[0.4, 0.3, 0.3]), rule(1.0, 2, 1.0, 0.0));
    assert!(probs.iter().all(|&p| p > 0.0), "{probs:?}");
    assert_eq!(reference_draw(&probs, 0.2), Some(0));
    assert_eq!(reference_draw(&probs, 0.55), Some(1));
    assert_eq!(reference_draw(&probs, 0.95), Some(2));
    assert_eq!(reference_draw(&probs, 0.4), None, "0.4 sits on a step");
}

#[test]
fn test_logits_to_token_seeded_tie_parity() {
    #[cfg(feature = "cuda")]
    with_cuda_backend(|client, device| {
        check_seeded_tokens::<numr::runtime::cuda::CudaRuntime>(&client, &device, "cuda");
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|client, device| {
        check_seeded_tokens::<numr::runtime::wgpu::WgpuRuntime>(&client, &device, "wgpu");
    });
}

#[test]
fn test_sample_token_tie_parity() {
    #[cfg(feature = "cuda")]
    with_cuda_backend(|client, device| {
        check_unseeded_tokens::<numr::runtime::cuda::CudaRuntime>(&client, &device, "cuda");
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|client, device| {
        check_unseeded_tokens::<numr::runtime::wgpu::WgpuRuntime>(&client, &device, "wgpu");
    });
}
