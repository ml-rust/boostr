//! Backend parity tests for SamplingProbsOps.
//!
//! The CPU result is the reference. Each case checks the kept set of every
//! row exactly and the probabilities within a float tolerance.

#![cfg_attr(not(feature = "wgpu"), allow(dead_code, unused_imports))]

use super::helpers::*;
use boostr::ops::traits::inference::sampling_probs::{FilterRule, SamplingProbsOps};
use numr::runtime::Runtime;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

/// The Llama 3 vocabulary size: exercises the global sort path on WebGPU.
const LARGE_VOCAB: usize = 128_256;

fn rule(temperature: f32, top_k: usize, top_p: f32, min_p: f32) -> FilterRule {
    FilterRule {
        temperature,
        top_k,
        top_p,
        min_p,
    }
}

/// Deterministic logits with distinct values.
fn logits_data(n: usize, phase: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 * 0.37 + phase).sin() * 4.0) + (i % 7) as f32 * 0.01)
        .collect()
}

/// `sampling_probs` of `data` shaped `shape` on one backend.
fn probs_on<R: Runtime<DType = numr::dtype::DType>>(
    client: &R::Client,
    device: &R::Device,
    data: &[f32],
    shape: &[usize],
    r: FilterRule,
) -> Vec<f32>
where
    R::Client: SamplingProbsOps<R>,
{
    let logits = Tensor::<R>::from_slice(data, shape, device).unwrap();
    let out = client.sampling_probs(&logits, &r).unwrap();
    assert_eq!(out.shape(), shape, "{r:?}");
    out.to_vec::<f32>()
}

fn kept(row: &[f32]) -> Vec<usize> {
    (0..row.len()).filter(|&i| row[i] > 0.0).collect()
}

fn assert_rows_sum_to_one(out: &[f32], vocab: usize, what: &str) {
    for (r, row) in out.chunks(vocab).enumerate() {
        let s: f64 = row.iter().map(|&p| f64::from(p)).sum();
        assert!((s - 1.0).abs() <= 1e-6, "{what}: row {r} sums to {s}");
    }
}

/// Asserts the backend rows keep the CPU tokens with matching probabilities.
fn assert_matches_cpu(gpu: &[f32], cpu: &[f32], vocab: usize, what: &str) {
    assert_eq!(gpu.len(), cpu.len(), "{what}: length");
    for (r, (g, c)) in gpu.chunks(vocab).zip(cpu.chunks(vocab)).enumerate() {
        assert_eq!(kept(g), kept(c), "{what}: row {r} kept set");
    }
    assert_parity_f32_tol(gpu, cpu, what, 1e-4, 1e-7);
    assert_rows_sum_to_one(gpu, vocab, what);
}

/// Runs `data` through the CPU and every enabled backend, then compares them.
fn check(data: &[f32], shape: &[usize], r: FilterRule, what: &str) -> Vec<f32> {
    let (cpu_client, cpu_device) = setup_cpu();
    let vocab = *shape.last().unwrap();
    let cpu = probs_on::<CpuRuntime>(&cpu_client, &cpu_device, data, shape, r);
    assert_rows_sum_to_one(&cpu, vocab, what);

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|client, device| {
        let gpu = probs_on::<numr::runtime::wgpu::WgpuRuntime>(&client, &device, data, shape, r);
        assert_matches_cpu(&gpu, &cpu, vocab, &format!("wgpu {what} {r:?}"));
    });
    cpu
}

#[test]
fn test_sampling_probs_top_k_keeps_ties() {
    let data = [0.4f32.ln(), 0.3f32.ln(), 0.3f32.ln()];
    let cpu = check(&data, &[3], rule(1.0, 2, 1.0, 0.0), "top-k tie");
    assert_eq!(
        kept(&cpu),
        vec![0, 1, 2],
        "kth largest is 0.3: both ties stay"
    );
}

#[test]
fn test_sampling_probs_top_k_one_keeps_the_argmax() {
    let data = logits_data(1000, 0.2);
    let argmax = (0..data.len())
        .max_by(|&a, &b| data[a].total_cmp(&data[b]))
        .unwrap();
    let cpu = check(&data, &[1000], rule(0.8, 1, 1.0, 0.0), "k = 1");
    assert_eq!(kept(&cpu), vec![argmax]);
    assert_eq!(cpu[argmax], 1.0);
}

#[test]
fn test_sampling_probs_top_p_measures_the_mass_top_k_left() {
    let data = [0.5f32.ln(), 0.3f32.ln(), 0.2f32.ln()];
    let after_k = check(&data, &[3], rule(1.0, 2, 0.6, 0.0), "top-p after top-k");
    assert_eq!(kept(&after_k), vec![0], "target 0.6 * 0.8 = 0.48");
    let full = check(&data, &[3], rule(1.0, 0, 0.6, 0.0), "top-p alone");
    assert_eq!(kept(&full), vec![0, 1], "target 0.6");
}

#[test]
fn test_sampling_probs_min_p() {
    let data = [0.5f32.ln(), 0.3f32.ln(), 0.15f32.ln(), 0.05f32.ln()];
    let cpu = check(&data, &[4], rule(1.0, 0, 1.0, 0.25), "min-p");
    assert_eq!(kept(&cpu), vec![0, 1, 2], "threshold 0.125");
}

#[test]
fn test_sampling_probs_each_filter_disabled() {
    let data = logits_data(777, 0.9);
    for r in [
        rule(1.0, 0, 1.0, 0.0),
        rule(1.0, 777, 1.0, 0.0),
        rule(1.0, 0, 1.0, -1.0),
        rule(1.0, 0, 2.0, 0.0),
    ] {
        let cpu = check(&data, &[777], r, "disabled");
        assert_eq!(kept(&cpu).len(), 777, "{r:?} keeps every token");
    }
}

#[test]
fn test_sampling_probs_temperature() {
    let data = logits_data(300, 0.4);
    let hot = check(&data, &[300], rule(2.0, 0, 1.0, 0.1), "temperature 2");
    let cold = check(&data, &[300], rule(0.5, 0, 1.0, 0.1), "temperature 0.5");
    assert!(
        kept(&hot).len() > kept(&cold).len(),
        "a higher temperature flattens the row past more min-p thresholds"
    );
}

#[test]
fn test_sampling_probs_combined_batched() {
    let rules = [
        rule(1.0, 5, 0.9, 0.05),
        rule(0.7, 40, 0.8, 0.0),
        rule(1.3, 0, 0.5, 0.1),
        rule(0.6, 3, 1.0, 0.3),
    ];
    for (shape, phase) in [(vec![4usize, 33], 0.1f32), (vec![2, 3, 1000], 1.7)] {
        let n: usize = shape.iter().product();
        let data = logits_data(n, phase);
        for r in rules {
            check(&data, &shape, r, &format!("combined {shape:?}"));
        }
    }
}

#[test]
fn test_sampling_probs_large_vocab() {
    let data = logits_data(2 * LARGE_VOCAB, 0.5);
    check(
        &data,
        &[2, LARGE_VOCAB],
        rule(0.9, 50, 0.9, 0.02),
        "large vocab",
    );
}

#[test]
fn test_sampling_probs_dominant_token() {
    let mut data = vec![0.0f32; 3000];
    data[2048] = 60.0;
    let cpu = check(&data, &[3000], rule(1.0, 0, 0.9, 0.05), "dominant");
    assert_eq!(kept(&cpu), vec![2048]);
    assert_eq!(cpu[2048], 1.0);
}

#[cfg(feature = "f16")]
#[test]
fn test_sampling_probs_half_input_matches_f32() {
    let (client, device) = setup_cpu();
    let data = logits_data(2 * 257, 0.3);
    let r = rule(0.9, 7, 0.85, 0.02);
    let f16: Vec<half::f16> = data.iter().map(|&x| half::f16::from_f32(x)).collect();
    let bf16: Vec<half::bf16> = data.iter().map(|&x| half::bf16::from_f32(x)).collect();
    let cases = [
        (
            Tensor::<CpuRuntime>::from_slice(&f16, &[2, 257], &device).unwrap(),
            f16.iter().map(|x| x.to_f32()).collect::<Vec<f32>>(),
        ),
        (
            Tensor::<CpuRuntime>::from_slice(&bf16, &[2, 257], &device).unwrap(),
            bf16.iter().map(|x| x.to_f32()).collect::<Vec<f32>>(),
        ),
    ];
    for (half_t, wide) in cases {
        let got = client.sampling_probs(&half_t, &r).unwrap().to_vec::<f32>();
        let want = check(&wide, &[2, 257], r, &format!("{:?} upcast", half_t.dtype()));
        assert_eq!(got, want, "{:?}", half_t.dtype());
    }
}
