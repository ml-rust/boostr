//! The NeuCodec encode path on CUDA, held to the SAME bar as the CPU port:
//! FSQ code indices must match upstream `NeuCodec.encode_code` EXACTLY.
//!
//! Run with:
//!   `cd boostr && cargo test --release --features cuda --test neucodec_encoder_cuda`
//!
//! Why this test exists separately from `neucodec_encoder_parity.rs`: every
//! struct in the encode path is generic over `R: Runtime` and every op it calls
//! has a CUDA kernel, so `NeuCodecEncoder<CudaRuntime>` *should* work with no
//! porting at all. "Should" is not evidence — a differing conv padding
//! convention, a strided-copy bug, or a reduction ordering difference would all
//! surface as shifted codes rather than as a compile error. This runs the real
//! checkpoint on the GPU and compares against the same upstream fixtures.
//!
//! Indices are DISCRETE: the bar is exact equality, not a tolerance. A float
//! difference that stays under any sane epsilon can still flip a code at a
//! quantization boundary, so the index check is the one that matters and the
//! float checks are only there to localize a failure.
//!
//! Note the mel frontend (`seamless_fbank`) is host-side f64 by design, so the
//! semantic branch's input is computed identically on both backends; this test
//! therefore exercises the *tensor* pipeline on CUDA, which is the part that
//! could diverge.

#![cfg(feature = "cuda")]

use boostr::model::audio::neucodec::NeuCodecEncoder;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::runtime::cuda::{CudaDevice, CudaRuntime};
use std::path::PathBuf;

fn read_f32(path: &PathBuf) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert!(
        bytes.len().is_multiple_of(4),
        "{} is not a whole number of f32s",
        path.display()
    );
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_i32(path: &PathBuf) -> Vec<i32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert!(
        bytes.len().is_multiple_of(4),
        "{} is not a whole number of i32s",
        path.display()
    );
    bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    (worst, at)
}

fn fixtures() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var("NEUCODEC_REF_DIR").ok()?);
    dir.join("enc_full_a_wave.f32").exists().then_some(dir)
}

fn checkpoint() -> Option<PathBuf> {
    let p = PathBuf::from(
        std::env::var("NEUCODEC_CHECKPOINT")
            .unwrap_or_else(|_| "/home/farhan/Projects/models/neucodec/model.safetensors".into()),
    );
    p.exists().then_some(p)
}

/// Compare, for both fixture clips: CUDA vs upstream (exact indices) and CUDA
/// vs CPU (exact indices). The CPU leg makes a failure readable — if CUDA
/// disagrees with BOTH, the GPU pipeline is wrong; if CUDA and CPU agree with
/// each other but not upstream, the fixtures or the shared code moved.
#[test]
fn cuda_encode_matches_upstream_and_cpu() {
    let Some(dir) = fixtures() else {
        eprintln!("skipping: set NEUCODEC_REF_DIR (run dump_full_encode.py)");
        return;
    };
    let Some(ckpt) = checkpoint() else {
        eprintln!("skipping: checkpoint absent");
        return;
    };

    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let encoder =
        NeuCodecEncoder::<CudaRuntime>::from_safetensors(&ckpt, &device).expect("load on cuda");

    let cpu_device = CpuDevice::new();
    let cpu_client = CpuClient::new(cpu_device.clone());
    let cpu_encoder =
        NeuCodecEncoder::<CpuRuntime>::from_safetensors(&ckpt, &cpu_device).expect("load on cpu");

    for clip in ["a", "b"] {
        let wave_path = dir.join(format!("enc_full_{clip}_wave.f32"));
        let idx_path = dir.join(format!("enc_full_{clip}_indices.i32"));
        let prior_path = dir.join(format!("enc_full_{clip}_prior.f32"));
        if !wave_path.exists() || !idx_path.exists() || !prior_path.exists() {
            eprintln!("skipping clip {clip}: fixtures absent");
            continue;
        }
        let wave = read_f32(&wave_path);

        let gpu = encoder
            .encode_stages(&client, &wave, &device)
            .unwrap_or_else(|e| panic!("clip {clip}: cuda encode failed: {e}"));
        let cpu = cpu_encoder
            .encode_stages(&cpu_client, &wave, &cpu_device)
            .unwrap_or_else(|e| panic!("clip {clip}: cpu encode failed: {e}"));

        // --- the prior, to localize any divergence before the quantizer ----
        let want_prior = read_f32(&prior_path);
        let got_prior: Vec<f32> = gpu.prior.contiguous().expect("contiguous prior").to_vec();
        assert_eq!(
            got_prior.len(),
            want_prior.len(),
            "clip {clip}: cuda prior length mismatch (shape {:?})",
            gpu.prior.shape()
        );
        let (d, i) = max_abs_diff(&got_prior, &want_prior);
        let scale =
            (want_prior.iter().map(|v| v * v).sum::<f32>() / want_prior.len() as f32).sqrt();
        eprintln!("clip {clip}: cuda prior vs upstream max|d|={d:.3e} at {i}, rms={scale:.3e}");

        let cpu_prior: Vec<f32> = cpu
            .prior
            .contiguous()
            .expect("contiguous cpu prior")
            .to_vec();
        let (dc, ic) = max_abs_diff(&got_prior, &cpu_prior);
        eprintln!("clip {clip}: cuda prior vs cpu       max|d|={dc:.3e} at {ic}");

        // --- the indices: exact, against upstream ---------------------------
        let want_idx = read_i32(&idx_path);
        let gpu_idx: Vec<i32> = gpu.indices.contiguous().expect("contiguous idx").to_vec();
        let cpu_idx: Vec<i32> = cpu
            .indices
            .contiguous()
            .expect("contiguous cpu idx")
            .to_vec();
        assert_eq!(
            gpu_idx.len(),
            want_idx.len(),
            "clip {clip}: cuda index count mismatch (shape {:?})",
            gpu.indices.shape()
        );

        let report = |label: &str, got: &[i32], want: &[i32]| {
            let mism: Vec<(usize, i32, i32)> = got
                .iter()
                .zip(want.iter())
                .enumerate()
                .filter_map(|(i, (&g, &w))| (g != w).then_some((i, g, w)))
                .collect();
            let frac = mism.len() as f64 / got.len() as f64 * 100.0;
            eprintln!(
                "clip {clip}: {label} {}/{} mismatched ({frac:.2}%)",
                mism.len(),
                got.len()
            );
            assert!(
                mism.is_empty(),
                "clip {clip}: {label} diverges: {}/{} ({frac:.2}%); \
                 first few (position, got, want): {:?}",
                mism.len(),
                got.len(),
                mism.iter().take(5).collect::<Vec<_>>(),
            );
        };
        report("cuda vs upstream", &gpu_idx, &want_idx);
        report("cuda vs cpu     ", &gpu_idx, &cpu_idx);
    }
}
