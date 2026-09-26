//! Numerical parity for the VITS building blocks (`VitsAttention`,
//! `VitsDilatedDepthSeparableConv`, `VitsWaveNet`) against HF
//! `transformers.models.vits.modeling_vits`.
//!
//! Fixtures are checked in under `tests/fixtures/vits_blocks/` (~70 KB, a
//! hidden_size=8/heads=2/window=2 config — small enough to commit). Regenerate
//! with `python3 tests/tools/vits_blocks_oracle/vits_blocks_oracle.py
//! tests/fixtures/vits_blocks` (requires torch, transformers, safetensors).
//!
//! What this pins that unit tests cannot: the exact windowed relative-position
//! bias added to both the attention scores and the attention-weighted values,
//! the additive-mask broadcast over the head axis, the dilation schedule and
//! padding for the depthwise conv (`kernel_size^i` for DDS, `dilation_rate^i`
//! for WaveNet — two different bases), the erf-vs-tanh GELU choice, the
//! tanh*sigmoid gate split point, and the weight-norm fusion of the WaveNet
//! convs read through the real `VarBuilder` loader path.

use boostr::format::SafeTensors;
use boostr::model::audio::vits::{VitsAttention, VitsDilatedDepthSeparableConv, VitsWaveNet};
use boostr::nn::VarBuilder;
use boostr::nn::varmap::VarMap;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;
use std::path::{Path, PathBuf};

const TOL: f32 = 1e-5;

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/vits_blocks")
}

#[derive(serde::Deserialize)]
struct Config {
    hidden_size: usize,
    num_heads: usize,
    window_size: usize,
    kernel_size: usize,
    dds_layers: usize,
    wavenet_layers: usize,
    dilation_rate: usize,
    #[allow(dead_code)]
    speaker_dim: usize,
    #[allow(dead_code)]
    batch: usize,
    #[allow(dead_code)]
    seq: usize,
}

fn config() -> Config {
    let text = std::fs::read_to_string(fixtures_dir().join("config.json"))
        .expect("read tests/fixtures/vits_blocks/config.json");
    serde_json::from_str(&text).expect("parse config.json")
}

fn load_tensor(st: &mut SafeTensors, name: &str, device: &CpuDevice) -> Tensor<CpuRuntime> {
    st.load_tensor::<CpuRuntime>(name, device)
        .unwrap_or_else(|e| panic!("missing tensor '{name}': {e}"))
}

/// Largest absolute difference, plus the index where it occurs.
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

fn assert_allclose(name: &str, got: &Tensor<CpuRuntime>, want: &Tensor<CpuRuntime>) {
    assert_eq!(got.shape(), want.shape(), "{name}: shape mismatch");
    let g: Vec<f32> = got.to_vec();
    let w: Vec<f32> = want.to_vec();
    let (diff, at) = max_abs_diff(&g, &w);
    assert!(
        diff <= TOL,
        "{name}: max abs diff {diff} at index {at} (got {} vs want {})",
        g[at],
        w[at]
    );
}

#[test]
fn attention_matches_hf_oracle() {
    let cfg = config();
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    let dir = fixtures_dir().join("attention");

    let mut map = VarMap::<CpuRuntime>::from_safetensors(dir.join("weights.safetensors"), &device)
        .expect("load attention weights.safetensors");
    let mut vb = VarBuilder::new(&mut map, &device);
    let attn = VitsAttention::<CpuRuntime>::from_varbuilder(
        &mut vb,
        cfg.hidden_size,
        cfg.num_heads,
        Some(cfg.window_size),
        true,
    )
    .expect("build VitsAttention from checkpoint");

    let mut io =
        SafeTensors::open(dir.join("io.safetensors")).expect("open attention io.safetensors");
    let input = load_tensor(&mut io, "input", &device);
    let attention_mask = load_tensor(&mut io, "attention_mask", &device);
    let want_output = load_tensor(&mut io, "output", &device);
    let want_output_masked = load_tensor(&mut io, "output_masked", &device);

    let got = attn
        .forward(&client, &input, None)
        .expect("attention forward (no mask)");
    assert_allclose("attention (no mask)", &got, &want_output);

    let got_masked = attn
        .forward(&client, &input, Some(&attention_mask))
        .expect("attention forward (masked)");
    assert_allclose("attention (masked)", &got_masked, &want_output_masked);
}

#[test]
fn dds_conv_matches_hf_oracle() {
    let cfg = config();
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    let dir = fixtures_dir().join("dds_conv");

    let mut map = VarMap::<CpuRuntime>::from_safetensors(dir.join("weights.safetensors"), &device)
        .expect("load dds_conv weights.safetensors");
    let mut vb = VarBuilder::new(&mut map, &device);
    let dds = VitsDilatedDepthSeparableConv::<CpuRuntime>::from_varbuilder(
        &mut vb,
        cfg.hidden_size,
        cfg.kernel_size,
        cfg.dds_layers,
    )
    .expect("build VitsDilatedDepthSeparableConv from checkpoint");

    let mut io =
        SafeTensors::open(dir.join("io.safetensors")).expect("open dds_conv io.safetensors");
    let input = load_tensor(&mut io, "input", &device);
    let padding_mask = load_tensor(&mut io, "padding_mask", &device);
    let want_output = load_tensor(&mut io, "output", &device);
    let want_output_masked = load_tensor(&mut io, "output_masked", &device);

    // The oracle's "output" case runs with an all-ones mask (no padding).
    let full_mask = Tensor::<CpuRuntime>::from_slice(
        &vec![1.0f32; padding_mask.shape().iter().product()],
        padding_mask.shape(),
        &device,
    )
    .unwrap();
    let got = dds
        .forward(&client, &input, &full_mask, None)
        .expect("dds_conv forward (full mask)");
    assert_allclose("dds_conv (full mask)", &got, &want_output);

    let got_masked = dds
        .forward(&client, &input, &padding_mask, None)
        .expect("dds_conv forward (padding mask)");
    assert_allclose("dds_conv (padding mask)", &got_masked, &want_output_masked);
}

#[test]
fn wavenet_matches_hf_oracle() {
    let cfg = config();
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    let dir = fixtures_dir().join("wavenet");

    let mut map = VarMap::<CpuRuntime>::from_safetensors(dir.join("weights.safetensors"), &device)
        .expect("load wavenet weights.safetensors");
    let mut vb = VarBuilder::new(&mut map, &device);
    let wn = VitsWaveNet::<CpuRuntime>::from_varbuilder(
        &client,
        &mut vb,
        cfg.hidden_size,
        cfg.kernel_size,
        cfg.dilation_rate,
        cfg.wavenet_layers,
        0,
    )
    .expect("build VitsWaveNet (no speaker conditioning) from checkpoint");

    let mut io =
        SafeTensors::open(dir.join("io.safetensors")).expect("open wavenet io.safetensors");
    let input = load_tensor(&mut io, "input", &device);
    let padding_mask = load_tensor(&mut io, "padding_mask", &device);
    let want_output = load_tensor(&mut io, "output", &device);
    let want_output_masked = load_tensor(&mut io, "output_masked", &device);

    let full_mask = Tensor::<CpuRuntime>::from_slice(
        &vec![1.0f32; padding_mask.shape().iter().product()],
        padding_mask.shape(),
        &device,
    )
    .unwrap();
    let got = wn
        .forward(&client, &input, &full_mask, None)
        .expect("wavenet forward (full mask)");
    assert_allclose("wavenet (full mask)", &got, &want_output);

    let got_masked = wn
        .forward(&client, &input, &padding_mask, None)
        .expect("wavenet forward (padding mask)");
    assert_allclose("wavenet (padding mask)", &got_masked, &want_output_masked);
}

#[test]
fn wavenet_with_global_conditioning_matches_hf_oracle() {
    let cfg = config();
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    let dir = fixtures_dir().join("wavenet_cond");

    let mut map = VarMap::<CpuRuntime>::from_safetensors(dir.join("weights.safetensors"), &device)
        .expect("load wavenet_cond weights.safetensors");
    let mut vb = VarBuilder::new(&mut map, &device);
    let wn = VitsWaveNet::<CpuRuntime>::from_varbuilder(
        &client,
        &mut vb,
        cfg.hidden_size,
        cfg.kernel_size,
        cfg.dilation_rate,
        cfg.wavenet_layers,
        cfg.speaker_dim,
    )
    .expect("build VitsWaveNet (with speaker conditioning) from checkpoint");

    let mut io =
        SafeTensors::open(dir.join("io.safetensors")).expect("open wavenet_cond io.safetensors");
    let input = load_tensor(&mut io, "input", &device);
    let padding_mask = load_tensor(&mut io, "padding_mask", &device);
    let global_conditioning = load_tensor(&mut io, "global_conditioning", &device);
    let want_output = load_tensor(&mut io, "output", &device);

    let got = wn
        .forward(&client, &input, &padding_mask, Some(&global_conditioning))
        .expect("wavenet forward (global conditioning)");
    assert_allclose("wavenet (global conditioning)", &got, &want_output);
}
