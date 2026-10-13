//! Shared test-construction helpers for the `whisper_model` module's test
//! suites: a tiny random Whisper model small enough to run a full greedy
//! decode in a unit test.

use super::model::WhisperModel;
use crate::model::config::AudioConfig;
use crate::nn::{VarBuilder, VarMap};
use numr::runtime::Runtime as _;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

pub(crate) const HIDDEN: usize = 8;
pub(crate) const HEADS: usize = 2;
pub(crate) const MEL_BINS: usize = 4;
pub(crate) const VOCAB: usize = 12;
pub(crate) const MAX_TARGET: usize = 16;
pub(crate) const DECODER_LAYERS: usize = 2;

pub(crate) fn cpu_setup() -> (CpuClient, CpuDevice) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    (client, device)
}

pub(crate) fn tiny_config() -> AudioConfig {
    AudioConfig {
        encoder_type: "whisper".to_string(),
        hidden_size: HIDDEN,
        num_layers: 1,
        num_heads: HEADS,
        num_mel_bins: MEL_BINS,
        max_audio_len: 8,
        projector_type: "linear".to_string(),
        vocab_size: VOCAB,
        decoder_layers: Some(DECODER_LAYERS),
        max_target_positions: MAX_TARGET,
        intermediate_size: None,
    }
}

/// Deterministic pseudo-random fill (no `rand` dependency), scaled small to
/// keep the tiny model numerically stable across a few layers.
pub(crate) fn fill(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i as u32).wrapping_add(seed).wrapping_mul(2_654_435_761);
            ((x >> 8) as f32 / u32::MAX as f32 - 0.5) * 0.2
        })
        .collect()
}

fn insert(
    varmap: &mut VarMap<CpuRuntime>,
    device: &CpuDevice,
    name: &str,
    shape: &[usize],
    seed: u32,
) {
    let data = fill(shape.iter().product(), seed);
    let tensor = Tensor::<CpuRuntime>::from_slice(&data, shape, device).expect("build tensor");
    varmap.insert(name.to_string(), tensor);
}

fn insert_linear(
    varmap: &mut VarMap<CpuRuntime>,
    device: &CpuDevice,
    prefix: &str,
    out_dim: usize,
    in_dim: usize,
    seed: u32,
) {
    insert(
        varmap,
        device,
        &format!("{prefix}.weight"),
        &[out_dim, in_dim],
        seed,
    );
    insert(
        varmap,
        device,
        &format!("{prefix}.bias"),
        &[out_dim],
        seed.wrapping_add(1),
    );
}

/// Identity-ish LayerNorm init (weight 1, bias 0) keeps the residual stream
/// well-scaled across several layers; a near-zero random gamma collapses it
/// toward zero instead.
fn insert_layernorm(varmap: &mut VarMap<CpuRuntime>, device: &CpuDevice, prefix: &str, dim: usize) {
    varmap.insert(
        format!("{prefix}.weight"),
        Tensor::<CpuRuntime>::from_slice(&vec![1.0f32; dim], &[dim], device).expect("ln weight"),
    );
    varmap.insert(
        format!("{prefix}.bias"),
        Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; dim], &[dim], device).expect("ln bias"),
    );
}

fn insert_attn(
    varmap: &mut VarMap<CpuRuntime>,
    device: &CpuDevice,
    prefix: &str,
    dim: usize,
    seed_base: u32,
) {
    insert_linear(
        varmap,
        device,
        &format!("{prefix}.q_proj"),
        dim,
        dim,
        seed_base,
    );
    insert_linear(
        varmap,
        device,
        &format!("{prefix}.k_proj"),
        dim,
        dim,
        seed_base + 10,
    );
    insert_linear(
        varmap,
        device,
        &format!("{prefix}.v_proj"),
        dim,
        dim,
        seed_base + 20,
    );
    insert_linear(
        varmap,
        device,
        &format!("{prefix}.out_proj"),
        dim,
        dim,
        seed_base + 30,
    );
}

/// Every tensor name `WhisperModel::from_varbuilder` reads, filled with
/// small deterministic values. The encoder is never run by `generate`, so
/// its weights only need to exist, not to be meaningful.
fn tiny_varmap(device: &CpuDevice) -> VarMap<CpuRuntime> {
    let mut varmap = VarMap::<CpuRuntime>::new();
    let ffn = HIDDEN * 4;

    insert(
        &mut varmap,
        device,
        "model.encoder.conv1.weight",
        &[HIDDEN, MEL_BINS, 3],
        1,
    );
    insert(
        &mut varmap,
        device,
        "model.encoder.conv1.bias",
        &[HIDDEN],
        2,
    );
    insert(
        &mut varmap,
        device,
        "model.encoder.conv2.weight",
        &[HIDDEN, HIDDEN, 3],
        3,
    );
    insert(
        &mut varmap,
        device,
        "model.encoder.conv2.bias",
        &[HIDDEN],
        4,
    );
    insert(
        &mut varmap,
        device,
        "model.encoder.embed_positions.weight",
        &[1, HIDDEN],
        5,
    );
    insert_attn(
        &mut varmap,
        device,
        "model.encoder.layers.0.self_attn",
        HIDDEN,
        100,
    );
    insert_layernorm(
        &mut varmap,
        device,
        "model.encoder.layers.0.self_attn_layer_norm",
        HIDDEN,
    );
    insert_linear(
        &mut varmap,
        device,
        "model.encoder.layers.0.fc1",
        ffn,
        HIDDEN,
        200,
    );
    insert_linear(
        &mut varmap,
        device,
        "model.encoder.layers.0.fc2",
        HIDDEN,
        ffn,
        210,
    );
    insert_layernorm(
        &mut varmap,
        device,
        "model.encoder.layers.0.final_layer_norm",
        HIDDEN,
    );
    insert_layernorm(&mut varmap, device, "model.encoder.layer_norm", HIDDEN);

    insert(
        &mut varmap,
        device,
        "model.decoder.embed_tokens.weight",
        &[VOCAB, HIDDEN],
        300,
    );
    insert(
        &mut varmap,
        device,
        "model.decoder.embed_positions.weight",
        &[MAX_TARGET, HIDDEN],
        301,
    );
    for i in 0..DECODER_LAYERS {
        let base = 1000 + i as u32 * 100;
        let p = format!("model.decoder.layers.{i}");
        insert_attn(&mut varmap, device, &format!("{p}.self_attn"), HIDDEN, base);
        insert_layernorm(
            &mut varmap,
            device,
            &format!("{p}.self_attn_layer_norm"),
            HIDDEN,
        );
        insert_attn(
            &mut varmap,
            device,
            &format!("{p}.encoder_attn"),
            HIDDEN,
            base + 40,
        );
        insert_layernorm(
            &mut varmap,
            device,
            &format!("{p}.encoder_attn_layer_norm"),
            HIDDEN,
        );
        insert_linear(
            &mut varmap,
            device,
            &format!("{p}.fc1"),
            ffn,
            HIDDEN,
            base + 80,
        );
        insert_linear(
            &mut varmap,
            device,
            &format!("{p}.fc2"),
            HIDDEN,
            ffn,
            base + 85,
        );
        insert_layernorm(
            &mut varmap,
            device,
            &format!("{p}.final_layer_norm"),
            HIDDEN,
        );
    }
    insert_layernorm(&mut varmap, device, "model.decoder.layer_norm", HIDDEN);

    varmap
}

pub(crate) fn tiny_model(device: &CpuDevice) -> WhisperModel<CpuRuntime> {
    let config = tiny_config();
    let mut varmap = tiny_varmap(device);
    let mut vb = VarBuilder::new(&mut varmap, device);
    WhisperModel::from_varbuilder(&mut vb, &config).expect("build tiny whisper model")
}

pub(crate) fn tiny_encoder_out(
    device: &CpuDevice,
    src_len: usize,
    seed: u32,
) -> Tensor<CpuRuntime> {
    Tensor::<CpuRuntime>::from_slice(&fill(src_len * HIDDEN, seed), &[1, src_len, HIDDEN], device)
        .expect("encoder_out")
}
