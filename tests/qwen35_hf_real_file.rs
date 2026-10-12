//! End-to-end `qwen35` load + CPU forward on a real Hugging Face Qwen3.5
//! text checkpoint: config through `load_config_auto`, weights through
//! `SafeTensorsLoader` with the MTP and vision tensors skipped before they
//! load, the layout through `restructure_hf_checkpoint`, the model through
//! `LoadedModel::load`, then a 4-token prefill and a 1-token decode.
//!
//! # Fixture directory
//!
//! `BOOSTR_QWEN35_HF_DIR` must hold the checkpoint's `config.json` and its
//! SafeTensors files, single or sharded. `Qwen/Qwen3.5-4B-Base` is one,
//! with two value heads per key head, so the value-head reorder runs.
//! `Qwen/Qwen3.5-0.8B` has one value head per key head, so the reorder is
//! the identity there. Unset or missing, every test prints one `skip:` line
//! and passes.
//!
//! ```bash
//! BOOSTR_QWEN35_HF_DIR=/path/to/dir cargo nextest run -p boostr \
//!   --test qwen35_hf_real_file --no-capture
//! ```
//!
//! # Memory
//!
//! Every float tensor is cast to F32 on load, so expect peak RSS near four
//! bytes per parameter of the text model.

use std::path::PathBuf;

use boostr::format::hf_qwen35::is_dropped;
use boostr::format::{SafeTensorsLoader, restructure_hf_checkpoint};
use boostr::inference::{LayeredGdnState, LayeredKvCache, LayeredKvCacheConfig};
use boostr::model::{LoadedModel, Qwen35Model, UniversalConfig, load_config_auto};
use boostr::nn::{VarBuilder, VarMap};
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

const ENV_DIR: &str = "BOOSTR_QWEN35_HF_DIR";
const PROMPT: [i64; 4] = [248_044, 17, 42, 99];
const NEXT: i64 = 7;

/// The checkpoint directory, or `None` after one `skip:` line.
fn require_dir() -> Option<PathBuf> {
    let Some(dir) = std::env::var(ENV_DIR).ok().map(PathBuf::from) else {
        println!("skip: {ENV_DIR} not set");
        return None;
    };
    if !dir.join("config.json").is_file() {
        println!("skip: {} has no config.json", dir.display());
        return None;
    }
    Some(dir)
}

fn config(dir: &std::path::Path) -> UniversalConfig {
    let path = dir.join("config.json");
    load_config_auto(&path).unwrap_or_else(|e| panic!("load_config_auto {}: {e}", path.display()))
}

/// Every text-model tensor of `dir`, as F32, in Hugging Face naming.
fn load_text_tensors(
    dir: &std::path::Path,
    client: &CpuClient,
    device: &CpuDevice,
) -> VarMap<CpuRuntime> {
    let mut loader = SafeTensorsLoader::open(dir)
        .unwrap_or_else(|e| panic!("SafeTensorsLoader::open {}: {e}", dir.display()));
    let mut map = VarMap::<CpuRuntime>::new();
    for name in loader.tensor_names() {
        if is_dropped(&name) {
            continue;
        }
        let t = loader
            .load_tensor::<CpuRuntime>(&name, device)
            .unwrap_or_else(|e| panic!("load {name}: {e}"));
        let t = if matches!(t.dtype(), DType::F16 | DType::BF16) {
            client
                .cast(&t, DType::F32)
                .unwrap_or_else(|e| panic!("cast {name}: {e}"))
        } else {
            t
        };
        map.insert(name, t);
    }
    map
}

fn caches(
    model: &Qwen35Model<CpuRuntime>,
    config: &UniversalConfig,
    device: &CpuDevice,
) -> (LayeredKvCache<CpuRuntime>, LayeredGdnState<CpuRuntime>) {
    let attn = model.attention_config();
    let kv_config = LayeredKvCacheConfig {
        batch_size: 1,
        num_kv_heads: attn.num_kv_heads,
        initial_capacity: 16,
        max_seq_len: config.max_seq_len,
        head_dim: attn.head_dim,
        dtype: DType::F32,
    };
    let kv = LayeredKvCache::<CpuRuntime>::new(model.num_attention_layers(), &kv_config, device)
        .unwrap_or_else(|e| panic!("LayeredKvCache::new: {e}"));
    let gdn = LayeredGdnState::<CpuRuntime>::zeros(
        model.num_gdn_layers(),
        model.gdn_config(),
        1,
        DType::F32,
        device,
    )
    .unwrap_or_else(|e| panic!("LayeredGdnState::zeros: {e}"));
    (kv, gdn)
}

fn finite(logits: &Tensor<CpuRuntime>, label: &str) -> Vec<f32> {
    let data = logits.to_vec::<f32>();
    let bad = data.iter().position(|v| !v.is_finite());
    assert!(
        bad.is_none(),
        "{label}: non-finite logit at flat index {bad:?}"
    );
    data
}

#[test]
fn config_json_derives_a_qwen35_config() {
    let Some(dir) = require_dir() else {
        return;
    };
    let config = config(&dir);
    assert_eq!(config.model_type, "qwen35");
    assert!(config.vision.is_none());
    let gdn = config.gdn.as_ref().expect("gdn config");
    assert_eq!(gdn.inner_size, gdn.value_heads * gdn.state_size);
    let attn = config.qwen35_attention.as_ref().expect("attention config");
    assert_eq!(attn.rope_sections.iter().sum::<usize>() * 2, attn.rope_dim);
    let layers = config.hybrid_layers.as_ref().expect("hybrid_layers");
    assert_eq!(
        layers.attention_layers.len() + layers.ssm_layers.len(),
        config.num_layers
    );
}

/// The model built from `dir` through the full Hugging Face load path.
fn build_model(
    dir: &std::path::Path,
    config: &UniversalConfig,
    client: &CpuClient,
    device: &CpuDevice,
) -> Box<Qwen35Model<CpuRuntime>> {
    let mut map = load_text_tensors(dir, client, device);
    restructure_hf_checkpoint(&mut map, config, device)
        .unwrap_or_else(|e| panic!("restructure_hf_checkpoint: {e}"));
    let mut vb = VarBuilder::new(&mut map, device);
    let loaded = LoadedModel::<CpuRuntime>::load(config, &mut vb)
        .unwrap_or_else(|e| panic!("LoadedModel::load: {e}"));
    let model = match loaded {
        LoadedModel::Qwen35(model) => model,
        other => panic!("LoadedModel::load returned {other:?}, expected Qwen35"),
    };
    assert_eq!(
        map.len(),
        0,
        "loader left {} tensors behind: {:?}",
        map.len(),
        map.names().take(8).collect::<Vec<_>>()
    );
    model
}

/// A long prompt's logits must match a reference at positions around the
/// chunk boundaries of the recurrent layers.
///
/// `BOOSTR_QWEN35_HF_LONG` is a path prefix: `<prefix>.json` holds `ids` and
/// `positions`, `<prefix>.f32` holds the reference logits at those positions,
/// `[positions, vocab]` little-endian F32.
#[test]
fn long_prefill_matches_reference_logits() {
    let Some(dir) = require_dir() else {
        return;
    };
    let Some(prefix) = std::env::var("BOOSTR_QWEN35_HF_LONG").ok() else {
        println!("skip: BOOSTR_QWEN35_HF_LONG not set");
        return;
    };
    let meta: serde_json::Value = serde_json::from_slice(
        &std::fs::read(format!("{prefix}.json")).unwrap_or_else(|e| panic!("read ids: {e}")),
    )
    .unwrap_or_else(|e| panic!("parse ids: {e}"));
    let ids: Vec<i64> = meta["ids"]
        .as_array()
        .expect("ids")
        .iter()
        .map(|v| v.as_i64().expect("id"))
        .collect();
    let positions: Vec<usize> = meta["positions"]
        .as_array()
        .expect("positions")
        .iter()
        .map(|v| v.as_u64().expect("position") as usize)
        .collect();

    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    let config = config(&dir);
    let model = build_model(&dir, &config, &client, &device);
    let vocab = config.vocab_size;

    let (mut kv, mut gdn) = caches(&model, &config, &device);
    let ids_t = Tensor::<CpuRuntime>::from_slice(&ids, &[1, ids.len()], &device)
        .unwrap_or_else(|e| panic!("prompt tensor: {e}"));
    let logits = model
        .forward_qwen35(&client, &ids_t, &mut kv, &mut gdn, 0)
        .unwrap_or_else(|e| panic!("prefill forward: {e}"));
    let got = finite(&logits, "long prefill");

    let bytes = std::fs::read(format!("{prefix}.f32")).unwrap_or_else(|e| panic!("read ref: {e}"));
    let (words, tail) = bytes.as_chunks::<4>();
    assert!(tail.is_empty(), "reference file is not whole f32 values");
    let want: Vec<f32> = words.iter().map(|b| f32::from_le_bytes(*b)).collect();
    assert_eq!(
        want.len(),
        positions.len() * vocab,
        "reference size mismatch"
    );

    let mut worst = 0.0f32;
    for (i, &pos) in positions.iter().enumerate() {
        let g = &got[pos * vocab..(pos + 1) * vocab];
        let w = &want[i * vocab..(i + 1) * vocab];
        let diff = g
            .iter()
            .zip(w)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("pos {pos}: max abs diff {diff}");
        worst = worst.max(diff);
    }
    assert!(
        worst < 5e-2,
        "long prefill differs from the reference by {worst}"
    );
}

/// Prefill logits must match a reference dump of the same checkpoint.
///
/// `BOOSTR_QWEN35_HF_REF` names a little-endian F32 file holding the
/// reference logits, `[PROMPT + NEXT, vocab]` row-major, produced by running
/// the Hugging Face model on those token ids in F32.
#[test]
fn prefill_matches_reference_logits() {
    let Some(dir) = require_dir() else {
        return;
    };
    let Some(reference) = std::env::var("BOOSTR_QWEN35_HF_REF")
        .ok()
        .map(PathBuf::from)
    else {
        println!("skip: BOOSTR_QWEN35_HF_REF not set");
        return;
    };
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    let config = config(&dir);
    let model = build_model(&dir, &config, &client, &device);

    let mut all = PROMPT.to_vec();
    all.push(NEXT);
    let vocab = config.vocab_size;
    let (mut kv, mut gdn) = caches(&model, &config, &device);
    let ids = Tensor::<CpuRuntime>::from_slice(&all, &[1, all.len()], &device)
        .unwrap_or_else(|e| panic!("prompt tensor: {e}"));
    let logits = model
        .forward_qwen35(&client, &ids, &mut kv, &mut gdn, 0)
        .unwrap_or_else(|e| panic!("prefill forward: {e}"));
    let got = finite(&logits, "prefill");

    let bytes =
        std::fs::read(&reference).unwrap_or_else(|e| panic!("read {}: {e}", reference.display()));
    let (words, tail) = bytes.as_chunks::<4>();
    assert!(
        tail.is_empty(),
        "reference file is not a whole number of f32"
    );
    let want: Vec<f32> = words.iter().map(|b| f32::from_le_bytes(*b)).collect();
    assert_eq!(want.len(), all.len() * vocab, "reference size mismatch");

    for pos in 0..all.len() {
        let g = &got[pos * vocab..(pos + 1) * vocab];
        let w = &want[pos * vocab..(pos + 1) * vocab];
        let diff = g
            .iter()
            .zip(w)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let argmax = |row: &[f32]| {
            row.iter()
                .enumerate()
                .fold((0, f32::MIN), |m, (i, &v)| if v > m.1 { (i, v) } else { m })
                .0
        };
        println!(
            "pos {pos}: max abs diff {diff}, argmax {} vs {}",
            argmax(g),
            argmax(w)
        );
        assert_eq!(argmax(g), argmax(w), "pos {pos}: argmax differs");
        assert!(diff < 5e-2, "pos {pos}: logits differ by {diff}");
    }
}

#[test]
fn loads_and_decode_matches_prefill() {
    let Some(dir) = require_dir() else {
        return;
    };
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    let config = config(&dir);
    let model = build_model(&dir, &config, &client, &device);

    let vocab = config.vocab_size;
    let (mut kv, mut gdn) = caches(&model, &config, &device);
    let ids = Tensor::<CpuRuntime>::from_slice(&PROMPT, &[1, PROMPT.len()], &device)
        .unwrap_or_else(|e| panic!("prompt tensor: {e}"));
    let logits = model
        .forward_qwen35(&client, &ids, &mut kv, &mut gdn, 0)
        .unwrap_or_else(|e| panic!("prefill forward: {e}"));
    assert_eq!(logits.shape(), &[1, PROMPT.len(), vocab]);
    finite(&logits, "prefill");

    let next = Tensor::<CpuRuntime>::from_slice(&[NEXT], &[1, 1], &device)
        .unwrap_or_else(|e| panic!("decode tensor: {e}"));
    let decoded = model
        .forward_qwen35(&client, &next, &mut kv, &mut gdn, PROMPT.len())
        .unwrap_or_else(|e| panic!("decode forward: {e}"));
    assert_eq!(decoded.shape(), &[1, 1, vocab]);
    let decoded = finite(&decoded, "decode");

    let mut all = PROMPT.to_vec();
    all.push(NEXT);
    let (mut kv, mut gdn) = caches(&model, &config, &device);
    let ids = Tensor::<CpuRuntime>::from_slice(&all, &[1, all.len()], &device)
        .unwrap_or_else(|e| panic!("full prompt tensor: {e}"));
    let full = model
        .forward_qwen35(&client, &ids, &mut kv, &mut gdn, 0)
        .unwrap_or_else(|e| panic!("full prefill forward: {e}"));
    let full = finite(&full, "full prefill");
    let last = &full[PROMPT.len() * vocab..];
    let diff = last
        .iter()
        .zip(&decoded)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(diff < 1e-2, "decode step differs from prefill by {diff}");
}
