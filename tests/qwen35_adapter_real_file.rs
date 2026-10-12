//! A real PEFT LoRA adapter on a real Hugging Face Qwen3.5 text checkpoint:
//! the base through `SafeTensorsLoader`, `restructure_hf_checkpoint` and
//! `LoadedModel::load_with_adapter`, the adapter through
//! `PeftAdapter::load`, attached unmerged, then a 5-token prefill checked
//! against reference logits.
//!
//! # Fixtures
//!
//! - `BOOSTR_QWEN35_HF_DIR`: the base checkpoint's `config.json` and its
//!   SafeTensors files, as for `qwen35_hf_real_file`.
//! - `BOOSTR_QWEN35_ADAPTER_DIR`: the adapter's `adapter_config.json` and
//!   `adapter_model.safetensors`, trained on that base.
//! - `BOOSTR_QWEN35_ADAPTER_REF`: a little-endian F32 file holding the
//!   reference logits, `[5, vocab]` row-major, produced by running the
//!   Hugging Face base with the PEFT adapter applied unmerged, in F32, on
//!   the token ids `PROMPT`.
//!
//! Unset or missing, each test prints one `skip:` line and passes.
//!
//! ```bash
//! BOOSTR_QWEN35_HF_DIR=/path/to/base BOOSTR_QWEN35_ADAPTER_DIR=/path/to/adapter \
//!   BOOSTR_QWEN35_ADAPTER_REF=/path/to/ref.f32 cargo nextest run -p boostr \
//!   --test qwen35_adapter_real_file --no-capture
//! ```
//!
//! # Memory
//!
//! Every float tensor of the base is cast to F32 on load, so expect peak
//! RSS near four bytes per parameter of the text model. The adapter keeps
//! its file dtype until the attach casts it to F32.

use std::path::{Path, PathBuf};

use boostr::format::hf_qwen35::is_dropped;
use boostr::format::{PeftAdapter, SafeTensorsLoader, restructure_hf_checkpoint};
use boostr::inference::{LayeredGdnState, LayeredKvCache, LayeredKvCacheConfig};
use boostr::model::{LoadedModel, Qwen35Model, UniversalConfig, load_config_auto};
use boostr::nn::{VarBuilder, VarMap};
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

const ENV_DIR: &str = "BOOSTR_QWEN35_HF_DIR";
const ENV_ADAPTER: &str = "BOOSTR_QWEN35_ADAPTER_DIR";
const ENV_REF: &str = "BOOSTR_QWEN35_ADAPTER_REF";
const PROMPT: [i64; 5] = [248_044, 17, 42, 99, 7];

/// The value of `var` as a path, or `None` after one `skip:` line.
fn env_path(var: &str) -> Option<PathBuf> {
    let path = std::env::var(var).ok().map(PathBuf::from);
    if path.is_none() {
        println!("skip: {var} not set");
    }
    path
}

/// The base directory, or `None` after one `skip:` line.
fn require_dir() -> Option<PathBuf> {
    let dir = env_path(ENV_DIR)?;
    if !dir.join("config.json").is_file() {
        println!("skip: {} has no config.json", dir.display());
        return None;
    }
    Some(dir)
}

/// The adapter directory, or `None` after one `skip:` line.
fn require_adapter() -> Option<PathBuf> {
    let dir = env_path(ENV_ADAPTER)?;
    if !dir.join("adapter_config.json").is_file() {
        println!("skip: {} has no adapter_config.json", dir.display());
        return None;
    }
    Some(dir)
}

fn config(dir: &Path) -> UniversalConfig {
    let path = dir.join("config.json");
    load_config_auto(&path).unwrap_or_else(|e| panic!("load_config_auto {}: {e}", path.display()))
}

/// Every text-model tensor of `dir`, as F32, in Hugging Face naming.
fn load_text_tensors(dir: &Path, client: &CpuClient, device: &CpuDevice) -> VarMap<CpuRuntime> {
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

/// The base built from `dir` through the full Hugging Face load path, with
/// `adapter` attached unmerged.
fn build_model(
    dir: &Path,
    config: &UniversalConfig,
    adapter: &PeftAdapter<CpuRuntime>,
    client: &CpuClient,
    device: &CpuDevice,
) -> Box<Qwen35Model<CpuRuntime>> {
    let mut map = load_text_tensors(dir, client, device);
    restructure_hf_checkpoint(&mut map, config, device)
        .unwrap_or_else(|e| panic!("restructure_hf_checkpoint: {e}"));
    let mut vb = VarBuilder::new(&mut map, device);
    let loaded = LoadedModel::<CpuRuntime>::load_with_adapter(config, &mut vb, Some(adapter))
        .unwrap_or_else(|e| panic!("LoadedModel::load_with_adapter: {e}"));
    let model = match loaded {
        LoadedModel::Qwen35(model) => model,
        other => panic!("LoadedModel::load_with_adapter returned {other:?}, expected Qwen35"),
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

/// Every module of the adapter attaches, and prefill logits match a
/// reference dump of the same base with the same adapter.
#[test]
fn adapted_prefill_matches_reference_logits() {
    let Some(dir) = require_dir() else {
        return;
    };
    let Some(adapter_dir) = require_adapter() else {
        return;
    };
    let Some(reference) = env_path(ENV_REF) else {
        return;
    };
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    let config = config(&dir);
    let adapter = PeftAdapter::<CpuRuntime>::load(&adapter_dir, &device)
        .unwrap_or_else(|e| panic!("PeftAdapter::load {}: {e}", adapter_dir.display()));
    let modules = adapter.layers.len();
    let model = build_model(&dir, &config, &adapter, &client, &device);
    println!("adapted {} modules", model.adapted_modules().len());
    assert_eq!(model.adapted_modules().len(), modules);

    let vocab = config.vocab_size;
    let (mut kv, mut gdn) = caches(&model, &config, &device);
    let ids = Tensor::<CpuRuntime>::from_slice(&PROMPT, &[1, PROMPT.len()], &device)
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
    assert_eq!(want.len(), PROMPT.len() * vocab, "reference size mismatch");

    for pos in 0..PROMPT.len() {
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
