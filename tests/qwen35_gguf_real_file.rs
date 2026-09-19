//! End-to-end `qwen35` load + CPU forward on a real Ternary-Bonsai-2-27B
//! `PQ2_0` checkpoint: config from the header, weights through
//! `VarMap::from_gguf`, model through `LoadedModel::load`, then a 4-token
//! prefill and a 1-token decode.
//!
//! # Fixture file
//!
//! `BOOSTR_BONSAI2_DIR` must hold `Ternary-Bonsai-2-27B-PQ2_0.gguf`. Unset or
//! missing, every test prints one `skip:` line and passes.
//!
//! ```bash
//! BOOSTR_BONSAI2_DIR=/path/to/dir cargo nextest run -p boostr \
//!   --test qwen35_gguf_real_file --nocapture
//! ```
//!
//! # Memory
//!
//! The file is 7.2 GB and every dense tensor is decoded to F32, so expect
//! peak RSS above the file size. Each test prints elapsed time and `VmHWM`
//! from `/proc/self/status`.

use std::path::PathBuf;
use std::time::Instant;

use boostr::format::gguf::Gguf;
use boostr::inference::{LayeredGdnState, LayeredKvCache, LayeredKvCacheConfig};
use boostr::model::{LoadedModel, qwen35_config_from_gguf};
use boostr::nn::{VarBuilder, VarMap};
use numr::dtype::DType;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

const ENV_DIR: &str = "BOOSTR_BONSAI2_DIR";
const PQ2_0_FILE: &str = "Ternary-Bonsai-2-27B-PQ2_0.gguf";

const VOCAB: usize = 248_320;
const BOS: i64 = 248_044;
const PROMPT: [i64; 4] = [BOS, 17, 42, 99];

/// The PQ2_0 file, or `None` after one `skip:` line.
fn require_file() -> Option<PathBuf> {
    let Some(dir) = std::env::var(ENV_DIR).ok().map(PathBuf::from) else {
        println!("skip: {ENV_DIR} not set");
        return None;
    };
    let path = dir.join(PQ2_0_FILE);
    if !path.is_file() {
        println!("skip: {} not found", path.display());
        return None;
    }
    Some(path)
}

/// `VmHWM` (peak resident set) from `/proc/self/status`, as printed there.
fn peak_rss() -> String {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .map(|l| l.trim_start_matches("VmHWM:").trim().to_string())
        })
        .unwrap_or_else(|| "unavailable".to_string())
}

fn report(label: &str, start: Instant) {
    println!(
        "{label}: {:.1?} elapsed, peak RSS {}",
        start.elapsed(),
        peak_rss()
    );
}

fn assert_finite(logits: &Tensor<CpuRuntime>, label: &str) {
    let data = logits.to_vec::<f32>();
    let bad = data.iter().position(|v| !v.is_finite());
    assert!(
        bad.is_none(),
        "{label}: non-finite logit at flat index {}",
        bad.unwrap_or(0)
    );
}

#[test]
fn header_derives_bonsai_config() {
    let Some(path) = require_file() else {
        return;
    };
    let gguf = Gguf::open(&path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let config = qwen35_config_from_gguf(gguf.metadata())
        .unwrap_or_else(|e| panic!("qwen35_config_from_gguf on {}: {e}", path.display()));

    assert_eq!(config.model_type, "qwen35");
    assert_eq!(config.vocab_size, VOCAB);
    assert_eq!(config.hidden_size, 5120);
    assert_eq!(config.num_layers, 64);
    assert_eq!(config.max_seq_len, 262_144);
    assert_eq!(config.intermediate_size, Some(17_408));
    assert!(!config.tie_word_embeddings);

    let gdn = config.gdn.as_ref().expect("gdn config");
    assert_eq!(gdn.conv_kernel, 4);
    assert_eq!(gdn.state_size, 128);
    assert_eq!(gdn.key_heads, 16);
    assert_eq!(gdn.value_heads, 48);
    assert_eq!(gdn.inner_size, 6144);

    let attn = config.qwen35_attention.as_ref().expect("attention config");
    assert_eq!(attn.num_heads, 24);
    assert_eq!(attn.num_kv_heads, 4);
    assert_eq!(attn.head_dim, 256);
    assert_eq!(attn.rope_dim, 64);
    assert_eq!(attn.rope_sections, [11, 11, 10, 0]);
    assert_eq!(attn.rope_theta, 1e7);

    let layers = config.hybrid_layers.as_ref().expect("hybrid_layers");
    assert_eq!(layers.attention_layers.len(), 16);
    assert_eq!(layers.ssm_layers.len(), 48);
    assert!(layers.is_attention_layer(3));
    assert!(layers.is_attention_layer(63));
    assert!(layers.is_ssm_layer(0));

    let hadamard = config.hadamard.as_ref().expect("prism.hadamard block");
    assert!(hadamard.rotates("output.weight"));
    assert_eq!(gdn.v_grouped, hadamard.gdn_v_grouped);
}

#[test]
fn loads_and_runs_prefill_then_decode() {
    let Some(path) = require_file() else {
        return;
    };
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());

    let start = Instant::now();
    let gguf = Gguf::open(&path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let config = qwen35_config_from_gguf(gguf.metadata())
        .unwrap_or_else(|e| panic!("qwen35_config_from_gguf on {}: {e}", path.display()));
    drop(gguf);

    let mut varmap = VarMap::<CpuRuntime>::from_gguf(&path, &device)
        .unwrap_or_else(|e| panic!("VarMap::from_gguf on {}: {e}", path.display()));
    report("weights loaded", start);

    let mut vb = VarBuilder::new(&mut varmap, &device);
    let loaded = LoadedModel::<CpuRuntime>::load(&config, &mut vb)
        .unwrap_or_else(|e| panic!("LoadedModel::load: {e}"));
    let num_attention = loaded
        .num_attention_layers()
        .expect("qwen35 attention layers");
    let num_gdn = loaded.num_gdn_layers().expect("qwen35 gdn layers");
    assert_eq!(num_attention, 16);
    assert_eq!(num_gdn, 48);
    assert_eq!(
        loaded.quant_formats(),
        &[boostr::quant::QuantFormat::PQ2_0],
        "Bonsai-2 is single-format PQ2_0; quant_formats must report only that"
    );
    let model = match loaded {
        LoadedModel::Qwen35(model) => model,
        other => panic!("LoadedModel::load returned {other:?}, expected Qwen35"),
    };
    assert_eq!(
        varmap.len(),
        0,
        "loader left {} tensors behind: {:?}",
        varmap.len(),
        varmap.names().take(8).collect::<Vec<_>>()
    );
    report("model built", start);

    let attn = model.attention_config();
    let kv_config = LayeredKvCacheConfig {
        batch_size: 1,
        num_kv_heads: attn.num_kv_heads,
        initial_capacity: 16,
        max_seq_len: config.max_seq_len,
        head_dim: attn.head_dim,
        dtype: DType::F32,
    };
    let mut kv = LayeredKvCache::<CpuRuntime>::new(num_attention, &kv_config, &device)
        .unwrap_or_else(|e| panic!("LayeredKvCache::new: {e}"));
    let mut gdn =
        LayeredGdnState::<CpuRuntime>::zeros(num_gdn, model.gdn_config(), 1, DType::F32, &device)
            .unwrap_or_else(|e| panic!("LayeredGdnState::zeros: {e}"));

    let ids = Tensor::<CpuRuntime>::from_slice(&PROMPT, &[1, PROMPT.len()], &device)
        .unwrap_or_else(|e| panic!("prompt tensor: {e}"));
    let step = Instant::now();
    let logits = model
        .forward_qwen35(&client, &ids, &mut kv, &mut gdn, 0)
        .unwrap_or_else(|e| panic!("prefill forward: {e}"));
    report("prefill (4 tokens)", step);
    assert_eq!(logits.shape(), &[1, PROMPT.len(), VOCAB]);
    assert_finite(&logits, "prefill");

    let next = Tensor::<CpuRuntime>::from_slice(&[7i64], &[1, 1], &device)
        .unwrap_or_else(|e| panic!("decode tensor: {e}"));
    let step = Instant::now();
    let logits = model
        .forward_qwen35(&client, &next, &mut kv, &mut gdn, PROMPT.len())
        .unwrap_or_else(|e| panic!("decode forward: {e}"));
    report("decode (1 token)", step);
    assert_eq!(logits.shape(), &[1, 1, VOCAB]);
    assert_finite(&logits, "decode");
    assert_eq!(kv.seq_len(), PROMPT.len() + 1);

    report("total", start);
}
