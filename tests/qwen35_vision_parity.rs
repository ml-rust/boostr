//! Image-prompt logits parity between our `qwen35` CPU forward over spliced
//! vision-tower rows and the reference decoder, on the real
//! Ternary-Bonsai-2-27B `PQ2_0` checkpoint and its Q8_0 mmproj.
//!
//! # Fixture files
//!
//! `BOOSTR_BONSAI2_DIR` must hold `Ternary-Bonsai-2-27B-PQ2_0.gguf` and
//! `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf`. Unset dir or a missing file:
//! the test prints one `skip:` line and passes.
//!
//! `tests/fixtures/vision/img_96x64.logits` and `img_96x64.logits.json`
//! come from `tests/tools/mtmd_logits_dump`, run with the text model on
//! the GPU and the vision tower on the CPU:
//!
//! ```bash
//! ./mtmd_logits_dump Ternary-Bonsai-2-27B-PQ2_0.gguf \
//!   Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf img_96x64.png 99 \
//!   img_96x64.logits img_96x64.logits.json
//! ```
//!
//! ```bash
//! BOOSTR_BONSAI2_DIR=/path/to/dir cargo test --release \
//!   --test qwen35_vision_parity -- --nocapture
//! ```

use std::path::PathBuf;
use std::time::Instant;

use boostr::format::gguf::Gguf;
use boostr::inference::{LayeredGdnState, LayeredKvCache, LayeredKvCacheConfig};
use boostr::model::qwen35::{ImageEmbeds, ImageGrid, Qwen35Model, Qwen35PromptPlan, VisionMarkers};
use boostr::model::vision::qwen3vl::{decode_image, load_qwen3vl_vision_gguf, preprocess};
use boostr::model::{LoadedModel, qwen35_config_from_gguf};
use boostr::nn::{VarBuilder, VarMap};
use numr::dtype::DType;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;
use serde::Deserialize;

const ENV_DIR: &str = "BOOSTR_BONSAI2_DIR";
const MODEL_FILE: &str = "Ternary-Bonsai-2-27B-PQ2_0.gguf";
const MMPROJ_FILE: &str = "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf";
const FIXTURE: &str = "img_96x64";

const COSINE_MIN: f64 = 0.999;

/// Sidecar of the logits fixture.
#[derive(Deserialize)]
struct Oracle {
    ids: Vec<u32>,
    nx: usize,
    ny: usize,
    n_image_tokens: usize,
    vision_start: u32,
    vision_end: u32,
    image_pad: u32,
    n_past: usize,
    n_vocab: usize,
    argmax: usize,
    greedy: Vec<u32>,
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/vision")
}

fn read_oracle() -> (Oracle, Vec<f32>) {
    let json =
        std::fs::read_to_string(fixture_dir().join(format!("{FIXTURE}.logits.json"))).unwrap();
    let oracle: Oracle = serde_json::from_str(&json).unwrap();
    let bytes = std::fs::read(fixture_dir().join(format!("{FIXTURE}.logits"))).unwrap();
    let n_vocab = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    assert_eq!(n_vocab, oracle.n_vocab, "logits header n_vocab");
    let row: Vec<f32> = bytes[4..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    assert_eq!(row.len(), n_vocab, "logits payload length");
    (oracle, row)
}

/// The model and mmproj paths, or `None` after one `skip:` line.
fn require_files() -> Option<(PathBuf, PathBuf)> {
    let Some(dir) = std::env::var(ENV_DIR).ok().map(PathBuf::from) else {
        println!("skip: {ENV_DIR} not set");
        return None;
    };
    let model = dir.join(MODEL_FILE);
    let mmproj = dir.join(MMPROJ_FILE);
    for path in [&model, &mmproj] {
        if !path.is_file() {
            println!("skip: {} not found", path.display());
            return None;
        }
    }
    Some((model, mmproj))
}

fn argmax_f32(row: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, &v) in row.iter().enumerate().skip(1) {
        if v > row[best] {
            best = i;
        }
    }
    best
}

/// `(cosine, max abs diff)` in `f64`.
fn row_stats(ours: &[f32], theirs: &[f32]) -> (f64, f64) {
    let (mut dot, mut na, mut nb, mut max_abs) = (0f64, 0f64, 0f64, 0f64);
    for (&a, &b) in ours.iter().zip(theirs) {
        let (a, b) = (a as f64, b as f64);
        dot += a * b;
        na += a * a;
        nb += b * b;
        max_abs = max_abs.max((a - b).abs());
    }
    (dot / (na.sqrt() * nb.sqrt()), max_abs)
}

fn load_model(
    model_path: &PathBuf,
    device: &CpuDevice,
) -> (
    Qwen35Model<CpuRuntime>,
    LayeredKvCache<CpuRuntime>,
    LayeredGdnState<CpuRuntime>,
) {
    let gguf =
        Gguf::open(model_path).unwrap_or_else(|e| panic!("open {}: {e}", model_path.display()));
    let config = qwen35_config_from_gguf(gguf.metadata())
        .unwrap_or_else(|e| panic!("qwen35_config_from_gguf: {e}"));
    drop(gguf);

    let mut varmap = VarMap::<CpuRuntime>::from_gguf(model_path, device)
        .unwrap_or_else(|e| panic!("VarMap::from_gguf: {e}"));
    let mut vb = VarBuilder::new(&mut varmap, device);
    let loaded = LoadedModel::<CpuRuntime>::load(&config, &mut vb)
        .unwrap_or_else(|e| panic!("LoadedModel::load: {e}"));
    let model = match loaded {
        LoadedModel::Qwen35(model) => *model,
        other => panic!("LoadedModel::load returned {other:?}, expected Qwen35"),
    };

    let attn = model.attention_config();
    let kv_config = LayeredKvCacheConfig {
        batch_size: 1,
        num_kv_heads: attn.num_kv_heads,
        initial_capacity: 64,
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
    (model, kv, gdn)
}

#[test]
fn image_prompt_logits_match_fork() {
    let Some((model_path, mmproj_path)) = require_files() else {
        return;
    };
    let (oracle, fork_row) = read_oracle();
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());

    // Vision tower: preprocess and encode the fixture image.
    let start = Instant::now();
    let tower = load_qwen3vl_vision_gguf::<CpuRuntime, _>(&mmproj_path, &device)
        .unwrap_or_else(|e| panic!("load mmproj: {e}"));
    let png = std::fs::read(fixture_dir().join(format!("{FIXTURE}.png"))).unwrap();
    let image = preprocess(&decode_image(&png).unwrap(), tower.config()).unwrap();
    assert_eq!((image.nx, image.ny), (oracle.nx, oracle.ny), "token grid");
    assert_eq!(image.n_tokens(), oracle.n_image_tokens);
    let embeds = tower.encode_image(&client, &image).unwrap();
    println!(
        "vision tower: {:.1?} elapsed, {:?}",
        start.elapsed(),
        embeds.shape()
    );
    drop(tower);

    let start = Instant::now();
    let (model, mut kv, mut gdn) = load_model(&model_path, &device);
    println!("model load: {:.1?} elapsed", start.elapsed());

    let markers = VisionMarkers {
        vision_start: oracle.vision_start,
        vision_end: oracle.vision_end,
        image_pad: oracle.image_pad,
    };
    let images = [ImageEmbeds {
        embeds,
        grid: ImageGrid {
            nx: oracle.nx,
            ny: oracle.ny,
        },
    }];
    let plan = Qwen35PromptPlan::build(&client, &model, &oracle.ids, &images, &markers, 0)
        .unwrap_or_else(|e| panic!("prompt plan: {e}"));
    let seq_len = oracle.ids.len() - 1 + oracle.n_image_tokens;
    assert_eq!(plan.seq_len(), seq_len);
    assert_eq!(
        plan.next_rope_pos(),
        oracle.n_past,
        "rope position after the prompt"
    );

    let start = Instant::now();
    let logits = model
        .forward_qwen35_embeds(&client, &plan.embeds, &plan.positions, &mut kv, &mut gdn)
        .unwrap_or_else(|e| panic!("prefill forward: {e}"));
    println!("prefill ({seq_len} rows): {:.1?} elapsed", start.elapsed());
    assert_eq!(logits.shape(), &[1, seq_len, oracle.n_vocab]);
    assert_eq!(kv.seq_len(), seq_len);

    let flat = logits.to_vec::<f32>();
    let ours = &flat[(seq_len - 1) * oracle.n_vocab..];
    let our_argmax = argmax_f32(ours);
    let (cosine, max_abs) = row_stats(ours, &fork_row);
    println!(
        "last row: our argmax {our_argmax}, fork argmax {}, cosine {cosine:.6}, \
         max_abs_diff {max_abs:.4}",
        oracle.argmax
    );
    assert_eq!(our_argmax, oracle.argmax, "last-row argmax");
    assert_eq!(
        argmax_f32(&fork_row),
        oracle.argmax,
        "fixture row vs sidecar argmax"
    );
    assert!(cosine >= COSINE_MIN, "cosine {cosine:.6} < {COSINE_MIN}");

    // Greedy continuation: the KV slot runs ahead of the rope position.
    let mut token = oracle.greedy[0];
    for (step, &want) in oracle.greedy[1..].iter().enumerate() {
        let rope_pos = oracle.n_past + step;
        assert!(kv.seq_len() > rope_pos, "kv slot must run ahead of rope");
        let ids = Tensor::<CpuRuntime>::from_slice(&[i64::from(token)], &[1, 1], &device).unwrap();
        let start = Instant::now();
        let row = model
            .forward_qwen35(&client, &ids, &mut kv, &mut gdn, rope_pos)
            .unwrap_or_else(|e| panic!("decode step {step}: {e}"))
            .to_vec::<f32>();
        let got = argmax_f32(&row) as u32;
        println!(
            "decode step {step} (rope {rope_pos}): ours {got}, fork {want}, {:.1?}",
            start.elapsed()
        );
        assert_eq!(got, want, "decode step {step} argmax");
        token = got;
    }
}
