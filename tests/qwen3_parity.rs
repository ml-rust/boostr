//! Numerical parity for the DECODER path against HuggingFace `transformers`,
//! on real Qwen3-1.7B weights.
//!
//! Run with:
//!   `cd boostr && QWEN3_REF_DIR=<dir> cargo test --release --test qwen3_parity -- --nocapture`
//!
//! Fixtures come from `dump_qwen3.py`; the test skips when they or the weights
//! are absent. **A skip is not a pass** — trust a run only when the `max|d|`
//! lines below actually print.
//!
//! ## Why this test exists
//!
//! Before it, `boostr` had NO numerical parity test for the Llama decoder
//! against any reference — the family it serves (Llama, Mistral, Qwen2/3) was
//! verified only by shape checks and "the output looks like text". That is not
//! enough: a wrong RoPE pairing convention keeps every shape valid, keeps the
//! loss finite, and produces fluent-looking garbage.
//!
//! Qwen3 is the right vehicle because it exercises the decoder's less-travelled
//! options all at once: per-head QK-norm before RoPE, an explicit `head_dim`
//! that is NOT `hidden_size / num_heads` by definition, GQA at 16 query heads
//! over 8 KV heads, and tied embeddings.
//!
//! The checks run innermost-first so a failure localizes:
//!   1. layer-0 Q/K immediately after QK-norm + RoPE  — isolates the RoPE
//!      convention with everything upstream still matching
//!   2. layer-0 block output                          — isolates attention/MLP
//!   3. final hidden state after `model.norm`         — isolates depth accumulation
//!   4. logits + top-1 token                          — the end-to-end claim

use boostr::format::SafeTensorsLoader;
use boostr::model::config::huggingface::HuggingFaceConfig;
use boostr::model::llama::Llama;
use boostr::model::traits::Model;
use boostr::nn::{VarBuilder, VarMap};
use numr::autograd::Var;
use numr::ops::TypeConversionOps;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;
use std::path::PathBuf;

mod common;
use common::model_fixture;

fn read_f32(path: &PathBuf) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert!(
        bytes.len().is_multiple_of(4),
        "{} is not a whole number of f32s",
        path.display()
    );
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_i32(path: &PathBuf) -> Vec<i32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    bytes
        .as_chunks::<4>()
        .0
        .iter()
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

fn rms(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt()
}

fn fixtures() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var("QWEN3_REF_DIR").ok()?);
    dir.join("qwen3_logits.f32").exists().then_some(dir)
}

fn model_dir() -> Option<PathBuf> {
    let p = model_fixture("QWEN3_MODEL", "qwen3-1.7b")?;
    p.join("config.json").exists().then_some(p)
}

/// Load every tensor in the checkpoint into a `VarMap`, then build the decoder.
fn load(dir: &PathBuf, device: &CpuDevice) -> (Llama<CpuRuntime>, usize) {
    let cfg_text = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let hf = HuggingFaceConfig::from_json(&cfg_text).expect("parse config.json");
    let config = hf.to_universal();

    // Cast every weight to F32. The checkpoint is BF16; the HF reference was
    // produced in F32. Comparing against a BF16 forward would conflate
    // algorithm error with 8-bit-mantissa noise, and this test exists to find
    // the former. The upcast is lossless.
    let client = CpuClient::new(device.clone());
    let mut loader = SafeTensorsLoader::open(dir).expect("open safetensors");
    let mut var_map = VarMap::<CpuRuntime>::new();
    for name in loader.tensor_names() {
        let t = loader
            .load_tensor::<CpuRuntime>(&name, device)
            .unwrap_or_else(|e| panic!("load {name}: {e}"));
        let t = if t.dtype() == numr::dtype::DType::F32 {
            t
        } else {
            client
                .cast(&t, numr::dtype::DType::F32)
                .unwrap_or_else(|e| panic!("cast {name} to f32: {e}"))
        };
        var_map.insert(name, t);
    }

    let mut vb = VarBuilder::new(&mut var_map, device);
    let model = Llama::<CpuRuntime>::from_varbuilder(&mut vb, &config).expect("build decoder");
    let vocab = config.vocab_size;
    (model, vocab)
}

/// What `boostr` actually parsed out of Qwen3's `config.json`.
///
/// Runs first because a mis-parsed scalar — `rms_norm_eps`, `intermediate_size`,
/// `head_dim`, `rope_theta` — reproduces as a large, position-independent
/// activation error that looks exactly like an algorithmic bug. Cheap to check,
/// and it removes a whole class of suspects.
#[test]
fn qwen3_config_parses_correctly() {
    let Some(model_path) = model_dir() else {
        common::skip_notice("Qwen3 weights", "QWEN3_MODEL");
        return;
    };
    let cfg_text = std::fs::read_to_string(model_path.join("config.json")).expect("config.json");
    let hf = HuggingFaceConfig::from_json(&cfg_text).expect("parse config.json");
    let c = hf.to_universal();
    let attn = c.attention.as_ref().expect("attention config");

    eprintln!("hidden_size        {}", c.hidden_size);
    eprintln!("num_layers         {}", c.num_layers);
    eprintln!("vocab_size         {}", c.vocab_size);
    eprintln!("intermediate_size  {:?}", c.intermediate_size);
    eprintln!("rms_norm_eps       {:e}", c.rms_norm_eps);
    eprintln!("tie_word_embeddings {}", c.tie_word_embeddings);
    eprintln!("num_heads          {}", attn.num_heads);
    eprintln!("num_kv_heads       {:?}", attn.num_kv_heads);
    eprintln!("head_dim           {:?}", attn.head_dim);
    eprintln!("head_dim(hidden)   {}", attn.head_dim(c.hidden_size));
    eprintln!("rope_theta         {:e}", attn.rope_theta);

    assert_eq!(c.hidden_size, 2048);
    assert_eq!(c.num_layers, 28);
    assert_eq!(c.vocab_size, 151936);
    assert_eq!(c.intermediate_size, Some(6144), "SwiGLU width");
    assert_eq!(attn.num_heads, 16);
    assert_eq!(attn.num_kv_heads, Some(8), "GQA: 8 KV heads");
    assert_eq!(
        attn.head_dim(c.hidden_size),
        128,
        "head_dim is an explicit config field, not hidden/heads"
    );
    assert!(c.tie_word_embeddings, "Qwen3 ties lm_head to embeddings");
    assert!(
        (c.rms_norm_eps - 1e-6).abs() < 1e-12,
        "rms_norm_eps must be 1e-6, got {:e}",
        c.rms_norm_eps
    );
    assert!(
        (attn.rope_theta - 1e6).abs() < 1.0,
        "rope_theta must be 1e6, got {:e}",
        attn.rope_theta
    );
}

/// The innermost probe: does the embedding lookup itself match?
///
/// Layer 0's input IS the embedding output, so if this differs, every
/// downstream comparison is measuring the same defect twice. It reads the
/// weight straight from the checkpoint and gathers rows by hand — no model, no
/// `Embedding` module — so it also validates the loader's BF16→F32 path.
#[test]
fn qwen3_embedding_matches_huggingface() {
    let Some(dir) = fixtures() else {
        eprintln!("skipping: set QWEN3_REF_DIR (run dump_qwen3.py)");
        return;
    };
    let Some(model_path) = model_dir() else {
        common::skip_notice("Qwen3 weights", "QWEN3_MODEL");
        return;
    };
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());

    let mut loader = SafeTensorsLoader::open(&model_path).expect("open safetensors");
    let w = loader
        .load_tensor::<CpuRuntime>("model.embed_tokens.weight", &device)
        .expect("embed_tokens.weight");
    let w = client
        .cast(&w, numr::dtype::DType::F32)
        .expect("cast embeddings");
    let hidden = w.shape()[1];
    let table: Vec<f32> = w.contiguous().expect("contiguous").to_vec();

    let ids = read_i32(&dir.join("qwen3_input_ids.i32"));
    let got: Vec<f32> = ids
        .iter()
        .flat_map(|&id| {
            let r = id as usize * hidden;
            table[r..r + hidden].to_vec()
        })
        .collect();

    let want = read_f32(&dir.join("qwen3_l0_hidden_in.f32"));
    assert_eq!(got.len(), want.len(), "embedding output length mismatch");
    let (d, i) = max_abs_diff(&got, &want);
    let scale = rms(&want);
    eprintln!("embedding: max|d|={d:.3e} at {i}, reference rms={scale:.3e}");
    assert!(
        d < 1e-5,
        "embedding lookup differs from HuggingFace: max|d|={d} at {i} — \
         the loader or the BF16 cast is wrong, before any layer runs"
    );
}

/// Trunk parity: the hidden state after `model.norm`, BEFORE `lm_head`.
///
/// This splits the model in two. `lm_head` here is provably not a suspect —
/// the checkpoint's `lm_head.weight` and `model.embed_tokens.weight` are
/// bit-identical, so tied-or-not cannot change the result. Therefore any logit
/// divergence must originate in the trunk, and this test says so directly
/// instead of inferring it through a 151936-wide projection.
#[test]
fn qwen3_hidden_state_matches_huggingface() {
    let Some(dir) = fixtures() else {
        eprintln!("skipping: set QWEN3_REF_DIR (run dump_qwen3.py)");
        return;
    };
    let Some(model_path) = model_dir() else {
        common::skip_notice("Qwen3 weights", "QWEN3_MODEL");
        return;
    };

    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    let (model, _) = load(&model_path, &device);

    let ids = read_i32(&dir.join("qwen3_input_ids.i32"));
    let seq = ids.len();
    let input = Tensor::<CpuRuntime>::try_from_slice(&ids, &[1, seq], &device).unwrap();

    let hidden = model
        .forward_hidden(&client, &input)
        .expect("forward_hidden");
    let got: Vec<f32> = hidden.tensor().contiguous().expect("contiguous").to_vec();
    let want = read_f32(&dir.join("qwen3_hidden_final.f32"));
    assert_eq!(got.len(), want.len(), "hidden state length mismatch");

    let hidden_dim = want.len() / seq;
    let (d, i) = max_abs_diff(&got, &want);
    let scale = rms(&want);
    eprintln!("hidden after model.norm: max|d|={d:.3e} at {i}, reference rms={scale:.3e}");
    eprintln!("  pos  max|d|");
    for p in 0..seq {
        let (pd, _) = max_abs_diff(
            &got[p * hidden_dim..(p + 1) * hidden_dim],
            &want[p * hidden_dim..(p + 1) * hidden_dim],
        );
        eprintln!("  {p:>3}  {pd:>9.3e}");
    }
    assert!(
        d < 1e-2 * scale.max(1.0),
        "trunk diverges from HuggingFace before lm_head: max|d|={d} at {i} (rms {scale}) — \
         the bug is in the layers, not the output projection"
    );
}

/// End-to-end logits parity. This is the claim that matters: if RoPE, QK-norm,
/// GQA or the tied `lm_head` is wrong, the logits diverge and the argmax moves.
#[test]
fn qwen3_logits_match_huggingface() {
    let Some(dir) = fixtures() else {
        eprintln!("skipping: set QWEN3_REF_DIR (run dump_qwen3.py)");
        return;
    };
    let Some(model_path) = model_dir() else {
        common::skip_notice("Qwen3 weights", "QWEN3_MODEL");
        return;
    };

    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    let (model, vocab) = load(&model_path, &device);
    eprintln!("weights cast to f32 to match the reference dump");

    let ids = read_i32(&dir.join("qwen3_input_ids.i32"));
    let seq = ids.len();
    eprintln!("prompt is {seq} tokens, vocab {vocab}");
    let input = Var::new(
        Tensor::<CpuRuntime>::try_from_slice(&ids, &[1, seq], &device).unwrap(),
        false,
    );

    let logits = model.forward(&client, &input).expect("decoder forward");
    assert_eq!(logits.shape(), &[1, seq, vocab], "unexpected logits shape");
    let got: Vec<f32> = logits.tensor().contiguous().expect("contiguous").to_vec();
    let want = read_f32(&dir.join("qwen3_logits.f32"));

    let (d, i) = max_abs_diff(&got, &want);
    let scale = rms(&want);
    eprintln!("logits: max|d|={d:.3e} at {i}, reference rms={scale:.3e}");

    // Per-position breakdown. Error that grows with position points at the
    // causal mask or the RoPE offset; error flat across positions points at a
    // weight or a norm. Reporting both the diff and whether the argmax agrees
    // separates "slightly off" from "predicting something else".
    eprintln!("  pos  max|d|     top1 got/want");
    for p in 0..seq {
        let (g, w) = (
            &got[p * vocab..(p + 1) * vocab],
            &want[p * vocab..(p + 1) * vocab],
        );
        let (pd, _) = max_abs_diff(g, w);
        let top = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(usize::MAX)
        };
        let (tg, tw) = (top(g), top(w));
        let flag = if tg == tw { "" } else { "  <-- DIFFERS" };
        eprintln!("  {p:>3}  {pd:>9.3e}  {tg:>6}/{tw:<6}{flag}");
    }

    // The token the model actually predicts — the human-legible form of the
    // same claim, and the one a RoPE bug breaks unmistakably.
    let last = (seq - 1) * vocab;
    let argmax = |v: &[f32]| {
        v[last..last + vocab]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, val)| (idx, *val))
            .expect("non-empty logits")
    };
    let (got_id, got_val) = argmax(&got);
    let (want_id, want_val) = argmax(&want);
    eprintln!("next token: got {got_id} ({got_val:.4}), want {want_id} ({want_val:.4})");

    assert_eq!(
        got_id, want_id,
        "predicted token differs from HuggingFace: got {got_id}, want {want_id} — \
         the decoder is numerically wrong, not merely imprecise"
    );
    assert!(
        d < 5e-2 * scale.max(1.0),
        "logits diverge from HuggingFace: max|d|={d} at {i} (rms {scale})"
    );
}
