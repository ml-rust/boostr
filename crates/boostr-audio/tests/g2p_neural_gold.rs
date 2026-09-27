//! Gates the out-of-vocabulary G2P model against 50 held-out gold words.
//!
//! The model directory (`config.json`, one `*.safetensors`, `vocab.json`)
//! comes from `$G2P_LM_MODEL_DIR`, else `$BOOSTR_MODELS_DIR/g2p-lm`. The gold
//! words come from `build_dataset.py`'s `dev.jsonl`: `$G2P_LM_DEV_JSONL`, else
//! `$BOOSTR_MODELS_DIR/g2p_lm/data/dev.jsonl`. Skips when either is absent.
#![cfg(feature = "misaki")]

use std::fs;
use std::path::Path;

use boostr_audio::g2p::neural::NeuralG2pFallback;

mod common;
use common::{model_fixture, skip_notice};

/// Minimum exact-match accuracy over the 50 held-out words below. Greedy
/// decoding is deterministic, so a checkpoint scores the same on every run.
/// A new checkpoint must reach this before it replaces the published one.
const GOLD_WORD_ACCURACY_THRESHOLD: f64 = 0.66;

/// Reads the first 50 `{"dialect","word","phonemes"}` lines of `dev.jsonl`.
fn gold_words(dev_jsonl: &Path) -> Vec<(String, String, String)> {
    let text = fs::read_to_string(dev_jsonl)
        .unwrap_or_else(|e| panic!("reading {}: {e}", dev_jsonl.display()));
    text.lines()
        .take(50)
        .map(|line| {
            let v: serde_json::Value =
                serde_json::from_str(line).unwrap_or_else(|e| panic!("parsing {line:?}: {e}"));
            let get = |key: &str| v[key].as_str().unwrap_or_default().to_string();
            (get("dialect"), get("word"), get("phonemes"))
        })
        .collect()
}

#[test]
fn reproduces_fifty_held_out_gold_words() {
    let Some(model_dir) = model_fixture("G2P_LM_MODEL_DIR", "g2p-lm") else {
        skip_notice("g2p-lm model directory", "G2P_LM_MODEL_DIR");
        return;
    };
    let Some(dev_jsonl) = model_fixture("G2P_LM_DEV_JSONL", "g2p_lm/data/dev.jsonl") else {
        skip_notice("g2p_lm dev.jsonl", "G2P_LM_DEV_JSONL");
        return;
    };

    let model = NeuralG2pFallback::from_dir(&model_dir, false)
        .unwrap_or_else(|e| panic!("NeuralG2pFallback::from_dir({}): {e}", model_dir.display()));

    let words = gold_words(&dev_jsonl);
    assert_eq!(
        words.len(),
        50,
        "expected 50 gold words in {}",
        dev_jsonl.display()
    );

    let mut correct = 0usize;
    let mut mismatches = Vec::new();
    for (dialect, word, want) in &words {
        let got = model
            .predict(dialect, word)
            .unwrap_or_else(|e| panic!("predict({dialect:?}, {word:?}): {e}"));
        match got {
            Some(ps) if &ps == want => correct += 1,
            Some(ps) => mismatches.push(format!("{word} ({dialect}): got {ps:?}, want {want:?}")),
            None => mismatches.push(format!("{word} ({dialect}): got nothing, want {want:?}")),
        }
    }

    let accuracy = correct as f64 / words.len() as f64;
    eprintln!(
        "g2p_lm gold-word accuracy: {accuracy:.4} ({correct}/{})",
        words.len()
    );
    for m in &mismatches {
        eprintln!("  mismatch: {m}");
    }
    assert!(
        accuracy >= GOLD_WORD_ACCURACY_THRESHOLD,
        "gold-word accuracy {accuracy:.4} is below the threshold {GOLD_WORD_ACCURACY_THRESHOLD:.4}"
    );
}
