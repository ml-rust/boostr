//! Gates the g2p_lm checkpoint against 50 held-out gold words.
//!
//! Skips when the checkpoint or the `dev.jsonl`/`vocab.json` fixtures are
//! unavailable (`$G2P_LM_CHECKPOINT_DIR`/`$G2P_LM_VOCAB`/`$G2P_LM_DEV_JSONL`,
//! else `$BOOSTR_MODELS_DIR/g2p_lm/{checkpoints/g2p-char,data/vocab.json,
//! data/dev.jsonl}`) — no model is checked into this repo yet, only the
//! fixture data `build_dataset.py` writes.
//!
//! `NeuralG2pFallback::from_dir` reads `config.json`, one `*.safetensors`
//! file and `vocab.json` from the SAME directory; oxidizr's checkpoint
//! directory and `assets/g2p_lm/build_dataset.py`'s `vocab.json` are two
//! different directories, so this test stages a temporary directory with
//! `config.json` and the checkpoint symlinked in and `vocab.json` copied
//! in, rather than requiring the fixture layout to change.
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

/// Stages `config.json` + checkpoint (symlinked) + `vocab.json` (copied)
/// into one temporary directory, the layout `NeuralG2pFallback::from_dir`
/// expects.
fn stage_model_dir(
    checkpoint_dir: &Path,
    vocab_path: &Path,
    staging: &Path,
) -> std::io::Result<()> {
    fs::create_dir_all(staging)?;
    for entry in fs::read_dir(checkpoint_dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let is_relevant = name == std::ffi::OsStr::new("config.json")
            || entry.path().extension().is_some_and(|e| e == "safetensors");
        if is_relevant {
            #[cfg(unix)]
            std::os::unix::fs::symlink(entry.path(), staging.join(&name))?;
            #[cfg(not(unix))]
            fs::copy(entry.path(), staging.join(&name)).map(|_| ())?;
        }
    }
    fs::copy(vocab_path, staging.join("vocab.json")).map(|_| ())
}

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
    let Some(checkpoint_dir) =
        model_fixture("G2P_LM_CHECKPOINT_DIR", "g2p_lm/checkpoints/g2p-char")
    else {
        skip_notice("g2p_lm checkpoint", "G2P_LM_CHECKPOINT_DIR");
        return;
    };
    let Some(vocab_path) = model_fixture("G2P_LM_VOCAB", "g2p_lm/data/vocab.json") else {
        skip_notice("g2p_lm vocab.json", "G2P_LM_VOCAB");
        return;
    };
    let Some(dev_jsonl) = model_fixture("G2P_LM_DEV_JSONL", "g2p_lm/data/dev.jsonl") else {
        skip_notice("g2p_lm dev.jsonl", "G2P_LM_DEV_JSONL");
        return;
    };

    let staging = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
    stage_model_dir(&checkpoint_dir, &vocab_path, staging.path())
        .unwrap_or_else(|e| panic!("staging model dir: {e}"));

    let model = NeuralG2pFallback::from_dir(staging.path(), false)
        .unwrap_or_else(|e| panic!("NeuralG2pFallback::from_dir: {e}"));

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
