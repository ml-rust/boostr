//! Phoneme error rate for the g2p_lm checkpoint over `dev.jsonl`/`test.jsonl`
//! (`assets/g2p_lm/build_dataset.py`), plus an out-of-vocabulary comparison
//! against misaki's own espeak fallback.
//!
//! ```text
//! cargo run --release --features misaki --example g2p_per_eval -- \
//!     MODEL_DIR DEV_JSONL TEST_JSONL [OOV_ORACLE_JSON]
//! ```
//!
//! `MODEL_DIR` holds `config.json`, one `*.safetensors` checkpoint and
//! `vocab.json` — the layout [`NeuralG2pFallback::from_dir`] reads.
//! `OOV_ORACLE_JSON`, when given, is
//! `tests/tools/g2p_oov_oracle/g2p_oov_oracle.py`'s output: the corpus's
//! out-of-vocabulary words with the espeak phonemes misaki's own fallback
//! produced for them. This scores the model's output against THAT, not
//! against ground truth — espeak's own pronunciation is not gold, so this
//! reports agreement, not accuracy.
//!
//! PER is a per-CHARACTER error rate (each phoneme symbol is one character
//! in Kokoro's vocab), reported per dialect and as a corpus-level total
//! ([`boostr_audio::eval::total`]), alongside exact-match word accuracy.

use std::path::PathBuf;

use boostr_audio::eval::{align, total};
use boostr_audio::g2p::neural::NeuralG2pFallback;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct GoldRow {
    dialect: String,
    word: String,
    phonemes: String,
}

#[derive(Debug, Deserialize)]
struct OovRow {
    word: String,
    dialect: String,
    espeak_phonemes: String,
}

/// `tests/tools/g2p_oov_oracle/g2p_oov_oracle.py`'s output: one JSON
/// document (not JSONL, unlike `dev.jsonl`/`test.jsonl`), `{"rows": [...]}`.
#[derive(Debug, Deserialize)]
struct OovFile {
    rows: Vec<OovRow>,
}

fn read_oov(path: &std::path::Path) -> Vec<OovRow> {
    let text =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    let file: OovFile =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("parsing {}: {e}", path.display()));
    file.rows
}

fn read_jsonl<T: for<'a> Deserialize<'a>>(path: &std::path::Path) -> Vec<T> {
    let text =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| serde_json::from_str(line).unwrap_or_else(|e| panic!("parsing {line:?}: {e}")))
        .collect()
}

/// Reports PER (per-character) and exact-match word accuracy for one file,
/// overall and per dialect.
fn report(label: &str, model: &NeuralG2pFallback, rows: &[GoldRow]) {
    let mut predictions: Vec<(String, String, String)> = Vec::with_capacity(rows.len());
    for row in rows {
        let predicted = model
            .predict(&row.dialect, &row.word)
            .unwrap_or_else(|e| panic!("predict({:?}, {:?}): {e}", row.dialect, row.word))
            .unwrap_or_default();
        predictions.push((row.dialect.clone(), row.phonemes.clone(), predicted));
    }

    for dialect in ["us", "gb"] {
        let subset: Vec<(&str, &str)> = predictions
            .iter()
            .filter(|(d, _, _)| d == dialect)
            .map(|(_, want, got)| (want.as_str(), got.as_str()))
            .collect();
        if subset.is_empty() {
            continue;
        }
        let rate = total(subset.iter().copied(), |r, h| {
            align(
                &r.chars().collect::<Vec<_>>(),
                &h.chars().collect::<Vec<_>>(),
            )
        });
        let exact = subset.iter().filter(|(want, got)| want == got).count();
        println!(
            "{label} [{dialect}]: PER {:.4} ({} words, {} S {} D {} I), word accuracy {:.4} ({exact}/{})",
            rate.rate(),
            subset.len(),
            rate.substitutions,
            rate.deletions,
            rate.insertions,
            exact as f64 / subset.len() as f64,
            subset.len(),
        );
    }

    let all: Vec<(&str, &str)> = predictions
        .iter()
        .map(|(_, want, got)| (want.as_str(), got.as_str()))
        .collect();
    let rate = total(all.iter().copied(), |r, h| {
        align(
            &r.chars().collect::<Vec<_>>(),
            &h.chars().collect::<Vec<_>>(),
        )
    });
    let exact = all.iter().filter(|(want, got)| want == got).count();
    println!(
        "{label} [total]: PER {:.4} ({} words), word accuracy {:.4} ({exact}/{})",
        rate.rate(),
        all.len(),
        exact as f64 / all.len() as f64,
        all.len(),
    );
}

/// Scores the model against misaki's espeak fallback over the corpus's OOV
/// words — agreement with espeak, not a gold-accuracy figure.
fn report_oov(model: &NeuralG2pFallback, rows: &[OovRow]) {
    let mut pairs: Vec<(String, String)> = Vec::with_capacity(rows.len());
    for row in rows {
        let predicted = model
            .predict(&row.dialect, &row.word)
            .unwrap_or_else(|e| panic!("predict({:?}, {:?}): {e}", row.dialect, row.word))
            .unwrap_or_default();
        pairs.push((row.espeak_phonemes.clone(), predicted));
    }
    let refs: Vec<(&str, &str)> = pairs
        .iter()
        .map(|(r, h)| (r.as_str(), h.as_str()))
        .collect();
    let rate = total(refs.iter().copied(), |r, h| {
        align(
            &r.chars().collect::<Vec<_>>(),
            &h.chars().collect::<Vec<_>>(),
        )
    });
    println!(
        "oov-vs-espeak [total]: agreement PER {:.4} over {} corpus OOV words (espeak fallback \
         output taken as the reference, not as ground truth)",
        rate.rate(),
        pairs.len(),
    );
}

fn main() {
    let mut args = std::env::args().skip(1);
    let model_dir = PathBuf::from(
        args.next()
            .expect("usage: g2p_per_eval MODEL_DIR DEV_JSONL TEST_JSONL [OOV_ORACLE_JSON]"),
    );
    let dev_path = PathBuf::from(args.next().expect("missing DEV_JSONL"));
    let test_path = PathBuf::from(args.next().expect("missing TEST_JSONL"));
    let oov_path = args.next().map(PathBuf::from);

    // `false` (US-instance dialect) is irrelevant here: every call goes
    // through `predict(dialect, word)`, which takes the dialect explicitly
    // per row rather than from the instance.
    let model = NeuralG2pFallback::from_dir(&model_dir, false)
        .unwrap_or_else(|e| panic!("NeuralG2pFallback::from_dir({}): {e}", model_dir.display()));

    let dev_rows: Vec<GoldRow> = read_jsonl(&dev_path);
    let test_rows: Vec<GoldRow> = read_jsonl(&test_path);
    report("dev", &model, &dev_rows);
    report("test", &model, &test_rows);

    if let Some(oov_path) = oov_path {
        let oov_rows = read_oov(&oov_path);
        report_oov(&model, &oov_rows);
    }
}
