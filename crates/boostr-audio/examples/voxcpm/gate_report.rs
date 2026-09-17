//! Human-readable stderr output and the summary's `source` block for
//! `voxcpm_quality_gate`.

use std::collections::HashMap;

use boostr_audio::tts_eval::{RowScore, Summary, Thresholds};

use crate::Args;

/// Provenance for the summary: the render run's fields plus this gate's.
pub fn source_json(run: &serde_json::Value, args: &Args) -> serde_json::Value {
    let pick = |key: &str| run.get(key).cloned().unwrap_or(serde_json::Value::Null);
    serde_json::json!({
        "reference": pick("reference"),
        "model_path": pick("model_path"),
        "lora": pick("lora"),
        "n_timesteps": pick("n_timesteps"),
        "cfg": pick("cfg"),
        "whisper": args.whisper.display().to_string(),
        "vad": args.vad.display().to_string(),
        "audiovae": args.audiovae.as_ref().map(|p| p.display().to_string()),
        "gate_ref": args.reference.as_ref().map(|p| p.display().to_string()),
        "language": args.language,
    })
}

fn fmt_opt(v: Option<f64>) -> String {
    match v {
        Some(x) if x.is_finite() => format!("{x:.3}"),
        Some(_) => "inf".to_string(),
        None => "-".to_string(),
    }
}

pub fn report_row(row: &RowScore) {
    let m: HashMap<&str, f64> = row.metrics().into_iter().collect();
    let get = |k: &str| fmt_opt(m.get(k).copied());
    eprintln!(
        "{:<6} {:<9} wer {} cer {} hnr {} lufs {} lead {} trail {} w/s {} cos {}{}{}",
        row.id,
        row.axis_key(),
        get("wer"),
        get("cer"),
        get("hnr_db"),
        get("lufs"),
        get("leading_silence_s"),
        get("trailing_silence_s"),
        get("words_per_s"),
        get("latent_cosine"),
        if row.violations.is_empty() {
            String::new()
        } else {
            format!("  VIOLATIONS: {}", row.violations.join(" | "))
        },
        row.error
            .as_ref()
            .map(|e| format!("  ERROR: {e}"))
            .unwrap_or_default(),
    );
}

pub fn report_summary(s: &Summary, t: &Thresholds) {
    eprintln!("--- summary: {} row(s), {} error(s) ---", s.rows, s.errors);
    eprintln!(
        "{:<12} {:>9} {:>8} {:>8}",
        "axis", "ref_words", "wer", "cer"
    );
    for (axis, wer) in &s.wer_by_axis {
        let cer = s.cer_by_axis.get(axis).copied().unwrap_or_default();
        eprintln!(
            "{axis:<12} {:>9} {:>8} {:>8}",
            wer.reference_len(),
            fmt_opt(Some(wer.rate())),
            fmt_opt(Some(cer.rate()))
        );
    }
    eprintln!(
        "{:<12} {:>9} {:>8} {:>8}",
        "overall",
        s.wer_total.reference_len(),
        fmt_opt(Some(s.wer_total.rate())),
        fmt_opt(Some(s.cer_total.rate()))
    );
    eprintln!(
        "{:<20} {:>4} {:>10} {:>10}",
        "metric", "n", "mean", "median"
    );
    for name in [
        "wer",
        "cer",
        "hnr_db",
        "lufs",
        "peak_dbfs",
        "leading_silence_s",
        "trailing_silence_s",
        "speech_ratio",
        "words_per_s",
        "f0_mean_hz",
        "latent_cosine",
        "f0_mean_delta_hz",
        "f0_std_ratio",
    ] {
        if let Some(st) = s.stats.get(name) {
            eprintln!(
                "{name:<20} {:>4} {:>10} {:>10}",
                st.n,
                fmt_opt(Some(st.mean)),
                fmt_opt(Some(st.median))
            );
        }
    }
    if !s.checks_failed.is_empty() {
        eprintln!("checks_passed=false: {}", s.checks_failed.join(", "));
    }
    if !s.max_len.is_empty() {
        eprintln!("stop_reason=max_len: {}", s.max_len.join(", "));
    }
    if t.is_empty() {
        eprintln!("no thresholds given: no violations possible");
    } else {
        eprintln!(
            "rows with violations: {} ({:?})",
            s.rows_with_violations, s.violation_counts
        );
    }
}
