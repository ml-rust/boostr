//! Per-row scores, thresholds, and the batch summary.
//!
//! [`RowScore`] carries one render's provenance and whichever metric structs
//! the gate could compute; `thresholds::violations` checks it against limits;
//! [`Summary`] folds a batch into corpus error rates per axis, means and
//! medians of every numeric metric, and a count of rows failing each
//! threshold.
//!
//! Every numeric metric has one stable snake_case name, listed by
//! [`RowScore::metrics`]. The JSON forms and the summary key on those same
//! names, so a score file written today joins one written next month.
//!
//! A violation string starts with the metric name it belongs to, followed by
//! a space: `wer 0.250 > max 0.100`. [`Summary`] counts violations by that
//! first word.

use std::collections::BTreeMap;

use serde_json::{Value, json};

#[cfg(feature = "whisper")]
use crate::eval::{ErrorRate, by_group, character_error_rate, grand_total, word_error_rate};
#[cfg(feature = "whisper")]
use crate::tts_eval::intelligibility::Intelligibility;
#[cfg(feature = "vad")]
use crate::tts_eval::pace::Pace;
use crate::tts_eval::signal::SignalStats;
#[cfg(feature = "voxcpm")]
use crate::tts_eval::timbre::TimbreProxy;

/// Axis key for a row that carries none.
pub const UNLABELLED_AXIS: &str = "unlabelled";

/// One render's provenance and metrics.
#[derive(Debug, Clone, PartialEq)]
pub struct RowScore {
    /// Prompt id, the join key back to the render log.
    pub id: String,
    /// The text the model was asked to say.
    pub text: String,
    /// Path of the scored wav.
    pub wav: String,
    /// Prompt class, for per-axis totals.
    pub axis: Option<String>,
    /// Language code the prompt was tagged with.
    pub lang: Option<String>,
    /// Render length as the renderer reported it.
    pub audio_seconds: Option<f64>,
    /// `stop_token` or `max_len`, from the renderer.
    pub stop_reason: Option<String>,
    /// The renderer's structural self-checks.
    pub checks_passed: Option<bool>,
    /// Reference clip F0 as the renderer measured it.
    pub ref_f0_hz: Option<f64>,
    /// Native-rate signal statistics.
    pub signal: Option<SignalStats>,
    /// VAD pace, when the gate ran one.
    #[cfg(feature = "vad")]
    pub pace: Option<Pace>,
    /// Whisper transcript and error rates, when the gate ran one.
    #[cfg(feature = "whisper")]
    pub intelligibility: Option<Intelligibility>,
    /// Timbre proxy against the reference, when the gate had one.
    #[cfg(feature = "voxcpm")]
    pub timbre: Option<TimbreProxy>,
    /// Why the row could not be scored, when it could not.
    pub error: Option<String>,
    /// Threshold violations, from [`crate::tts_eval::violations`].
    pub violations: Vec<String>,
}

impl RowScore {
    /// A row with provenance only; metrics start empty.
    pub fn new(id: impl Into<String>, text: impl Into<String>, wav: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            wav: wav.into(),
            axis: None,
            lang: None,
            audio_seconds: None,
            stop_reason: None,
            checks_passed: None,
            ref_f0_hz: None,
            signal: None,
            #[cfg(feature = "vad")]
            pace: None,
            #[cfg(feature = "whisper")]
            intelligibility: None,
            #[cfg(feature = "voxcpm")]
            timbre: None,
            error: None,
            violations: Vec::new(),
        }
    }

    /// Axis for grouping: the row's own or [`UNLABELLED_AXIS`].
    pub fn axis_key(&self) -> &str {
        self.axis.as_deref().unwrap_or(UNLABELLED_AXIS)
    }

    /// Every numeric metric present, under its stable name.
    pub fn metrics(&self) -> Vec<(&'static str, f64)> {
        let mut out = Vec::new();
        if let Some(s) = &self.signal {
            out.push(("duration_s", s.duration_s));
            out.push(("peak_dbfs", s.peak_dbfs));
            out.push(("rms_dbfs", s.rms_dbfs));
            out.push(("lufs", s.lufs));
            out.push(("floor_dbfs", s.floor_dbfs));
            out.push(("snr_db", s.snr_db));
            out.push(("clipped_samples", s.clipped_samples as f64));
            push_opt(&mut out, "f0_mean_hz", s.f0_mean_hz);
            push_opt(&mut out, "f0_std_hz", s.f0_std_hz);
            out.push(("voiced_fraction", s.voiced_fraction));
            push_opt(&mut out, "hnr_db", s.hnr_db);
        }
        #[cfg(feature = "vad")]
        if let Some(p) = &self.pace {
            out.push(("speech_s", p.speech_s));
            out.push(("leading_silence_s", p.leading_silence_s));
            out.push(("trailing_silence_s", p.trailing_silence_s));
            out.push(("speech_ratio", p.speech_ratio));
            out.push(("words", p.words as f64));
            push_opt(&mut out, "words_per_s", p.words_per_speech_s);
            out.push(("segments", p.segments as f64));
        }
        #[cfg(feature = "whisper")]
        if let Some(i) = &self.intelligibility {
            out.push(("wer", i.wer.rate()));
            out.push(("cer", i.cer.rate()));
        }
        #[cfg(feature = "voxcpm")]
        if let Some(t) = &self.timbre {
            out.push(("latent_cosine", t.latent_cosine));
            push_opt(&mut out, "f0_mean_delta_hz", t.f0_mean_delta_hz);
            push_opt(&mut out, "f0_std_ratio", t.f0_std_ratio);
        }
        out
    }

    /// One `"record":"score"` object: provenance, every metric flattened,
    /// the transcript, and the violations.
    pub fn to_json(&self) -> Value {
        let mut obj = serde_json::Map::new();
        obj.insert("record".into(), json!("score"));
        obj.insert("id".into(), json!(self.id));
        obj.insert("axis".into(), json!(self.axis));
        obj.insert("lang".into(), json!(self.lang));
        obj.insert("text".into(), json!(self.text));
        obj.insert("wav".into(), json!(self.wav));
        obj.insert("audio_seconds".into(), json!(self.audio_seconds));
        obj.insert("stop_reason".into(), json!(self.stop_reason));
        obj.insert("checks_passed".into(), json!(self.checks_passed));
        obj.insert("ref_f0_hz".into(), json!(self.ref_f0_hz));
        for (name, value) in self.metrics() {
            obj.insert(name.into(), json_number(value));
        }
        #[cfg(feature = "whisper")]
        if let Some(i) = &self.intelligibility {
            obj.insert("transcript".into(), json!(i.transcript));
            obj.insert("wer_counts".into(), error_rate_json(&i.wer));
            obj.insert("cer_counts".into(), error_rate_json(&i.cer));
        }
        obj.insert("error".into(), json!(self.error));
        obj.insert("violations".into(), json!(self.violations));
        Value::Object(obj)
    }
}

fn push_opt(out: &mut Vec<(&'static str, f64)>, name: &'static str, value: Option<f64>) {
    if let Some(v) = value {
        out.push((name, v));
    }
}

/// JSON has no infinity; a non-finite metric serialises as `null`.
fn json_number(value: f64) -> Value {
    if value.is_finite() {
        json!(value)
    } else {
        Value::Null
    }
}

#[cfg(feature = "whisper")]
fn error_rate_json(rate: &ErrorRate) -> Value {
    json!({
        "substitutions": rate.substitutions,
        "deletions": rate.deletions,
        "insertions": rate.insertions,
        "hits": rate.hits,
        "rate": json_number(rate.rate()),
    })
}

/// Count, mean and median of one metric over the rows that carry it finite.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetricStat {
    pub n: usize,
    pub mean: f64,
    pub median: f64,
}

/// Batch-level view of a slice of [`RowScore`]s.
#[derive(Debug, Clone, PartialEq)]
pub struct Summary {
    /// Rows seen, scored or not.
    pub rows: usize,
    /// Rows with `error` set.
    pub errors: usize,
    /// Mean and median per metric name.
    pub stats: BTreeMap<&'static str, MetricStat>,
    /// Corpus word error rate per axis.
    #[cfg(feature = "whisper")]
    pub wer_by_axis: BTreeMap<String, ErrorRate>,
    /// Corpus character error rate per axis.
    #[cfg(feature = "whisper")]
    pub cer_by_axis: BTreeMap<String, ErrorRate>,
    /// Corpus word error rate over every axis.
    #[cfg(feature = "whisper")]
    pub wer_total: ErrorRate,
    /// Corpus character error rate over every axis.
    #[cfg(feature = "whisper")]
    pub cer_total: ErrorRate,
    /// Violations per metric name, summed over rows.
    pub violation_counts: BTreeMap<String, usize>,
    /// Rows with at least one violation.
    pub rows_with_violations: usize,
    /// Ids whose renderer self-checks failed.
    pub checks_failed: Vec<String>,
    /// Ids the renderer stopped at `max_len`.
    pub max_len: Vec<String>,
}

impl Summary {
    /// Fold `rows`. Rows with `error` set contribute to `errors` only.
    pub fn from_rows(rows: &[RowScore]) -> Self {
        let mut samples: BTreeMap<&'static str, Vec<f64>> = BTreeMap::new();
        let mut violation_counts: BTreeMap<String, usize> = BTreeMap::new();
        let mut rows_with_violations = 0usize;
        let mut checks_failed = Vec::new();
        let mut max_len = Vec::new();
        let mut errors = 0usize;
        for row in rows {
            if row.error.is_some() {
                errors += 1;
            }
            for (name, value) in row.metrics() {
                if value.is_finite() {
                    samples.entry(name).or_default().push(value);
                }
            }
            if !row.violations.is_empty() {
                rows_with_violations += 1;
            }
            for v in &row.violations {
                let name = v.split_whitespace().next().unwrap_or("").to_string();
                *violation_counts.entry(name).or_default() += 1;
            }
            if row.checks_passed == Some(false) {
                checks_failed.push(row.id.clone());
            }
            if row.stop_reason.as_deref() == Some("max_len") {
                max_len.push(row.id.clone());
            }
        }
        let stats = samples
            .into_iter()
            .map(|(name, values)| (name, metric_stat(values)))
            .collect();

        #[cfg(feature = "whisper")]
        let scored = rows.iter().filter_map(|r| {
            r.intelligibility.as_ref().map(|i| {
                (
                    r.axis_key().to_string(),
                    r.text.as_str(),
                    i.transcript.as_str(),
                )
            })
        });
        #[cfg(feature = "whisper")]
        let (wer_by_axis, cer_by_axis) = {
            let items: Vec<(String, &str, &str)> = scored.collect();
            let wer = by_group(
                items.iter().map(|(k, r, h)| (k.clone(), *r, *h)),
                word_error_rate,
            );
            let cer = by_group(
                items.iter().map(|(k, r, h)| (k.clone(), *r, *h)),
                character_error_rate,
            );
            (wer, cer)
        };

        Self {
            rows: rows.len(),
            errors,
            stats,
            #[cfg(feature = "whisper")]
            wer_total: grand_total(&wer_by_axis),
            #[cfg(feature = "whisper")]
            cer_total: grand_total(&cer_by_axis),
            #[cfg(feature = "whisper")]
            wer_by_axis,
            #[cfg(feature = "whisper")]
            cer_by_axis,
            violation_counts,
            rows_with_violations,
            checks_failed,
            max_len,
        }
    }

    /// One `"record":"summary"` object.
    pub fn to_json(&self) -> Value {
        let stats: serde_json::Map<String, Value> = self
            .stats
            .iter()
            .map(|(name, s)| {
                (
                    (*name).to_string(),
                    json!({ "n": s.n, "mean": json_number(s.mean), "median": json_number(s.median) }),
                )
            })
            .collect();
        let mut obj = serde_json::Map::new();
        obj.insert("record".into(), json!("summary"));
        obj.insert("rows".into(), json!(self.rows));
        obj.insert("errors".into(), json!(self.errors));
        obj.insert("stats".into(), Value::Object(stats));
        #[cfg(feature = "whisper")]
        {
            let by_axis: serde_json::Map<String, Value> = self
                .wer_by_axis
                .iter()
                .map(|(axis, wer)| {
                    let cer = self.cer_by_axis.get(axis).copied().unwrap_or_default();
                    (
                        axis.clone(),
                        json!({ "wer": error_rate_json(wer), "cer": error_rate_json(&cer) }),
                    )
                })
                .collect();
            obj.insert("by_axis".into(), Value::Object(by_axis));
            obj.insert("wer".into(), error_rate_json(&self.wer_total));
            obj.insert("cer".into(), error_rate_json(&self.cer_total));
        }
        obj.insert("violation_counts".into(), json!(self.violation_counts));
        obj.insert(
            "rows_with_violations".into(),
            json!(self.rows_with_violations),
        );
        obj.insert("checks_failed".into(), json!(self.checks_failed));
        obj.insert("max_len".into(), json!(self.max_len));
        Value::Object(obj)
    }
}

fn metric_stat(mut values: Vec<f64>) -> MetricStat {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    let mean = values.iter().sum::<f64>() / n as f64;
    let median = if n.is_multiple_of(2) {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    } else {
        values[n / 2]
    };
    MetricStat { n, mean, median }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signal(lufs: f64, hnr: Option<f64>) -> SignalStats {
        SignalStats {
            duration_s: 1.0,
            peak_dbfs: -1.0,
            rms_dbfs: -10.0,
            lufs,
            floor_dbfs: -60.0,
            snr_db: 50.0,
            clipped_samples: 0,
            f0_mean_hz: Some(120.0),
            f0_std_hz: Some(10.0),
            voiced_fraction: 0.5,
            hnr_db: hnr,
        }
    }

    #[test]
    fn summary_counts_violations_and_skips_non_finite() {
        let mut a = RowScore::new("a", "hello", "a.wav");
        a.signal = Some(signal(-20.0, Some(5.0)));
        a.violations = vec!["hnr_db 5.000 < min 10.000".into()];
        a.checks_passed = Some(false);
        let mut b = RowScore::new("b", "world", "b.wav");
        b.signal = Some(signal(f64::NEG_INFINITY, None));
        b.stop_reason = Some("max_len".into());
        let mut c = RowScore::new("c", "again", "c.wav");
        c.error = Some("unreadable".into());

        let s = Summary::from_rows(&[a, b, c]);
        assert_eq!(s.rows, 3);
        assert_eq!(s.errors, 1);
        assert_eq!(s.rows_with_violations, 1);
        assert_eq!(s.violation_counts.get("hnr_db"), Some(&1));
        assert_eq!(s.checks_failed, vec!["a".to_string()]);
        assert_eq!(s.max_len, vec!["b".to_string()]);
        let lufs = s.stats.get("lufs").expect("lufs");
        assert_eq!(lufs.n, 1);
        assert_eq!(lufs.mean, -20.0);
        let hnr = s.stats.get("hnr_db").expect("hnr");
        assert_eq!((hnr.n, hnr.median), (1, 5.0));
        assert_eq!(s.to_json()["record"], json!("summary"));
    }

    #[test]
    fn median_of_even_and_odd_counts() {
        assert_eq!(metric_stat(vec![3.0, 1.0, 2.0]).median, 2.0);
        assert_eq!(metric_stat(vec![4.0, 1.0, 2.0, 3.0]).median, 2.5);
    }

    #[test]
    fn row_json_flattens_metrics_and_nulls_infinity() {
        let mut row = RowScore::new("a", "hello", "a.wav");
        row.axis = Some("ms_only".into());
        row.signal = Some(signal(f64::NEG_INFINITY, Some(12.0)));
        let v = row.to_json();
        assert_eq!(v["record"], json!("score"));
        assert_eq!(v["axis"], json!("ms_only"));
        assert_eq!(v["hnr_db"], json!(12.0));
        assert!(v["lufs"].is_null());
        assert_eq!(v["violations"], json!([]));
    }

    #[cfg(feature = "whisper")]
    #[test]
    fn corpus_rates_group_by_axis() {
        let mut a = RowScore::new("a", "one two", "a.wav");
        a.axis = Some("x".into());
        a.intelligibility = Some(Intelligibility {
            transcript: "one two".into(),
            wer: word_error_rate("one two", "one two"),
            cer: character_error_rate("one two", "one two"),
        });
        let mut b = RowScore::new("b", "one two", "b.wav");
        b.intelligibility = Some(Intelligibility {
            transcript: "one".into(),
            wer: word_error_rate("one two", "one"),
            cer: character_error_rate("one two", "one"),
        });
        let s = Summary::from_rows(&[a, b]);
        assert_eq!(s.wer_by_axis["x"].rate(), 0.0);
        assert_eq!(s.wer_by_axis[UNLABELLED_AXIS].rate(), 0.5);
        assert_eq!(s.wer_total.rate(), 0.25);
        assert_eq!(s.stats["wer"].n, 2);
    }
}
