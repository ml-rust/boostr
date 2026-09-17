//! Limits a scored row must respect, and the check that applies them.
//!
//! A violation string starts with the metric name, then a space, then the
//! value and the limit: `wer 0.250 > max 0.100`. The summary counts
//! violations by that first word, so the name is part of the contract.

use std::collections::BTreeMap;

use crate::tts_eval::summary::RowScore;

/// Limits a row must respect. Every field is optional; an unset limit is
/// never checked, so a default `Thresholds` reports no violations.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Thresholds {
    pub max_wer: Option<f64>,
    pub max_cer: Option<f64>,
    pub min_hnr_db: Option<f64>,
    pub min_lufs: Option<f64>,
    pub max_lufs: Option<f64>,
    pub max_leading_silence_s: Option<f64>,
    pub max_trailing_silence_s: Option<f64>,
    /// `(min, max)` words per second of speech.
    pub words_per_s: Option<(f64, f64)>,
    pub min_latent_cosine: Option<f64>,
}

impl Thresholds {
    /// Whether any limit is set.
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

/// Check `row` against `t`. A limit whose metric the row lacks is a
/// violation too: a gate cannot pass what it did not measure.
pub fn violations(row: &RowScore, t: &Thresholds) -> Vec<String> {
    let metrics: BTreeMap<&str, f64> = row.metrics().into_iter().collect();
    let mut out = Vec::new();
    let mut check = |name: &str, limit: Option<f64>, ok: fn(f64, f64) -> bool, rel: &str| {
        let Some(limit) = limit else { return };
        match metrics.get(name) {
            Some(&v) if ok(v, limit) => {}
            Some(&v) => out.push(format!("{name} {v:.3} {rel} {limit:.3}")),
            None => out.push(format!("{name} missing, limit {limit:.3}")),
        }
    };
    check("wer", t.max_wer, |v, l| v <= l, "> max");
    check("cer", t.max_cer, |v, l| v <= l, "> max");
    check("hnr_db", t.min_hnr_db, |v, l| v >= l, "< min");
    check("lufs", t.min_lufs, |v, l| v >= l, "< min");
    check("lufs", t.max_lufs, |v, l| v <= l, "> max");
    check(
        "leading_silence_s",
        t.max_leading_silence_s,
        |v, l| v <= l,
        "> max",
    );
    check(
        "trailing_silence_s",
        t.max_trailing_silence_s,
        |v, l| v <= l,
        "> max",
    );
    if let Some((lo, hi)) = t.words_per_s {
        check("words_per_s", Some(lo), |v, l| v >= l, "< min");
        check("words_per_s", Some(hi), |v, l| v <= l, "> max");
    }
    check("latent_cosine", t.min_latent_cosine, |v, l| v >= l, "< min");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tts_eval::signal::SignalStats;

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
    fn empty_thresholds_never_violate() {
        let mut row = RowScore::new("a", "hello", "a.wav");
        row.signal = Some(signal(-20.0, Some(15.0)));
        assert!(violations(&row, &Thresholds::default()).is_empty());
        assert!(Thresholds::default().is_empty());
    }

    #[test]
    fn violations_name_the_metric_first() {
        let mut row = RowScore::new("a", "hello", "a.wav");
        row.signal = Some(signal(-20.0, Some(5.0)));
        let t = Thresholds {
            min_hnr_db: Some(10.0),
            max_lufs: Some(-23.0),
            max_wer: Some(0.1),
            ..Default::default()
        };
        let v = violations(&row, &t);
        assert_eq!(v.len(), 3, "{v:?}");
        assert!(v[0].starts_with("wer missing"), "{v:?}");
        assert!(v[1].starts_with("hnr_db 5.000 < min 10.000"), "{v:?}");
        assert!(v[2].starts_with("lufs -20.000 > max -23.000"), "{v:?}");
    }
}
