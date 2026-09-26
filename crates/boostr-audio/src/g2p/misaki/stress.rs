//! Stress-mark rewriting: misaki `apply_stress` and `stress_weight`.

use super::chars::{DIPHTHONGS, PRIMARY_STRESS, SECONDARY_STRESS, is_vowel};

fn has_stress(ps: &str) -> bool {
    ps.contains(PRIMARY_STRESS) || ps.contains(SECONDARY_STRESS)
}

/// Places `mark` before the first vowel of an unstressed `ps`.
///
/// misaki prepends the mark and sorts it to half a position before the first
/// vowel. The caller checks that `ps` holds a vowel and no stress marks.
fn restress(mark: char, ps: &str) -> String {
    let mut out = String::with_capacity(ps.len() + mark.len_utf8());
    let mut placed = false;
    for c in ps.chars() {
        if !placed && is_vowel(c) {
            out.push(mark);
            placed = true;
        }
        out.push(c);
    }
    out
}

/// misaki `apply_stress(ps, stress)`.
///
/// `None` keeps `ps`. Below -1 strips every mark. -1, or 0 and -0.5 when a
/// primary mark exists, demote primary to secondary. 0, 0.5 and 1 add a
/// secondary mark to an unmarked word. 1 and above promote a lone secondary
/// mark. Above 1 adds a primary mark to an unmarked word.
pub(crate) fn apply_stress(ps: &str, stress: Option<f64>) -> String {
    let Some(stress) = stress else {
        return ps.to_string();
    };
    if stress < -1.0 {
        return ps
            .chars()
            .filter(|&c| c != PRIMARY_STRESS && c != SECONDARY_STRESS)
            .collect();
    }
    if stress == -1.0 || ((stress == 0.0 || stress == -0.5) && ps.contains(PRIMARY_STRESS)) {
        return ps
            .chars()
            .filter(|&c| c != SECONDARY_STRESS)
            .map(|c| {
                if c == PRIMARY_STRESS {
                    SECONDARY_STRESS
                } else {
                    c
                }
            })
            .collect();
    }
    if (stress == 0.0 || stress == 0.5 || stress == 1.0) && !has_stress(ps) {
        if !ps.chars().any(is_vowel) {
            return ps.to_string();
        }
        return restress(SECONDARY_STRESS, ps);
    }
    if stress >= 1.0 && !ps.contains(PRIMARY_STRESS) && ps.contains(SECONDARY_STRESS) {
        return ps
            .chars()
            .map(|c| {
                if c == SECONDARY_STRESS {
                    PRIMARY_STRESS
                } else {
                    c
                }
            })
            .collect();
    }
    if stress > 1.0 && !has_stress(ps) {
        if !ps.chars().any(is_vowel) {
            return ps.to_string();
        }
        return restress(PRIMARY_STRESS, ps);
    }
    ps.to_string()
}

/// `apply_stress` over an optional phoneme string. misaki raises on `None`
/// with a stress value set. The port returns `None` there.
pub(crate) fn apply_stress_opt(ps: Option<String>, stress: Option<f64>) -> Option<String> {
    ps.map(|p| apply_stress(&p, stress))
}

/// misaki `stress_weight`: diphthongs and affricates count two, the rest one.
pub(crate) fn stress_weight(ps: &str) -> usize {
    ps.chars()
        .map(|c| if DIPHTHONGS.contains(c) { 2 } else { 1 })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_misaki_apply_stress() {
        let cases: &[(&str, Option<f64>, &str)] = &[
            ("ˈæpəl", Some(-1.0), "ˌæpəl"),
            ("ˈæpəlˌtɹi", Some(-2.0), "æpəltɹi"),
            ("æpəl", Some(0.0), "ˌæpəl"),
            ("æpəl", Some(2.0), "ˈæpəl"),
            ("ˌæpəl", Some(1.0), "ˈæpəl"),
            ("ˈæpəl", Some(0.0), "ˌæpəl"),
            ("ˌæpəl", Some(0.0), "ˌæpəl"),
            ("æpəl", Some(0.5), "ˌæpəl"),
            ("ptk", Some(2.0), "ptk"),
            ("ˈæpəl", Some(-0.5), "ˌæpəl"),
            ("ˈæpəl", None, "ˈæpəl"),
            ("bæt", Some(1.0), "bˌæt"),
        ];
        for &(ps, stress, want) in cases {
            assert_eq!(apply_stress(ps, stress), want, "{ps} {stress:?}");
        }
    }

    #[test]
    fn weight_counts_diphthongs_twice() {
        assert_eq!(stress_weight("bAt"), 4);
        assert_eq!(stress_weight(""), 0);
        assert_eq!(stress_weight("ʧɪp"), 4);
    }
}
