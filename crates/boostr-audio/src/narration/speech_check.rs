//! Checks one spoken take against its sentence: speech the sentence does
//! not contain (a generated voice can babble on after the text, or add a
//! stray word), sentence words with no speech at all, and numbers heard
//! other than written.

use serde::Serialize;

use super::heard::HeardWord;
use super::js::js_string;
use super::numbers::{is_number_word, numbers_in};
use super::pattern::Patterns;
use super::speech_match::{key, match_words, norm, script_words, split_units};
use crate::error::Result;

/// Extra speech over this many seconds fails a take.
const EXTRA_SECONDS: f64 = 0.35;
/// This many extra words fail a take.
const EXTRA_WORDS: usize = 2;
/// Missing words over this share of the sentence (and over one word) fail a
/// take.
const MISSING_SHARE: f64 = 0.15;
/// A take with fewer of its words recognized than this share fails.
const SURE_SHARE: f64 = 0.6;
/// Badness per extra word, on top of one per second of extra speech.
const BAD_EXTRA_WORD: f64 = 0.3;
/// Badness per missing word.
const BAD_MISSING_WORD: f64 = 0.3;
/// Badness of a wrong number.
const BAD_WRONG_NUMBER: f64 = 1.0;
/// Badness for a take with none of its words recognized, scaled by the
/// unrecognized share.
const BAD_UNSURE: f64 = 2.0;
/// A script word the recognizer split into pieces counts as heard when its
/// pieces run together are longer than this and appear in the heard words
/// run together.
const COMPOUND_LETTERS: usize = 4;

/// Speech the sentence does not contain.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Extra {
    /// The extra words, space-separated.
    pub text: String,
    /// Their length in seconds: 0 for words heard in place of fewer script
    /// words.
    pub seconds: f64,
}

/// The result of [`check_speech`].
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Verdict {
    /// The take says its sentence.
    pub ok: bool,
    /// Speech the sentence does not contain.
    pub extra: Vec<Extra>,
    /// Sentence words with no speech at all.
    pub missing: Vec<String>,
    /// Sentence words (numbers aside) the recognizer did not hear as
    /// written: mispronounced or misheard. `None` when nothing was
    /// recognized.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unsure: Option<Vec<String>>,
    /// Why the take fails, comma-separated; empty when it passes.
    pub why: String,
    /// How far the take is from its sentence, to keep the best of several
    /// takes. `None` when nothing was recognized.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub badness: Option<f64>,
}

/// Values joined as JavaScript's `Array.prototype.join` writes numbers.
fn join_numbers(values: &[f64], sep: &str) -> String {
    values
        .iter()
        .map(|&v| js_string(v))
        .collect::<Vec<_>>()
        .join(sep)
}

/// Checks the words heard in one take against its sentence.
///
/// Heard words between sure matches stand for the script words there, so a
/// number written as digits covers its spoken words. Numbers are checked
/// apart: every number in the sentence must be heard, as digits or words,
/// and no other.
pub fn check_speech(heard: &[HeardWord], sentence: &str) -> Result<Verdict> {
    let p = Patterns::get()?;
    let words = split_units(p, heard);
    let said = script_words(p, sentence);
    let mut extra = Vec::new();
    let mut missing: Vec<String> = Vec::new();
    if words.is_empty() {
        return Ok(Verdict {
            ok: false,
            extra,
            missing: said.iter().map(|w| w.to_string()).collect(),
            unsure: None,
            why: "no speech recognized".to_string(),
            badness: None,
        });
    }
    let matched = match_words(p, &words, &said);
    let mut last_script = 0;
    let mut last_heard = 0;
    // The script and heard words between the last sure pair and the next.
    let mut gap = |script: &[&str], heard: &[HeardWord]| -> Result<()> {
        // Misheard words pair up one to one. Heard words beyond the
        // script's own words (numbers aside) are extra speech.
        let mut other = Vec::new();
        for w in heard {
            if !is_number_word(&w.text)? {
                other.push(w.text.as_str());
            }
        }
        let mut plain_script = 0usize;
        for w in script {
            if !is_number_word(w)? {
                plain_script += 1;
            }
        }
        let surplus = other.len() as i64 - plain_script as i64;
        if let (Some(first), Some(last)) = (heard.first(), heard.last())
            && script.is_empty()
        {
            let text: Vec<&str> = heard.iter().map(|w| w.text.as_str()).collect();
            extra.push(Extra {
                text: text.join(" "),
                seconds: last.end - first.start,
            });
        } else if surplus >= EXTRA_WORDS as i64 {
            extra.push(Extra {
                text: other.join(" "),
                seconds: 0.0,
            });
        }
        if !script.is_empty() && heard.is_empty() {
            missing.extend(script.iter().map(|w| w.to_string()));
        }
        Ok(())
    };
    for (i, (&sure, pair)) in matched.sure.iter().zip(&matched.pair).enumerate() {
        let Some(j) = pair.filter(|_| sure) else {
            continue;
        };
        gap(&said[last_script..i], &words[last_heard.min(j)..j])?;
        last_script = i + 1;
        last_heard = j + 1;
    }
    gap(
        &said[last_script.min(said.len())..],
        &words[last_heard.min(words.len())..],
    )?;

    let extra_seconds = extra.iter().fold(0.0, |t, e| t + e.seconds);
    let extra_words = extra.iter().fold(0usize, |n, e| n + norm(p, &e.text).len());
    // Words are scored apart from numbers, which have their own exact
    // check: a phrase with a long spoken year and a garbled rest must not
    // pass on its number.
    let mut plain = Vec::new();
    for (i, w) in said.iter().enumerate() {
        if !is_number_word(w)? {
            plain.push((i, *w));
        }
    }
    let sure_share = if plain.is_empty() {
        1.0
    } else {
        plain.iter().filter(|(i, _)| matched.sure[*i]).count() as f64 / plain.len() as f64
    };
    let want = numbers_in(sentence)?;
    let heard_text: Vec<&str> = words.iter().map(|w| w.text.as_str()).collect();
    let got = numbers_in(&heard_text.join(" "))?;
    // `want.join() !== got.join()`: a NaN equals a NaN there.
    let wrong_number = want.len() != got.len()
        || want
            .iter()
            .zip(&got)
            .any(|(a, b)| a != b && !(a.is_nan() && b.is_nan()));

    let mut why = Vec::new();
    if wrong_number {
        let or_none = |s: String| if s.is_empty() { "none".to_string() } else { s };
        why.push(format!(
            "numbers heard as {}, the script says {}",
            or_none(join_numbers(&got, ", ")),
            or_none(join_numbers(&want, ", "))
        ));
    }
    if extra_seconds > EXTRA_SECONDS || extra_words >= EXTRA_WORDS {
        let texts: Vec<&str> = extra.iter().map(|e| e.text.as_str()).collect();
        why.push(format!("extra speech \"{}\"", texts.join(" … ")));
    }
    if missing.len() as f64 > (said.len() as f64 * MISSING_SHARE).max(1.0) {
        why.push(format!("missing \"{}\"", missing.join(" ")));
    }
    // Recognition mishears some words, so a take passes with a few unsure
    // ones, but not with most of them.
    if sure_share < SURE_SHARE {
        why.push(format!(
            "only {}% of its words recognized",
            js_string((sure_share * 100.0).round())
        ));
    }
    let badness = extra_seconds
        + BAD_EXTRA_WORD * extra_words as f64
        + BAD_MISSING_WORD * missing.len() as f64
        + (if wrong_number { BAD_WRONG_NUMBER } else { 0.0 })
        + BAD_UNSURE * (1.0 - sure_share).max(0.0);
    // A compound heard as two words ("Wi -Fi" for "Wi-Fi") was heard right:
    // its pieces run together appear in the heard words run together.
    let stream: String = words.iter().map(|w| key(p, &w.text)).collect();
    let unsure = plain
        .iter()
        .filter(|(i, w)| {
            let pieces = norm(p, w);
            let joined = pieces.concat();
            !matched.sure[*i]
                && !(joined.encode_utf16().count() > COMPOUND_LETTERS
                    && pieces.len() > 1
                    && stream.contains(&joined))
        })
        .map(|(_, w)| w.to_string())
        .collect();
    Ok(Verdict {
        ok: why.is_empty(),
        extra,
        missing,
        unsure: Some(unsure),
        why: why.join(", "),
        badness: Some(badness),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Words 0.4 s apart, each 0.35 s long.
    fn heard(text: &str) -> Vec<HeardWord> {
        text.split(' ')
            .enumerate()
            .map(|(i, t)| HeardWord {
                text: t.to_string(),
                start: i as f64 * 0.4,
                end: i as f64 * 0.4 + 0.35,
            })
            .collect()
    }

    #[test]
    fn clean_take_passes() -> Result<()> {
        let v = check_speech(
            &heard("Masuk tahun 1896, Kuala Lumpur jadi ibu kota."),
            "Masuk tahun lapan belas sembilan puluh enam, Kuala Lumpur jadi ibu kota.",
        )?;
        assert!(v.ok, "{}", v.why);
        assert_eq!(v.unsure, Some(vec![]));
        Ok(())
    }

    #[test]
    fn wrong_year_and_babble_fail() -> Result<()> {
        let v = check_speech(
            &heard("Masuk tahun 1860, Kuala Lumpur jadi ibu kota."),
            "Masuk tahun lapan belas sembilan puluh enam, Kuala Lumpur jadi ibu kota.",
        )?;
        assert!(
            v.why
                .contains("numbers heard as 1860, the script says 1896")
        );
        let v = check_speech(
            &heard("It opens at nine. marbulan nambar"),
            "It opens at nine.",
        )?;
        assert!(!v.ok);
        assert!(v.why.contains("extra speech \"marbulan nambar\""));
        assert_eq!(v.extra.len(), 1);
        Ok(())
    }

    #[test]
    fn nothing_heard() -> Result<()> {
        let v = check_speech(&[], "Two words.")?;
        assert!(!v.ok);
        assert_eq!(v.missing, ["Two", "words."]);
        assert_eq!((v.unsure, v.badness), (None, None));
        assert_eq!(v.why, "no speech recognized");
        Ok(())
    }

    #[test]
    fn split_compound_is_not_unsure() -> Result<()> {
        let v = check_speech(
            &heard("The cafe has free Wi -Fi for every guest."),
            "The cafe has free Wi-Fi for every guest.",
        )?;
        assert!(v.ok, "{}", v.why);
        assert_eq!(v.unsure, Some(vec![]));
        Ok(())
    }
}
