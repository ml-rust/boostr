//! Word matching between a script and what a recognizer heard: word
//! normalization, the recognizer's split words rejoined, units written onto
//! numbers split off, and the edit-distance alignment of the two lists.

use unicode_normalization::UnicodeNormalization;

use super::heard::HeardWord;
use crate::narration::pattern::{
    Patterns, group, is_space, replace_all, replace_all_with, split, trim,
};

/// Cost of pairing two words that are not alike. It is a little less than
/// a skipped word on each side: a lone misheard word still pairs up, but
/// alike words never shift out of line to pair two mismatches.
pub(crate) const SUB: f64 = 1.9;

/// Words longer than this may differ by a few letters and still match.
/// Shorter words match only exactly: one letter changes them ("tahi" is not
/// "tapi").
const SHORT_WORD: usize = 4;

/// Units a recognizer writes short: "L" for "liter". Each set is one word.
const UNITS: [&[&str]; 13] = [
    &["l", "liter", "litre", "liters", "litres"],
    &["km", "kilometer", "kilometre", "kilometers", "kilometres"],
    &["m", "meter", "metre", "meters", "metres"],
    &["kg", "kilogram", "kilograms"],
    &["g", "gram", "grams"],
    &["cm", "sentimeter", "centimeter", "centimetre"],
    &["%", "peratus", "percent", "persen"],
    &["kb", "kilobyte", "kilobytes"],
    &["mb", "megabyte", "megabytes"],
    &["gb", "gigabyte", "gigabytes"],
    &["tb", "terabyte", "terabytes"],
    &["mhz", "megahertz"],
    &["ghz", "gigahertz"],
];

/// Spoken short forms a recognizer writes out in full, and loanword
/// spellings. Each set is one word.
const SAME: [&[&str]; 13] = [
    &["ni", "ini"],
    &["tu", "itu"],
    &["tak", "tidak"],
    &["dah", "sudah"],
    &["je", "saja", "sahaja"],
    &["kat", "dekat"],
    &["nak", "hendak"],
    &["org", "orang"],
    &["yg", "yang"],
    &["dgn", "dengan"],
    &["utk", "untuk"],
    &["saintis", "scientist"],
    &["teknologi", "technology"],
];

/// The set a word belongs to. No word is in two sets.
fn set_of(sets: &[&[&str]], w: &str) -> Option<usize> {
    sets.iter().position(|set| set.contains(&w))
}

fn unit_of(w: &str) -> Option<usize> {
    set_of(&UNITS, w)
}

/// A word's normalized pieces: lowercased, NFKD, anything but letters,
/// digits, whitespace and `%` made a space, and a number split from a unit
/// written onto it ("5L" is "5 l"). NFKD splits an accent off its letter
/// and the accent then becomes a space: "crème" is "cre me".
pub(crate) fn norm(p: &Patterns, t: &str) -> Vec<String> {
    let t: String = t.to_lowercase().nfkd().collect();
    let t = replace_all_with(&p.not_word, &t, " ");
    let t = replace_all(&p.digit_unit, &t, |caps| {
        format!(
            "{} {}",
            group(caps, 1).unwrap_or_default(),
            group(caps, 2).unwrap_or_default()
        )
    });
    split(&p.spaces, &t)
        .into_iter()
        .filter(|w| !w.is_empty())
        .map(str::to_string)
        .collect()
}

/// `norm(t).join('')`: a word's pieces run together.
pub(crate) fn key(p: &Patterns, t: &str) -> String {
    norm(p, t).concat()
}

/// A script's words: split at whitespace, words with no pieces dropped.
pub(crate) fn script_words<'t>(p: &Patterns, script: &'t str) -> Vec<&'t str> {
    split(&p.spaces, script)
        .into_iter()
        .filter(|w| !norm(p, w).is_empty())
        .collect()
}

/// A recognizer splits some written words apart: "non-stop" comes back as
/// "non" and "-stop", "1,100" as "1" and ",100". A piece that starts with a
/// hyphen before a letter, or a comma before a digit, rejoins the word
/// before it and takes its end.
pub(crate) fn join_fragments(p: &Patterns, words: &[HeardWord]) -> Vec<HeardWord> {
    let mut out: Vec<HeardWord> = Vec::with_capacity(words.len());
    for w in words {
        if let Some(prev) = out.last_mut()
            && p.fragment.is_match(&w.text)
        {
            prev.text = format!("{}{}", prev.text.trim_end_matches(is_space), trim(&w.text));
            prev.end = w.end;
        } else {
            out.push(w.clone());
        }
    }
    out
}

/// Fragments rejoined, then a word with a unit written onto its number
/// ("5L") made two words at the same time, so the unit can match the
/// script's word for it. End punctuation stays with the unit.
pub(crate) fn split_units(p: &Patterns, words: &[HeardWord]) -> Vec<HeardWord> {
    let mut out = Vec::with_capacity(words.len());
    for w in join_fragments(p, words) {
        let parts = p.number_unit.captures(&w.text).and_then(|caps| {
            let number = group(&caps, 1)?;
            let unit = group(&caps, 2)?;
            let tail = group(&caps, 3).unwrap_or_default();
            unit_of(&unit.to_lowercase()).map(|_| (number.to_string(), format!("{unit}{tail}")))
        });
        match parts {
            Some((number, unit)) => {
                out.push(HeardWord {
                    text: number,
                    ..w.clone()
                });
                out.push(HeardWord { text: unit, ..w });
            }
            None => out.push(w),
        }
    }
    out
}

/// Edit distance over UTF-16 code units, for words the recognizer spelled
/// slightly wrong ("iscribe" for "describe").
fn distance(a: &[u16], b: &[u16]) -> usize {
    let mut d: Vec<usize> = (0..=b.len()).collect();
    for (i, &ca) in a.iter().enumerate() {
        let mut prev = d[0];
        d[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let t = d[j + 1];
            d[j + 1] = (t + 1).min(d[j] + 1).min(prev + usize::from(ca != cb));
            prev = t;
        }
    }
    d[b.len()]
}

/// Two normalized words that stand for the same spoken word: equal, one
/// unit, one short form, or a spelling within a quarter of the longer
/// word's length. A number is a fact: a word with a digit matches only
/// exactly, so 1982 never matches 1985. Words of 4 letters or fewer match
/// only exactly.
pub(crate) fn alike(a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }
    if unit_of(a).is_some_and(|i| unit_of(b) == Some(i)) {
        return true;
    }
    if set_of(&SAME, a).is_some_and(|i| set_of(&SAME, b) == Some(i)) {
        return true;
    }
    let digit = |s: &str| s.bytes().any(|c| c.is_ascii_digit());
    if digit(a) || digit(b) {
        return false;
    }
    let a: Vec<u16> = a.encode_utf16().collect();
    let b: Vec<u16> = b.encode_utf16().collect();
    let longer = a.len().max(b.len());
    longer > SHORT_WORD && distance(&a, &b) <= longer / 4
}

/// Script words aligned to recognized words.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Matched {
    /// The recognized word each script word lines up with, if any.
    pub(crate) pair: Vec<Option<usize>>,
    /// Pairs of alike words.
    pub(crate) sure: Vec<bool>,
}

/// Aligns script words to recognized words by edit distance over words,
/// alike words pairing for free and others for [`SUB`].
///
/// Each word is compared as its normalized pieces run together, not piece
/// by piece: "Wi-Fi" is "wifi" on both sides. The backtrack prefers the
/// diagonal on ties, then a skipped script word.
pub(crate) fn match_words(p: &Patterns, words: &[HeardWord], said: &[&str]) -> Matched {
    let a: Vec<String> = said.iter().map(|w| key(p, w)).collect();
    let b: Vec<String> = words.iter().map(|w| key(p, &w.text)).collect();
    let (n, m) = (a.len(), b.len());
    let same: Vec<bool> = a
        .iter()
        .flat_map(|x| b.iter().map(move |y| alike(x, y)))
        .collect();
    let sub = |i: usize, j: usize| if same[(i - 1) * m + j - 1] { 0.0 } else { SUB };
    let w = m + 1;
    let mut cost = vec![0.0f64; (n + 1) * w];
    for i in 0..=n {
        cost[i * w] = i as f64;
    }
    for (j, c) in cost.iter_mut().take(w).enumerate() {
        *c = j as f64;
    }
    for i in 1..=n {
        for j in 1..=m {
            cost[i * w + j] = (cost[(i - 1) * w + j - 1] + sub(i, j))
                .min(cost[(i - 1) * w + j] + 1.0)
                .min(cost[i * w + j - 1] + 1.0);
        }
    }
    let mut pair = vec![None; n];
    let (mut i, mut j) = (n, m);
    while i > 0 {
        if j > 0 && cost[i * w + j] == cost[(i - 1) * w + j - 1] + sub(i, j) {
            pair[i - 1] = Some(j - 1);
            i -= 1;
            j -= 1;
        } else if cost[i * w + j] == cost[(i - 1) * w + j] + 1.0 {
            i -= 1;
        } else if j > 0 {
            j -= 1;
        } else {
            break;
        }
    }
    let sure = pair
        .iter()
        .enumerate()
        .map(|(i, j)| j.is_some_and(|j| same[i * m + j]))
        .collect();
    Matched { pair, sure }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result;

    fn heard(texts: &[&str]) -> Vec<HeardWord> {
        texts
            .iter()
            .enumerate()
            .map(|(i, t)| HeardWord {
                text: (*t).to_string(),
                start: i as f64,
                end: i as f64 + 0.5,
            })
            .collect()
    }

    fn texts(words: &[HeardWord]) -> Vec<&str> {
        words.iter().map(|w| w.text.as_str()).collect()
    }

    #[test]
    fn norm_splits_units_and_accents() -> Result<()> {
        let p = Patterns::get()?;
        assert_eq!(norm(p, "5L"), ["5", "l"]);
        assert_eq!(norm(p, "20%,"), ["20", "%"]);
        assert_eq!(norm(p, "Wi-Fi"), ["wi", "fi"]);
        assert_eq!(norm(p, "crème"), ["cre", "me"]);
        assert_eq!(norm(p, "café"), ["cafe"]);
        assert!(norm(p, "—").is_empty());
        assert_eq!(key(p, "non-stop"), "nonstop");
        Ok(())
    }

    #[test]
    fn fragments_rejoin_and_units_split() -> Result<()> {
        let p = Patterns::get()?;
        let joined = join_fragments(p, &heard(&["non", "-stop", "1", ",100", "-", "x"]));
        assert_eq!(texts(&joined), ["non-stop", "1,100", "-", "x"]);
        assert_eq!(joined[0].end, 1.5);
        let split = split_units(p, &heard(&["5L", "20%.", "3.5kg!", "5x", "Ltd"]));
        assert_eq!(
            texts(&split),
            ["5", "L", "20", "%.", "3.5", "kg!", "5x", "Ltd"]
        );
        assert_eq!((split[1].start, split[1].end), (0.0, 0.5));
        Ok(())
    }

    #[test]
    fn alike_rules() {
        assert!(alike("describe", "iscribe"));
        assert!(!alike("tapi", "tahi"));
        assert!(!alike("1982", "1985"));
        assert!(alike("l", "liter"));
        assert!(alike("tak", "tidak"));
        assert!(!alike("tak", "liter"));
        // 5 letters allow one edit, 8 allow two.
        assert!(alike("north", "forth"));
        assert!(!alike("north", "fourth"));
        assert_eq!(distance(&[1, 2, 3], &[1, 3]), 1);
    }

    #[test]
    fn match_prefers_the_diagonal() -> Result<()> {
        let p = Patterns::get()?;
        let m = match_words(
            p,
            &heard(&["the", "river", "walks", "north"]),
            &["The", "river", "runs", "north."],
        );
        assert_eq!(m.pair, [Some(0), Some(1), Some(2), Some(3)]);
        assert_eq!(m.sure, [true, true, false, true]);
        let m = match_words(p, &heard(&["a"]), &["a", "b"]);
        assert_eq!(m.pair, [Some(0), None]);
        let m = match_words(p, &[], &["a"]);
        assert_eq!(m.pair, [None::<usize>]);
        Ok(())
    }
}
