//! Phonetic tokenizer: what the text says out loud, not how it is spelled.

use crate::eval::align::{ErrorRate, align};
use crate::eval::normalize::normalize;

/// Malay affixes that attach to the token after them during canonicalization.
const AFFIXES: [&str; 12] = [
    "di", "ter", "ber", "me", "ke", "se", "pe", "men", "mem", "meng", "peng", "pem",
];

/// Tokenize `text` for scoring, collapsing spellings that sound the same.
///
/// Steps, in order:
///
/// 1. `RM23.50` becomes `23 ringgit 50 sen`, `RM250` becomes `250 ringgit`,
///    and `3.30` becomes `3 30`. Case-insensitive, space after `RM` optional.
/// 2. Tokenize with [`crate::eval::normalize()`].
/// 3. Replace Malay and English number words with digits, so `dua belas`,
///    `twelve` and `12` are one token.
/// 4. Join a standalone Malay affix onto the token after it, unless that
///    token is all digits.
///
/// Step 4 is not linguistically correct: `di rumah` becomes `dirumah`. That is
/// intended. The rule runs over the reference and the hypothesis alike, so a
/// consistent transformation cannot change whether the two agree. It buys a
/// match on `di-restart` against `di restart` and `dimerge`, which is a
/// spelling difference and never a pronunciation difference.
///
/// Use [`crate::eval::normalize()`] instead wherever the token count must track
/// spoken words, such as a speaking-pace ratio.
pub fn canonicalize(text: &str) -> Vec<String> {
    let tokens = normalize(&expand_numerals(text));
    join_affixes(substitute_number_words(&tokens))
}

/// Word error rate of `hypothesis` against `reference` over [`canonicalize`].
pub fn canonical_word_error_rate(reference: &str, hypothesis: &str) -> ErrorRate {
    align(&canonicalize(reference), &canonicalize(hypothesis))
}

/// Character error rate of `hypothesis` against `reference` over
/// [`canonicalize`], tokens rejoined with single spaces.
pub fn canonical_character_error_rate(reference: &str, hypothesis: &str) -> ErrorRate {
    let r: Vec<char> = canonicalize(reference).join(" ").chars().collect();
    let h: Vec<char> = canonicalize(hypothesis).join(" ").chars().collect();
    align(&r, &h)
}

/// Index one past the run of ASCII digits starting at `start`.
fn digits_end(chars: &[char], start: usize) -> usize {
    let mut end = start;
    while chars.get(end).is_some_and(char::is_ascii_digit) {
        end += 1;
    }
    end
}

/// A currency amount at `i`, spelled out, with the index just past it.
fn ringgit_at(chars: &[char], i: usize) -> Option<(String, usize)> {
    if i > 0 && chars[i - 1].is_alphanumeric() {
        return None;
    }
    let (Some(r), Some(m)) = (chars.get(i), chars.get(i + 1)) else {
        return None;
    };
    if !r.eq_ignore_ascii_case(&'r') || !m.eq_ignore_ascii_case(&'m') {
        return None;
    }
    let mut start = i + 2;
    while chars.get(start) == Some(&' ') {
        start += 1;
    }
    let int_end = digits_end(chars, start);
    if int_end == start {
        return None;
    }
    let major: String = chars[start..int_end].iter().collect();
    if chars.get(int_end) == Some(&'.') {
        let frac_end = digits_end(chars, int_end + 1);
        if frac_end == int_end + 3 {
            let minor: String = chars[int_end + 1..frac_end].iter().collect();
            return Some((format!("{major} ringgit {minor} sen"), frac_end));
        }
    }
    Some((format!("{major} ringgit"), int_end))
}

/// Spell out currency amounts and split a decimal point into two runs.
fn expand_numerals(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        if let Some((spelled, next)) = ringgit_at(&chars, i) {
            out.push_str(&spelled);
            i = next;
            continue;
        }
        if !chars[i].is_ascii_digit() {
            out.push(chars[i]);
            i += 1;
            continue;
        }
        let int_end = digits_end(&chars, i);
        out.extend(chars[i..int_end].iter());
        i = int_end;
        if chars.get(int_end) == Some(&'.')
            && chars.get(int_end + 1).is_some_and(char::is_ascii_digit)
        {
            let frac_end = digits_end(&chars, int_end + 1);
            out.push(' ');
            out.extend(chars[int_end + 1..frac_end].iter());
            i = frac_end;
        }
    }
    out
}

/// Malay unit digit, or `None`. `se` is a prefix and never the value 1.
fn malay_unit(token: &str) -> Option<u64> {
    match token {
        "kosong" => Some(0),
        "satu" => Some(1),
        "dua" => Some(2),
        "tiga" => Some(3),
        "empat" => Some(4),
        "lima" => Some(5),
        "enam" => Some(6),
        "tujuh" => Some(7),
        "lapan" | "delapan" => Some(8),
        "sembilan" => Some(9),
        _ => None,
    }
}

/// Longest Malay number starting at `start`, with the token count it spans.
///
/// A scale word (`ratus`, `ribu`, `seratus`, `seribu`) lets the next group
/// join the same number, so `seribu dua ratus` is 1200. A bare value ends the
/// number, so two adjacent bare values stay two numbers.
fn parse_malay(tokens: &[String], start: usize) -> Option<(u64, usize)> {
    let mut value = 0u64;
    let mut idx = start;
    while idx < tokens.len() {
        match tokens[idx].as_str() {
            "sepuluh" => {
                value += 10;
                idx += 1;
                break;
            }
            "sebelas" => {
                value += 11;
                idx += 1;
                break;
            }
            "seratus" => {
                value += 100;
                idx += 1;
            }
            "seribu" => {
                value += 1000;
                idx += 1;
            }
            token => {
                let Some(unit) = malay_unit(token) else { break };
                match tokens.get(idx + 1).map(String::as_str) {
                    Some("belas") => {
                        value += 10 + unit;
                        idx += 2;
                        break;
                    }
                    Some("puluh") => {
                        value += 10 * unit;
                        idx += 2;
                    }
                    Some("ratus") => {
                        value += 100 * unit;
                        idx += 2;
                    }
                    Some("ribu") => {
                        value += 1000 * unit;
                        idx += 2;
                    }
                    _ => {
                        value += unit;
                        idx += 1;
                        break;
                    }
                }
            }
        }
    }
    (idx > start).then_some((value, idx - start))
}

/// One English number word's role.
#[derive(Clone, Copy, PartialEq, Eq)]
enum EnWord {
    /// A unit or teen, 0 through 19.
    Small(u64),
    /// A multiple of ten, 20 through 90.
    Tens(u64),
    /// `hundred`, which multiplies the pending group.
    Hundred,
    /// `thousand` or `million`, which flushes the pending group.
    Scale(u64),
}

fn english_word(token: &str) -> Option<EnWord> {
    let small = |v| Some(EnWord::Small(v));
    let tens = |v| Some(EnWord::Tens(v));
    match token {
        "zero" => small(0),
        "one" => small(1),
        "two" => small(2),
        "three" => small(3),
        "four" => small(4),
        "five" => small(5),
        "six" => small(6),
        "seven" => small(7),
        "eight" => small(8),
        "nine" => small(9),
        "ten" => small(10),
        "eleven" => small(11),
        "twelve" => small(12),
        "thirteen" => small(13),
        "fourteen" => small(14),
        "fifteen" => small(15),
        "sixteen" => small(16),
        "seventeen" => small(17),
        "eighteen" => small(18),
        "nineteen" => small(19),
        "twenty" => tens(20),
        "thirty" => tens(30),
        "forty" => tens(40),
        "fifty" => tens(50),
        "sixty" => tens(60),
        "seventy" => tens(70),
        "eighty" => tens(80),
        "ninety" => tens(90),
        "hundred" => Some(EnWord::Hundred),
        "thousand" => Some(EnWord::Scale(1000)),
        "million" => Some(EnWord::Scale(1_000_000)),
        _ => None,
    }
}

/// What the pending English group will accept next.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Slot {
    /// Nothing pending.
    Empty,
    /// A complete value pending. A scale word can follow, another value cannot.
    Bare,
    /// A hundreds group pending, which a smaller value can still be added to.
    AfterHundred,
}

/// Longest English number starting at `start`, with the token count it spans.
///
/// Only a tens word takes a following unit, so `nine twenty` is 9 then 20 and
/// never 29. `and` is skipped between a scale word and the next number word.
fn parse_english(tokens: &[String], start: usize) -> Option<(u64, usize)> {
    let mut total = 0u64;
    let mut pending = 0u64;
    let mut slot = Slot::Empty;
    let mut after_scale = false;
    let mut idx = start;
    while idx < tokens.len() {
        let token = tokens[idx].as_str();
        if token == "and" {
            let next_is_number = tokens
                .get(idx + 1)
                .and_then(|t| english_word(t.as_str()))
                .is_some_and(|w| matches!(w, EnWord::Small(_) | EnWord::Tens(_)));
            if after_scale && next_is_number {
                idx += 1;
                continue;
            }
            break;
        }
        let Some(word) = english_word(token) else {
            break;
        };
        match word {
            EnWord::Small(v) => {
                if slot == Slot::Bare {
                    break;
                }
                pending += v;
                slot = Slot::Bare;
                idx += 1;
            }
            EnWord::Tens(v) => {
                if slot == Slot::Bare {
                    break;
                }
                pending += v;
                idx += 1;
                let next = tokens.get(idx).and_then(|t| english_word(t.as_str()));
                if let Some(EnWord::Small(u)) = next
                    && (1..=9).contains(&u)
                {
                    pending += u;
                    idx += 1;
                }
                slot = Slot::Bare;
            }
            EnWord::Hundred => {
                if slot != Slot::Bare {
                    break;
                }
                pending *= 100;
                slot = Slot::AfterHundred;
                idx += 1;
            }
            EnWord::Scale(m) => {
                if slot == Slot::Empty {
                    break;
                }
                total += pending * m;
                pending = 0;
                slot = Slot::Empty;
                idx += 1;
            }
        }
        after_scale = matches!(word, EnWord::Hundred | EnWord::Scale(_));
    }
    (idx > start).then_some((total + pending, idx - start))
}

/// Replace number words with their decimal value, longest parse first.
///
/// A tie between the two grammars goes to Malay.
fn substitute_number_words(tokens: &[String]) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(tokens.len());
    let mut i = 0;
    while i < tokens.len() {
        let malay = parse_malay(tokens, i);
        let english = parse_english(tokens, i);
        let parsed = match (malay, english) {
            (Some((_, mc)), Some((ev, ec))) if ec > mc => Some((ev, ec)),
            (Some(m), _) => Some(m),
            (None, other) => other,
        };
        match parsed {
            Some((value, span)) => {
                out.push(value.to_string());
                i += span;
            }
            None => {
                out.push(tokens[i].clone());
                i += 1;
            }
        }
    }
    out
}

/// Glue each standalone affix onto the token after it, digits excepted.
fn join_affixes(tokens: Vec<String>) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(tokens.len());
    let mut i = 0;
    while i < tokens.len() {
        let next_is_word = tokens
            .get(i + 1)
            .is_some_and(|t| !t.chars().all(|c| c.is_ascii_digit()));
        if next_is_word && AFFIXES.contains(&tokens[i].as_str()) {
            out.push(format!("{}{}", tokens[i], tokens[i + 1]));
            i += 2;
            continue;
        }
        out.push(tokens[i].clone());
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn same(reference: &str, hypothesis: &str) {
        assert_eq!(
            canonicalize(reference),
            canonicalize(hypothesis),
            "{reference:?} vs {hypothesis:?}"
        );
    }

    #[test]
    fn currency_with_sen_matches_its_spoken_form() {
        same("Dua puluh tiga ringgit lima puluh sen", "RM23.50");
        assert_eq!(canonicalize("RM23.50"), ["23", "ringgit", "50", "sen"]);
    }

    #[test]
    fn whole_ringgit_matches_an_english_number_word_prompt() {
        same(
            "It cost me two hundred and fifty ringgit",
            "It cost me RM250",
        );
    }

    #[test]
    fn malay_teens_and_units_become_digits() {
        same("dua belas", "12");
        same("dalam pukul tiga petang", "dalam pukul 3 petang");
    }

    #[test]
    fn english_units_become_digits() {
        same(
            "which gives us three working days",
            "which gives us 3 working days",
        );
    }

    #[test]
    fn hyphenated_affixes_match_split_and_joined_spellings() {
        same(
            "Server dah di-restart tiga kali",
            "Server dah di restart 3 kali",
        );
        same(
            "Fail tu dah ter-delete semalam",
            "Fail tu dah terdelete semalam",
        );
        same(
            "Kenapa branch ni tak di-merge lagi",
            "Kenapa branch ni tak dimerge lagi",
        );
    }

    #[test]
    fn two_bare_english_numbers_stay_two_numbers() {
        let tokens = canonicalize("be there by nine twenty");
        assert_eq!(tokens, ["be", "there", "by", "9", "20"]);
    }

    #[test]
    fn a_tens_word_still_takes_its_unit() {
        assert_eq!(canonicalize("twenty one"), ["21"]);
    }

    #[test]
    fn malay_scale_words_compose_left_to_right() {
        assert_eq!(canonicalize("seribu dua ratus"), ["1200"]);
    }

    #[test]
    fn se_alone_is_a_prefix_and_never_the_number_one() {
        assert_eq!(canonicalize("se rumah"), ["serumah"]);
        assert_eq!(canonicalize("se 5"), ["se", "5"]);
    }

    #[test]
    fn plain_text_tokenizes_the_same_as_normalize() {
        let text = "Boleh hantar fail tu sekarang?";
        assert_eq!(canonicalize(text), normalize(text));
    }

    #[test]
    fn a_decimal_point_splits_into_its_parts() {
        assert_eq!(canonicalize("pukul 3.30"), ["pukul", "3", "30"]);
    }

    #[test]
    fn canonical_rates_score_equal_utterances_as_clean() {
        let wer = canonical_word_error_rate("dua belas", "12");
        assert_eq!(wer.errors(), 0);
        let cer = canonical_character_error_rate("dua belas", "12");
        assert_eq!(cer.errors(), 0);
    }
}
