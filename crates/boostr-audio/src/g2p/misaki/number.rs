//! misaki number reading: `is_number`, `get_number`, currency and suffixes.

use super::chars::{is_ascii_digit, is_digit_str};
use super::lexicon::{Lexicon, Lookup};
use super::num2words;
use super::stem::{suffix_ed, suffix_ing, suffix_s};

/// misaki `CURRENCIES`: symbol to (unit, sub-unit).
pub(crate) const CURRENCIES: [(&str, (&str, &str)); 3] = [
    ("$", ("dollar", "cent")),
    ("£", ("pound", "pence")),
    ("€", ("euro", "cent")),
];

/// misaki `ORDINALS`.
const ORDINALS: [&str; 4] = ["st", "nd", "rd", "th"];

pub(crate) fn currency_units(symbol: &str) -> Option<(&'static str, &'static str)> {
    CURRENCIES
        .iter()
        .find(|(s, _)| *s == symbol)
        .map(|(_, u)| *u)
}

fn is_ordinal(suffix: Option<&str>) -> bool {
    suffix.is_some_and(|s| ORDINALS.contains(&s))
}

/// misaki `Lexicon.is_number`: a digit somewhere, and after dropping one
/// known suffix only digits, commas, dots, and a leading minus on a head
/// token.
pub(crate) fn is_number(word: &str, is_head: bool) -> bool {
    if !word.chars().any(is_ascii_digit) {
        return false;
    }
    let suffixes = ["ing", "'d", "ed", "'s", "st", "nd", "rd", "th", "s"];
    let body = suffixes
        .iter()
        .find_map(|s| word.strip_suffix(*s))
        .unwrap_or(word);
    body.chars()
        .enumerate()
        .all(|(i, c)| is_ascii_digit(c) || c == ',' || c == '.' || (is_head && i == 0 && c == '-'))
}

/// misaki `Lexicon.is_currency`: at most one dot and under three cents
/// digits.
fn is_currency(word: &str) -> bool {
    match word.split_once('.') {
        None => true,
        Some((_, cents)) => !cents.contains('.') && cents.chars().count() < 3,
    }
}

/// The trailing run of `[a-z']`, as `re.search(r"[a-z']+$", word)` finds it.
fn trailing_suffix(word: &str) -> Option<&str> {
    let start = word
        .char_indices()
        .rev()
        .take_while(|&(_, c)| c.is_ascii_lowercase() || c == '\'')
        .last()
        .map(|(i, _)| i)?;
    Some(&word[start..])
}

/// Accumulates misaki `get_number`'s `result` list.
struct NumberWords<'a> {
    lex: &'a Lexicon,
    flags: &'a str,
    result: Vec<Lookup>,
}

impl NumberWords<'_> {
    /// misaki `extend_num` on text already in words (`escape=True`).
    fn words(&mut self, text: &str, first: bool) {
        let splits = split_non_letters(text);
        for (i, w) in splits.iter().enumerate() {
            if *w != "and" || self.flags.contains('&') {
                if first && i == 0 && splits.len() > 1 && *w == "one" && self.flags.contains('a') {
                    self.result.push((Some("ə".to_string()), Some(4)));
                } else {
                    let stress = if *w == "point" { Some(-2.0) } else { None };
                    self.result.push(self.lex.lookup(w, None, stress, None));
                }
            } else if self.flags.contains('n') {
                let last = self.result.last_mut();
                if let Some((Some(ps), _)) = last {
                    ps.push_str("ən");
                }
            }
        }
    }

    /// misaki `extend_num` on digits (`escape=False`): reads the cardinal.
    /// A value num2words refuses leaves an unresolved entry, which fails the
    /// whole number.
    fn digits(&mut self, digits: &str, first: bool) {
        match num2words::cardinal(digits) {
            Some(text) => self.words(&text, first),
            None => self.result.push((None, None)),
        }
    }
}

/// `re.split(r'[^a-z]+', text)`: pieces between maximal runs of non-letters,
/// with an empty piece at either end when the text starts or ends with one.
fn split_non_letters(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut in_sep = false;
    for (i, c) in text.char_indices() {
        let sep = !c.is_ascii_lowercase();
        if sep && !in_sep {
            out.push(&text[start..i]);
            in_sep = true;
        } else if !sep && in_sep {
            start = i;
            in_sep = false;
        }
    }
    out.push(if in_sep { "" } else { &text[start..] });
    out
}

impl Lexicon {
    /// misaki `get_number`.
    pub(crate) fn get_number(
        &self,
        word: &str,
        currency: Option<&str>,
        is_head: bool,
        num_flags: &str,
    ) -> Lookup {
        let suffix = trailing_suffix(word);
        let mut word = &word[..word.len() - suffix.map_or(0, str::len)];
        let mut nw = NumberWords {
            lex: self,
            flags: num_flags,
            result: Vec::new(),
        };
        if let Some(rest) = word.strip_prefix('-') {
            nw.result.push(self.lookup("minus", None, None, None));
            word = rest;
        }
        let units = currency.and_then(currency_units);

        if is_digit_str(word) && is_ordinal(suffix) {
            let Some(text) = num2words::ordinal(word) else {
                return (None, None);
            };
            nw.words(&text, true);
        } else if nw.result.is_empty()
            && word.chars().count() == 4
            && units.is_none()
            && is_digit_str(word)
        {
            let Some(text) = num2words::year(word) else {
                return (None, None);
            };
            nw.words(&text, true);
        } else if !is_head && !word.contains('.') {
            let num = word.replace(',', "");
            let chars: Vec<char> = num.chars().collect();
            let Some(&lead) = chars.first() else {
                return (None, None);
            };
            if lead == '0' || chars.len() > 3 {
                for c in &chars {
                    nw.digits(&c.to_string(), false);
                }
            } else if chars.len() == 3 && !num.ends_with("00") {
                nw.digits(&lead.to_string(), true);
                if chars[1] == '0' {
                    nw.result.push(self.lookup("O", None, Some(-2.0), None));
                    nw.digits(&chars[2].to_string(), false);
                } else {
                    nw.digits(&num[1..], false);
                }
            } else {
                nw.digits(&num, true);
            }
        } else if word.matches('.').count() > 1 || !is_head {
            let mut first = true;
            for num in word.replace(',', "").split('.') {
                if num.is_empty() {
                    // misaki skips empty parts but still clears `first`.
                } else if num.starts_with('0')
                    || (num.chars().count() != 2 && num.chars().skip(1).any(|c| c != '0'))
                {
                    for c in num.chars() {
                        nw.digits(&c.to_string(), false);
                    }
                } else {
                    nw.digits(num, first);
                }
                first = false;
            }
        } else if let Some(units) = units.filter(|_| is_currency(word)) {
            let cleaned = word.replace(',', "");
            let mut pairs: Vec<(String, &str)> = cleaned
                .split('.')
                .zip([units.0, units.1])
                .map(|(num, unit)| {
                    let digits = num.trim_start_matches('0');
                    (
                        if digits.is_empty() { "0" } else { digits }.to_string(),
                        unit,
                    )
                })
                .collect();
            if pairs.len() > 1 {
                if pairs[1].0 == "0" {
                    pairs.truncate(1);
                } else if pairs[0].0 == "0" {
                    pairs.remove(0);
                }
            }
            for (i, (num, unit)) in pairs.iter().enumerate() {
                if i > 0 {
                    nw.result.push(self.lookup("and", None, None, None));
                }
                nw.digits(num, i == 0);
                if num != "1" && *unit != "pence" {
                    nw.result
                        .push(self.stem_s(&format!("{unit}s"), None, None, None));
                } else {
                    nw.result.push(self.lookup(unit, None, None, None));
                }
            }
        } else {
            let text = if is_digit_str(word) {
                num2words::cardinal(word)
            } else if !word.contains('.') {
                let digits = word.replace(',', "");
                if is_ordinal(suffix) {
                    num2words::ordinal(&digits)
                } else {
                    num2words::cardinal(&digits)
                }
            } else {
                let cleaned = word.replace(',', "");
                match cleaned.strip_prefix('.') {
                    Some(frac) => {
                        let mut parts = vec!["point".to_string()];
                        for c in frac.chars() {
                            match num2words::cardinal(&c.to_string()) {
                                Some(w) => parts.push(w),
                                None => return (None, None),
                            }
                        }
                        Some(parts.join(" "))
                    }
                    None => num2words::float(&cleaned),
                }
            };
            let Some(text) = text else {
                return (None, None);
            };
            nw.words(&text, true);
        }

        if nw.result.is_empty() || nw.result.iter().any(|(p, _)| p.is_none()) {
            return (None, None);
        }
        let rating = nw.result.iter().filter_map(|(_, r)| *r).min();
        let joined = nw
            .result
            .iter()
            .filter_map(|(p, _)| p.as_deref())
            .collect::<Vec<_>>()
            .join(" ");
        let ps = match suffix {
            Some("s" | "'s") => suffix_s(Some(&joined), self.british),
            Some("ed" | "'d") => suffix_ed(Some(&joined), self.british),
            Some("ing") => suffix_ing(Some(&joined), self.british),
            _ => Some(joined),
        };
        (ps, rating)
    }

    /// misaki `append_currency`: appends the plural unit name.
    pub(crate) fn append_currency(&self, ps: &str, currency: Option<&str>) -> String {
        let unit = currency
            .filter(|c| !c.is_empty())
            .and_then(currency_units)
            .and_then(|(unit, _)| self.stem_s(&format!("{unit}s"), None, None, None).0)
            .filter(|u| !u.is_empty());
        match unit {
            Some(u) => format!("{ps} {u}"),
            None => ps.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// (british, word, currency, is_head, num_flags, expected phonemes).
    type NumberCase = (
        bool,
        &'static str,
        Option<&'static str>,
        bool,
        &'static str,
        &'static str,
    );

    #[test]
    fn number_detection() {
        assert!(is_number("1990s", true) && is_number("-7", true) && is_number("3rd", true));
        assert!(!is_number("-7", false) && !is_number("100km", true) && !is_number("abc", true));
        assert!(is_currency("99.99") && is_currency("5") && !is_currency("1.234"));
        assert!(!is_currency("1.2.3"));
        assert_eq!(trailing_suffix("1990s"), Some("s"));
        assert_eq!(trailing_suffix("5's"), Some("'s"));
        assert_eq!(trailing_suffix("42"), None);
        assert_eq!(
            split_non_letters("one thousand, two"),
            ["one", "thousand", "two"]
        );
        assert_eq!(split_non_letters("-one-"), ["", "one", ""]);
    }

    #[test]
    fn numbers_match_misaki() {
        let cases: &[NumberCase] = &[
            (false, "1990", None, true, "", "nˌIntˈin nˈIndi"),
            (false, "1990s", None, true, "", "nˌIntˈin nˈIndiz"),
            (false, "42", Some("$"), true, "", "fˈɔɹɾi tˈu dˈɑləɹz"),
            (
                false,
                "99.99",
                Some("$"),
                true,
                "",
                "nˈIndi nˈIn dˈɑləɹz ænd nˈIndi nˈIn sˈɛnts",
            ),
            (
                false,
                "1.5",
                Some("$"),
                true,
                "",
                "wˈʌn dˈɑləɹ ænd fˈIv sˈɛnts",
            ),
            (false, "3rd", None, true, "", "θˈɜɹd"),
            (false, "1,024", None, true, "", "wˈʌn θˈWzᵊnd twˈɛnti fˈɔɹ"),
            (false, "3.14", None, true, "", "θɹˈi pYnt wˈʌn fˈɔɹ"),
            (false, "-7", None, true, "", "mˈInəs sˈɛvən"),
            (false, ".5", None, true, "", "pYnt fˈIv"),
            (false, "1.2.3", None, true, "", "wˈʌn tˈu θɹˈi"),
            (false, "007", None, false, "", "zˈɪɹO zˈɪɹO sˈɛvən"),
            (false, "123", None, false, "", "wˈʌn twˈɛnti θɹˈi"),
            (false, "105", None, false, "", "wˈʌn O fˈIv"),
            (false, "1,000,000", None, true, "", "wˈʌn mˈɪljᵊn"),
            (false, "101", None, true, "&", "wˈʌn hˈʌndɹəd ænd wˈʌn"),
            (
                false,
                "150",
                Some("£"),
                true,
                "",
                "wˈʌn hˈʌndɹəd fˈɪfti pˈWndz",
            ),
            (false, "0.01", Some("€"), true, "", "wˈʌn sˈɛnt"),
            (false, "1", Some("$"), true, "", "wˈʌn dˈɑləɹ"),
            (false, "5.00", Some("$"), true, "", "fˈIv dˈɑləɹz"),
            (false, "21st", None, true, "", "twˈɛnti fˈɜɹst"),
            (false, "1", None, true, "a", "wˈʌn"),
            (false, "100", None, true, "a", "ə hˈʌndɹəd"),
            (false, "101", None, true, "n", "wˈʌn hˈʌndɹədən wˈʌn"),
            (true, "1990", None, true, "", "nˌIntˈiːn nˈInti"),
            (true, "1990s", None, true, "", "nˌIntˈiːn nˈIntiz"),
            (true, "42", Some("$"), true, "", "fˈɔːti tˈuː dˈɒləz"),
            (
                true,
                "99.99",
                Some("$"),
                true,
                "",
                "nˈInti nˈIn dˈɒləz and nˈInti nˈIn sˈɛnts",
            ),
            (
                true,
                "1.5",
                Some("$"),
                true,
                "",
                "wˈʌn dˈɒlə and fˈIv sˈɛnts",
            ),
            (true, "3rd", None, true, "", "θˈɜːd"),
            (true, "1,024", None, true, "", "wˈʌn θˈWzᵊnd twˈɛnti fˈɔː"),
            (true, "3.14", None, true, "", "θɹˈiː pYnt wˈʌn fˈɔː"),
            (true, "-7", None, true, "", "mˈInəs sˈɛvᵊn"),
            (true, ".5", None, true, "", "pYnt fˈIv"),
            (true, "1.2.3", None, true, "", "wˈʌn tˈuː θɹˈiː"),
            (true, "007", None, false, "", "zˈɪəɹQ zˈɪəɹQ sˈɛvᵊn"),
            (true, "123", None, false, "", "wˈʌn twˈɛnti θɹˈiː"),
            (true, "105", None, false, "", "wˈʌn Q fˈIv"),
            (true, "1,000,000", None, true, "", "wˈʌn mˈɪljən"),
            (true, "101", None, true, "&", "wˈʌn hˈʌndɹəd and wˈʌn"),
            (
                true,
                "150",
                Some("£"),
                true,
                "",
                "wˈʌn hˈʌndɹəd fˈɪfti pˈWndz",
            ),
            (true, "0.01", Some("€"), true, "", "wˈʌn sˈɛnt"),
            (true, "1", Some("$"), true, "", "wˈʌn dˈɒlə"),
            (true, "5.00", Some("$"), true, "", "fˈIv dˈɒləz"),
            (true, "21st", None, true, "", "twˈɛnti fˈɜːst"),
            (true, "1", None, true, "a", "wˈʌn"),
            (true, "100", None, true, "a", "ə hˈʌndɹəd"),
            (true, "101", None, true, "n", "wˈʌn hˈʌndɹədən wˈʌn"),
        ];
        let us = Lexicon::new(false);
        let gb = Lexicon::new(true);
        let (Ok(us), Ok(gb)) = (us, gb) else {
            panic!("embedded lexicons failed to load");
        };
        for &(british, word, currency, is_head, flags, want) in cases {
            let lex = if british { &gb } else { &us };
            let got = lex.get_number(word, currency, is_head, flags).0;
            assert_eq!(
                got.as_deref(),
                Some(want),
                "{word} {currency:?} british={british}"
            );
        }
    }
}
