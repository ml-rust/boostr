//! misaki suffix stemming: `-s`, `-ed`, `-ing` on the spelling, and the
//! matching phoneme suffixes on the stem's pronunciation.

use super::chars::{US_TAUS, char_len, drop_last};
use super::lexicon::{Lexicon, Lookup};
use super::token::TokenContext;

fn last_char(s: &str) -> Option<char> {
    s.chars().next_back()
}

fn second_last_char(s: &str) -> Option<char> {
    s.chars().rev().nth(1)
}

/// misaki `_s`: the plural or third-person `-s` sound after `stem`.
pub(crate) fn suffix_s(stem: Option<&str>, british: bool) -> Option<String> {
    let stem = stem.filter(|s| !s.is_empty())?;
    let last = last_char(stem)?;
    Some(if "ptkfθ".contains(last) {
        format!("{stem}s")
    } else if "szʃʒʧʤ".contains(last) {
        format!("{stem}{}z", if british { 'ɪ' } else { 'ᵻ' })
    } else {
        format!("{stem}z")
    })
}

/// misaki `_ed`: the past-tense `-ed` sound after `stem`.
pub(crate) fn suffix_ed(stem: Option<&str>, british: bool) -> Option<String> {
    let stem = stem.filter(|s| !s.is_empty())?;
    let last = last_char(stem)?;
    Some(if "pkfθʃsʧ".contains(last) {
        format!("{stem}t")
    } else if last == 'd' {
        format!("{stem}{}d", if british { 'ɪ' } else { 'ᵻ' })
    } else if last != 't' {
        format!("{stem}d")
    } else if british || char_len(stem) < 2 {
        format!("{stem}ɪd")
    } else if second_last_char(stem).is_some_and(|c| US_TAUS.contains(c)) {
        format!("{}ɾᵻd", drop_last(stem, 1))
    } else {
        format!("{stem}ᵻd")
    })
}

/// misaki `_ing`: the `-ing` sound after `stem`. GB English refuses a stem
/// ending in `ə` or `ː`.
pub(crate) fn suffix_ing(stem: Option<&str>, british: bool) -> Option<String> {
    let stem = stem.filter(|s| !s.is_empty())?;
    let last = last_char(stem)?;
    if british {
        if last == 'ə' || last == 'ː' {
            return None;
        }
    } else if char_len(stem) > 1
        && last == 't'
        && second_last_char(stem).is_some_and(|c| US_TAUS.contains(c))
    {
        return Some(format!("{}ɾɪŋ", drop_last(stem, 1)));
    }
    Some(format!("{stem}ɪŋ"))
}

/// `re.search(r'([bcdgklmnprstvxz])\1ing$|cking$', word)`.
fn doubled_consonant_ing(word: &str) -> bool {
    if word.ends_with("cking") {
        return true;
    }
    let Some(stem) = word.strip_suffix("ing") else {
        return false;
    };
    let mut it = stem.chars().rev();
    match (it.next(), it.next()) {
        (Some(a), Some(b)) => a == b && "bcdgklmnprstvxz".contains(a),
        _ => false,
    }
}

impl Lexicon {
    /// misaki `stem_s`.
    pub(crate) fn stem_s(
        &self,
        word: &str,
        tag: Option<&str>,
        stress: Option<f64>,
        ctx: Option<&TokenContext>,
    ) -> Lookup {
        let n = char_len(word);
        if n < 3 || !word.ends_with('s') {
            return (None, None);
        }
        let stem = if !word.ends_with("ss") && self.is_known(drop_last(word, 1)) {
            drop_last(word, 1).to_string()
        } else if (word.ends_with("'s")
            || (n > 4 && word.ends_with("es") && !word.ends_with("ies")))
            && self.is_known(drop_last(word, 2))
        {
            drop_last(word, 2).to_string()
        } else if n > 4
            && word.ends_with("ies")
            && self.is_known(&format!("{}y", drop_last(word, 3)))
        {
            format!("{}y", drop_last(word, 3))
        } else {
            return (None, None);
        };
        let (ps, rating) = self.lookup(&stem, tag, stress, ctx);
        (suffix_s(ps.as_deref(), self.british), rating)
    }

    /// misaki `stem_ed`.
    pub(crate) fn stem_ed(
        &self,
        word: &str,
        tag: Option<&str>,
        stress: Option<f64>,
        ctx: Option<&TokenContext>,
    ) -> Lookup {
        let n = char_len(word);
        if n < 4 || !word.ends_with('d') {
            return (None, None);
        }
        let stem = if !word.ends_with("dd") && self.is_known(drop_last(word, 1)) {
            drop_last(word, 1)
        } else if n > 4
            && word.ends_with("ed")
            && !word.ends_with("eed")
            && self.is_known(drop_last(word, 2))
        {
            drop_last(word, 2)
        } else {
            return (None, None);
        };
        let (ps, rating) = self.lookup(stem, tag, stress, ctx);
        (suffix_ed(ps.as_deref(), self.british), rating)
    }

    /// misaki `stem_ing`.
    pub(crate) fn stem_ing(
        &self,
        word: &str,
        tag: Option<&str>,
        stress: Option<f64>,
        ctx: Option<&TokenContext>,
    ) -> Lookup {
        let n = char_len(word);
        if n < 5 || !word.ends_with("ing") {
            return (None, None);
        }
        let base = drop_last(word, 3);
        let stem = if n > 5 && self.is_known(base) {
            base.to_string()
        } else if self.is_known(&format!("{base}e")) {
            format!("{base}e")
        } else if n > 5 && doubled_consonant_ing(word) && self.is_known(drop_last(word, 4)) {
            drop_last(word, 4).to_string()
        } else {
            return (None, None);
        };
        let (ps, rating) = self.lookup(&stem, tag, stress, ctx);
        (suffix_ing(ps.as_deref(), self.british), rating)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phoneme_suffixes() {
        assert_eq!(suffix_s(Some("kˈæt"), false).as_deref(), Some("kˈæts"));
        assert_eq!(suffix_s(Some("bˈʌs"), false).as_deref(), Some("bˈʌsᵻz"));
        assert_eq!(suffix_s(Some("bˈʌs"), true).as_deref(), Some("bˈʌsɪz"));
        assert_eq!(suffix_s(Some(""), false), None);
        assert_eq!(suffix_ed(Some("wˈɔk"), false).as_deref(), Some("wˈɔkt"));
        assert_eq!(suffix_ed(Some("nˈid"), false).as_deref(), Some("nˈidᵻd"));
        assert_eq!(suffix_ed(Some("wˈɑnt"), false).as_deref(), Some("wˈɑntᵻd"));
        assert_eq!(suffix_ed(Some("sˈit"), false).as_deref(), Some("sˈiɾᵻd"));
        assert_eq!(suffix_ed(Some("sˈit"), true).as_deref(), Some("sˈitɪd"));
        assert_eq!(suffix_ing(Some("sˈɪt"), false).as_deref(), Some("sˈɪɾɪŋ"));
        assert_eq!(suffix_ing(Some("sˈɪt"), true).as_deref(), Some("sˈɪtɪŋ"));
        assert_eq!(suffix_ing(Some("kˈʌlə"), true), None);
    }

    #[test]
    fn doubled_consonants() {
        assert!(doubled_consonant_ing("running") && doubled_consonant_ing("kicking"));
        assert!(!doubled_consonant_ing("seeing") && !doubled_consonant_ing("ing"));
    }

    #[test]
    fn stems_match_misaki() {
        let ctx = TokenContext::default();
        for (british, cases) in [
            (
                false,
                [
                    ("cats", "kˈæts"),
                    ("buses", "bˈʌsᵻz"),
                    ("flies", "flˈIz"),
                    ("walked", "wˈɔkt"),
                    ("wanted", "wˈɑntᵻd"),
                    ("sitting", "sˈɪɾɪŋ"),
                    ("making", "mˌAkɪŋ"),
                ],
            ),
            (
                true,
                [
                    ("cats", "kˈats"),
                    ("buses", "bˈʌsɪz"),
                    ("flies", "flˈIz"),
                    ("walked", "wˈɔːkt"),
                    ("wanted", "wˈɒntɪd"),
                    ("sitting", "sˈɪtɪŋ"),
                    ("making", "mˌAkɪŋ"),
                ],
            ),
        ] {
            let lex = match Lexicon::new(british) {
                Ok(l) => l,
                Err(e) => panic!("lexicon: {e}"),
            };
            for (word, want) in cases {
                let got = if word.ends_with("ing") {
                    lex.stem_ing(word, Some("VBG"), Some(0.5), Some(&ctx)).0
                } else if word.ends_with('d') {
                    lex.stem_ed(word, Some("VBD"), None, Some(&ctx)).0
                } else {
                    lex.stem_s(word, Some("NNS"), None, Some(&ctx)).0
                };
                assert_eq!(got.as_deref(), Some(want), "{word} british={british}");
            }
        }
    }
}
