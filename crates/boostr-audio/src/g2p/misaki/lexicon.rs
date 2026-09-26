//! misaki `Lexicon`: gold and silver lookup, capitalization, and the
//! per-token entry point.
//!
//! Special cases live in `special.rs`, suffix stemming in `stem.rs`, and
//! number reading in `number.rs`; each adds an `impl Lexicon` block.

use unicode_normalization::UnicodeNormalization;

use super::chars::{
    PRIMARY_STRESS, SECONDARY_STRESS, char_len, digit_value, drop_last, is_alpha, is_letter,
    is_lexicon_char, lower, tail, upper,
};
use super::data::{self, Dictionary, Entry};
use super::number::is_number;
use super::stress::{apply_stress, apply_stress_opt};
use super::token::{MToken, TokenContext};
use crate::error::Result;

/// Phonemes and rating, as misaki's `(ps, rating)` pairs.
pub(crate) type Lookup = (Option<String>, Option<u8>);

/// Symbols read as words: misaki `SYMBOLS`.
pub(crate) const SYMBOLS: [(&str, &str); 4] =
    [("%", "percent"), ("&", "and"), ("+", "plus"), ("@", "at")];

/// Stress for a capitalized word, then for an all-caps word.
const CAP_STRESSES: (f64, f64) = (0.5, 2.0);

/// misaki `Lexicon` for one dialect.
pub(crate) struct Lexicon {
    pub(crate) british: bool,
    golds: Dictionary,
    silvers: Dictionary,
}

/// misaki `Lexicon.get_parent_tag`.
pub(crate) fn get_parent_tag(tag: Option<&str>) -> Option<&str> {
    let t = tag?;
    Some(if t.starts_with("VB") {
        "VERB"
    } else if t.starts_with("NN") {
        "NOUN"
    } else if t.starts_with("ADV") || t.starts_with("RB") {
        "ADV"
    } else if t.starts_with("ADJ") || t.starts_with("JJ") {
        "ADJ"
    } else {
        t
    })
}

pub(crate) fn symbol_word(word: &str) -> Option<&'static str> {
    SYMBOLS.iter().find(|(s, _)| *s == word).map(|(_, w)| *w)
}

/// misaki `Lexicon.numeric_if_needed`: a digit of any script becomes its
/// ASCII digit.
fn numeric_if_needed(c: char) -> char {
    match digit_value(c).and_then(|v| char::from_digit(v, 10)) {
        Some(d) => d,
        None => c,
    }
}

impl Lexicon {
    /// Decodes the embedded lexicon for US (`british = false`) or GB English.
    pub(crate) fn new(british: bool) -> Result<Self> {
        let (golds, silvers) = data::load(british)?;
        Ok(Self {
            british,
            golds,
            silvers,
        })
    }

    pub(crate) fn in_golds(&self, word: &str) -> bool {
        self.golds.contains_key(word)
    }

    pub(crate) fn in_silvers(&self, word: &str) -> bool {
        self.silvers.contains_key(word)
    }

    /// `self.golds[key]` for a plain entry.
    pub(crate) fn gold_plain(&self, key: &str) -> Option<String> {
        match self.golds.get(key) {
            Some(Entry::Plain(p)) => Some(p.clone()),
            _ => None,
        }
    }

    /// `self.golds[key][tag]` for a tagged entry.
    pub(crate) fn gold_tagged(&self, key: &str, tag: &str) -> Option<String> {
        self.golds
            .get(key)
            .and_then(|e| e.tagged_get(tag))
            .flatten()
            .map(str::to_string)
    }

    /// misaki `get_NNP`: spells the letters of `word` from the gold letter
    /// names and moves primary stress to the last letter.
    pub(crate) fn get_nnp(&self, word: &str) -> Lookup {
        let mut joined = String::new();
        for c in word.chars().filter(|&c| is_letter(c)) {
            let key: String = c.to_uppercase().collect();
            match self.golds.get(&key) {
                Some(Entry::Plain(p)) => joined.push_str(p),
                _ => return (None, None),
            }
        }
        let ps = apply_stress(&joined, Some(0.0));
        let ps = match ps.rfind(SECONDARY_STRESS) {
            Some(i) => format!(
                "{}{}{}",
                &ps[..i],
                PRIMARY_STRESS,
                &ps[i + SECONDARY_STRESS.len_utf8()..]
            ),
            None => ps,
        };
        (Some(ps), Some(3))
    }

    /// misaki `is_known`. The tag argument is unused upstream.
    pub(crate) fn is_known(&self, word: &str) -> bool {
        if self.in_golds(word) || symbol_word(word).is_some() || self.in_silvers(word) {
            return true;
        }
        if !is_alpha(word) || !word.chars().all(is_lexicon_char) {
            return false;
        }
        if char_len(word) == 1 {
            return true;
        }
        if word == upper(word) && self.in_golds(&lower(word)) {
            return true;
        }
        let t = tail(word);
        t == upper(t)
    }

    /// misaki `lookup`: gold, then silver unless the word reads as an
    /// all-caps proper noun, then tag selection, then letter spelling.
    pub(crate) fn lookup(
        &self,
        word: &str,
        tag: Option<&str>,
        stress: Option<f64>,
        ctx: Option<&TokenContext>,
    ) -> Lookup {
        let mut word = word.to_string();
        let mut is_nnp: Option<bool> = None;
        if word == upper(&word) && !self.in_golds(&word) {
            word = lower(&word);
            is_nnp = Some(tag == Some("NNP"));
        }
        let mut entry = self.golds.get(&word);
        let mut rating = 4u8;
        if entry.is_none() && is_nnp != Some(true) {
            entry = self.silvers.get(&word);
            rating = 3;
        }
        let ps: Option<String> = match entry {
            None => None,
            Some(Entry::Plain(p)) => Some(p.clone()),
            Some(e @ Entry::Tagged(_)) => {
                let key = if ctx.is_some_and(|c| c.future_vowel.is_none())
                    && e.tagged_get("None").is_some()
                {
                    Some("None")
                } else if tag.is_none_or(|t| e.tagged_get(t).is_none()) {
                    get_parent_tag(tag)
                } else {
                    tag
                };
                match key.and_then(|k| e.tagged_get(k)) {
                    Some(v) => v.map(str::to_string),
                    None => e.tagged_get("DEFAULT").flatten().map(str::to_string),
                }
            }
        };
        let needs_nnp = match &ps {
            None => true,
            Some(p) => is_nnp == Some(true) && !p.contains(PRIMARY_STRESS),
        };
        if needs_nnp {
            // misaki overwrites `(ps, rating)` with the spelling result.
            return self.get_nnp(&word);
        }
        (apply_stress_opt(ps, stress), Some(rating))
    }

    /// misaki `get_word`: special cases, lowercase retry, known-word lookup,
    /// possessive forms, then -s, -ed, -ing stemming.
    pub(crate) fn get_word(
        &self,
        word: &str,
        tag: &str,
        stress: Option<f64>,
        ctx: &TokenContext,
    ) -> Lookup {
        let special = self.get_special_case(word, tag, stress, ctx);
        if special.0.is_some() {
            return special;
        }
        let wl = lower(word);
        let found = |r: Lookup| r.0.is_some_and(|p| !p.is_empty());
        let use_lower = char_len(word) > 1
            && is_alpha(&word.replace('\'', ""))
            && word != wl
            && (tag != "NNP" || char_len(word) > 7)
            && !self.in_golds(word)
            && !self.in_silvers(word)
            && (word == upper(word) || tail(word) == lower(tail(word)))
            && (self.in_golds(&wl)
                || self.in_silvers(&wl)
                || found(self.stem_s(&wl, Some(tag), stress, Some(ctx)))
                || found(self.stem_ed(&wl, Some(tag), stress, Some(ctx)))
                || found(self.stem_ing(&wl, Some(tag), stress, Some(ctx))));
        let word: &str = if use_lower { &wl } else { word };

        if self.is_known(word) {
            return self.lookup(word, Some(tag), stress, Some(ctx));
        }
        if word.ends_with("s'") {
            let possessive = format!("{}'s", drop_last(word, 2));
            if self.is_known(&possessive) {
                return self.lookup(&possessive, Some(tag), stress, Some(ctx));
            }
        }
        if word.ends_with('\'') {
            let bare = drop_last(word, 1);
            if self.is_known(bare) {
                return self.lookup(bare, Some(tag), stress, Some(ctx));
            }
        }
        let s = self.stem_s(word, Some(tag), stress, Some(ctx));
        if s.0.is_some() {
            return s;
        }
        let ed = self.stem_ed(word, Some(tag), stress, Some(ctx));
        if ed.0.is_some() {
            return ed;
        }
        let ing_stress = Some(stress.unwrap_or(0.5));
        let ing = self.stem_ing(word, Some(tag), ing_stress, Some(ctx));
        if ing.0.is_some() {
            return ing;
        }
        (None, None)
    }

    /// misaki `Lexicon.__call__`: normalizes the token text and resolves it
    /// as a word, then as a number.
    pub(crate) fn resolve(&self, tk: &MToken, ctx: &TokenContext) -> Lookup {
        let source = tk.alias.as_deref().unwrap_or(&tk.text);
        let quoted = source.replace(['\u{2018}', '\u{2019}'], "'");
        let word: String = quoted.nfkc().map(numeric_if_needed).collect();
        let stress = if word == lower(&word) {
            None
        } else if word == upper(&word) {
            Some(CAP_STRESSES.1)
        } else {
            Some(CAP_STRESSES.0)
        };
        let (ps, rating) = self.get_word(&word, &tk.tag, stress, ctx);
        if let Some(ps) = ps {
            let ps = self.append_currency(&ps, tk.currency.as_deref());
            return (Some(apply_stress(&ps, tk.stress)), rating);
        }
        if is_number(&word, tk.is_head) {
            let (ps, rating) =
                self.get_number(&word, tk.currency.as_deref(), tk.is_head, &tk.num_flags);
            return (apply_stress_opt(ps, tk.stress), rating);
        }
        (None, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lexicon(british: bool) -> Lexicon {
        match Lexicon::new(british) {
            Ok(l) => l,
            Err(e) => panic!("lexicon: {e}"),
        }
    }

    fn word(lex: &Lexicon, text: &str, tag: &str) -> Option<String> {
        lex.resolve(&MToken::new(text, tag, ""), &TokenContext::default())
            .0
    }

    #[test]
    fn parent_tags() {
        assert_eq!(get_parent_tag(Some("VBD")), Some("VERB"));
        assert_eq!(get_parent_tag(Some("NNS")), Some("NOUN"));
        assert_eq!(get_parent_tag(Some("RBR")), Some("ADV"));
        assert_eq!(get_parent_tag(Some("JJ")), Some("ADJ"));
        assert_eq!(get_parent_tag(Some("DT")), Some("DT"));
        assert_eq!(get_parent_tag(None), None);
    }

    #[test]
    fn heteronyms_follow_tags() {
        let us = lexicon(false);
        assert_eq!(word(&us, "read", "VBD").as_deref(), Some("ɹˈɛd"));
        assert_eq!(word(&us, "read", "VB").as_deref(), Some("ɹˈid"));
        assert_eq!(word(&us, "lead", "NN").as_deref(), Some("lˈid"));
    }

    #[test]
    fn caps_spell_unknown_acronyms() {
        let us = lexicon(false);
        assert_eq!(word(&us, "SQL", "NNP").as_deref(), Some("sˈikwəl"));
        assert_eq!(word(&us, "XQZ", "NNP").as_deref(), Some("ˌɛkskjˌuzˈi"));
        assert_eq!(word(&us, "U.S.", "NNP").as_deref(), Some("jˌuˈɛs"));
        assert_eq!(word(&us, "Hello", "UH").as_deref(), Some("həlˈO"));
    }
}
