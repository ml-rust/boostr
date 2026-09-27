//! `vocab.json`: the g2p_lm char-level token layout written by
//! `assets/g2p_lm/build_dataset.py`.
//!
//! Layout: specials `pad`=0 `bos`=1 `sep`=2 `eos`=3 `us`=4 `gb`=5, then
//! graphemes (case-kept ASCII `[A-Za-z'-.]`) at ids 6-60, then phonemes
//! (single Kokoro vocab characters) at ids 61-110.

use std::collections::HashMap;

use serde::Deserialize;

use crate::error::{Error, Result};

/// Raw shape of `vocab.json`, before the char-keyed maps this module builds.
#[derive(Debug, Deserialize)]
struct RawVocab {
    specials: HashMap<String, u32>,
    graphemes: HashMap<String, u32>,
    phonemes: HashMap<String, u32>,
    target_base: u32,
    eos: u32,
    vocab_size: u32,
}

/// The g2p_lm token vocabulary: specials, graphemes, phonemes.
#[derive(Debug, Clone)]
pub struct G2pVocab {
    specials: HashMap<String, u32>,
    graphemes: HashMap<char, u32>,
    id_to_phoneme: HashMap<u32, char>,
    /// First phoneme id. Every generated id `>= target_base` is a phoneme;
    /// anything below it (a special or a stray grapheme id) ends decoding.
    pub target_base: u32,
    /// `<eos>` token id: the stop token passed to `generate_greedy`.
    pub eos: u32,
    /// Total id count, `pad` through the last phoneme.
    pub vocab_size: u32,
}

impl G2pVocab {
    /// Parses `vocab.json`. Every grapheme and phoneme key must be exactly
    /// one character, matching what `build_dataset.py` asserts on the way in.
    pub fn from_json(text: &str) -> Result<Self> {
        let raw: RawVocab = serde_json::from_str(text).map_err(|e| Error::ModelError {
            reason: format!("g2p_lm vocab.json does not parse: {e}"),
        })?;

        let mut graphemes = HashMap::with_capacity(raw.graphemes.len());
        for (key, id) in &raw.graphemes {
            let ch = one_char(key, "graphemes")?;
            graphemes.insert(ch, *id);
        }

        let mut id_to_phoneme = HashMap::with_capacity(raw.phonemes.len());
        for (key, id) in &raw.phonemes {
            let ch = one_char(key, "phonemes")?;
            id_to_phoneme.insert(*id, ch);
        }

        Ok(Self {
            specials: raw.specials,
            graphemes,
            id_to_phoneme,
            target_base: raw.target_base,
            eos: raw.eos,
            vocab_size: raw.vocab_size,
        })
    }

    /// `<bos>` token id.
    pub fn bos(&self) -> Result<u32> {
        self.special("bos")
    }

    /// `<sep>` token id.
    pub fn sep(&self) -> Result<u32> {
        self.special("sep")
    }

    /// Dialect token id: `dialect` is `"us"` or `"gb"`.
    pub fn dialect(&self, dialect: &str) -> Result<u32> {
        self.special(dialect)
    }

    fn special(&self, name: &str) -> Result<u32> {
        self.specials
            .get(name)
            .copied()
            .ok_or_else(|| Error::ModelError {
                reason: format!("g2p_lm vocab.json has no special token {name:?}"),
            })
    }

    /// Grapheme id for one character, or `None` outside the trained set
    /// (`[A-Za-z'-.]`).
    pub fn grapheme(&self, ch: char) -> Option<u32> {
        self.graphemes.get(&ch).copied()
    }

    /// Encodes every character of `word` as a grapheme id. `None` on the
    /// first character outside the trained set.
    pub fn encode_word(&self, word: &str) -> Option<Vec<u32>> {
        word.chars().map(|c| self.grapheme(c)).collect()
    }

    /// Phoneme symbol for a generated id, or `None` when `id` is not one of
    /// the trained phoneme ids (a decoding error, since generation is meant
    /// to stay in `target_base..vocab_size` until `eos`).
    pub fn phoneme_symbol(&self, id: u32) -> Option<char> {
        self.id_to_phoneme.get(&id).copied()
    }
}

/// Extracts the single character a vocab key must hold, or a
/// [`Error::ModelError`] naming which section and key broke that assumption.
fn one_char(key: &str, section: &str) -> Result<char> {
    let mut chars = key.chars();
    let ch = chars.next().ok_or_else(|| Error::ModelError {
        reason: format!("g2p_lm vocab.json {section} has an empty key"),
    })?;
    if chars.next().is_some() {
        return Err(Error::ModelError {
            reason: format!("g2p_lm vocab.json {section} key {key:?} is not one character"),
        });
    }
    Ok(ch)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> &'static str {
        r#"{
            "specials": {"pad": 0, "bos": 1, "sep": 2, "eos": 3, "us": 4, "gb": 5},
            "graphemes": {"a": 6, "b": 7},
            "phonemes": {"ˈ": 8, "b": 9},
            "target_base": 8,
            "eos": 3,
            "vocab_size": 10
        }"#
    }

    #[test]
    fn parses_specials_graphemes_and_phonemes() {
        let vocab = G2pVocab::from_json(sample()).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(vocab.bos().unwrap_or_else(|e| panic!("{e}")), 1);
        assert_eq!(vocab.sep().unwrap_or_else(|e| panic!("{e}")), 2);
        assert_eq!(vocab.dialect("us").unwrap_or_else(|e| panic!("{e}")), 4);
        assert_eq!(vocab.grapheme('a'), Some(6));
        assert_eq!(vocab.grapheme('z'), None);
        assert_eq!(vocab.phoneme_symbol(9), Some('b'));
        assert_eq!(vocab.target_base, 8);
        assert_eq!(vocab.eos, 3);
        assert_eq!(vocab.vocab_size, 10);
    }

    #[test]
    fn encode_word_stops_at_the_first_unknown_character() {
        let vocab = G2pVocab::from_json(sample()).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(vocab.encode_word("ab"), Some(vec![6, 7]));
        assert_eq!(vocab.encode_word("az"), None);
    }

    #[test]
    fn unknown_dialect_names_the_missing_special() {
        let vocab = G2pVocab::from_json(sample()).unwrap_or_else(|e| panic!("{e}"));
        let err = vocab.dialect("fr").unwrap_err();
        assert!(err.to_string().contains("fr"));
    }

    #[test]
    fn multi_character_key_is_rejected() {
        let bad = r#"{
            "specials": {"pad": 0, "bos": 1, "sep": 2, "eos": 3, "us": 4, "gb": 5},
            "graphemes": {"ab": 6},
            "phonemes": {},
            "target_base": 7,
            "eos": 3,
            "vocab_size": 7
        }"#;
        let err = G2pVocab::from_json(bad).unwrap_err();
        assert!(err.to_string().contains("graphemes"));
    }
}
