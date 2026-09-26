//! Character-level tokenizer for VITS / MMS-TTS.
//!
//! Exact port of HF `transformers.models.vits.tokenization_vits.VitsTokenizer`
//! for the case this crate supports: a checkpoint with `phonemize: false` and
//! `is_uroman: false` (character-vocabulary MMS-TTS checkpoints such as
//! `facebook/mms-tts-*`). [`VitsTokenizer::from_dir`] refuses any checkpoint
//! that needs phonemization or uroman romanization.

use std::collections::HashMap;
use std::path::Path;

use crate::error::{Error, Result};
use crate::tts::vits::load;

/// Character-level VITS/MMS-TTS tokenizer: text in, model input ids out.
///
/// Holds the vocabulary, the special-token strings, and the flags read from
/// `tokenizer_config.json` that shape [`VitsTokenizer::encode`]:
/// `normalize` (lower-case and drop out-of-vocabulary characters) and
/// `add_blank` (intersperse the pad token id between characters).
#[derive(Debug, Clone)]
pub struct VitsTokenizer {
    encoder: HashMap<String, u32>,
    pad_token: String,
    unk_token: String,
    language: Option<String>,
    add_blank: bool,
    normalize: bool,
    /// Id of the pad token, used as the blank-interspersion id. HF always
    /// re-derives this as `encoder[decoder[0]]`, which is the pad token's own
    /// id whenever the vocabulary assigns id 0 to exactly one token.
    blank_id: u32,
    /// Vocabulary words (encoder keys plus any special token absent from the
    /// vocabulary) sorted by descending character length, for the
    /// longest-match-first scan `normalize_text` runs. Sorting by length
    /// rather than replaying the source file's key order is behavior
    /// preserving here because every candidate match is copied back
    /// verbatim: only how far the scan advances depends on which word
    /// matched, and among equal-length candidates any match advances the
    /// same distance.
    match_words: Vec<Vec<char>>,
    /// Distinct special-token strings (pad, unk), sorted by descending
    /// character length. HF registers both as "added tokens" even when one
    /// already sits in the main vocabulary, and its base `tokenize()` splits
    /// the prepared text around them with a trie *before* handing the
    /// remaining runs to `_tokenize`. A special token therefore becomes one
    /// atomic id with no blank interspersed around it, while each run of
    /// ordinary characters between/around special tokens gets its own
    /// independent blank interspersion.
    special_tokens: Vec<Vec<char>>,
}

/// One chunk of text produced by [`VitsTokenizer::split_on_special_tokens`]:
/// either a literal special-token string or a run of ordinary characters.
enum Segment {
    Special(String),
    Regular(Vec<char>),
}

impl VitsTokenizer {
    /// Load a tokenizer from a checkpoint directory containing `vocab.json`,
    /// `tokenizer_config.json`, and optionally `special_tokens_map.json`.
    ///
    /// Errors when the checkpoint needs phonemization (`phonemize: true`) or
    /// uroman romanization (`is_uroman: true`): this crate has no G2P for
    /// VITS, and uroman romanization is a queued unit.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let load::Loaded {
            encoder,
            pad_token,
            unk_token,
            language,
            add_blank,
            normalize,
        } = load::load(dir)?;

        let blank_id = if add_blank {
            let mut zero_token = encoder
                .iter()
                .filter(|(_, id)| **id == 0)
                .map(|(token, _)| token.clone());
            let token = zero_token.next().ok_or_else(|| Error::ModelError {
                reason: format!(
                    "{}: add_blank is true but no vocabulary entry has id 0",
                    dir.display()
                ),
            })?;
            // Round-trip through the encoder, matching HF's
            // `_convert_token_to_id(_convert_id_to_token(0))`.
            *encoder.get(&token).ok_or_else(|| Error::ModelError {
                reason: format!(
                    "{}: token {token:?} disappeared from its own encoder",
                    dir.display()
                ),
            })?
        } else {
            0
        };

        let mut match_words: Vec<Vec<char>> = encoder.keys().map(|w| w.chars().collect()).collect();
        for extra in [&pad_token, &unk_token] {
            if !encoder.contains_key(extra) {
                match_words.push(extra.chars().collect());
            }
        }
        match_words.sort_by_key(|w| std::cmp::Reverse(w.len()));

        let mut special_tokens: Vec<Vec<char>> = Vec::new();
        for tok in [&pad_token, &unk_token] {
            let chars: Vec<char> = tok.chars().collect();
            if !special_tokens.contains(&chars) {
                special_tokens.push(chars);
            }
        }
        special_tokens.sort_by_key(|w| std::cmp::Reverse(w.len()));

        Ok(Self {
            encoder,
            pad_token,
            unk_token,
            language,
            add_blank,
            normalize,
            blank_id,
            match_words,
            special_tokens,
        })
    }

    /// Encode `text` into VITS model input ids.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let normalized = if self.normalize {
            self.normalize_text(text)
        } else {
            text.to_string()
        };
        let preprocessed = self.preprocess_char(&normalized);

        let filtered = if self.normalize {
            let kept: String = preprocessed
                .chars()
                .filter(|c| self.encoder.contains_key(&c.to_string()))
                .collect();
            kept.trim().to_string()
        } else {
            preprocessed
        };

        // HF's base `tokenize()` splits the prepared text into chunks and
        // skips any chunk that comes out empty before calling `_tokenize` —
        // it never reaches the blank-interspersion step. A filtered text
        // that normalizes away to nothing (all-whitespace, all
        // out-of-vocabulary, or an empty input) therefore yields no ids at
        // all, not a single blank.
        if filtered.is_empty() {
            return Ok(Vec::new());
        }

        let mut ids = Vec::new();
        for segment in self.split_on_special_tokens(&filtered) {
            match segment {
                Segment::Special(token) => ids.push(self.convert_special_to_id(&token)?),
                Segment::Regular(chars) => self.push_regular_run(&chars, &mut ids)?,
            }
        }

        Ok(ids)
    }

    /// Split `text` around any occurrence of a special-token string (longest
    /// match first at each position), alternating regular and special
    /// segments and dropping empty regular runs. Mirrors the effect of HF's
    /// `tokens_trie.split` ahead of `_tokenize`.
    fn split_on_special_tokens(&self, text: &str) -> Vec<Segment> {
        let chars: Vec<char> = text.chars().collect();
        let mut segments = Vec::new();
        let mut regular_start = 0;
        let mut i = 0;
        while i < chars.len() {
            let mut matched_len = 0;
            for tok in &self.special_tokens {
                let wlen = tok.len();
                if wlen > 0 && i + wlen <= chars.len() && chars[i..i + wlen] == tok[..] {
                    matched_len = wlen;
                    break;
                }
            }
            if matched_len > 0 {
                if i > regular_start {
                    segments.push(Segment::Regular(chars[regular_start..i].to_vec()));
                }
                segments.push(Segment::Special(chars[i..i + matched_len].iter().collect()));
                i += matched_len;
                regular_start = i;
            } else {
                i += 1;
            }
        }
        if regular_start < chars.len() {
            segments.push(Segment::Regular(chars[regular_start..].to_vec()));
        }
        segments
    }

    /// Convert one ordinary (non-special) character run to ids, interspersing
    /// the blank id around/between its characters when `add_blank` is set.
    /// Each run is interspersed independently of any neighboring run.
    fn push_regular_run(&self, chars: &[char], ids: &mut Vec<u32>) -> Result<()> {
        if self.add_blank {
            ids.push(self.blank_id);
            for &ch in chars {
                ids.push(self.convert_char_to_id(ch)?);
                ids.push(self.blank_id);
            }
        } else {
            for &ch in chars {
                ids.push(self.convert_char_to_id(ch)?);
            }
        }
        Ok(())
    }

    /// Convert a whole special-token segment to its id via the main
    /// vocabulary. A special token absent from the vocabulary (this
    /// checkpoint's `unk_token` never appears there) has no id to fall back
    /// to, matching HF's own tokenizer, which produces a `None` id for that
    /// case rather than a usable one.
    fn convert_special_to_id(&self, token: &str) -> Result<u32> {
        self.encoder
            .get(token)
            .copied()
            .ok_or_else(|| Error::ModelError {
                reason: format!(
                    "special token {token:?} has no id in the vocabulary and no usable fallback"
                ),
            })
    }

    /// Lower-case `input`, leaving any vocabulary or special-token substring
    /// that matches at the current position untouched (longest match wins).
    fn normalize_text(&self, input: &str) -> String {
        let chars: Vec<char> = input.chars().collect();
        let mut out = String::with_capacity(input.len());
        let mut i = 0;
        while i < chars.len() {
            let mut matched_len = 0;
            for word in &self.match_words {
                let wlen = word.len();
                if wlen > 0 && i + wlen <= chars.len() && chars[i..i + wlen] == word[..] {
                    for &c in word {
                        out.push(c);
                    }
                    matched_len = wlen;
                    break;
                }
            }
            if matched_len > 0 {
                i += matched_len;
            } else {
                out.extend(chars[i].to_lowercase());
                i += 1;
            }
        }
        out
    }

    /// Language-specific character substitution. Only Romanian remaps
    /// anything (ț → ţ), matching `VitsTokenizer._preprocess_char`.
    fn preprocess_char(&self, text: &str) -> String {
        if self.language.as_deref() == Some("ron") {
            text.replace('ț', "ţ")
        } else {
            text.to_string()
        }
    }

    fn convert_char_to_id(&self, ch: char) -> Result<u32> {
        let token = ch.to_string();
        self.encoder
            .get(&token)
            .or_else(|| self.encoder.get(&self.unk_token))
            .copied()
            .ok_or_else(|| Error::ModelError {
                reason: format!(
                    "character {token:?} is not in the vocabulary and unk token {:?} is not \
                     either",
                    self.unk_token
                ),
            })
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.encoder.len()
    }

    /// Id of the pad token as read from `special_tokens_map.json` (or
    /// `tokenizer_config.json`, or the `<pad>` default), for callers that
    /// need to strip or locate blank frames outside `encode`.
    pub fn pad_token(&self) -> &str {
        &self.pad_token
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny synthetic vocabulary: `y`=0 (pad), space=1, a=2, b=3, c=4,
    /// `'`=5, `<unk>` absent from the vocabulary (added token).
    fn tiny() -> VitsTokenizer {
        let encoder: HashMap<String, u32> = [
            ("y".to_string(), 0),
            (" ".to_string(), 1),
            ("a".to_string(), 2),
            ("b".to_string(), 3),
            ("c".to_string(), 4),
            ("'".to_string(), 5),
        ]
        .into_iter()
        .collect();
        let pad_token = "y".to_string();
        let unk_token = "<unk>".to_string();
        let mut match_words: Vec<Vec<char>> = encoder.keys().map(|w| w.chars().collect()).collect();
        match_words.push(unk_token.chars().collect());
        match_words.sort_by_key(|w| std::cmp::Reverse(w.len()));
        let mut special_tokens: Vec<Vec<char>> =
            vec![pad_token.chars().collect(), unk_token.chars().collect()];
        special_tokens.sort_by_key(|w| std::cmp::Reverse(w.len()));
        VitsTokenizer {
            encoder,
            pad_token,
            unk_token,
            language: None,
            add_blank: true,
            normalize: true,
            blank_id: 0,
            match_words,
            special_tokens,
        }
    }

    #[test]
    fn add_blank_interspersion() {
        let tok = tiny();
        // "abc" -> chars a,b,c -> ids 2,3,4 with blank (0) around and between.
        let ids = tok.encode("abc").expect("encode");
        assert_eq!(ids, vec![0, 2, 0, 3, 0, 4, 0]);
    }

    #[test]
    fn empty_string_yields_no_ids() {
        let tok = tiny();
        let ids = tok.encode("").expect("encode");
        assert_eq!(ids, Vec::<u32>::new());
    }

    #[test]
    fn all_out_of_vocab_text_yields_no_ids() {
        let tok = tiny();
        // 'z' is not in the tiny vocab and whitespace-only strips to empty.
        let ids = tok.encode("zzz   ").expect("encode");
        assert_eq!(ids, Vec::<u32>::new());
    }

    #[test]
    fn no_blank_when_disabled() {
        let mut tok = tiny();
        tok.add_blank = false;
        let ids = tok.encode("abc").expect("encode");
        assert_eq!(ids, vec![2, 3, 4]);
    }

    #[test]
    fn lowercases_and_filters_out_of_vocab_chars() {
        let tok = tiny();
        // 'Z' is not in vocab even lower-cased, so it is dropped; case of
        // 'A'/'B' is folded down to the vocab's lower-case entries.
        let ids = tok.encode("AZb").expect("encode");
        assert_eq!(ids, vec![0, 2, 0, 3, 0]);
    }

    #[test]
    fn strips_leading_and_trailing_whitespace_after_filtering() {
        let tok = tiny();
        let ids = tok.encode("  a  ").expect("encode");
        assert_eq!(ids, vec![0, 2, 0]);
    }

    #[test]
    fn keeps_interior_repeated_spaces() {
        let tok = tiny();
        let ids = tok.encode("a  b").expect("encode");
        // a, space, space, b with blanks interspersed.
        assert_eq!(ids, vec![0, 2, 0, 1, 0, 1, 0, 3, 0]);
    }

    #[test]
    fn apostrophe_survives_normalization() {
        let tok = tiny();
        let ids = tok.encode("a'b").expect("encode");
        assert_eq!(ids, vec![0, 2, 0, 5, 0, 3, 0]);
    }

    #[test]
    fn pad_token_character_is_atomic_with_no_own_blank() {
        let tok = tiny();
        // "y" is both a normal vocab character and the pad token, so it
        // splits out as its own segment with no blank interspersed around
        // it: a single id 0, not [0, 0, 0].
        let ids = tok.encode("y").expect("encode");
        assert_eq!(ids, vec![0]);
    }

    #[test]
    fn pad_token_character_splits_surrounding_runs() {
        let tok = tiny();
        // "ay" -> regular run "a" (blank-interspersed on its own) followed
        // by the atomic pad-token segment "y".
        let ids = tok.encode("ay").expect("encode");
        assert_eq!(ids, vec![0, 2, 0, 0]);
    }

    #[test]
    fn repeated_pad_token_characters_have_no_blanks_between_them() {
        let tok = tiny();
        let ids = tok.encode("yy").expect("encode");
        assert_eq!(ids, vec![0, 0]);
    }

    #[test]
    fn pad_token_between_two_regular_runs() {
        let tok = tiny();
        // "ayb" -> regular "a", atomic "y", regular "b", each blank-run
        // independent of the others.
        let ids = tok.encode("ayb").expect("encode");
        assert_eq!(ids, vec![0, 2, 0, 0, 0, 3, 0]);
    }
}
