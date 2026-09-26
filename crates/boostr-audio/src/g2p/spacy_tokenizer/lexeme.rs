//! spaCy's lexeme attribute getters for English: `NORM`, `PREFIX`,
//! `SUFFIX`, `SHAPE` and `IS_SPACE`.

use std::collections::HashMap;

use super::hash::hash_string;
use crate::g2p::pystr::{is_letter, is_py_digit, is_py_space, is_upper_letter};

/// The lexeme `NORM` getter of `en_core_web_sm`: the `lexeme_norm` table,
/// then `BASE_NORMS`, then `str.lower`.
#[derive(Debug)]
pub(crate) struct Norms {
    lexeme_norm: HashMap<u64, String>,
    base_norms: HashMap<String, String>,
}

impl Norms {
    pub(crate) fn new(lexeme_norm: Vec<(u64, String)>, base_norms: Vec<(String, String)>) -> Self {
        Self {
            lexeme_norm: lexeme_norm.into_iter().collect(),
            base_norms: base_norms.into_iter().collect(),
        }
    }

    /// `Lexeme.norm_` of `orth`.
    pub(crate) fn norm(&self, orth: &str) -> String {
        if let Some(norm) = self.lexeme_norm.get(&hash_string(orth)) {
            return norm.clone();
        }
        if let Some(norm) = self.base_norms.get(orth) {
            return norm.clone();
        }
        orth.to_lowercase()
    }
}

/// `spacy.lang.lex_attrs.prefix`: `string[0]`.
pub(crate) fn prefix(orth: &str) -> String {
    orth.chars().take(1).collect()
}

/// `spacy.lang.lex_attrs.suffix`: `string[-3:]`.
pub(crate) fn suffix(orth: &str) -> String {
    match orth.char_indices().rev().nth(2) {
        Some((i, _)) => orth[i..].to_string(),
        None => orth.to_string(),
    }
}

/// `spacy.lang.lex_attrs.is_space`: `string.isspace()`.
pub(crate) fn is_space(orth: &str) -> bool {
    !orth.is_empty() && orth.chars().all(is_py_space)
}

/// `spacy.lang.lex_attrs.word_shape`.
///
/// Letters map to `X` or `x`, digits to `d`, anything else to itself; a run
/// of one shape character is cut after four. 100 or more characters give
/// `LONG`.
pub(crate) fn word_shape(orth: &str) -> String {
    if orth.chars().count() >= 100 {
        return "LONG".to_string();
    }
    let mut shape = String::new();
    let mut last: Option<char> = None;
    let mut seq = 0usize;
    for c in orth.chars() {
        let shape_char = if is_letter(c) {
            if is_upper_letter(c) { 'X' } else { 'x' }
        } else if is_py_digit(c) {
            'd'
        } else {
            c
        };
        if last == Some(shape_char) {
            seq += 1;
        } else {
            seq = 0;
            last = Some(shape_char);
        }
        if seq < 4 {
            shape.push(shape_char);
        }
    }
    shape
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes_match_spacy() {
        assert_eq!(word_shape("Apple"), "Xxxxx");
        assert_eq!(word_shape("U.S."), "X.X.");
        assert_eq!(word_shape("3.14159"), "d.dddd");
        assert_eq!(word_shape("$1.5M"), "$d.dX");
        assert_eq!(word_shape("xxxxxxxx"), "xxxx");
        assert_eq!(word_shape("Straße²"), "Xxxxxd");
        assert_eq!(word_shape(&"a".repeat(100)), "LONG");
    }

    #[test]
    fn affixes_count_characters() {
        assert_eq!(prefix("éa"), "é");
        assert_eq!(suffix("naïve"), "ïve");
        assert_eq!(suffix("ab"), "ab");
        assert!(is_space(" \n\u{a0}") && !is_space("") && !is_space(" a"));
    }

    #[test]
    fn norm_order_is_table_then_base_then_lower() {
        let norms = Norms::new(
            vec![(hash_string("colour"), "color".to_string())],
            vec![("’s".to_string(), "'s".to_string())],
        );
        assert_eq!(norms.norm("colour"), "color");
        assert_eq!(norms.norm("’s"), "'s");
        assert_eq!(norms.norm("Hello"), "hello");
    }
}
