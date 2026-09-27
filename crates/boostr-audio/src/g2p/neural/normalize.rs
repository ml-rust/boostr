//! Folds a word to the g2p_lm model's trained character set
//! (`[A-Za-z'-.]`), or reports it out of range.

use unicode_general_category::{GeneralCategory, get_general_category};
use unicode_normalization::UnicodeNormalization;

/// `true` for a combining mark general category (`Mn`, `Mc`, `Me`), the
/// categories Unicode NFKD peels off a precomposed letter.
fn is_combining_mark(c: char) -> bool {
    matches!(
        get_general_category(c),
        GeneralCategory::NonspacingMark
            | GeneralCategory::SpacingMark
            | GeneralCategory::EnclosingMark
    )
}

/// `true` for a character the g2p_lm vocab was trained on.
fn is_model_char(c: char) -> bool {
    c.is_ascii_alphabetic() || matches!(c, '\'' | '-' | '.')
}

/// NFKD-folds `word` (`café` -> `cafe`) and drops the combining marks the
/// decomposition exposes, then checks every remaining character sits in the
/// model's trained set.
///
/// Returns `None` when a character survives folding outside that set
/// (digits, other scripts) — nothing for the model to say, so the caller's
/// unk path applies, the same outcome misaki's own fallback reaches when it
/// has nothing to contribute.
pub fn fold_to_model_chars(word: &str) -> Option<String> {
    let folded: String = word.nfkd().filter(|c| !is_combining_mark(*c)).collect();
    if folded.chars().all(is_model_char) {
        Some(folded)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::fold_to_model_chars;

    #[test]
    fn strips_combining_marks_after_decomposition() {
        assert_eq!(fold_to_model_chars("café").as_deref(), Some("cafe"));
        assert_eq!(fold_to_model_chars("naïve").as_deref(), Some("naive"));
    }

    #[test]
    fn keeps_the_trained_punctuation() {
        assert_eq!(fold_to_model_chars("don't").as_deref(), Some("don't"));
        assert_eq!(
            fold_to_model_chars("Anglo-American").as_deref(),
            Some("Anglo-American")
        );
        assert_eq!(fold_to_model_chars("U.S.").as_deref(), Some("U.S."));
    }

    #[test]
    fn rejects_characters_outside_the_trained_set() {
        assert_eq!(fold_to_model_chars("2024"), None);
        assert_eq!(fold_to_model_chars("北京"), None);
        assert_eq!(fold_to_model_chars("naïve5"), None);
    }
}
