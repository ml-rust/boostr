//! Orthographic tokenizer: what the text says, character for character.

/// Normalize `text` into comparison tokens.
///
/// Rules, applied in order:
///
/// 1. Lowercase (`char::to_lowercase`, so it is Unicode-aware, not ASCII-only).
/// 2. Every character that is neither alphanumeric nor `'` becomes a space.
///    This strips `.,?!:;"` and turns `-` and `/` into token boundaries.
/// 3. Leading and trailing `'` are trimmed from each token, so `'word'`
///    normalizes to `word` while `don't` keeps its apostrophe.
/// 4. Split on whitespace; empty tokens are dropped.
///
/// Digits stay digits and number words stay words, so `250` and
/// `two hundred and fifty` are four errors apart. Use
/// [`crate::eval::canonicalize`] to score speech, where that difference is
/// not a pronunciation difference.
pub fn normalize(text: &str) -> Vec<String> {
    let spaced: String = text
        .chars()
        .flat_map(|c| {
            if c.is_alphanumeric() || c == '\'' {
                c.to_lowercase().collect::<Vec<char>>()
            } else {
                vec![' ']
            }
        })
        .collect();

    spaced
        .split_whitespace()
        .map(|t| t.trim_matches('\'').to_string())
        .filter(|t| !t.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::normalize;

    #[test]
    fn punctuation_and_hyphens_become_boundaries() {
        assert_eq!(normalize("Ter-delete, okay?"), ["ter", "delete", "okay"]);
    }

    #[test]
    fn inner_apostrophes_survive_and_outer_ones_do_not() {
        assert_eq!(normalize("'don't'"), ["don't"]);
    }

    #[test]
    fn digits_are_left_alone() {
        assert_eq!(normalize("RM23.50"), ["rm23", "50"]);
    }
}
