//! misaki's `MToken` and the token merge the pipeline applies to groups.

use super::chars::{char_is_lower_fixed, is_py_space};

/// One token, as misaki's `MToken` carries it.
///
/// The first three fields come from the tagger: the token text, its Penn
/// Treebank fine tag (spaCy `tag_`), and the whitespace after it. The rest is
/// misaki's `MToken._` namespace, flattened.
#[derive(Debug, Clone, PartialEq)]
pub struct MToken {
    /// Token text.
    pub text: String,
    /// Penn Treebank fine tag, e.g. `NN`, `VBD`, `$`, `CD`.
    pub tag: String,
    /// Whitespace that follows the token in the source text.
    pub whitespace: String,
    /// Phonemes. `None` means unresolved.
    pub phonemes: Option<String>,
    /// `false` joins the token to the one before it before lookup.
    pub is_head: bool,
    /// Text to look up in place of `text`.
    pub alias: Option<String>,
    /// Stress override passed to `apply_stress`.
    pub stress: Option<f64>,
    /// Currency symbol that precedes this number.
    pub currency: Option<String>,
    /// Number-reading flags: `a` reads a leading "one" as "a", `&` keeps
    /// "and", `n` appends "and" to the previous word.
    pub num_flags: String,
    /// `true` puts a space before this token's phonemes when merged.
    pub prespace: bool,
    /// misaki `MToken._.rating`: 5 user-supplied, 4 gold or rule, 3 silver
    /// or derived. misaki leaves it unset on a standalone token the lexicon
    /// resolves, and the port does the same.
    pub rating: Option<u8>,
}

impl MToken {
    /// A token as misaki builds it from one spaCy token: head, no flags.
    pub fn new(
        text: impl Into<String>,
        tag: impl Into<String>,
        whitespace: impl Into<String>,
    ) -> Self {
        Self {
            text: text.into(),
            tag: tag.into(),
            whitespace: whitespace.into(),
            phonemes: None,
            is_head: true,
            alias: None,
            stress: None,
            currency: None,
            num_flags: String::new(),
            prespace: false,
            rating: None,
        }
    }
}

/// Right-hand context misaki threads backwards through the sentence.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct TokenContext {
    /// Whether the next resolved word starts with a vowel. `None` when the
    /// next phoneme is punctuation or no word follows.
    pub future_vowel: Option<bool>,
    /// Whether the next word is "to".
    pub future_to: bool,
}

/// `sum(1 if c == c.lower() else 2 for c in text)`.
fn case_weight(text: &str) -> usize {
    text.chars()
        .map(|c| if char_is_lower_fixed(c) { 1 } else { 2 })
        .sum()
}

/// misaki `merge_tokens(tokens, unk)`. `tokens` is non-empty; an empty slice
/// returns `None`.
pub(crate) fn merge_tokens(tokens: &[MToken], unk: Option<&str>) -> Option<MToken> {
    let first = tokens.first()?;
    let last = tokens.last()?;

    let mut stresses: Vec<f64> = Vec::new();
    for s in tokens.iter().filter_map(|tk| tk.stress) {
        if !stresses.contains(&s) {
            stresses.push(s);
        }
    }
    let currency = tokens.iter().filter_map(|tk| tk.currency.clone()).max();
    let rating = if tokens.iter().any(|tk| tk.rating.is_none()) {
        None
    } else {
        tokens.iter().filter_map(|tk| tk.rating).min()
    };

    let phonemes = unk.map(|unk| {
        let mut out = String::new();
        for tk in tokens {
            let ends_in_space = out.chars().next_back().is_some_and(is_py_space);
            if tk.prespace
                && !out.is_empty()
                && !ends_in_space
                && tk.phonemes.as_deref().is_some_and(|p| !p.is_empty())
            {
                out.push(' ');
            }
            out.push_str(tk.phonemes.as_deref().unwrap_or(unk));
        }
        out
    });

    let mut text = String::new();
    for tk in &tokens[..tokens.len() - 1] {
        text.push_str(&tk.text);
        text.push_str(&tk.whitespace);
    }
    text.push_str(&last.text);

    // Python `max` keeps the first of equal keys.
    let mut tag_source = first;
    let mut best = case_weight(&first.text);
    for tk in &tokens[1..] {
        let w = case_weight(&tk.text);
        if w > best {
            best = w;
            tag_source = tk;
        }
    }

    let mut flags: Vec<char> = tokens.iter().flat_map(|tk| tk.num_flags.chars()).collect();
    flags.sort_unstable();
    flags.dedup();

    Some(MToken {
        text,
        tag: tag_source.tag.clone(),
        whitespace: last.whitespace.clone(),
        phonemes,
        is_head: first.is_head,
        alias: None,
        stress: if stresses.len() == 1 {
            Some(stresses[0])
        } else {
            None
        },
        currency,
        num_flags: flags.into_iter().collect(),
        prespace: first.prespace,
        rating,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok(text: &str, ws: &str, ps: Option<&str>) -> MToken {
        let mut t = MToken::new(text, "NN", ws);
        t.phonemes = ps.map(str::to_string);
        t
    }

    #[test]
    fn merge_joins_text_and_phonemes() {
        let mut b = tok("B", " ", Some("bˈi"));
        b.tag = "NNP".into();
        b.prespace = true;
        b.rating = Some(3);
        let mut a = tok("a", "", Some("ə"));
        a.rating = Some(4);
        let m = merge_tokens(&[a, b], Some("❓")).unwrap_or_else(|| MToken::new("", "", ""));
        assert_eq!(m.text, "aB");
        assert_eq!(m.whitespace, " ");
        assert_eq!(m.phonemes.as_deref(), Some("ə bˈi"));
        assert_eq!(m.tag, "NNP");
        assert_eq!(m.rating, Some(3));
    }

    #[test]
    fn merge_uses_unk_for_unresolved() {
        let m = merge_tokens(&[tok("x", "", None), tok("y", "", Some("wˈI"))], Some("?"))
            .unwrap_or_else(|| MToken::new("", "", ""));
        assert_eq!(m.phonemes.as_deref(), Some("?wˈI"));
        assert_eq!(m.rating, None);
        let none = merge_tokens(&[tok("x", "", None)], None);
        assert_eq!(none.and_then(|m| m.phonemes), None);
    }
}
