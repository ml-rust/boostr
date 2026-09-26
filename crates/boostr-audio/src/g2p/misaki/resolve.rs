//! misaki `G2P.token_context` and `G2P.resolve_tokens`: the right-hand
//! context carried between words, and spacing and stress within a group.

use super::chars::{
    CONSONANTS, NON_QUOTE_PUNCTS, PRIMARY_STRESS, SUBTOKEN_JUNKS, char_len, is_ascii_digit,
    is_letter, is_vowel,
};
use super::stress::{apply_stress, stress_weight};
use super::token::{MToken, TokenContext};

/// misaki `token_context`: the context the word before `token` sees.
pub(crate) fn token_context(ctx: TokenContext, ps: Option<&str>, token: &MToken) -> TokenContext {
    let mut vowel = ctx.future_vowel;
    if let Some(ps) = ps.filter(|p| !p.is_empty()) {
        let first = ps
            .chars()
            .find(|&c| is_vowel(c) || CONSONANTS.contains(c) || NON_QUOTE_PUNCTS.contains(c));
        if let Some(c) = first {
            vowel = if NON_QUOTE_PUNCTS.contains(c) {
                None
            } else {
                Some(is_vowel(c))
            };
        }
    }
    let future_to = matches!(token.text.as_str(), "to" | "To")
        || (token.text == "TO" && matches!(token.tag.as_str(), "TO" | "IN"));
    TokenContext {
        future_vowel: vowel,
        future_to,
    }
}

fn is_junk(text: &str) -> bool {
    text.chars().all(|c| SUBTOKEN_JUNKS.contains(c))
}

/// Whether `text` is one character of `NON_QUOTE_PUNCTS`.
fn is_single_punct(text: &str) -> bool {
    let mut it = text.chars();
    matches!((it.next(), it.next()), (Some(c), None) if NON_QUOTE_PUNCTS.contains(c))
}

/// misaki `resolve_tokens`: fills silent and punctuation sub-tokens, marks
/// which sub-tokens take a leading space, and demotes stress in
/// multi-part words read without spaces.
pub(crate) fn resolve_tokens(tokens: &mut [MToken]) {
    let Some(last) = tokens.len().checked_sub(1) else {
        return;
    };
    let mut text = String::new();
    for tk in &tokens[..last] {
        text.push_str(&tk.text);
        text.push_str(&tk.whitespace);
    }
    text.push_str(&tokens[last].text);

    let mut classes = [false; 3];
    for c in text.chars().filter(|&c| !SUBTOKEN_JUNKS.contains(c)) {
        let class = if is_letter(c) {
            0
        } else if is_ascii_digit(c) {
            1
        } else {
            2
        };
        classes[class] = true;
    }
    let prespace =
        text.contains(' ') || text.contains('/') || classes.iter().filter(|&&b| b).count() > 1;

    for (i, tk) in tokens.iter_mut().enumerate() {
        if tk.phonemes.is_none() {
            if i == last && is_single_punct(&tk.text) {
                tk.phonemes = Some(tk.text.clone());
                tk.rating = Some(3);
            } else if is_junk(&tk.text) {
                tk.phonemes = Some(String::new());
                tk.rating = Some(3);
            }
        } else if i > 0 {
            tk.prespace = prespace;
        }
    }
    if prespace {
        return;
    }

    let mut indices: Vec<(bool, usize, usize)> = tokens
        .iter()
        .enumerate()
        .filter_map(|(i, tk)| {
            let ps = tk.phonemes.as_deref().filter(|p| !p.is_empty())?;
            Some((ps.contains(PRIMARY_STRESS), stress_weight(ps), i))
        })
        .collect();
    if indices.len() == 2 && char_len(&tokens[indices[0].2].text) == 1 {
        let i = indices[1].2;
        demote(&mut tokens[i]);
        return;
    }
    let primaries = indices.iter().filter(|(b, _, _)| *b).count();
    if indices.len() < 2 || primaries <= indices.len().div_ceil(2) {
        return;
    }
    indices.sort_unstable();
    let half = indices.len() / 2;
    for &(_, _, i) in &indices[..half] {
        demote(&mut tokens[i]);
    }
}

fn demote(tk: &mut MToken) {
    if let Some(ps) = tk.phonemes.as_deref() {
        tk.phonemes = Some(apply_stress(ps, Some(-0.5)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sub(text: &str, ps: Option<&str>) -> MToken {
        let mut t = MToken::new(text, "NNP", "");
        t.phonemes = ps.map(str::to_string);
        t
    }

    #[test]
    fn context_reads_first_phoneme() {
        let ctx = TokenContext::default();
        let tk = MToken::new("apple", "NN", " ");
        assert_eq!(
            token_context(ctx, Some("ˈæpəl"), &tk).future_vowel,
            Some(true)
        );
        assert_eq!(
            token_context(ctx, Some("bˈæt"), &tk).future_vowel,
            Some(false)
        );
        assert_eq!(token_context(ctx, Some(","), &tk).future_vowel, None);
        let to = MToken::new("to", "TO", " ");
        assert!(token_context(ctx, Some("tə"), &to).future_to);
    }

    #[test]
    fn mixed_classes_take_prespace() {
        let mut toks = vec![sub("v", Some("vˈi")), sub("2", Some("tˈu"))];
        resolve_tokens(&mut toks);
        assert!(toks[1].prespace);
    }

    #[test]
    fn letter_pair_demotes_second_stress() {
        let mut toks = vec![sub("i", Some("ˈI")), sub("Phone", Some("fˈOn"))];
        resolve_tokens(&mut toks);
        assert_eq!(toks[1].phonemes.as_deref(), Some("fˌOn"));
        assert!(!toks[1].prespace);
    }
}
