//! misaki `G2P.fold_left` and `G2P.retokenize`: joins non-head tokens,
//! splits tokens into sub-tokens, assigns punctuation and currency, and
//! groups sub-tokens written without whitespace between them.

use super::chars::{PUNCTS, is_alpha};
use super::number::currency_units;
use super::subtokenize::subtokenize;
use super::token::{MToken, merge_tokens};

/// spaCy tags misaki treats as punctuation.
const PUNCT_TAGS: [&str; 11] = [
    ".", ",", "-LRB-", "-RRB-", "``", "\"\"", "''", ":", "$", "#", "NFP",
];

/// One retokenized word: a standalone token, or sub-tokens looked up as a
/// group.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Word {
    Single(MToken),
    Group(Vec<MToken>),
}

/// misaki `PUNCT_TAG_PHONEMES`.
fn punct_tag_phonemes(tag: &str) -> Option<&'static str> {
    match tag {
        "-LRB-" => Some("("),
        "-RRB-" => Some(")"),
        "``" => Some("\u{201C}"),
        "\"\"" | "''" => Some("\u{201D}"),
        _ => None,
    }
}

/// `all(97 <= ord(c.lower()) <= 122 for c in text)`. A character whose
/// lowercase is more than one character counts as outside the range.
fn all_ascii_letters(text: &str) -> bool {
    text.chars().all(|c| {
        let mut it = c.to_lowercase();
        match (it.next(), it.next()) {
            (Some(l), None) => l.is_ascii_lowercase(),
            _ => false,
        }
    })
}

/// misaki `G2P.fold_left`: merges each non-head token into the token before
/// it.
pub(crate) fn fold_left(tokens: Vec<MToken>, unk: &str) -> Vec<MToken> {
    let mut result: Vec<MToken> = Vec::with_capacity(tokens.len());
    for tk in tokens {
        let merged = if tk.is_head {
            None
        } else {
            result
                .pop()
                .and_then(|prev| merge_tokens(&[prev, tk.clone()], Some(unk)))
        };
        result.push(merged.unwrap_or(tk));
    }
    result
}

/// misaki `G2P.retokenize`.
pub(crate) fn retokenize(tokens: Vec<MToken>) -> Vec<Word> {
    let next_tags: Vec<Option<String>> = (0..tokens.len())
        .map(|i| tokens.get(i + 1).map(|t| t.tag.clone()))
        .collect();
    let mut words: Vec<Word> = Vec::new();
    let mut currency: Option<String> = None;

    for (i, token) in tokens.into_iter().enumerate() {
        let whitespace = token.whitespace.clone();
        let mut tks: Vec<MToken> = if token.alias.is_none() && token.phonemes.is_none() {
            subtokenize(&token.text)
                .into_iter()
                .map(|t| MToken {
                    text: t,
                    whitespace: String::new(),
                    is_head: true,
                    alias: None,
                    currency: None,
                    prespace: false,
                    rating: None,
                    ..token.clone()
                })
                .collect()
        } else {
            vec![token]
        };
        let Some(last) = tks.last_mut() else {
            continue;
        };
        last.whitespace = whitespace;

        let n = tks.len();
        for j in 0..n {
            let resolved = tks[j].alias.is_some() || tks[j].phonemes.is_some();
            if !resolved {
                let alias_to = j > 0 && j + 1 < n && tks[j].text == "2" && {
                    let prev = tks[j - 1].text.chars().next_back();
                    let next = tks[j + 1].text.chars().next();
                    match (prev, next) {
                        (Some(a), Some(b)) => is_alpha(&format!("{a}{b}")),
                        _ => false,
                    }
                };
                let tk = &mut tks[j];
                if tk.tag == "$" && currency_units(&tk.text).is_some() {
                    currency = Some(tk.text.clone());
                    tk.phonemes = Some(String::new());
                    tk.rating = Some(4);
                } else if tk.tag == ":" && (tk.text == "-" || tk.text == "–") {
                    tk.phonemes = Some("—".to_string());
                    tk.rating = Some(3);
                } else if PUNCT_TAGS.contains(&tk.tag.as_str()) && !all_ascii_letters(&tk.text) {
                    let ps = match punct_tag_phonemes(&tk.tag) {
                        Some(p) => p.to_string(),
                        None => tk.text.chars().filter(|&c| PUNCTS.contains(c)).collect(),
                    };
                    tk.phonemes = Some(ps);
                    tk.rating = Some(4);
                } else if currency.is_some() {
                    if tk.tag != "CD" {
                        currency = None;
                    } else if j + 1 == n && next_tags[i].as_deref() != Some("CD") {
                        tk.currency = currency.clone();
                    }
                } else if alias_to {
                    tk.alias = Some("to".to_string());
                }
            }

            let mut tk = tks[j].clone();
            if tk.alias.is_some() || tk.phonemes.is_some() {
                words.push(Word::Single(tk));
                continue;
            }
            let joins_group = matches!(
                words.last(),
                Some(Word::Group(g)) if g.last().is_some_and(|x| x.whitespace.is_empty())
            );
            if joins_group {
                tk.is_head = false;
                if let Some(Word::Group(group)) = words.last_mut() {
                    group.push(tk);
                }
            } else if tk.whitespace.is_empty() {
                words.push(Word::Group(vec![tk]));
            } else {
                words.push(Word::Single(tk));
            }
        }
    }

    words
        .into_iter()
        .map(|w| match w {
            Word::Group(mut g) if g.len() == 1 => match g.pop() {
                Some(t) => Word::Single(t),
                None => Word::Group(g),
            },
            other => other,
        })
        .collect()
}
