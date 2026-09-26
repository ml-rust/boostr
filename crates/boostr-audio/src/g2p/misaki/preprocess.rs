//! misaki `G2P.preprocess` (markdown-link features) and the feature pass of
//! `G2P.tokenize`, with spaCy's `Alignment.from_strings`.
//!
//! A link `[text](feature)` keeps `text` in the sentence and attaches
//! `feature` to it: an integer or `±0.5` stress, `/phonemes/`, or
//! `#flags#`. `tokenize` finds the spaCy tokens of the link through a
//! character alignment of the whitespace-split words against spaCy's tokens.

use super::chars::is_py_space;
use super::token::MToken;
use crate::error::{Error, Result};

/// One link feature, as misaki's `preprocess` converts it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Feature {
    /// An integer or `±0.5`: `MToken._.stress`.
    Stress(f64),
    /// `/phonemes/`, stored as `/` + the text with trailing `/` removed.
    Phonemes(String),
    /// `#flags#`, stored as `#` + the text with trailing `#` removed.
    NumFlags(String),
}

/// Output of [`preprocess`].
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Preprocessed {
    /// The text with every link replaced by its link text.
    pub text: String,
    /// Whitespace-split words, one per link text.
    pub words: Vec<String>,
    /// Feature per index into `words`, ascending.
    pub features: Vec<(usize, Feature)>,
}

/// misaki `is_digit`: `re.match(r'^[0-9]+$', text)`, where `$` also
/// matches before one final newline.
fn is_digit(text: &str) -> bool {
    let body = text.strip_suffix('\n').unwrap_or(text);
    !body.is_empty() && body.bytes().all(|b| b.is_ascii_digit())
}

/// misaki's conversion of a link target; `None` when it carries no feature.
fn feature(f: &str) -> Option<Feature> {
    let unsigned = f.strip_prefix(['-', '+']).unwrap_or(f);
    if is_digit(unsigned) {
        // `int(f)`: sign, digits, and the newline `int` ignores.
        let digits: f64 = unsigned.trim_end_matches('\n').parse().ok()?;
        let negative = f.starts_with('-') && digits != 0.0;
        return Some(Feature::Stress(if negative { -digits } else { digits }));
    }
    match f {
        "0.5" | "+0.5" => return Some(Feature::Stress(0.5)),
        "-0.5" => return Some(Feature::Stress(-0.5)),
        _ => {}
    }
    let edged = |c: char| f.chars().count() > 1 && f.starts_with(c) && f.ends_with(c);
    if edged('/') {
        Some(Feature::Phonemes(format!(
            "/{}",
            f[1..].trim_end_matches('/')
        )))
    } else if edged('#') {
        Some(Feature::NumFlags(format!(
            "#{}",
            f[1..].trim_end_matches('#')
        )))
    } else {
        None
    }
}

/// The next `\[([^\]]+)\]\(([^\)]*)\)` match at or after byte `from`:
/// `(start, end, link text, target)`.
fn next_link(text: &str, from: usize) -> Option<(usize, usize, &str, &str)> {
    let mut search = from;
    while let Some(rel) = text[search..].find('[') {
        let open = search + rel;
        let inner = open + 1;
        let close = text[inner..].find(']').map(|i| inner + i);
        if let Some(close) = close.filter(|&c| c > inner)
            && text[close + 1..].starts_with('(')
        {
            let target = close + 2;
            if let Some(end) = text[target..].find(')').map(|i| target + i) {
                return Some((open, end + 1, &text[inner..close], &text[target..end]));
            }
        }
        search = inner;
    }
    None
}

/// Python `str.split()`.
fn py_split(s: &str) -> impl Iterator<Item = &str> {
    s.split(is_py_space).filter(|w| !w.is_empty())
}

/// misaki `G2P.preprocess(text)`.
pub(crate) fn preprocess(text: &str) -> Preprocessed {
    let text = text.trim_start_matches(is_py_space);
    let mut out = Preprocessed {
        text: String::with_capacity(text.len()),
        words: Vec::new(),
        features: Vec::new(),
    };
    let mut last_end = 0;
    while let Some((start, end, link, target)) = next_link(text, last_end) {
        out.text.push_str(&text[last_end..start]);
        out.words
            .extend(py_split(&text[last_end..start]).map(str::to_string));
        if let Some(f) = feature(target) {
            out.features.push((out.words.len(), f));
        }
        out.text.push_str(link);
        out.words.push(link.to_string());
        last_end = end;
    }
    if last_end < text.len() {
        out.text.push_str(&text[last_end..]);
        out.words
            .extend(py_split(&text[last_end..]).map(str::to_string));
    }
    out
}

fn misaligned(a: &[String], b: &[String]) -> Error {
    let head = |v: &[String]| v.iter().take(10).cloned().collect::<Vec<_>>();
    Error::DataError {
        reason: format!(
            "misaki cannot align words {:?} with spaCy tokens {:?}",
            head(a),
            head(b)
        ),
    }
}

/// spaCy `get_alignments(A, B)[1]` (`y2x`) flattened: the sorted `A`
/// indices of each `B` token, concatenated, as `Ragged.data` holds them.
pub(crate) fn y2x_data(a: &[String], b: &[String]) -> Result<Vec<usize>> {
    let owners = |v: &[String]| -> Vec<usize> {
        v.iter()
            .enumerate()
            .flat_map(|(i, x)| std::iter::repeat_n(i, x.to_lowercase().chars().count()))
            .collect()
    };
    let (char_a, char_b) = (owners(a), owners(b));
    let str_a: Vec<char> = a.concat().to_lowercase().chars().collect();
    let str_b: Vec<char> = b.concat().to_lowercase().chars().collect();
    let squeeze = |s: &[char]| s.iter().filter(|c| !is_py_space(**c)).collect::<String>();
    if squeeze(&str_a) != squeeze(&str_b)
        || str_a.len() != char_a.len()
        || str_b.len() != char_b.len()
    {
        return Err(misaligned(a, b));
    }
    let a_chars: Vec<usize> = a.iter().map(|x| x.chars().count()).collect();
    let b_chars: Vec<usize> = b.iter().map(|x| x.chars().count()).collect();
    let (mut ia, mut ib) = (0usize, 0usize);
    let mut prev_b = usize::MAX;
    let mut b2a: Vec<Vec<usize>> = Vec::new();
    while ia < str_a.len() && ib < str_b.len() {
        let (ta, tb) = (char_a[ia], char_b[ib]);
        if prev_b != tb {
            b2a.push(Vec::new());
        }
        let starts_a = ia == 0 || char_a[ia - 1] < ta;
        let starts_b = ib == 0 || char_b[ib - 1] < tb;
        let current = b2a.last_mut().ok_or_else(|| misaligned(a, b))?;
        if a[ta] == b[tb] && starts_a && starts_b {
            current.push(ta);
            ia += a_chars[ta];
            ib += b_chars[tb];
        } else if str_a[ia] == str_b[ib] {
            current.push(ta);
            ia += 1;
            ib += 1;
        } else if is_py_space(str_a[ia]) {
            ia += 1;
        } else if is_py_space(str_b[ib]) {
            ib += 1;
        } else {
            return Err(misaligned(a, b));
        }
        prev_b = tb;
    }
    Ok(b2a
        .into_iter()
        .flat_map(|mut xs| {
            xs.sort_unstable();
            xs.dedup();
            xs
        })
        .collect())
}

/// The feature pass of misaki `G2P.tokenize`: applies `features` to the
/// spaCy tokens `tokens` through the alignment with `words`.
///
/// Like misaki, it looks up each feature's word index in the flattened
/// `y2x` data and uses the position found there as the token index.
pub(crate) fn apply_features(
    tokens: &mut [MToken],
    words: &[String],
    features: &[(usize, Feature)],
) -> Result<()> {
    if features.is_empty() {
        return Ok(());
    }
    let texts: Vec<String> = tokens.iter().map(|t| t.text.clone()).collect();
    let data = y2x_data(words, &texts)?;
    for (k, f) in features {
        let hits = data
            .iter()
            .enumerate()
            .filter(|(_, x)| *x == k)
            .map(|(j, _)| j);
        for (i, j) in hits.enumerate() {
            let Some(tk) = tokens.get_mut(j) else {
                continue;
            };
            match f {
                Feature::Stress(s) => tk.stress = Some(*s),
                Feature::Phonemes(p) => {
                    tk.is_head = i == 0;
                    tk.phonemes = Some(if i == 0 {
                        p.trim_start_matches('/').to_string()
                    } else {
                        String::new()
                    });
                    tk.rating = Some(5);
                }
                Feature::NumFlags(n) => tk.num_flags = n.trim_start_matches('#').to_string(),
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    // Expected values from misaki 0.9.4 `G2P.preprocess`.
    #[test]
    fn preprocess_extracts_links() {
        let p = preprocess("  [Kokoro](/kˈOkəɹO/) is great.");
        assert_eq!(p.text, "Kokoro is great.");
        assert_eq!(p.words, words(&["Kokoro", "is", "great."]));
        assert_eq!(p.features, [(0, Feature::Phonemes("/kˈOkəɹO".into()))]);

        let p = preprocess(
            "[Hello world](/həlˈO/) and [read](-1) [it](+2) [a](0.5) [b](-0.5) [c](#&#) [d](x)",
        );
        assert_eq!(p.text, "Hello world and read it a b c d");
        assert_eq!(
            p.words,
            words(&["Hello world", "and", "read", "it", "a", "b", "c", "d"])
        );
        assert_eq!(
            p.features,
            [
                (0, Feature::Phonemes("/həlˈO".into())),
                (2, Feature::Stress(-1.0)),
                (3, Feature::Stress(2.0)),
                (4, Feature::Stress(0.5)),
                (5, Feature::Stress(-0.5)),
                (6, Feature::NumFlags("#&".into())),
            ]
        );

        let p = preprocess("x [a]() y [5](//ab//) [](z) [[q]](1)");
        assert_eq!(p.text, "x a y 5 [](z) [[q]](1)");
        assert_eq!(p.words, words(&["x", "a", "y", "5", "[](z)", "[[q]](1)"]));
        assert_eq!(p.features, [(3, Feature::Phonemes("//ab".into()))]);

        let p = preprocess("[a](-0) [b](3\n) [c\nd](/x/)");
        assert_eq!(p.text, "a b c\nd");
        assert_eq!(p.words, words(&["a", "b", "c\nd"]));
        assert_eq!(
            p.features,
            [
                (0, Feature::Stress(0.0)),
                (1, Feature::Stress(3.0)),
                (2, Feature::Phonemes("/x".into()))
            ]
        );
    }

    #[test]
    fn preprocess_without_links_is_lstrip() {
        let p = preprocess(" \t Plain text here. ");
        assert_eq!(p.text, "Plain text here. ");
        assert!(p.features.is_empty());
    }

    #[test]
    fn digit_targets_follow_python_int() {
        assert_eq!(feature("007"), Some(Feature::Stress(7.0)));
        assert_eq!(feature("3\n"), Some(Feature::Stress(3.0)));
        assert_eq!(feature("-"), None);
        assert_eq!(feature("1.5"), None);
        assert_eq!(feature("/"), None);
    }

    #[test]
    fn alignment_matches_spacy() {
        // `spacy.training.Alignment.from_strings(A, B).y2x.data`.
        let a = words(&["Say", "don't", "now"]);
        let b = words(&["Say", "do", "n't", "now"]);
        assert_eq!(y2x_data(&a, &b).ok(), Some(vec![0, 1, 1, 2]));
        let a = words(&["New York", "rocks"]);
        let b = words(&["New", "York", "rocks"]);
        assert_eq!(y2x_data(&a, &b).ok(), Some(vec![0, 0, 1]));
        let a = words(&["a", "b"]);
        let b = words(&["a", "\n\n", "b"]);
        assert_eq!(y2x_data(&a, &b).ok(), Some(vec![0, 1]));
        let b = words(&["a", "c"]);
        assert!(y2x_data(&a, &b).is_err());
    }

    #[test]
    fn features_reach_the_link_tokens() {
        let mut tokens = vec![
            MToken::new("New", "NNP", " "),
            MToken::new("York", "NNP", " "),
            MToken::new("rocks", "NNS", ""),
        ];
        let p = preprocess("[New York](/nuː jɔːk/) rocks");
        apply_features(&mut tokens, &p.words, &p.features)
            .unwrap_or_else(|e| panic!("apply_features: {e}"));
        assert_eq!(tokens[0].phonemes.as_deref(), Some("nuː jɔːk"));
        assert!(tokens[0].is_head);
        assert_eq!(tokens[1].phonemes.as_deref(), Some(""));
        assert!(!tokens[1].is_head);
        assert_eq!((tokens[0].rating, tokens[1].rating), (Some(5), Some(5)));
        assert_eq!(tokens[2].phonemes, None);
    }
}
