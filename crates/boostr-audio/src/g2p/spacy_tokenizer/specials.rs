//! Special cases (`Tokenizer.rules`) and the phrase-matcher pass
//! `Tokenizer._apply_special_cases` runs after affix splitting.

use std::collections::{BTreeSet, HashMap};

use super::data::SpecialPiece;

/// A token while the tokenizer builds a doc: text, the `NORM` a special
/// case sets, and the `SPACY` flag.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RawToken {
    pub text: String,
    pub norm: Option<String>,
    pub spacy: bool,
}

impl RawToken {
    /// A token from a lexeme: no `NORM` override, no trailing space.
    pub(crate) fn lex(text: &str) -> Self {
        Self {
            text: text.to_string(),
            norm: None,
            spacy: false,
        }
    }
}

/// `Tokenizer._specials`: chunk to replacement tokens.
#[derive(Debug, Default)]
pub(crate) struct Specials {
    map: HashMap<String, Vec<SpecialPiece>>,
}

impl Specials {
    pub(crate) fn new(rules: &[(String, Vec<SpecialPiece>)]) -> Self {
        Self {
            map: rules.iter().cloned().collect(),
        }
    }

    pub(crate) fn contains(&self, chunk: &str) -> bool {
        self.map.contains_key(chunk)
    }

    /// Pushes the tokens of special case `chunk`, each with `SPACY` unset.
    /// Returns `false` when `chunk` is no special case.
    pub(crate) fn push(&self, chunk: &str, doc: &mut Vec<RawToken>) -> bool {
        let Some(pieces) = self.map.get(chunk) else {
            return false;
        };
        doc.extend(pieces.iter().map(|(orth, norm)| RawToken {
            text: orth.clone(),
            norm: norm.clone(),
            spacy: false,
        }));
        true
    }
}

/// `Tokenizer._special_matcher`: a `PhraseMatcher` on `ORTH` holding, for
/// each special case that passes the `faster_heuristics` check, the tokens
/// the chunk splits into without special cases.
#[derive(Debug, Default)]
pub(crate) struct SpecialMatcher {
    /// First token to the patterns that start with it.
    by_first: HashMap<String, Vec<Vec<String>>>,
}

impl SpecialMatcher {
    pub(crate) fn add(&mut self, pattern: Vec<String>) {
        if let Some(first) = pattern.first() {
            self.by_first
                .entry(first.clone())
                .or_default()
                .push(pattern);
        }
    }

    /// `PhraseMatcher.find_matches`: every `(start, end)` whose token texts
    /// equal a pattern, once per matching pattern.
    fn find_matches(&self, doc: &[RawToken]) -> Vec<(usize, usize)> {
        let mut matches = Vec::new();
        for start in 0..doc.len() {
            let Some(patterns) = self.by_first.get(&doc[start].text) else {
                continue;
            };
            for pattern in patterns {
                let end = start + pattern.len();
                if end <= doc.len()
                    && doc[start..end]
                        .iter()
                        .zip(pattern)
                        .all(|(t, p)| t.text == *p)
                {
                    matches.push((start, end));
                }
            }
        }
        matches
    }

    /// `Tokenizer._apply_special_cases`: replaces each surviving match whose
    /// text is a special case with that case's tokens.
    pub(crate) fn apply(&self, doc: Vec<RawToken>, specials: &Specials) -> Vec<RawToken> {
        let matches = self.find_matches(&doc);
        if matches.is_empty() {
            return doc;
        }
        let spans = filter_spans(matches);

        let mut out = Vec::with_capacity(doc.len());
        let mut span_iter = spans.into_iter().peekable();
        let mut i = 0;
        while i < doc.len() {
            let Some(&(start, end)) = span_iter.peek().filter(|(s, _)| *s == i) else {
                out.push(doc[i].clone());
                i += 1;
                continue;
            };
            span_iter.next();
            let text = span_text(&doc[start..end]);
            let before = out.len();
            if specials.push(&text, &mut out) {
                if let Some(last) = out[before..].last_mut() {
                    last.spacy = doc[end - 1].spacy;
                }
            } else {
                out.extend_from_slice(&doc[start..end]);
            }
            i = end;
        }
        out
    }
}

/// `Span.text`: the tokens with the spaces between them, without the
/// trailing one.
fn span_text(tokens: &[RawToken]) -> String {
    let mut text = String::new();
    for (k, t) in tokens.iter().enumerate() {
        text.push_str(&t.text);
        if t.spacy && k + 1 < tokens.len() {
            text.push(' ');
        }
    }
    text
}

/// `Tokenizer._filter_special_spans`: longest first, then leftmost; a span
/// survives when neither its first nor its last token is taken, and every
/// span visited marks its tokens taken whether it survives or not. The
/// survivors come back sorted by start.
fn filter_spans(mut spans: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    spans.sort_by(|a, b| {
        let (la, lb) = (a.1 - a.0, b.1 - b.0);
        la.cmp(&lb).then(b.0.cmp(&a.0))
    });
    let mut seen = BTreeSet::new();
    let mut kept = Vec::new();
    for &(start, end) in spans.iter().rev() {
        if !seen.contains(&start) && !seen.contains(&(end - 1)) {
            kept.push((start, end));
        }
        seen.extend(start..end);
    }
    kept.sort_by_key(|&(start, _)| start);
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(texts: &[(&str, bool)]) -> Vec<RawToken> {
        texts
            .iter()
            .map(|&(t, spacy)| RawToken {
                text: t.to_string(),
                norm: None,
                spacy,
            })
            .collect()
    }

    #[test]
    fn filter_prefers_longest_then_leftmost() {
        assert_eq!(filter_spans(vec![(0, 2), (1, 3), (0, 1)]), vec![(0, 2)]);
        assert_eq!(
            filter_spans(vec![(2, 3), (0, 1), (1, 2)]),
            vec![(0, 1), (1, 2), (2, 3)]
        );
        // A rejected span still marks its tokens: (1, 4) loses to (0, 3)
        // and then blocks (3, 4).
        assert_eq!(filter_spans(vec![(0, 3), (1, 4), (3, 4)]), vec![(0, 3)]);
    }

    #[test]
    fn replaces_only_exact_text() {
        let specials = Specials::new(&[(":)".to_string(), vec![(":)".to_string(), None)])]);
        let mut m = SpecialMatcher::default();
        m.add(vec![":".to_string(), ")".to_string()]);

        let joined = m.apply(toks(&[("a", true), (":", false), (")", true)]), &specials);
        assert_eq!(joined, toks(&[("a", true), (":)", true)]));

        let spaced = toks(&[(":", true), (")", false)]);
        assert_eq!(m.apply(spaced.clone(), &specials), spaced);
    }
}
