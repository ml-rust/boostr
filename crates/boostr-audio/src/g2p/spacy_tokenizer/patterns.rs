//! The tokenizer's five regular expressions, called the way `Tokenizer`
//! calls the Python ones.

use regexr::Regex;

use super::data::TokenizerData;
use crate::error::{Error, Result};

/// `prefix_search`, `suffix_search`, `infix_finditer`, `token_match` and
/// `url_match`. A missing pattern matches nothing, as `None` does in spaCy.
#[derive(Debug)]
pub(crate) struct Patterns {
    prefix: Option<Regex>,
    suffix: Option<Regex>,
    infix: Option<Regex>,
    token_match: Option<Regex>,
    url_match: Option<Regex>,
}

fn compile(name: &str, pattern: Option<&str>) -> Result<Option<Regex>> {
    pattern
        .map(|p| {
            Regex::new(p).map_err(|e| Error::ModelError {
                reason: format!("spaCy tokenizer pattern `{name}` does not compile: {e}"),
            })
        })
        .transpose()
}

/// Byte offset of the character `n` characters before the end of `s`.
fn tail_start(s: &str, n: usize) -> usize {
    if n == 0 {
        return s.len();
    }
    s.char_indices().rev().nth(n - 1).map_or(0, |(i, _)| i)
}

/// Byte offset just past the first `n` characters of `s`.
fn head_end(s: &str, n: usize) -> usize {
    s.char_indices().nth(n).map_or(s.len(), |(i, _)| i)
}

impl Patterns {
    pub(crate) fn new(data: &TokenizerData) -> Result<Self> {
        Ok(Self {
            prefix: compile("prefix", data.prefix.as_deref())?,
            suffix: compile("suffix", data.suffix.as_deref())?,
            infix: compile("infix", data.infix.as_deref())?,
            token_match: compile("token_match", data.token_match.as_deref())?,
            url_match: compile("url_match", data.url_match.as_deref())?,
        })
    }

    /// `Tokenizer.find_prefix`: the prefix length in characters, returned
    /// as the byte offset that ends `s[:n]`. 0 when nothing matches.
    pub(crate) fn find_prefix(&self, s: &str) -> usize {
        let Some(m) = self.prefix.as_ref().and_then(|re| re.find(s)) else {
            return 0;
        };
        head_end(s, m.as_str().chars().count())
    }

    /// `Tokenizer.find_suffix`: the suffix length in characters, returned
    /// as the byte offset that starts `s[-n:]`. `s.len()` when nothing
    /// matches.
    ///
    /// spaCy slices by the match length, not the match position: `$` also
    /// matches before a final newline, so the two can differ.
    pub(crate) fn find_suffix(&self, s: &str) -> usize {
        let Some(m) = self.suffix.as_ref().and_then(|re| re.find(s)) else {
            return s.len();
        };
        tail_start(s, m.as_str().chars().count())
    }

    /// `Tokenizer.find_infix`: byte ranges of every non-overlapping match.
    pub(crate) fn find_infix(&self, s: &str) -> Vec<(usize, usize)> {
        match &self.infix {
            Some(re) => re.find_iter(s).map(|m| (m.start(), m.end())).collect(),
            None => Vec::new(),
        }
    }

    /// `token_match(s)` is truthy.
    pub(crate) fn token_match(&self, s: &str) -> bool {
        self.token_match.as_ref().is_some_and(|re| re.is_match(s))
    }

    /// `url_match(s)` is truthy.
    pub(crate) fn url_match(&self, s: &str) -> bool {
        self.url_match.as_ref().is_some_and(|re| re.is_match(s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slices_count_characters() {
        assert_eq!(tail_start("aé€", 2), 1);
        assert_eq!(tail_start("aé", 0), 3);
        assert_eq!(tail_start("aé", 5), 0);
        assert_eq!(head_end("é€b", 2), 5);
        assert_eq!(head_end("é", 4), 2);
    }

    #[test]
    fn english_affixes() {
        let data = TokenizerData::en_core_web_sm()
            .unwrap_or_else(|e| panic!("embedded tokenizer data: {e}"));
        let p = Patterns::new(&data).unwrap_or_else(|e| panic!("patterns: {e}"));
        assert_eq!(p.find_prefix("(hello"), 1);
        assert_eq!(p.find_prefix("hello"), 0);
        assert_eq!(p.find_suffix("hello."), 5);
        assert_eq!(p.find_suffix("hello"), 5);
        assert_eq!(p.find_suffix("10km"), 2);
        assert_eq!(p.find_infix("well-known"), vec![(4, 5)]);
        assert!(p.url_match("https://example.com/a?b=c"));
        assert!(!p.url_match("hello"));
        assert!(!p.token_match("hello"));
    }
}
