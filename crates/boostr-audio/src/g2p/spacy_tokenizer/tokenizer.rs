//! `spacy.tokenizer.Tokenizer.__call__` with the `en_core_web_sm` 3.8.0
//! settings.

use std::collections::HashMap;

use super::data::TokenizerData;
use super::lexeme::{Norms, is_space, prefix, suffix, word_shape};
use super::patterns::Patterns;
use super::specials::{RawToken, SpecialMatcher, Specials};
use super::token::SpacyToken;
use crate::error::Result;
use crate::g2p::pystr::is_py_space;

/// Chunk to tokens, as spaCy's tokenizer cache holds them.
type Cache = HashMap<String, Vec<RawToken>>;

/// Port of the spaCy tokenizer of `en_core_web_sm` 3.8.0 (spaCy 3.8.x).
///
/// Splits on whitespace, then per chunk applies special cases, prefix,
/// suffix and infix splitting and the URL match, then the phrase-matcher
/// pass for special cases that contain affixes. Returns spaCy's tokens with
/// the lexical attributes the tagger embeds.
#[derive(Debug)]
pub struct SpacyTokenizer {
    patterns: Patterns,
    specials: Specials,
    matcher: SpecialMatcher,
    norms: Norms,
    source: String,
    /// The cache entries spaCy's tokenizer holds after loading: the chunks
    /// of the last special case added to the phrase matcher, split without
    /// special cases. spaCy never flushes them, and it consults the cache
    /// before splitting a chunk, so they take part in tokenization.
    load_cache: Cache,
}

impl SpacyTokenizer {
    /// The tokenizer of spaCy's `en_core_web_sm` 3.8.0, from the embedded
    /// data.
    pub fn en_core_web_sm() -> Result<Self> {
        Self::from_data(TokenizerData::en_core_web_sm()?)
    }

    /// `Tokenizer.from_bytes`: sets the patterns, then loads the rules.
    fn from_data(data: TokenizerData) -> Result<Self> {
        let mut tok = Self {
            patterns: Patterns::new(&data)?,
            specials: Specials::new(&data.rules),
            matcher: SpecialMatcher::default(),
            norms: Norms::new(data.lexeme_norm, data.base_norms),
            source: data.source,
            load_cache: Cache::new(),
        };
        // `_load_special_cases` adds the rules in sorted order; each
        // `add_special_case` flushes the cache and then, when the chunk
        // passes the heuristic, tokenizes it without special cases, which
        // fills the cache again.
        let mut rules: Vec<&str> = data.rules.iter().map(|(chunk, _)| chunk.as_str()).collect();
        rules.sort_unstable();
        let mut cache = Cache::new();
        for chunk in rules {
            cache.clear();
            let p = &tok.patterns;
            if data.faster_heuristics
                && p.find_prefix(chunk) == 0
                && p.find_infix(chunk).is_empty()
                && p.find_suffix(chunk) == chunk.len()
                && !chunk.contains(' ')
            {
                continue;
            }
            let pattern = tok.tokenize_affixes(chunk, false, Some(&mut cache));
            tok.matcher
                .add(pattern.into_iter().map(|t| t.text).collect());
        }
        tok.load_cache = cache;
        Ok(tok)
    }

    /// The spaCy and pipeline versions the embedded data came from.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// `Tokenizer.__call__`: the tokens of `text` with their lexical
    /// attributes.
    pub fn tokenize(&self, text: &str) -> Vec<SpacyToken> {
        let doc = self.tokenize_affixes(text, true, None);
        let doc = self.matcher.apply(doc, &self.specials);
        doc.into_iter()
            .map(|t| SpacyToken {
                norm: t.norm.unwrap_or_else(|| self.norms.norm(&t.text)),
                prefix: prefix(&t.text),
                suffix: suffix(&t.text),
                shape: word_shape(&t.text),
                is_space: is_space(&t.text),
                spacy: t.spacy,
                text: t.text,
            })
            .collect()
    }

    /// `Tokenizer._tokenize_affixes`: splits `text` into chunks at
    /// whitespace boundaries and tokenizes each.
    ///
    /// A single space after a chunk sets that chunk's last token's `SPACY`
    /// flag; any other whitespace run becomes a chunk of its own.
    /// `record`, when given, receives each chunk tokenized from scratch, as
    /// spaCy's cache does.
    fn tokenize_affixes(
        &self,
        text: &str,
        with_specials: bool,
        mut record: Option<&mut Cache>,
    ) -> Vec<RawToken> {
        let mut doc = Vec::new();
        let Some(first) = text.chars().next() else {
            return doc;
        };
        let mut in_ws = is_py_space(first);
        let mut start = 0usize;
        for (i, c) in text.char_indices() {
            if is_py_space(c) == in_ws {
                continue;
            }
            if start < i {
                self.tokenize_chunk(
                    &text[start..i],
                    with_specials,
                    &mut doc,
                    record.as_deref_mut(),
                );
            }
            if c == ' ' {
                if let Some(last) = doc.last_mut() {
                    last.spacy = true;
                }
                start = i + 1;
            } else {
                start = i;
            }
            in_ws = !in_ws;
        }
        if start < text.len() {
            self.tokenize_chunk(&text[start..], with_specials, &mut doc, record);
            if let Some(last) = doc.last_mut() {
                last.spacy = text.ends_with(' ') && !in_ws;
            }
        }
        doc
    }

    /// `_try_specials_and_cache`, then `Tokenizer._tokenize` on a miss.
    fn tokenize_chunk(
        &self,
        chunk: &str,
        with_specials: bool,
        doc: &mut Vec<RawToken>,
        record: Option<&mut Cache>,
    ) {
        if self.try_specials_and_cache(chunk, with_specials, doc) {
            return;
        }
        let before = doc.len();
        let mut prefixes = Vec::new();
        let mut suffixes = Vec::new();
        let rest = self.split_affixes(chunk, with_specials, &mut prefixes, &mut suffixes);
        self.attach_tokens(doc, rest, &prefixes, &suffixes, with_specials);
        if let Some(cache) = record
            && doc.len() > before
        {
            cache
                .entry(chunk.to_string())
                .or_insert_with(|| doc[before..].to_vec());
        }
    }

    /// `Tokenizer._try_specials_and_cache`: pushes the special case or the
    /// load-time cache entry for `chunk`, if there is one.
    fn try_specials_and_cache(
        &self,
        chunk: &str,
        with_specials: bool,
        doc: &mut Vec<RawToken>,
    ) -> bool {
        if with_specials && self.specials.push(chunk, doc) {
            return true;
        }
        match self.load_cache.get(chunk) {
            Some(tokens) => {
                doc.extend(tokens.iter().cloned());
                true
            }
            None => false,
        }
    }

    /// `Tokenizer._split_affixes`: strips prefixes and suffixes off `s`
    /// until neither matches or a special case remains. Returns the core.
    fn split_affixes<'a>(
        &self,
        mut s: &'a str,
        with_specials: bool,
        prefixes: &mut Vec<&'a str>,
        suffixes: &mut Vec<&'a str>,
    ) -> &'a str {
        let p = &self.patterns;
        let special = |x: &str| with_specials && self.specials.contains(x);
        let mut last_size = 0usize;
        while !s.is_empty() && s.len() != last_size {
            if p.token_match(s) || special(s) {
                break;
            }
            last_size = s.len();
            let pre_end = p.find_prefix(s);
            if pre_end != 0 {
                let minus_pre = &s[pre_end..];
                if !minus_pre.is_empty() && special(minus_pre) {
                    prefixes.push(&s[..pre_end]);
                    s = minus_pre;
                    break;
                }
            }
            let suf_start = pre_end + p.find_suffix(&s[pre_end..]);
            if suf_start != s.len() {
                let minus_suf = &s[..suf_start];
                if !minus_suf.is_empty() && special(minus_suf) {
                    suffixes.push(&s[suf_start..]);
                    s = minus_suf;
                    break;
                }
            }
            let has_pre = pre_end != 0;
            let has_suf = suf_start != s.len();
            if has_pre && has_suf && pre_end <= suf_start {
                prefixes.push(&s[..pre_end]);
                suffixes.push(&s[suf_start..]);
                s = &s[pre_end..suf_start];
            } else if has_pre {
                prefixes.push(&s[..pre_end]);
                s = &s[pre_end..];
            } else if has_suf {
                suffixes.push(&s[suf_start..]);
                s = &s[..suf_start];
            }
        }
        s
    }

    /// `Tokenizer._attach_tokens`: prefixes, the core (special case, cache
    /// entry, URL, or infix split), then suffixes innermost first.
    fn attach_tokens(
        &self,
        doc: &mut Vec<RawToken>,
        core: &str,
        prefixes: &[&str],
        suffixes: &[&str],
        with_specials: bool,
    ) {
        doc.extend(prefixes.iter().map(|p| RawToken::lex(p)));
        if !core.is_empty() && !self.try_specials_and_cache(core, with_specials, doc) {
            let p = &self.patterns;
            let infixes = if p.token_match(core) || p.url_match(core) {
                Vec::new()
            } else {
                p.find_infix(core)
            };
            if infixes.is_empty() {
                doc.push(RawToken::lex(core));
            } else {
                let mut start = 0usize;
                for (infix_start, infix_end) in infixes {
                    // spaCy skips an infix at the very start of the core.
                    if infix_start == 0 {
                        continue;
                    }
                    if infix_start != start {
                        doc.push(RawToken::lex(&core[start..infix_start]));
                    }
                    if infix_start != infix_end {
                        doc.push(RawToken::lex(&core[infix_start..infix_end]));
                    }
                    start = infix_end;
                }
                if start < core.len() {
                    doc.push(RawToken::lex(&core[start..]));
                }
            }
        }
        doc.extend(suffixes.iter().rev().map(|s| RawToken::lex(s)));
    }
}
