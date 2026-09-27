//! misaki `en.G2P.__call__` from tagged tokens onward.
//!
//! [`MisakiG2p::phonemize`] takes the tokens misaki builds from spaCy (text,
//! Penn Treebank fine tag, trailing whitespace).
//! [`MisakiG2p::phonemize_text`] builds them from raw text with the embedded
//! spaCy tokenizer and tagger, as `G2P.__call__` does.

use super::chars::SUBTOKEN_JUNKS;
use super::front::TextFrontEnd;
use super::lexicon::Lexicon;
use super::resolve::{resolve_tokens, token_context};
use super::retokenize::{Word, fold_left, retokenize};
use super::token::{MToken, TokenContext, merge_tokens};
use crate::error::Result;

/// misaki's default placeholder for an unresolved word.
pub const DEFAULT_UNK: &str = "❓";
/// The placeholder Kokoro's pipeline passes: unresolved words read as
/// nothing.
pub const KOKORO_UNK: &str = "";

/// Pronounces a word the lexicon leaves unresolved: misaki's `fallback`
/// hook.
///
/// misaki calls it with a standalone token, or with the merge of a whole
/// group when one of its sub-tokens stays unresolved. It returns phonemes
/// and a rating; `None` phonemes leave the word unresolved. As in misaki, the
/// rating reaches [`MToken::rating`] for a group and is dropped for a
/// standalone token.
pub trait OovFallback {
    /// Phonemes and rating for `token`.
    fn phonemize(&self, token: &MToken) -> (Option<String>, Option<u8>);
}

/// Result of [`MisakiG2p::phonemize`].
#[derive(Debug, Clone, PartialEq)]
pub struct G2pOutput {
    /// Phoneme string for the whole input, whitespace preserved.
    pub phonemes: String,
    /// Final tokens: one per standalone token or merged group, with
    /// phonemes and rating.
    pub tokens: Vec<MToken>,
}

/// misaki English G2P (`misaki.en.G2P(trf=False)`) over pre-tagged tokens.
pub struct MisakiG2p {
    lexicon: Lexicon,
    unk: String,
    fallback: Option<Box<dyn OovFallback + Send + Sync>>,
}

impl std::fmt::Debug for MisakiG2p {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MisakiG2p")
            .field("british", &self.lexicon.british)
            .field("unk", &self.unk)
            .field("fallback", &self.fallback.is_some())
            .finish()
    }
}

impl MisakiG2p {
    /// US (`british = false`) or GB English, `unk` = [`DEFAULT_UNK`], no
    /// fallback. Decodes the embedded lexicon.
    pub fn new(british: bool) -> Result<Self> {
        Ok(Self {
            lexicon: Lexicon::new(british)?,
            unk: DEFAULT_UNK.to_string(),
            fallback: None,
        })
    }

    /// Replaces the placeholder written for unresolved words.
    pub fn with_unk(mut self, unk: impl Into<String>) -> Self {
        self.unk = unk.into();
        self
    }

    /// Installs the hook for words the lexicon leaves unresolved.
    pub fn with_fallback(mut self, fallback: Box<dyn OovFallback + Send + Sync>) -> Self {
        self.fallback = Some(fallback);
        self
    }

    /// `true` for GB English.
    pub fn british(&self) -> bool {
        self.lexicon.british
    }

    /// The placeholder written for unresolved words.
    pub fn unk(&self) -> &str {
        &self.unk
    }

    /// misaki `G2P.__call__(text)`: markdown-link preprocessing, the spaCy
    /// `en_core_web_sm` tokenizer and tagger, then [`Self::phonemize`].
    ///
    /// The first call across every `MisakiG2p` instance builds the embedded
    /// tokenizer and tagger; later calls reuse the shared instance.
    pub fn phonemize_text(&self, text: &str) -> Result<G2pOutput> {
        let tokens = TextFrontEnd::shared()?.tokens(text)?;
        Ok(self.phonemize(tokens))
    }

    /// Converts tagged tokens to phonemes, exactly as misaki's `G2P.__call__`
    /// does after tokenization.
    pub fn phonemize(&self, tokens: Vec<MToken>) -> G2pOutput {
        let tokens = fold_left(tokens, &self.unk);
        let mut words = retokenize(tokens);
        let mut ctx = TokenContext::default();
        for word in words.iter_mut().rev() {
            match word {
                Word::Single(w) => {
                    if w.phonemes.is_none() {
                        w.phonemes = self.lexicon.resolve(w, &ctx).0;
                    }
                    let fallback = self.fallback.as_ref().filter(|_| w.phonemes.is_none());
                    if let Some(fb) = fallback {
                        w.phonemes = fb.phonemize(w).0;
                    }
                    ctx = token_context(ctx, w.phonemes.as_deref(), w);
                }
                Word::Group(group) => ctx = self.resolve_group(group, ctx),
            }
        }

        let mut out_tokens: Vec<MToken> = Vec::with_capacity(words.len());
        for word in words {
            let tk = match word {
                Word::Single(t) => Some(t),
                Word::Group(g) => merge_tokens(&g, Some(&self.unk)),
            };
            if let Some(mut tk) = tk {
                if let Some(ps) = tk.phonemes.as_mut().filter(|p| !p.is_empty()) {
                    *ps = ps.replace('ɾ', "T").replace('ʔ', "t");
                }
                out_tokens.push(tk);
            }
        }
        let mut phonemes = String::new();
        for tk in &out_tokens {
            phonemes.push_str(tk.phonemes.as_deref().unwrap_or(&self.unk));
            phonemes.push_str(&tk.whitespace);
        }
        G2pOutput {
            phonemes,
            tokens: out_tokens,
        }
    }

    /// The group branch of misaki's main loop: finds the longest resolvable
    /// run ending at `right`, moving `right` left after each hit.
    fn resolve_group(&self, w: &mut [MToken], mut ctx: TokenContext) -> TokenContext {
        let (mut left, mut right) = (0usize, w.len());
        let mut should_fallback = false;
        while left < right {
            let span = &w[left..right];
            let merged = if span
                .iter()
                .any(|t| t.alias.is_some() || t.phonemes.is_some())
            {
                None
            } else {
                merge_tokens(span, None)
            };
            let (ps, rating) = match &merged {
                Some(tk) => self.lexicon.resolve(tk, &ctx),
                None => (None, None),
            };
            if let (Some(ps), Some(tk)) = (ps, merged.as_ref()) {
                ctx = token_context(ctx, Some(&ps), tk);
                w[left].phonemes = Some(ps);
                w[left].rating = rating;
                for x in &mut w[left + 1..right] {
                    x.phonemes = Some(String::new());
                }
                right = left;
                left = 0;
            } else if left + 1 < right {
                left += 1;
            } else {
                right -= 1;
                let tk = &mut w[right];
                if tk.phonemes.is_none() {
                    if tk.text.chars().all(|c| SUBTOKEN_JUNKS.contains(c)) {
                        tk.phonemes = Some(String::new());
                        tk.rating = Some(3);
                    } else if self.fallback.is_some() {
                        should_fallback = true;
                        break;
                    }
                }
                left = 0;
            }
        }
        match (should_fallback, &self.fallback) {
            (true, Some(fb)) => {
                if let Some(tk) = merge_tokens(w, None) {
                    let (ps, rating) = fb.phonemize(&tk);
                    if let Some((head, rest)) = w.split_first_mut() {
                        head.phonemes = ps;
                        head.rating = rating;
                        for x in rest {
                            x.phonemes = Some(String::new());
                            x.rating = rating;
                        }
                    }
                }
            }
            _ => resolve_tokens(w),
        }
        ctx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phonemize_text_applies_link_features() {
        // misaki 0.9.4 `en.G2P(trf=False, british=False)(text)[0]` with
        // spaCy 3.8.14 and en_core_web_sm 3.8.0.
        let g2p = MisakiG2p::new(false).unwrap_or_else(|e| panic!("MisakiG2p: {e}"));
        let cases = [
            ("  [Kokoro](/kˈOkəɹO/) is great.", "kˈOkəɹO ɪz ɡɹˈAt."),
            (
                "[Hello world](/həlˈO/) and [read](-1) [it](+2) [a](0.5) [b](-0.5) [c](#&#) [d](x)",
                "həlˈO ænd ɹˌɛd ˈɪt ɐ bˌi sˈi dˈi",
            ),
            ("Say [don't](/dˈOnt/) now", "sˈA dˈOnt nˈW"),
            ("[12](#a#)", "twˈɛlv"),
            ("x [a]() y", "ˈɛks ɐ wˈI"),
            ("[New York](/nuː jɔːk/) rocks", "nuː jɔːk ɹˈɑks"),
            (
                "Visit [U.S.](/juː ɛs/) today [5](//ab//)",
                "vˈɪzət juː ɛs tədˈA ab",
            ),
        ];
        for (text, want) in cases {
            let got = g2p
                .phonemize_text(text)
                .unwrap_or_else(|e| panic!("phonemize_text({text:?}): {e}"));
            assert_eq!(got.phonemes, want, "{text:?}");
        }
    }

    #[test]
    fn phonemize_text_of_empty_text() {
        let g2p = MisakiG2p::new(true).unwrap_or_else(|e| panic!("MisakiG2p: {e}"));
        for text in ["", "   "] {
            let got = g2p
                .phonemize_text(text)
                .unwrap_or_else(|e| panic!("phonemize_text({text:?}): {e}"));
            assert_eq!(got.phonemes, "");
            assert!(got.tokens.is_empty());
        }
    }
}
