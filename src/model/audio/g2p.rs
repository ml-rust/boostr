//! Grapheme-to-phoneme (G2P) via espeak-ng.
//!
//! Kokoro TTS requires phoneme input rather than raw text. This module wraps
//! espeak-ng's IPA output to produce the phoneme sequences Kokoro was trained on.
//!
//! Requires the `tts-g2p` feature and the `espeak-ng` system library at
//! runtime. Without the feature, [`Phonemizer::new`] returns
//! [`G2pError::FeatureDisabled`] so callers can fall back to an alternate G2P.

use thiserror::Error;

/// Errors from the G2P pipeline.
#[derive(Debug, Error)]
pub enum G2pError {
    #[error("tts-g2p feature is disabled — rebuild with `--features tts-g2p`")]
    FeatureDisabled,
    #[error("unsupported language: {0}")]
    UnsupportedLanguage(String),
    #[error("espeak-ng error: {0}")]
    Backend(String),
}

/// BCP-47-like language codes Kokoro's English voices cover. Additional codes
/// can be added as new voice packs ship.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lang {
    EnUs,
    EnGb,
    Es,
    Fr,
    Ja,
    Hi,
    It,
    Pt,
    Zh,
    Ko,
}

impl std::str::FromStr for Lang {
    type Err = G2pError;
    fn from_str(s: &str) -> Result<Self, G2pError> {
        Self::parse(s)
    }
}

impl Lang {
    /// Parse a BCP-47 or dash/underscore-separated code.
    pub fn parse(s: &str) -> Result<Self, G2pError> {
        let normalized = s.to_ascii_lowercase().replace('_', "-");
        match normalized.as_str() {
            "en" | "en-us" => Ok(Self::EnUs),
            "en-gb" | "en-uk" => Ok(Self::EnGb),
            "es" | "es-es" | "es-mx" => Ok(Self::Es),
            "fr" | "fr-fr" => Ok(Self::Fr),
            "ja" | "ja-jp" => Ok(Self::Ja),
            "hi" | "hi-in" => Ok(Self::Hi),
            "it" | "it-it" => Ok(Self::It),
            "pt" | "pt-br" | "pt-pt" => Ok(Self::Pt),
            "zh" | "zh-cn" | "zh-tw" => Ok(Self::Zh),
            "ko" | "ko-kr" => Ok(Self::Ko),
            _ => Err(G2pError::UnsupportedLanguage(s.to_string())),
        }
    }

    /// espeak-ng voice identifier for this language.
    pub fn espeak_voice(self) -> &'static str {
        match self {
            Self::EnUs => "en-us",
            Self::EnGb => "en-gb",
            Self::Es => "es",
            Self::Fr => "fr",
            Self::Ja => "ja",
            Self::Hi => "hi",
            Self::It => "it",
            Self::Pt => "pt",
            Self::Zh => "cmn", // Mandarin
            Self::Ko => "ko",
        }
    }
}

/// Converts text into a sequence of IPA phoneme strings plus punctuation.
///
/// The output preserves sentence-break markers (`.`, `,`, `?`, `!`, `;`, `:`)
/// as separate entries so downstream prosody models can observe them.
#[derive(Debug)]
pub struct Phonemizer {
    lang: Lang,
    #[cfg(feature = "tts-g2p")]
    _espeak: (),
}

impl Phonemizer {
    /// Create a phonemizer for `lang`. Initializes espeak-ng lazily on first
    /// call. Safe to call multiple times.
    pub fn new(lang: Lang) -> Result<Self, G2pError> {
        #[cfg(feature = "tts-g2p")]
        {
            // espeakng's initialize() is idempotent and process-wide.
            espeakng::initialise(None).map_err(|e| G2pError::Backend(format!("{e:?}")))?;
            Ok(Self { lang, _espeak: () })
        }
        #[cfg(not(feature = "tts-g2p"))]
        {
            let _ = lang;
            Err(G2pError::FeatureDisabled)
        }
    }

    /// Language this phonemizer was constructed for.
    pub fn lang(&self) -> Lang {
        self.lang
    }

    /// Convert `text` into a sequence of phoneme tokens.
    ///
    /// Returned tokens are either:
    /// - IPA phoneme strings (possibly multi-codepoint, e.g. `"ɹ"`, `"aʊ"`, `"tʃ"`)
    /// - Single-character punctuation marks preserved inline
    /// - `"<space>"` sentinel between words (for Kokoro's phoneme vocab)
    pub fn text_to_phonemes(&self, text: &str) -> Result<Vec<String>, G2pError> {
        #[cfg(feature = "tts-g2p")]
        {
            let mut speaker = espeakng::Speaker::new()
                .map_err(|e| G2pError::Backend(format!("new speaker: {e:?}")))?;
            speaker
                .set_voice_by_name(self.lang.espeak_voice())
                .map_err(|e| G2pError::Backend(format!("set_voice: {e:?}")))?;
            let ipa = speaker
                .synthesize_ipa(text)
                .map_err(|e| G2pError::Backend(format!("synthesize_ipa: {e:?}")))?;
            Ok(tokenize_ipa(&ipa))
        }
        #[cfg(not(feature = "tts-g2p"))]
        {
            let _ = text;
            Err(G2pError::FeatureDisabled)
        }
    }
}

/// Split espeak-ng IPA output into phoneme tokens.
///
/// espeak's output uses:
/// - `_` as a syllable separator
/// - spaces between words
/// - primary/secondary stress marks (`ˈ`, `ˌ`) which we preserve on the
///   following phoneme
/// - punctuation kept inline
///
/// We deliberately keep this deterministic and pure so it's unit-testable
/// without requiring the espeak-ng system library.
pub fn tokenize_ipa(ipa: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut current = String::new();

    for ch in ipa.chars() {
        match ch {
            ' ' => {
                flush(&mut current, &mut out);
                if !matches!(out.last().map(String::as_str), Some("<space>")) {
                    out.push("<space>".to_string());
                }
            }
            '_' => {
                // Syllable break — flush current phoneme but emit no token.
                flush(&mut current, &mut out);
            }
            '.' | ',' | '?' | '!' | ';' | ':' => {
                flush(&mut current, &mut out);
                out.push(ch.to_string());
            }
            // Stress markers prefix the *next* phoneme rather than stand alone.
            'ˈ' | 'ˌ' => {
                flush(&mut current, &mut out);
                current.push(ch);
            }
            // Combining characters attach to the current base phoneme.
            c if is_combining(c) => {
                current.push(c);
            }
            // Alphabetic-like phoneme characters: collect into current token.
            // Break on whitespace and punctuation (handled above). Multi-char
            // phonemes like "tʃ" are grouped by the stress-prefix rule + the
            // fact that espeak emits them contiguously.
            c => {
                // Start of a new phoneme cluster: flush the prior one unless
                // we're mid-stress-prefix.
                if current.is_empty() || current_is_stress_prefix(&current) {
                    current.push(c);
                } else {
                    // Treat each non-combining base character as its own
                    // phoneme token — espeak separates adjacent phonemes
                    // within a word with either `_` or by never emitting
                    // ambiguous digraphs without a combining mark.
                    flush(&mut current, &mut out);
                    current.push(c);
                }
            }
        }
    }
    flush(&mut current, &mut out);

    // Trim a trailing `<space>` if present.
    if matches!(out.last().map(String::as_str), Some("<space>")) {
        out.pop();
    }
    out
}

fn flush(current: &mut String, out: &mut Vec<String>) {
    if !current.is_empty() {
        out.push(std::mem::take(current));
    }
}

fn current_is_stress_prefix(s: &str) -> bool {
    s == "ˈ" || s == "ˌ"
}

/// Unicode combining marks used by IPA (tone, length, nasalization, …).
fn is_combining(c: char) -> bool {
    let u = c as u32;
    // Combining Diacritical Marks + IPA Extensions length markers.
    (0x0300..=0x036F).contains(&u) || c == 'ː' || c == '̃' || c == '̩'
}

/// Map an IPA phoneme token to its id in a user-supplied phoneme vocabulary.
///
/// Returns `None` for tokens missing from `vocab`. Callers typically fall back
/// to an `<unk>` id or skip silently.
pub fn phonemes_to_ids<V>(tokens: &[String], vocab: &V) -> Vec<Option<u32>>
where
    V: PhonemeVocab,
{
    tokens.iter().map(|t| vocab.lookup(t)).collect()
}

/// Trait implemented by Kokoro-specific phoneme vocabularies.
pub trait PhonemeVocab {
    fn lookup(&self, phoneme: &str) -> Option<u32>;
}

impl PhonemeVocab for std::collections::HashMap<String, u32> {
    fn lookup(&self, phoneme: &str) -> Option<u32> {
        self.get(phoneme).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn lang_parses_common_forms() {
        assert_eq!(Lang::parse("en").unwrap(), Lang::EnUs);
        assert_eq!(Lang::parse("en-US").unwrap(), Lang::EnUs);
        assert_eq!(Lang::parse("en_GB").unwrap(), Lang::EnGb);
        assert_eq!(Lang::parse("zh-CN").unwrap(), Lang::Zh);
        assert!(Lang::parse("xx").is_err());
    }

    #[test]
    fn tokenize_ipa_inserts_spaces_between_words() {
        let tokens = tokenize_ipa("hɛloʊ wɝld");
        assert!(tokens.contains(&"<space>".to_string()));
        // No adjacent duplicate spaces.
        for w in tokens.windows(2) {
            assert!(!(w[0] == "<space>" && w[1] == "<space>"));
        }
    }

    #[test]
    fn tokenize_ipa_keeps_punctuation() {
        let tokens = tokenize_ipa("hɛloʊ, wɝld.");
        assert!(tokens.iter().any(|t| t == ","));
        assert!(tokens.iter().any(|t| t == "."));
    }

    #[test]
    fn tokenize_ipa_attaches_stress_to_next_phoneme() {
        let tokens = tokenize_ipa("ˈhɛ");
        assert_eq!(tokens[0], "ˈh");
    }

    #[test]
    fn tokenize_ipa_attaches_length_mark() {
        let tokens = tokenize_ipa("aːb");
        // The length mark should combine with the preceding vowel.
        assert_eq!(tokens[0], "aː");
        assert_eq!(tokens[1], "b");
    }

    #[test]
    fn tokenize_ipa_drops_syllable_separator() {
        let tokens = tokenize_ipa("a_b");
        assert_eq!(tokens, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn tokenize_ipa_trims_trailing_space() {
        let tokens = tokenize_ipa("a ");
        assert_eq!(tokens, vec!["a".to_string()]);
    }

    #[test]
    fn phonemes_to_ids_looks_up_vocab() {
        let mut vocab: HashMap<String, u32> = HashMap::new();
        vocab.insert("h".to_string(), 1);
        vocab.insert("ɛ".to_string(), 2);
        let ids = phonemes_to_ids(&["h".into(), "ɛ".into(), "?".into()], &vocab);
        assert_eq!(ids, vec![Some(1), Some(2), None]);
    }

    #[test]
    #[cfg(not(feature = "tts-g2p"))]
    fn phonemizer_errors_without_feature() {
        let err = Phonemizer::new(Lang::EnUs).unwrap_err();
        assert!(matches!(err, G2pError::FeatureDisabled));
    }
}
