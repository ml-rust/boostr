//! The written script: the ear script's words with the punctuation grammar
//! wants.
//!
//! The voice speaks the ear script, whose commas and dashes set its pauses.
//! Captions show the written one. The two are matched letter by letter, with
//! case, punctuation and spacing ignored, so `motion-video` matches
//! `motion video`.

use super::phrase::Phrase;
use super::tags::spoken;
use crate::error::{Error, Result};
use crate::narration::pattern::{Patterns, replace_all_with, split, utf16_len};

/// A written word and where its letters start in the letter stream, in
/// UTF-16 code units.
#[derive(Debug, Clone, PartialEq)]
struct Token {
    text: String,
    at: usize,
}

/// A written script checked against its ear script.
#[derive(Debug, Clone, PartialEq)]
pub struct WrittenScript {
    /// The written script's spoken words on one line: markup removed,
    /// written forms of `{written|spoken}` pairs kept.
    pub text: String,
    tokens: Vec<Token>,
}

/// A word's letters and digits, lowercased.
fn key(p: &Patterns, word: &str) -> String {
    replace_all_with(&p.not_letters, &word.to_lowercase(), "")
}

/// The words of `text` split at whitespace, empty pieces included, each with
/// the UTF-16 length of its key.
fn keyed_words<'t>(p: &Patterns, text: &'t str) -> Vec<(&'t str, usize)> {
    split(&p.spaces, text)
        .into_iter()
        .map(|w| (w, utf16_len(&key(p, w))))
        .collect()
}

/// The word of `script` whose letters pass position `i` of its letter
/// stream, else `(the end)`.
fn word_at(p: &Patterns, script: &str, i: usize) -> String {
    let mut n = 0;
    for (w, len) in keyed_words(p, script) {
        n += len;
        if n > i {
            return w.to_string();
        }
    }
    "(the end)".to_string()
}

/// Checks that `written` holds the same words as `ear` and returns it ready
/// to align. Errors at the first word where the two differ.
pub fn written_script(ear: &str, written: &str) -> Result<WrittenScript> {
    let p = Patterns::get()?;
    let text = spoken(written)?;
    // A mark standing alone joins the word before it.
    let mut tokens: Vec<Token> = Vec::new();
    let mut at = 0;
    for (w, len) in keyed_words(p, &text) {
        if w.is_empty() {
            continue;
        }
        match tokens.last_mut() {
            Some(last) if len == 0 => {
                last.text.push(' ');
                last.text.push_str(w);
            }
            _ => tokens.push(Token {
                text: w.to_string(),
                at,
            }),
        }
        at += len;
    }
    let ear_text = spoken(ear)?;
    let a: Vec<u16> = key(p, &ear_text).encode_utf16().collect();
    let b: Vec<u16> = key(p, &text).encode_utf16().collect();
    if a != b {
        let i = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
        return Err(Error::InvalidArgument {
            arg: "written",
            reason: format!(
                "the scripts' words differ at \"{}\" in the ear script and \"{}\" in the written one. The written script holds the same words, with only punctuation, case or spacing changed.",
                word_at(p, &ear_text, i),
                word_at(p, &text, i),
            ),
        });
    }
    Ok(WrittenScript { text, tokens })
}

impl WrittenScript {
    /// The written words each spoken text covers, in order: `texts` are
    /// consecutive pieces of the ear script, such as its phrases.
    pub fn span_texts<S: AsRef<str>>(&self, texts: &[S]) -> Result<Vec<String>> {
        let p = Patterns::get()?;
        let mut from = 0;
        // Tokens are in non-decreasing `at` order and the spans are
        // contiguous from 0, so one cursor walks the tokens for all spans.
        let mut idx = 0;
        Ok(texts
            .iter()
            .map(|t| {
                let to = from + utf16_len(&key(p, t.as_ref()));
                let start = idx;
                while idx < self.tokens.len() && self.tokens[idx].at < to {
                    idx += 1;
                }
                let words: Vec<&str> = self.tokens[start..idx]
                    .iter()
                    .map(|tok| tok.text.as_str())
                    .collect();
                from = to;
                words.join(" ")
            })
            .collect())
    }

    /// The phrases with their text replaced by the written words they cover.
    pub fn spans(&self, phrases: &[Phrase]) -> Result<Vec<Phrase>> {
        let texts: Vec<&str> = phrases.iter().map(|p| p.text.as_str()).collect();
        let texts = self.span_texts(&texts)?;
        Ok(phrases
            .iter()
            .zip(texts)
            .map(|(p, text)| Phrase {
                text,
                pause: p.pause,
                kind: p.kind,
                delivery: p.delivery.clone(),
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EAR: &str = "Then, it gathers facts, and chooses a direction.\n\nThe {VoxCPM2|Vox C P M two} voice — calm.";

    #[test]
    fn written_words_cover_spoken_phrases() -> Result<()> {
        let w = written_script(
            EAR,
            "Then it gathers facts and chooses a direction.\n\nThe VoxCPM2 voice: calm.",
        )?;
        assert_eq!(
            w.text,
            "Then it gathers facts and chooses a direction. The VoxCPM2 voice: calm."
        );
        let got = w.span_texts(&[
            "Then, it gathers facts, and chooses a direction.",
            "The VoxCPM2 voice — calm.",
        ])?;
        assert_eq!(
            got,
            vec![
                "Then it gathers facts and chooses a direction.",
                "The VoxCPM2 voice: calm."
            ]
        );
        Ok(())
    }

    #[test]
    fn a_lone_mark_joins_the_word_before() -> Result<()> {
        let w = written_script("A word and more.", "A word — and more.")?;
        assert_eq!(
            w.span_texts(&["A word", "and more."])?,
            vec!["A word —", "and more."]
        );
        Ok(())
    }

    #[test]
    fn mismatch_quotes_both_words() {
        let err = written_script(
            EAR,
            "Then it collects facts and chooses a direction. The VoxCPM2 voice: calm.",
        )
        .err()
        .map(|e| e.to_string());
        assert_eq!(
            err.as_deref(),
            Some(
                "invalid argument 'written': the scripts' words differ at \"gathers\" in the ear script and \"collects\" in the written one. The written script holds the same words, with only punctuation, case or spacing changed."
            )
        );
    }

    #[test]
    fn a_missing_word_is_the_end() {
        let err = written_script("One two three.", "One two three four.")
            .err()
            .map(|e| e.to_string());
        assert!(
            err.is_some_and(|e| e.contains("at \"(the end)\" in the ear script and \"four.\""))
        );
    }
}
