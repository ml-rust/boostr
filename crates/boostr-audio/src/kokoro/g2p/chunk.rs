//! Text splitting and phoneme-budget chunking, ported from reference
//! Kokoro's `KPipeline.en_tokenize` / `KPipeline.waterfall_last` (misaki
//! 0.9.4, `kokoro` 0.9.4, Apache-2.0).
//!
//! A chunk holds at most 510 phoneme characters — the model's context
//! budget after the BOS/EOS pair reference Kokoro reserves. When a
//! sentence run would cross that budget, [`en_tokenize`] cuts at the last
//! sentence-ending punctuation it can find (then `:;`, then `,—`), so long
//! scripts split at the same boundaries the reference pipeline does.

use crate::g2p::misaki::MToken;

/// One chunk `KPipeline.en_tokenize` yields: source text and the phoneme
/// string built from its tokens, both trimmed.
#[derive(Debug, Clone, PartialEq)]
pub struct TextChunk {
    /// Original text this chunk covers.
    pub graphemes: String,
    /// Phoneme string for this chunk (whitespace collapsed to single
    /// spaces, as reference Kokoro's `tokens_to_ps` does).
    pub phonemes: String,
}

/// Splits `text` on runs of `\n`, matching reference Kokoro's default
/// `split_pattern = r'\n+'` applied to `text.strip()`. Segments that are
/// empty after trimming are dropped by the caller, not here — filtering
/// happens once, in [`super::KokoroG2p::chunks`], mirroring
/// `KPipeline.__call__`'s `if not graphemes.strip(): continue`.
pub(crate) fn split_segments(text: &str) -> Vec<&str> {
    text.trim().split('\n').collect()
}

/// `KPipeline.tokens_to_ps`: phonemes joined, a whitespace-bearing token's
/// gap collapsed to one space, then trimmed.
fn tokens_to_ps(tokens: &[MToken]) -> String {
    let mut out = String::new();
    for t in tokens {
        out.push_str(t.phonemes.as_deref().unwrap_or(""));
        if !t.whitespace.is_empty() {
            out.push(' ');
        }
    }
    out.trim().to_string()
}

/// `KPipeline.tokens_to_text`: source text and whitespace joined, trimmed.
fn tokens_to_text(tokens: &[MToken]) -> String {
    let mut out = String::new();
    for t in tokens {
        out.push_str(&t.text);
        out.push_str(&t.whitespace);
    }
    out.trim().to_string()
}

/// `true` when `phonemes` is exactly one of `candidates`' characters —
/// `KPipeline.waterfall_last`'s `t.phonemes in set(w)` test. A punctuation
/// token's phonemes is that single character; a resolved word's phonemes
/// never matches.
fn phonemes_is_one_of(phonemes: &Option<String>, candidates: &str) -> bool {
    let Some(p) = phonemes.as_deref() else {
        return false;
    };
    let mut chars = p.chars();
    match (chars.next(), chars.next()) {
        (Some(c), None) => candidates.contains(c),
        _ => false,
    }
}

/// `KPipeline.waterfall_last`: the split point nearest the end of `tokens`
/// that (a) sits right after a sentence-ending mark, colon/semicolon, or
/// comma/dash, in that preference order, and (b) leaves the resulting
/// chunk within the 510-character phoneme budget. Falls back to the whole
/// slice when no waterfall level fits.
fn waterfall_last(tokens: &[MToken], next_count: usize) -> usize {
    const WATERFALL: [&str; 3] = ["!.?…", ":;", ",—"];
    const BUMPS: &str = ")”";
    for level in WATERFALL {
        let Some(mut z) = tokens
            .iter()
            .enumerate()
            .rev()
            .find(|(_, t)| phonemes_is_one_of(&t.phonemes, level))
            .map(|(i, _)| i + 1)
        else {
            continue;
        };
        if z < tokens.len() && phonemes_is_one_of(&tokens[z].phonemes, BUMPS) {
            z += 1;
        }
        let head_len = tokens_to_ps(&tokens[..z]).chars().count();
        if next_count.saturating_sub(head_len) <= 510 {
            return z;
        }
    }
    tokens.len()
}

/// `KPipeline.en_tokenize`: cuts `tokens` into chunks of at most 510
/// phoneme characters, one call to [`waterfall_last`] per cut. `tokens`
/// carry `phonemes: None` for a word the G2P step left unresolved;
/// reference Kokoro treats that the same as an empty phoneme string here,
/// and so does this port.
pub(crate) fn en_tokenize(tokens: Vec<MToken>) -> Vec<TextChunk> {
    let mut out = Vec::new();
    let mut run: Vec<MToken> = Vec::new();
    let mut pcount: usize = 0;
    for mut t in tokens {
        if t.phonemes.is_none() {
            t.phonemes = Some(String::new());
        }
        let mut next_ps = t.phonemes.clone().unwrap_or_default();
        if !t.whitespace.is_empty() {
            next_ps.push(' ');
        }
        let next_pcount = pcount + next_ps.trim_end().chars().count();
        if next_pcount > 510 {
            let z = waterfall_last(&run, next_pcount);
            out.push(TextChunk {
                graphemes: tokens_to_text(&run[..z]),
                phonemes: tokens_to_ps(&run[..z]),
            });
            run = run.split_off(z);
            pcount = tokens_to_ps(&run).chars().count();
            if run.is_empty() {
                next_ps = next_ps.trim_start().to_string();
            }
        }
        pcount += next_ps.chars().count();
        run.push(t);
    }
    if !run.is_empty() {
        out.push(TextChunk {
            graphemes: tokens_to_text(&run),
            phonemes: tokens_to_ps(&run),
        });
    }
    out
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
    fn split_segments_drops_blank_runs() {
        assert_eq!(split_segments("a\n\nb"), vec!["a", "", "b"]);
        assert_eq!(split_segments("  a\nb  "), vec!["a", "b"]);
    }

    #[test]
    fn single_short_sentence_is_one_chunk() {
        let tokens = vec![tok("Hi", "", Some("hˈI")), tok(".", "", Some("."))];
        let chunks = en_tokenize(tokens);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].graphemes, "Hi.");
        assert_eq!(chunks[0].phonemes, "hˈI.");
    }

    #[test]
    fn unresolved_word_reads_as_empty_phonemes() {
        let tokens = vec![tok("word", "", None)];
        let chunks = en_tokenize(tokens);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].phonemes, "");
    }

    #[test]
    fn long_run_splits_at_sentence_boundary() {
        // Two "sentences" of 300 phoneme characters each, joined by a
        // period token, cross the 510 budget only once both are queued;
        // the cut lands right after the period.
        let tokens = vec![
            tok("a", " ", Some(&"x".repeat(300))),
            tok(".", " ", Some(".")),
            tok("b", "", Some(&"y".repeat(300))),
        ];
        let chunks = en_tokenize(tokens);
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].phonemes.ends_with('.'));
        assert_eq!(chunks[1].phonemes, "y".repeat(300));
    }

    #[test]
    fn no_waterfall_boundary_falls_back_to_whole_run() {
        // A single token already over budget, queued before any other
        // token exists: waterfall_last has nothing to cut at, so it
        // reports a zero-length head — an empty leading chunk, exactly as
        // reference Kokoro's generator does before its caller drops any
        // chunk with an empty phoneme string. The token itself then forms
        // its own (over-budget) chunk.
        let tokens = vec![tok("word", "", Some(&"x".repeat(600)))];
        let chunks = en_tokenize(tokens);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].phonemes, "");
        assert_eq!(chunks[1].phonemes.chars().count(), 600);
    }
}
