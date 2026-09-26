//! Word-level collation of a generated token sequence, matching
//! HuggingFace's `transformers.models.whisper.tokenization_whisper`:
//! `_combine_tokens_into_words`, `_split_tokens_on_unicode`,
//! `_split_tokens_on_spaces`, `_merge_punctuations`, and the timestamp
//! pairing `_collate_word_timestamps` does around them.
//!
//! HF splits on Unicode boundaries first (a token that decodes to a partial,
//! invalid-UTF-8 codepoint is folded into the next one), then — for
//! languages written without spaces — leaves that as the final split; every
//! other language further splits on whitespace/punctuation tokens. A closing
//! pass merges standalone punctuation into its neighboring word.

use splintr::AnyTokenizer;

/// One word's text and time span, seconds.
#[derive(Debug, Clone, PartialEq)]
pub struct WordTiming {
    pub text: String,
    pub start: f32,
    pub end: f32,
}

/// Languages HF treats as not using spaces to separate words: it splits these
/// on Unicode boundaries only, skipping the whitespace/punctuation pass.
/// Codes, not HF's English language names (`"chinese"`, `"cantonese"`, ...) —
/// this crate's language type carries BCP-47-ish codes throughout.
const CJK_LANGUAGE_CODES: [&str; 6] = ["zh", "ja", "th", "lo", "my", "yue"];

/// HF's default `prepend_punctuations` for `_merge_punctuations`.
const PREPEND_PUNCTUATIONS: &str = "\"'\u{201c}\u{a1}\u{bf}([{-";
/// HF's default `append_punctuations` for `_merge_punctuations`.
const APPEND_PUNCTUATIONS: &str =
    "\"'.\u{3002},\u{ff0c}!\u{ff01}?\u{ff1f}:\u{ff1a}\u{201d})]}\u{3001}";
/// The ASCII punctuation set `_split_tokens_on_spaces` checks a stripped
/// subword against (HF: `subword.strip() in "!\"#$%&'()*+,-./:;<=>?@[\\]^_\`{|}~"`).
const ASCII_PUNCTUATION_SET: &str = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";

/// Groups `tokens` into words, HF's `_combine_tokens_into_words`. Returns the
/// word texts and, per word, the local indices into `tokens` that make it up.
///
/// `language` is a BCP-47-ish code (`"en"`, `"zh"`, ...); `None` matches HF's
/// fallback to `"english"` when a tokenizer carries no language. `eos_token_id`
/// is the variant's own id (HF: `tokenizer.eos_token_id`), used only to flag a
/// leaked special token as its own "word" during the space-split pass.
pub fn combine_tokens_into_words(
    tokenizer: &AnyTokenizer,
    tokens: &[u32],
    language: Option<&str>,
    eos_token_id: u32,
) -> (Vec<String>, Vec<Vec<usize>>) {
    let (mut words, mut word_tokens, mut token_indices) = if is_cjk_language(language) {
        split_tokens_on_unicode(tokenizer, tokens)
    } else {
        split_tokens_on_spaces(tokenizer, tokens, eos_token_id)
    };

    merge_punctuations(
        &mut words,
        &mut word_tokens,
        &mut token_indices,
        PREPEND_PUNCTUATIONS,
        APPEND_PUNCTUATIONS,
    );

    (words, token_indices)
}

/// `language.is_none()` matches HF's fallback to `"english"`, which is not in
/// the CJK set.
fn is_cjk_language(language: Option<&str>) -> bool {
    matches!(language, Some(code) if CJK_LANGUAGE_CODES.contains(&code))
}

/// HF's `_split_tokens_on_unicode`: grow a token run one id at a time, and
/// close the current word the moment its lossily-decoded text stops
/// containing a "genuine" replacement character — one the full-sequence
/// decode also placed there, as opposed to one this prefix's decode
/// introduced only because it cut a multi-byte codepoint in half.
///
/// Indexing matches HF's Python `str` semantics exactly: by Unicode scalar
/// value (`char` in Rust), not by byte.
fn split_tokens_on_unicode(
    tokenizer: &AnyTokenizer,
    tokens: &[u32],
) -> (Vec<String>, Vec<Vec<u32>>, Vec<Vec<usize>>) {
    let decoded_full = tokenizer.decode_lossy(tokens);
    let decoded_full_chars: Vec<char> = decoded_full.chars().collect();
    const REPLACEMENT: char = '\u{fffd}';

    let mut words = Vec::new();
    let mut word_tokens = Vec::new();
    let mut token_indices = Vec::new();
    let mut current_tokens: Vec<u32> = Vec::new();
    let mut current_indices: Vec<usize> = Vec::new();
    let mut unicode_offset = 0usize;

    for (token_idx, &token) in tokens.iter().enumerate() {
        current_tokens.push(token);
        current_indices.push(token_idx);
        let decoded = tokenizer.decode_lossy(&current_tokens);

        let closes_word = match decoded.chars().position(|c| c == REPLACEMENT) {
            None => true,
            Some(idx_in_decoded) => {
                let full_idx = unicode_offset + idx_in_decoded;
                decoded_full_chars.get(full_idx) == Some(&REPLACEMENT)
            }
        };

        if closes_word {
            unicode_offset += decoded.chars().count();
            words.push(decoded);
            word_tokens.push(std::mem::take(&mut current_tokens));
            token_indices.push(std::mem::take(&mut current_indices));
        }
    }

    (words, word_tokens, token_indices)
}

/// HF's `_split_tokens_on_spaces`: further splits the Unicode-level subwords
/// on whitespace and standalone punctuation, gluing a subword onto the
/// previous word when it starts neither a special token, a leading space, nor
/// punctuation.
fn split_tokens_on_spaces(
    tokenizer: &AnyTokenizer,
    tokens: &[u32],
    eos_token_id: u32,
) -> (Vec<String>, Vec<Vec<u32>>, Vec<Vec<usize>>) {
    let (subwords, subword_tokens, subword_indices) = split_tokens_on_unicode(tokenizer, tokens);

    let mut words: Vec<String> = Vec::new();
    let mut word_tokens: Vec<Vec<u32>> = Vec::new();
    let mut token_indices: Vec<Vec<usize>> = Vec::new();

    for ((subword, sub_tokens), sub_indices) in subwords
        .into_iter()
        .zip(subword_tokens)
        .zip(subword_indices)
    {
        let special = sub_tokens.first().is_some_and(|&t| t >= eos_token_id);
        let with_space = subword.starts_with(' ');
        // HF: `subword.strip() in "<punct set>"` — Python substring
        // membership, true for an empty stripped subword too (`"" in s` is
        // always `True`), which `str::contains` reproduces directly.
        let punctuation = ASCII_PUNCTUATION_SET.contains(subword.trim());

        if special || with_space || punctuation || words.is_empty() {
            words.push(subword);
            word_tokens.push(sub_tokens);
            token_indices.push(sub_indices);
        } else {
            let last = words.len() - 1;
            words[last].push_str(&subword);
            word_tokens[last].extend(sub_tokens);
            token_indices[last].extend(sub_indices);
        }
    }

    (words, word_tokens, token_indices)
}

/// HF's `_merge_punctuations`: folds a standalone prepend-punctuation word
/// into the word after it, then a standalone append-punctuation word into
/// the word before it, and drops the now-empty slots.
fn merge_punctuations(
    words: &mut Vec<String>,
    word_tokens: &mut Vec<Vec<u32>>,
    token_indices: &mut Vec<Vec<usize>>,
    prepend: &str,
    append: &str,
) {
    // Prepend pass: walk right to left, folding word[i] into word[j] whenever
    // word[i] is exactly one prepend-punctuation character with a leading
    // space.
    if !words.is_empty() {
        let mut i = words.len() as isize - 2;
        let mut j = words.len() - 1;
        while i >= 0 {
            let ii = i as usize;
            if words[ii].starts_with(' ') && prepend.contains(words[ii].trim()) {
                let moved = std::mem::take(&mut words[ii]);
                words[j] = format!("{moved}{}", words[j]);
                let moved_tokens = std::mem::take(&mut word_tokens[ii]);
                word_tokens[j] = [moved_tokens, std::mem::take(&mut word_tokens[j])].concat();
                let moved_indices = std::mem::take(&mut token_indices[ii]);
                token_indices[j] = [moved_indices, std::mem::take(&mut token_indices[j])].concat();
            } else {
                j = ii;
            }
            i -= 1;
        }
    }

    // Append pass: walk left to right, folding word[j] into word[i] whenever
    // word[j] is exactly one append-punctuation character and word[i] has no
    // trailing space.
    if words.len() > 1 {
        let mut i = 0usize;
        let mut j = 1usize;
        while j < words.len() {
            if !words[i].ends_with(' ') && append.contains(words[j].as_str()) {
                let moved = std::mem::take(&mut words[j]);
                words[i].push_str(&moved);
                let moved_tokens = std::mem::take(&mut word_tokens[j]);
                word_tokens[i].extend(moved_tokens);
                let moved_indices = std::mem::take(&mut token_indices[j]);
                token_indices[i].extend(moved_indices);
            } else {
                i = j;
            }
            j += 1;
        }
    }

    let mut kept_words = Vec::with_capacity(words.len());
    let mut kept_tokens = Vec::with_capacity(word_tokens.len());
    let mut kept_indices = Vec::with_capacity(token_indices.len());
    for ((word, tokens), indices) in words
        .drain(..)
        .zip(word_tokens.drain(..))
        .zip(token_indices.drain(..))
    {
        if !word.is_empty() {
            kept_words.push(word);
            kept_tokens.push(tokens);
            kept_indices.push(indices);
        }
    }
    *words = kept_words;
    *word_tokens = kept_tokens;
    *token_indices = kept_indices;
}

/// Pairs each word's local token-index span with a start/end timestamp,
/// HF's `_collate_word_timestamps`.
///
/// `full_token_timestamps` is the array [`super::dtw::token_timestamps`]
/// returns: `prefix_len` zeros, one entry per generated token (`tokens`, the
/// same slice `combine_tokens_into_words` grouped), then one trailing
/// duplicate. A word spanning local indices `[first, ..., last]` gets
/// `(full_token_timestamps[prefix_len - 1 + first], full_token_timestamps[prefix_len + last])`
/// — the start of its first token and the end of its last, skipping any
/// interior token boundaries. `prefix_len` must be at least 1 (Whisper's SOT
/// prompt always is), so `prefix_len - 1 + first` never underflows.
pub fn collate_word_timestamps(
    words: &[String],
    token_indices: &[Vec<usize>],
    full_token_timestamps: &[f32],
    prefix_len: usize,
) -> Vec<WordTiming> {
    words
        .iter()
        .zip(token_indices.iter())
        .filter_map(|(word, indices)| {
            let first = *indices.first()?;
            let last = *indices.last()?;
            let start = full_token_timestamps.get(prefix_len - 1 + first).copied()?;
            let end = full_token_timestamps.get(prefix_len + last).copied()?;
            Some(WordTiming {
                text: word.clone(),
                start,
                end,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use splintr::{PretrainedVocab, from_vocab};

    fn tokenizer() -> AnyTokenizer {
        from_vocab(PretrainedVocab::WhisperV2).expect("bundled whisper vocab")
    }

    #[test]
    fn merges_standalone_leading_punctuation() {
        let mut words = vec![" (".to_string(), "hello".to_string()];
        let mut word_tokens = vec![vec![1u32], vec![2u32]];
        let mut token_indices = vec![vec![0usize], vec![1usize]];
        merge_punctuations(
            &mut words,
            &mut word_tokens,
            &mut token_indices,
            PREPEND_PUNCTUATIONS,
            APPEND_PUNCTUATIONS,
        );
        assert_eq!(words, vec![" (hello".to_string()]);
        assert_eq!(word_tokens, vec![vec![1u32, 2]]);
        assert_eq!(token_indices, vec![vec![0usize, 1]]);
    }

    #[test]
    fn merges_standalone_trailing_punctuation() {
        let mut words = vec![" hello".to_string(), ".".to_string()];
        let mut word_tokens = vec![vec![1u32], vec![2u32]];
        let mut token_indices = vec![vec![0usize], vec![1usize]];
        merge_punctuations(
            &mut words,
            &mut word_tokens,
            &mut token_indices,
            PREPEND_PUNCTUATIONS,
            APPEND_PUNCTUATIONS,
        );
        assert_eq!(words, vec![" hello.".to_string()]);
        assert_eq!(word_tokens, vec![vec![1u32, 2]]);
    }

    #[test]
    fn cjk_language_detection_uses_codes_not_names() {
        assert!(is_cjk_language(Some("zh")));
        assert!(is_cjk_language(Some("yue")));
        assert!(!is_cjk_language(Some("en")));
        assert!(!is_cjk_language(None));
    }

    #[test]
    fn collate_word_timestamps_skips_interior_boundaries() {
        // 3 generated tokens forming one word (indices 0,1,2), prefix_len=4.
        // full_token_timestamps layout: 4 zeros, then jump_times[0..3], then
        // a trailing duplicate: index 3 is the last prefix zero, indices
        // 4,5,6 are per-token jump times.
        let full_ts = [0.0f32, 0.0, 0.0, 0.0, 0.5, 0.8, 1.2, 1.2];
        let words = vec!["hello".to_string()];
        let token_indices = vec![vec![0usize, 1, 2]];
        let timings = collate_word_timestamps(&words, &token_indices, &full_ts, 4);
        assert_eq!(timings.len(), 1);
        assert_eq!(timings[0].start, 0.0); // full_ts[4 - 1 + 0] = full_ts[3]
        assert_eq!(timings[0].end, 1.2); // full_ts[4 + 2] = full_ts[6]
    }

    #[test]
    fn split_tokens_on_unicode_runs_on_real_tokenizer() {
        let tok = tokenizer();
        // Encode a short plain-English phrase and split it back into words;
        // asserts only that every token is accounted for and words concat
        // back into the original decode.
        let ids = tok.encode(" hello world");
        let (words, word_tokens, token_indices) = split_tokens_on_unicode(&tok, &ids);
        assert_eq!(words.len(), word_tokens.len());
        assert_eq!(words.len(), token_indices.len());
        let total_tokens: usize = word_tokens.iter().map(|t| t.len()).sum();
        assert_eq!(total_tokens, ids.len());
        assert_eq!(words.concat(), tok.decode_lossy(&ids));
    }
}
