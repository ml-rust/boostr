//! Load VoxCPM2's `tokenizer.json` through splintr.
//!
//! This is the same split as `format::gguf_vocab`: splintr owns tokenization,
//! boostr only hands it a file. Unlike the GGUF path there is no container to
//! extract here, `tokenizer.json` is already the format splintr's generic
//! HuggingFace loader (`splintr::from_json_path`) reads directly, so
//! [`load_tokenizer`] is a one-line call. It exists at all only to fold
//! splintr's `HfJsonError` into `crate::error::Error` the way every other
//! boostr loader does.
//!
//! # No-BOS is the caller's contract, not this function's
//!
//! The reference tokenizer tokenizes with `tokenizer.tokenize(text)` then
//! `convert_tokens_to_ids`, which does NOT prepend BOS. VoxCPM2's
//! `tokenizer.json` post-processor is `TemplateProcessing` prepending `<s>`
//! (id 1), so `AnyTokenizer::encode` WOULD add it. Callers must use
//! `encode_raw`, never `encode`, or every position in the causal context
//! shifts by one. `encode_raw` applies no post-processor template, matching
//! the reference exactly, and needs no boostr-side stripping of a leading
//! id. [`tokenize`] below is the encode_raw-only entry point for this
//! reason: it does not expose a way to call `encode` by mistake.
//!
//! # Why input is normalized before tokenizing
//!
//! `tokenizer.json` declares `normalizer: Sequence[Prepend "▁",
//! Replace(" ", "▁")]` with `pre_tokenizer: null`. splintr implements
//! that declared pipeline faithfully, verified against `tokenizers`' own
//! `Tokenizer.from_file` on the same file: this is NOT a splintr bug and
//! nothing here is a workaround for one.
//!
//! The divergence is upstream of splintr. `LlamaTokenizerFast.from_pretrained`
//! under transformers 5.x rewrites the loaded tokenizer to use a `Metaspace`
//! pre_tokenizer, which handles runs of whitespace differently from the
//! declared `Replace`-based normalizer above; transformers 4.57.3 does not
//! perform this rewrite. So which convention the reference pipeline follows
//! depends on which transformers version produced it, and both are
//! defensible readings of the same checkpoint. Measured across sampled
//! text, the two conventions diverge only on runs of two or more whitespace
//! characters or leading/trailing whitespace; on every sampled text, after
//! collapsing whitespace runs to one ASCII space and trimming the ends, both
//! conventions produce identical ids. Normalizing the input therefore makes
//! the ambiguity unreachable, and is also ordinary input hygiene for a TTS
//! frontend: leading, trailing, and repeated whitespace carry no speech
//! meaning.

use splintr::AnyTokenizer;

use crate::error::{Error, Result};

/// Load a VoxCPM2 `tokenizer.json` into a splintr [`AnyTokenizer`], with no
/// normalization applied to input text.
///
/// Kept for callers that want raw control over what text reaches splintr.
/// Most VoxCPM2 text-path code should call [`tokenize`] instead, which
/// normalizes first and always uses `encode_raw`. See the module docs for
/// why both of those matter here.
pub fn load_tokenizer(path: impl AsRef<std::path::Path>) -> Result<AnyTokenizer> {
    splintr::from_json_path(path).map_err(|e| Error::ModelError {
        reason: format!("VoxCPM2 tokenizer.json: {e}"),
    })
}

/// Collapse every run of Unicode whitespace in `text` to a single ASCII
/// space, and trim leading/trailing whitespace.
///
/// Whitespace is anything `char::is_whitespace` accepts, so a non-breaking
/// space, an ideographic space, or a tab/newline run collapses the same way
/// a run of ASCII spaces does. See the module docs for why this
/// normalization is applied before tokenizing VoxCPM2 text.
pub fn normalize_whitespace(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut pending_space = false;
    for ch in text.chars() {
        if ch.is_whitespace() {
            if !out.is_empty() {
                pending_space = true;
            }
        } else {
            if pending_space {
                out.push(' ');
                pending_space = false;
            }
            out.push(ch);
        }
    }
    out
}

/// Normalize `text` with [`normalize_whitespace`] and tokenize it with
/// `tokenizer`, returning the raw ids.
///
/// Always calls `AnyTokenizer::encode_raw`, never `encode`: see the module
/// docs section "No-BOS is the caller's contract, not this function's" for
/// why `encode` would insert a BOS id the reference pipeline does not add.
pub fn tokenize(tokenizer: &AnyTokenizer, text: &str) -> Vec<u32> {
    tokenizer.encode_raw(&normalize_whitespace(text))
}

#[cfg(test)]
mod tests {
    use super::{normalize_whitespace, tokenize};
    use splintr::from_json_bytes;

    /// Miniature Llama-2-shaped tokenizer.json: metaspace normalizer
    /// (`Prepend "▁"` then `Replace " " -> "▁"`), `pre_tokenizer: null`,
    /// the same declared shape as VoxCPM2's real `tokenizer.json`. Trimmed
    /// from splintr's own `tests/hf_json_bpe_nosplit.rs` fixture, which
    /// tracks each expected id against `tokenizers`' `Tokenizer.from_file`.
    const MINI_TOKENIZER_JSON: &str = r#"{
      "version": "1.0",
      "truncation": null,
      "padding": null,
      "added_tokens": [
        {"id": 0, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
      ],
      "normalizer": {"type": "Sequence", "normalizers": [
        {"type": "Prepend", "prepend": "▁"},
        {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}
      ]},
      "pre_tokenizer": null,
      "post_processor": null,
      "decoder": {"type": "Sequence", "decoders": [
        {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
        {"type": "Strip", "content": " ", "start": 1, "stop": 0}
      ]},
      "model": {
        "type": "BPE",
        "dropout": null,
        "unk_token": "<unk>",
        "continuing_subword_prefix": null,
        "end_of_word_suffix": null,
        "fuse_unk": false,
        "byte_fallback": false,
        "vocab": {"<unk>": 0, "a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7, "h": 8, "i": 9, "j": 10, "k": 11, "l": 12, "m": 13, "n": 14, "o": 15, "p": 16, "q": 17, "r": 18, "s": 19, "t": 20, "u": 21, "v": 22, "w": 23, "x": 24, "y": 25, "z": 26, "▁": 27, "▁t": 28, "▁a": 29, "in": 30, "▁th": 31, "▁s": 32, "er": 33, "▁o": 34, "▁the": 35, "re": 36, "▁w": 37, "▁c": 38, "on": 39, "en": 40, "▁b": 41, "▁f": 42, "at": 43, "▁p": 44, "▁m": 45, "es": 46, "it": 47, "or": 48, "nd": 49, "is": 50, "▁h": 51, "ing": 52, "ed": 53, "ou": 54, "ar": 55, "▁d": 56, "▁in": 57, "al": 58, "▁to": 59, "an": 60, "▁of": 61, "▁and": 62, "le": 63, "ic": 64, "▁g": 65, "as": 66, "om": 67, "▁n": 68, "ion": 69, "▁re": 70, "▁l": 71, "il": 72, "▁e": 73, "ent": 74, "ve": 75, "ro": 76, "us": 77, "et": 78, "▁i": 79, "ac": 80, "▁y": 81, "ay": 82, "▁be": 83, "▁on": 84, "▁for": 85, "id": 86, "ly": 87, "▁wh": 88, "oo": 89},
        "merges": ["▁ t", "▁ a", "i n", "▁t h", "▁ s", "e r", "▁ o", "▁th e", "r e", "▁ w", "▁ c", "o n", "e n", "▁ b", "▁ f", "a t", "▁ p", "▁ m", "e s", "i t", "o r", "n d", "i s", "▁ h", "in g", "e d", "o u", "a r", "▁ d", "▁ in", "a l", "▁t o", "a n", "▁o f", "▁a nd", "l e", "i c", "▁ g", "a s", "o m", "▁ n", "i on", "▁ re", "▁ l", "i l", "▁ e", "en t", "v e", "r o", "u s", "e t", "▁ i", "a c", "▁ y", "a y", "▁b e", "▁o n", "▁f or", "i d", "l y", "▁w h", "o o"]
      }
    }"#;

    #[test]
    fn tokenize_on_empty_and_whitespace_only_input_does_not_panic() {
        let tokenizer =
            from_json_bytes(MINI_TOKENIZER_JSON.as_bytes()).expect("mini tokenizer.json loads");
        assert_eq!(tokenize(&tokenizer, ""), Vec::<u32>::new());
        assert_eq!(tokenize(&tokenizer, "   \t\n  "), Vec::<u32>::new());
        // Sanity: a real word still tokenizes once normalized.
        assert_eq!(tokenize(&tokenizer, "  the  "), vec![35]);
    }

    #[test]
    fn collapses_interior_whitespace_runs() {
        assert_eq!(normalize_whitespace("  a   b  "), "a b");
    }

    #[test]
    fn collapses_mixed_whitespace_kinds_to_one_space() {
        // tab, newline, and NBSP (U+00A0) in one run.
        assert_eq!(normalize_whitespace("a\t\n\u{00A0}b"), "a b");
    }

    #[test]
    fn leaves_already_clean_string_unchanged() {
        assert_eq!(normalize_whitespace("a b c"), "a b c");
    }

    #[test]
    fn does_not_alter_interior_punctuation_or_non_latin_text() {
        let text = "Hello, world! \u{4f60}\u{597d}\u{ff0c}\u{4e16}\u{754c}\u{ff01}";
        assert_eq!(normalize_whitespace(text), text);
    }

    #[test]
    fn empty_and_whitespace_only_input_produce_empty_string() {
        assert_eq!(normalize_whitespace(""), "");
        assert_eq!(normalize_whitespace("   \t\n  "), "");
    }
}
