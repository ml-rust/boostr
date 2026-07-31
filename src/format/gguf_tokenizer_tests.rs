//! Vocabulary normalization and merge-rank construction for GGUF tokenizers.

use super::{build_merge_ranks, normalize_wordpiece_vocab};
use crate::format::gguf::value::GgufValue;

fn v(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| (*s).to_owned()).collect()
}

// ── WordPiece vocab normalization ────────────────────────────────────────────

/// The `nomic-embed-text-v1.5` shape: `▁`-marked word-initial pieces, bare
/// continuations, bracketed specials.
#[test]
fn sentencepiece_marked_bert_vocab_is_converted_to_wordpiece() {
    let got = normalize_wordpiece_vocab(v(&[
        "[PAD]", "[CLS]", "[SEP]", "[UNK]", "▁the", "▁hello", "s", "ing", "▁!", "▁1",
    ]));
    assert_eq!(
        got,
        v(&[
            "[PAD]", "[CLS]", "[SEP]", "[UNK]", "the", "hello", "##s", "##ing", "!", "1",
        ]),
        "▁X must become X, bare X must become ##X, specials must be untouched"
    );
}

/// A vocab already in WordPiece convention must be returned byte-identical —
/// otherwise fixing one model's tokenizer would break every other BERT GGUF.
#[test]
fn already_wordpiece_vocab_is_left_untouched() {
    let original = v(&["[PAD]", "[CLS]", "the", "##s", "hello", "!"]);
    assert_eq!(normalize_wordpiece_vocab(original.clone()), original);
}

/// Mixed marking (some `▁`, some `##`) means the file is already using the
/// WordPiece continuation marker, so rewriting would corrupt it.
#[test]
fn mixed_marking_is_left_untouched() {
    let original = v(&["▁the", "##s", "hello"]);
    assert_eq!(normalize_wordpiece_vocab(original.clone()), original);
}

/// No `▁` anywhere → nothing to convert.
#[test]
fn unmarked_vocab_is_left_untouched() {
    let original = v(&["the", "hello", "world"]);
    assert_eq!(normalize_wordpiece_vocab(original.clone()), original);
}

// ── Byte-level BPE merge ranks ───────────────────────────────────────────────

fn merges(items: &[&str]) -> Vec<GgufValue> {
    items
        .iter()
        .map(|s| GgufValue::String((*s).to_string()))
        .collect()
}

fn rank(map: &rustc_hash::FxHashMap<Vec<u8>, u32>, token: &str) -> u32 {
    *map.get(token.as_bytes())
        .unwrap_or_else(|| panic!("{token:?} has no merge rank"))
}

/// Merge priority comes from the `merges` list order, not from token id — the
/// two disagree in real vocabularies, and using ids silently changes every
/// tokenization.
#[test]
fn merge_priority_follows_list_order_not_token_id() {
    // Ids put "lo" before "he", but the merges list puts "he" first.
    let tokens = v(&["h", "e", "l", "o", "lo", "he", "hel", "hello"]);
    let ranks = build_merge_ranks(&merges(&["h e", "he l", "l o", "hel lo"]), &tokens);

    assert!(
        rank(&ranks, "he") < rank(&ranks, "lo"),
        "\"he\" is earlier in the merges list, so it must merge first regardless \
         of \"lo\" having the lower token id"
    );
    assert!(rank(&ranks, "he") < rank(&ranks, "hel"));
    assert!(rank(&ranks, "hel") < rank(&ranks, "hello"));
}

/// Single characters are never a merge result, so they must all rank below every
/// merge — multi-byte UTF-8 has to coalesce before any real merge runs.
#[test]
fn the_base_alphabet_outranks_every_merge() {
    let tokens = v(&["a", "b", "c", "ab", "abc"]);
    let ranks = build_merge_ranks(&merges(&["a b", "ab c"]), &tokens);

    let base_max = ["a", "b", "c"]
        .iter()
        .map(|t| rank(&ranks, t))
        .max()
        .unwrap();
    let merge_min = ["ab", "abc"].iter().map(|t| rank(&ranks, t)).min().unwrap();
    assert!(
        base_max < merge_min,
        "every base-alphabet token must rank below every merge"
    );
}

/// Byte-level tokens spell real spaces as `Ġ`, so only the first space in a
/// merge entry is the separator — splitting on all of them would corrupt any
/// merge involving a space token.
#[test]
fn only_the_first_space_separates_a_merge_entry() {
    let tokens = v(&["Ġ", "a", "Ġa"]);
    let ranks = build_merge_ranks(&merges(&["Ġ a"]), &tokens);
    assert!(
        ranks.contains_key("Ġa".as_bytes()),
        "the merge result must be the concatenation \"Ġa\""
    );
}

/// A merge naming a token that is not in the vocab must not displace or
/// renumber the ranks of tokens that are.
#[test]
fn merges_referencing_absent_tokens_do_not_disturb_the_rest() {
    let tokens = v(&["a", "b", "ab"]);
    let ranks = build_merge_ranks(&merges(&["a b", "z z"]), &tokens);
    assert!(rank(&ranks, "a") < rank(&ranks, "ab"));
    assert!(rank(&ranks, "b") < rank(&ranks, "ab"));
}
