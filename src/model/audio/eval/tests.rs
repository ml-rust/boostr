use super::*;

#[test]
fn identical_text_has_zero_errors() {
    let r = word_error_rate(
        "saya rasa lebih baik kita berbincang dulu",
        "saya rasa lebih baik kita berbincang dulu",
    );
    assert_eq!(r.errors(), 0);
    assert_eq!(r.hits, 7);
    assert_eq!(r.rate(), 0.0);
}

#[test]
fn punctuation_and_case_do_not_count_as_errors() {
    let r = word_error_rate(
        "Esok hari Khamis, dua belas haribulan. Jangan lupa!",
        "esok hari khamis dua belas haribulan jangan lupa",
    );
    assert_eq!(r.errors(), 0);
    assert_eq!(r.reference_len(), 8);
}

#[test]
fn one_of_each_edit_is_counted_separately() {
    // reference:  a b c d e
    // hypothesis: a x d e f   -> b->x substitution, c deletion, f insertion.
    //
    // The lengths are equal, so an all-substitution alignment also exists — but
    // it costs 4 against this one's 3, making the breakdown unique. Picking a
    // case where two optimal alignments tie would assert something `align` does
    // not promise (see its doc comment).
    let r = word_error_rate("a b c d e", "a x d e f");
    assert_eq!(r.substitutions, 1, "{r:?}");
    assert_eq!(r.deletions, 1, "{r:?}");
    assert_eq!(r.insertions, 1, "{r:?}");
    assert_eq!(r.hits, 3, "{r:?}");
    assert_eq!(r.reference_len(), 5);
    assert_eq!(r.rate(), 0.6);
}

#[test]
fn a_tied_alignment_reports_one_optimal_breakdown_with_the_right_total() {
    // `a b c d` vs `a x d e` is cost 3 two ways: three substitutions, or
    // substitution + deletion + insertion. `align` prefers the diagonal, so it
    // reports the former. The TOTAL is what the metric promises; the split is
    // not. Pinned so the tie-breaking preference cannot drift unnoticed.
    let r = word_error_rate("a b c d", "a x d e");
    assert_eq!(r.errors(), 3, "{r:?}");
    assert_eq!(r.substitutions, 3, "{r:?}");
    assert_eq!(r.hits, 1, "{r:?}");
}

#[test]
fn insertions_can_push_the_rate_above_one() {
    // A hallucinating ASR is not bounded at 100% WER, and clamping it would
    // hide exactly the failure mode we are looking for.
    let r = word_error_rate("satu", "satu dua tiga empat lima");
    assert_eq!(r.insertions, 4);
    assert_eq!(r.hits, 1);
    assert!(r.rate() > 1.0, "rate {} should exceed 1.0", r.rate());
    assert_eq!(r.rate(), 4.0);
}

#[test]
fn empty_reference_is_zero_only_when_the_hypothesis_is_also_empty() {
    assert_eq!(word_error_rate("", "").rate(), 0.0);
    assert!(word_error_rate("", "sesuatu").rate().is_infinite());
}

#[test]
fn deleting_everything_costs_the_whole_reference() {
    let r = word_error_rate("satu dua tiga", "");
    assert_eq!(r.deletions, 3);
    assert_eq!(r.hits, 0);
    assert_eq!(r.rate(), 1.0);
}

#[test]
fn hyphen_splits_a_malay_affix_off_its_english_root() {
    // Documented in the module docs: this is the rule that makes an ASR's
    // "ter delete" a match for the script's "ter-delete".
    assert_eq!(normalize("ter-delete"), vec!["ter", "delete"]);
    assert_eq!(
        word_error_rate("Fail tu dah ter-delete", "fail tu dah ter delete").errors(),
        0
    );
    // ...and the rule cuts both ways: a joined transcription is 2 errors, not 0.
    let joined = word_error_rate("Fail tu dah ter-delete", "fail tu dah terdelete");
    assert_eq!(joined.substitutions, 1, "{joined:?}");
    assert_eq!(joined.deletions, 1, "{joined:?}");
}

#[test]
fn apostrophes_survive_inside_a_word_but_not_around_it() {
    assert_eq!(normalize("don't"), vec!["don't"]);
    assert_eq!(normalize("'quoted'"), vec!["quoted"]);
    assert_eq!(word_error_rate("I don't know", "i don't know").errors(), 0);
}

#[test]
fn digits_are_not_reconciled_with_number_words() {
    // Deliberate: converting these needs a per-language number-word engine and
    // silently doing it flatters the ASR. Pinning the behaviour so nobody
    // "fixes" it into a normalizer without reading the module docs.
    let r = word_error_rate("two hundred and fifty ringgit", "250 ringgit");
    assert!(
        r.errors() > 0,
        "digit/word difference must register as errors"
    );
    assert_eq!(r.hits, 1, "only 'ringgit' matches: {r:?}");
}

#[test]
fn corpus_total_is_not_the_mean_of_per_utterance_rates() {
    // One short utterance wrong, one long utterance right. The mean of the two
    // rates is 0.5; the corpus rate is 1/5 = 0.2. The corpus rate is correct.
    let pairs = [("satu", "dua"), ("a b c d", "a b c d")];
    let agg = total(pairs, word_error_rate);
    assert_eq!(agg.reference_len(), 5);
    assert_eq!(agg.errors(), 1);
    assert_eq!(agg.rate(), 0.2);

    let mean: f64 = pairs
        .iter()
        .map(|(r, h)| word_error_rate(r, h).rate())
        .sum::<f64>()
        / 2.0;
    assert_eq!(mean, 0.5);
    assert_ne!(agg.rate(), mean);
}

#[test]
fn character_error_rate_counts_characters_not_words() {
    // One wrong letter in a 7-character reference is 1 CER error, but a whole
    // word wrong at the WER level.
    let cer = character_error_rate("kucing", "kucang");
    assert_eq!(cer.substitutions, 1, "{cer:?}");
    assert_eq!(cer.reference_len(), 6);
    let wer = word_error_rate("kucing", "kucang");
    assert_eq!(wer.substitutions, 1);
    assert_eq!(wer.reference_len(), 1);
    assert!(cer.rate() < wer.rate());
}

#[test]
fn code_switched_reference_scores_only_the_switched_span() {
    // The boundary case the eval set exists for: Malay frame intact, the
    // English insertion mis-transcribed. Errors must localize to the insertion.
    let r = word_error_rate(
        "Saya dah cuba semua cara, tapi honestly it just doesn't make any sense.",
        "saya dah cuba semua cara tapi honestly it just doesn't make any sent",
    );
    assert_eq!(r.substitutions, 1, "{r:?}");
    assert_eq!(r.deletions, 0, "{r:?}");
    assert_eq!(r.insertions, 0, "{r:?}");
}

#[test]
fn align_is_symmetric_in_cost_but_swaps_deletions_and_insertions() {
    let forward = word_error_rate("a b c", "a c");
    let backward = word_error_rate("a c", "a b c");
    assert_eq!(forward.errors(), backward.errors());
    assert_eq!(forward.deletions, 1);
    assert_eq!(backward.insertions, 1);
}
