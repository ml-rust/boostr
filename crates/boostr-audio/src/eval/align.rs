//! Levenshtein alignment and the error-rate counts it produces.

use crate::eval::normalize::normalize;

/// Counts behind one error rate, kept separately so they can be summed.
///
/// Aggregate a corpus by summing these and taking [`ErrorRate::rate`] once —
/// see [`total`]. Averaging per-utterance rates instead weights a three-word
/// utterance the same as a thirty-word one and is not the standard definition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ErrorRate {
    /// Reference tokens replaced by a different hypothesis token.
    pub substitutions: usize,
    /// Reference tokens absent from the hypothesis.
    pub deletions: usize,
    /// Hypothesis tokens absent from the reference.
    pub insertions: usize,
    /// Reference tokens matched exactly.
    pub hits: usize,
}

impl ErrorRate {
    /// Reference length: `hits + substitutions + deletions`.
    pub fn reference_len(&self) -> usize {
        self.hits + self.substitutions + self.deletions
    }

    /// `(S + D + I) / reference_len`.
    ///
    /// Can exceed 1.0 — insertions are unbounded by reference length, which is
    /// why a hallucinating ASR can score above 100%.
    ///
    /// An empty reference has no denominator. This returns `0.0` when the
    /// hypothesis is also empty (nothing was asked for, nothing was wrong) and
    /// [`f64::INFINITY`] when it is not (tokens appeared from nowhere). Prefer
    /// aggregating with [`total`] over a corpus, where one empty reference
    /// cannot poison the result.
    pub fn rate(&self) -> f64 {
        let denom = self.reference_len();
        let errors = self.substitutions + self.deletions + self.insertions;
        if denom == 0 {
            return if errors == 0 { 0.0 } else { f64::INFINITY };
        }
        errors as f64 / denom as f64
    }

    /// Sum of the three error counts.
    pub fn errors(&self) -> usize {
        self.substitutions + self.deletions + self.insertions
    }
}

impl std::ops::Add for ErrorRate {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            substitutions: self.substitutions + other.substitutions,
            deletions: self.deletions + other.deletions,
            insertions: self.insertions + other.insertions,
            hits: self.hits + other.hits,
        }
    }
}

impl std::iter::Sum for ErrorRate {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::default(), |a, b| a + b)
    }
}

/// Edit counts aligning `hypothesis` to `reference`, both already tokenized.
///
/// Every edit costs 1. Where several alignments share the minimum cost the
/// split across substitutions, deletions and insertions is one of the optimal
/// ones — the total is exact, the breakdown is not unique.
pub fn align<T: PartialEq>(reference: &[T], hypothesis: &[T]) -> ErrorRate {
    // Cell state is the full breakdown, not just the cost, so S/D/I fall out of
    // the same pass. `cost` is redundant with the counts but kept explicit to
    // keep the comparison below readable.
    #[derive(Clone, Copy, Default)]
    struct Cell {
        cost: usize,
        rate: ErrorRate,
    }

    let m = hypothesis.len();
    let mut prev: Vec<Cell> = Vec::with_capacity(m + 1);
    // Row 0: the whole hypothesis is insertions.
    for j in 0..=m {
        prev.push(Cell {
            cost: j,
            rate: ErrorRate {
                insertions: j,
                ..Default::default()
            },
        });
    }
    let mut curr: Vec<Cell> = vec![Cell::default(); m + 1];

    for (i, r) in reference.iter().enumerate() {
        // Column 0: the whole reference so far is deletions.
        curr[0] = Cell {
            cost: i + 1,
            rate: ErrorRate {
                deletions: i + 1,
                ..Default::default()
            },
        };
        for (j, h) in hypothesis.iter().enumerate() {
            let diagonal = prev[j];
            let mut best = if r == h {
                Cell {
                    cost: diagonal.cost,
                    rate: ErrorRate {
                        hits: diagonal.rate.hits + 1,
                        ..diagonal.rate
                    },
                }
            } else {
                Cell {
                    cost: diagonal.cost + 1,
                    rate: ErrorRate {
                        substitutions: diagonal.rate.substitutions + 1,
                        ..diagonal.rate
                    },
                }
            };

            let deletion = prev[j + 1];
            if deletion.cost + 1 < best.cost {
                best = Cell {
                    cost: deletion.cost + 1,
                    rate: ErrorRate {
                        deletions: deletion.rate.deletions + 1,
                        ..deletion.rate
                    },
                };
            }

            let insertion = curr[j];
            if insertion.cost + 1 < best.cost {
                best = Cell {
                    cost: insertion.cost + 1,
                    rate: ErrorRate {
                        insertions: insertion.rate.insertions + 1,
                        ..insertion.rate
                    },
                };
            }

            curr[j + 1] = best;
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[m].rate
}

/// Word error rate of `hypothesis` against `reference`, normalizing both.
pub fn word_error_rate(reference: &str, hypothesis: &str) -> ErrorRate {
    align(&normalize(reference), &normalize(hypothesis))
}

/// Character error rate of `hypothesis` against `reference`.
///
/// Characters are taken from the normalized token stream rejoined with single
/// spaces, so spacing differences in the raw text do not register but a
/// genuinely missing word still does.
pub fn character_error_rate(reference: &str, hypothesis: &str) -> ErrorRate {
    let r: Vec<char> = normalize(reference).join(" ").chars().collect();
    let h: Vec<char> = normalize(hypothesis).join(" ").chars().collect();
    align(&r, &h)
}

/// Corpus-level total over `(reference, hypothesis)` pairs.
///
/// Sums the counts and leaves the single division to [`ErrorRate::rate`]. This
/// is the standard corpus WER; it is NOT the mean of per-utterance rates.
pub fn total<'a, I, F>(pairs: I, mut metric: F) -> ErrorRate
where
    I: IntoIterator<Item = (&'a str, &'a str)>,
    F: FnMut(&str, &str) -> ErrorRate,
{
    pairs
        .into_iter()
        .map(|(r, h)| metric(r, h))
        .sum::<ErrorRate>()
}

/// Corpus-level total per group, over `(group_key, reference, hypothesis)`.
///
/// A number-bearing subset needs its own figure whenever the metric is scored
/// over [`normalize`], which leaves digits and number words unreconciled. This
/// is how a caller gets that subset scored without grouping the pairs by hand
/// first.
///
/// Each group sums its counts and divides once, the same corpus definition
/// [`total`] uses; averaging per-utterance rates within a group is still not
/// the standard WER. [`total`] over the same items, ungrouped, equals
/// [`grand_total`] over this map's values — both are sums of the same counts,
/// just grouped differently before the single division.
///
/// A `BTreeMap` orders by key so a report prints groups in a stable order
/// regardless of the input's iteration order.
pub fn by_group<'a, K, I, F>(items: I, mut metric: F) -> std::collections::BTreeMap<K, ErrorRate>
where
    K: Ord,
    I: IntoIterator<Item = (K, &'a str, &'a str)>,
    F: FnMut(&str, &str) -> ErrorRate,
{
    let mut groups: std::collections::BTreeMap<K, ErrorRate> = std::collections::BTreeMap::new();
    for (key, reference, hypothesis) in items {
        let rate = metric(reference, hypothesis);
        groups
            .entry(key)
            .and_modify(|acc| *acc = *acc + rate)
            .or_insert(rate);
    }
    groups
}

/// Overall figure across every group in `groups`.
///
/// This is exactly [`total`] over the same items before they were grouped —
/// both sum the same counts and divide once. It exists so a report scores the
/// groups a single time with [`by_group`] and derives the corpus-wide figure
/// from that map, instead of scoring every item a second time.
pub fn grand_total<K>(groups: &std::collections::BTreeMap<K, ErrorRate>) -> ErrorRate {
    groups.values().copied().sum::<ErrorRate>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_token_streams_are_all_hits() {
        let rate = word_error_rate("one two three", "one two three");
        assert_eq!(rate.hits, 3);
        assert_eq!(rate.errors(), 0);
        assert_eq!(rate.rate(), 0.0);
    }

    #[test]
    fn counts_split_into_substitution_deletion_insertion() {
        let rate = align(&["a", "b", "c"], &["a", "x", "c", "d"]);
        assert_eq!(rate.substitutions, 1);
        assert_eq!(rate.insertions, 1);
        assert_eq!(rate.deletions, 0);
        assert_eq!(rate.hits, 2);
    }

    #[test]
    fn an_empty_reference_with_output_has_no_denominator() {
        assert_eq!(word_error_rate("", "").rate(), 0.0);
        assert!(word_error_rate("", "hello").rate().is_infinite());
    }

    #[test]
    fn character_rate_ignores_spacing_in_the_raw_text() {
        assert_eq!(character_error_rate("a  b", "a b").errors(), 0);
    }

    #[test]
    fn grand_total_matches_the_ungrouped_total() {
        let items = [("x", "one two", "one two"), ("y", "three", "four")];
        let groups = by_group(items, word_error_rate);
        let flat = total([("one two", "one two"), ("three", "four")], word_error_rate);
        assert_eq!(grand_total(&groups), flat);
    }
}
