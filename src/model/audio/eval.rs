//! Error-rate metrics for ASR output and for the intelligibility leg of TTS eval.
//!
//! Two callers need this and neither should reimplement it:
//!
//! - **Corpus QC.** A transcript is training ground truth. Measuring an ASR's
//!   word error rate against a read script is how we learn whether an
//!   ASR-transcribed corpus is usable at all.
//! - **TTS eval.** Intelligibility of generated speech is ASR WER over that
//!   speech against the text it was asked to say.
//!
//! **CPU-only, no tensors.** Levenshtein over token sequences: two rows of
//! `O(hypothesis)` state, one pass per utterance.
//!
//! # Normalization is part of the metric, not a detail
//!
//! WER is meaningless without saying how text was normalized, because
//! normalization choices move the number by several points. [`normalize`]
//! documents every rule it applies. Two of them matter enough to state here:
//!
//! - **Hyphens become spaces**, so `ter-delete` scores as two tokens
//!   `["ter", "delete"]`. Malay affixes on English roots are usually written
//!   with a hyphen and transcribed without one; this rule makes
//!   `ter-delete`/`ter delete` a match and `ter-delete`/`terdelete` a
//!   two-error miss. There is no neutral choice here — it is recorded so the
//!   number can be interpreted.
//! - **Numbers are NOT converted between digits and words.** `250` and
//!   `two hundred and fifty` score as four errors, not zero. Converting them
//!   needs a per-language number-word engine, and getting it wrong silently
//!   flatters the ASR. Score number-bearing prompts as their own subset
//!   instead of normalizing the difference away.

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

/// Normalize `text` into comparison tokens.
///
/// Rules, applied in order:
///
/// 1. Lowercase (`char::to_lowercase`, so it is Unicode-aware, not ASCII-only).
/// 2. Every character that is neither alphanumeric nor `'` becomes a space.
///    This strips `.,?!:;"` and turns `-` and `/` into token boundaries.
/// 3. Leading and trailing `'` are trimmed from each token, so `'word'`
///    normalizes to `word` while `don't` keeps its apostrophe.
/// 4. Split on whitespace; empty tokens are dropped.
///
/// Digits are left as digits — see the module docs on why numbers are not
/// converted.
pub fn normalize(text: &str) -> Vec<String> {
    let spaced: String = text
        .chars()
        .flat_map(|c| {
            if c.is_alphanumeric() || c == '\'' {
                c.to_lowercase().collect::<Vec<char>>()
            } else {
                vec![' ']
            }
        })
        .collect();

    spaced
        .split_whitespace()
        .map(|t| t.trim_matches('\'').to_string())
        .filter(|t| !t.is_empty())
        .collect()
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

#[cfg(test)]
mod tests;
