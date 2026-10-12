//! Question metadata, answer shapes and the per-request decision result.

/// The kind of a decision question.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum QuestionKind {
    /// Pick one of the options.
    Choice,
    /// Yes or no over two options. `yes` is the index of the "yes" option.
    Noul {
        /// Index of the "yes" option (0 or 1).
        yes: usize,
    },
    /// An ordered scale of levels.
    Score {
        /// The value of each level. `None` gives levels the values `0..L`.
        values: Option<Vec<f64>>,
        /// Each level is read in its own yes/no row and the per-level yes
        /// probabilities are combined into one distribution.
        isolated: bool,
    },
}

/// What the answer stage needs to know about one question.
#[derive(Debug, Clone, PartialEq)]
pub struct QuestionMeta {
    /// The question kind.
    pub kind: QuestionKind,
    /// The number of options (levels for a Score question).
    pub options: usize,
}

/// One answered question.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Answer {
    /// The answer to a Choice question.
    Choice {
        /// The first option with the largest probability.
        index: usize,
        /// Chance-corrected confidence `(p_max - 1/K) / (1 - 1/K)` in `[0, 1]`.
        confidence: f64,
        /// The option distribution.
        probs: Vec<f64>,
    },
    /// The answer to a Noul question.
    Noul {
        /// The probability of "yes" (after the configured Noul map).
        p_yes: f64,
        /// The committed decision. `None` inside the abstention band, and
        /// always `Some` when no band is configured.
        decided: Option<bool>,
    },
    /// The answer to a Score question.
    Score {
        /// The expected value `sum_i p_i * v_i`.
        expected: f64,
        /// `expected` mapped to `[0, 1]` over the value range:
        /// `(expected - v_min) / (v_max - v_min)`.
        normalized: f64,
        /// The Score confidence in `[0, 1]` under the configured rule.
        confidence: f64,
        /// The level distribution.
        probs: Vec<f64>,
        /// The sum of the per-level yes probabilities. `Some` only for an
        /// isolated Score question.
        fit_mass: Option<f64>,
    },
}

/// The answers to every question of one request, with the inputs that produced them.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionResult {
    /// One answer per question, in question order.
    pub answers: Vec<Answer>,
    /// The raw option logits of each scoring row, before tempering.
    pub raw_logits: Vec<Vec<f32>>,
    /// The temperature applied to each scoring row.
    pub temperatures: Vec<f64>,
}
