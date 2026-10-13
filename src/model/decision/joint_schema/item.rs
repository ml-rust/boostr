//! [`JointSchemaItem`]: one field of a joint-schema record as the head
//! reads it, its [`FieldKind`] and [`JointOption`]s; the plan that reads a
//! whole record ([`full_sequence_plan`]); and the answer stage settings of
//! the head ([`JointSchemaItem::meta`], [`joint_answer_options`]).

use crate::error::{Error, Result};
use crate::inference::decision::{
    AnswerOptions, DecisionPlan, Gather, PrefixInput, QuestionKind, QuestionMeta, ScoreConfidence,
};
use std::ops::Range;

/// The type of a field, with its type-embedding row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldKind {
    /// Yes or no: options `true` then `false`.
    Noul,
    /// One of a set of options, in code-point order of their ids.
    Choice,
    /// One of an ordered scale of levels `0..n`.
    Score,
}

impl FieldKind {
    /// The row of the type embedding: 0 noul, 1 choice, 2 score.
    pub fn type_id(self) -> usize {
        match self {
            Self::Noul => 0,
            Self::Choice => 1,
            Self::Score => 2,
        }
    }
}

/// One option of a field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JointOption {
    /// The option's token positions in the record, half-open.
    pub span: Range<usize>,
    /// The token ids at `span`, whose output-projection rows form the
    /// option's lexical vector.
    pub ids: Vec<u32>,
}

/// One field of a record.
///
/// The head reads every row of the record's sequence: hand it the plan of
/// [`full_sequence_plan`], so hidden row `p` is position `p`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JointSchemaItem {
    /// The field type.
    pub kind: FieldKind,
    /// The positions of the field's instruction, half-open, non-empty.
    pub question: Range<usize>,
    /// The options in head order: `true`, `false` for a Noul field; the
    /// option ids in code-point order for a Choice field; levels `0..n`
    /// for a Score field. Non-empty; exactly two for a Noul field.
    pub options: Vec<JointOption>,
}

fn invalid(reason: String) -> Error {
    Error::InvalidArgument {
        arg: "items",
        reason,
    }
}

impl JointSchemaItem {
    /// A field whose option ids are read from the record's `tokens`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] as [`Self::check`] over `tokens.len()`
    /// positions.
    pub fn from_tokens(
        kind: FieldKind,
        question: Range<usize>,
        options: &[Range<usize>],
        tokens: &[u32],
    ) -> Result<Self> {
        let mut built = Vec::with_capacity(options.len());
        for (j, span) in options.iter().enumerate() {
            let ids = tokens.get(span.clone()).ok_or_else(|| {
                invalid(format!(
                    "option {j} span {span:?} is outside a record of {} tokens",
                    tokens.len()
                ))
            })?;
            built.push(JointOption {
                span: span.clone(),
                ids: ids.to_vec(),
            });
        }
        let item = Self {
            kind,
            question,
            options: built,
        };
        item.check(0, tokens.len())?;
        Ok(item)
    }

    /// Checks the field against a record of `len` positions: the
    /// instruction and every option span non-empty and inside it, each
    /// option's ids as long as its span, at least one option, and exactly
    /// two for a Noul field. `n` names the field in the errors.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] naming the field, the option and the
    /// value that fails.
    pub fn check(&self, n: usize, len: usize) -> Result<()> {
        let bad_span = |what: String, span: &Range<usize>| {
            invalid(format!(
                "field {n} {what} span {span:?} must be non-empty and inside a record of \
                 {len} positions"
            ))
        };
        if self.question.start >= self.question.end || self.question.end > len {
            return Err(bad_span("instruction".into(), &self.question));
        }
        if self.options.is_empty() {
            return Err(invalid(format!("field {n} has no options")));
        }
        if self.kind == FieldKind::Noul && self.options.len() != 2 {
            return Err(invalid(format!(
                "field {n} is a Noul field with {} options; it has true and false",
                self.options.len()
            )));
        }
        for (j, option) in self.options.iter().enumerate() {
            let span = &option.span;
            if span.start >= span.end || span.end > len {
                return Err(bad_span(format!("option {j}"), span));
            }
            if option.ids.len() != span.len() {
                return Err(invalid(format!(
                    "field {n} option {j} holds {} ids for a span of {} positions",
                    option.ids.len(),
                    span.len()
                )));
            }
        }
        Ok(())
    }

    /// What the answer stage needs: a Noul field answers `p(true)`, option
    /// 0; a Choice field its distribution; a Score field of two or more
    /// levels the expected level index `sum_i i * p_i`. A Score field of
    /// one level answers as a one-option Choice: its level has probability
    /// 1, so its expected index is 0 and its confidence 1.
    pub fn meta(&self) -> QuestionMeta {
        let options = self.options.len();
        let kind = match self.kind {
            FieldKind::Noul => QuestionKind::Noul { yes: 0 },
            FieldKind::Choice => QuestionKind::Choice,
            FieldKind::Score if options == 1 => QuestionKind::Choice,
            FieldKind::Score => QuestionKind::Score {
                values: None,
                isolated: false,
            },
        };
        QuestionMeta { kind, options }
    }
}

/// The answer settings of the head: no Noul band or map, and a Score
/// confidence of the largest level probability. Pair them with
/// `TemperaturePolicy::Fixed(1.0)`: the head's probabilities are a plain
/// softmax of its logits.
///
/// A Choice answer's `confidence` stays chance-corrected; the head's own
/// confidence is the chosen option's probability, `probs[index]`.
pub fn joint_answer_options() -> AnswerOptions {
    AnswerOptions {
        score_confidence: ScoreConfidence::MaxProbability,
        ..AnswerOptions::default()
    }
}

/// The plan that reads every position of `tokens`, in order: the whole
/// record as the prefix, no rows, one gather per position.
///
/// # Errors
///
/// [`Error::InvalidArgument`] when `tokens` is empty.
pub fn full_sequence_plan(tokens: &[u32]) -> Result<DecisionPlan> {
    if tokens.is_empty() {
        return Err(Error::InvalidArgument {
            arg: "tokens",
            reason: "an empty record has no rows to read".into(),
        });
    }
    Ok(DecisionPlan {
        prefix: PrefixInput::Tokens(tokens.to_vec()),
        rows: Vec::new(),
        gathers: (0..tokens.len())
            .map(|pos| Gather { row: None, pos })
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens() -> Vec<u32> {
        (0..20).map(|t| 100 + t).collect()
    }

    #[test]
    fn from_tokens_reads_ids_and_checks_spans() {
        let item = JointSchemaItem::from_tokens(FieldKind::Choice, 2..4, &[5..8, 9..10], &tokens())
            .unwrap();
        assert_eq!(item.options[0].ids, vec![105, 106, 107]);
        assert_eq!(item.options[1].ids, vec![109]);

        let e = JointSchemaItem::from_tokens(
            FieldKind::Choice,
            2..4,
            std::slice::from_ref(&(5..21)),
            &tokens(),
        )
        .unwrap_err()
        .to_string();
        assert!(
            e.contains("option 0 span 5..21 is outside a record of 20"),
            "{e}"
        );
        let e = JointSchemaItem::from_tokens(
            FieldKind::Noul,
            2..4,
            std::slice::from_ref(&(5..8)),
            &tokens(),
        )
        .unwrap_err()
        .to_string();
        assert!(e.contains("Noul field with 1 options"), "{e}");
        let e = JointSchemaItem::from_tokens(
            FieldKind::Score,
            4..4,
            std::slice::from_ref(&(5..8)),
            &tokens(),
        )
        .unwrap_err()
        .to_string();
        assert!(e.contains("field 0 instruction span 4..4"), "{e}");
        let e = JointSchemaItem::from_tokens(FieldKind::Score, 1..4, &[], &tokens())
            .unwrap_err()
            .to_string();
        assert!(e.contains("field 0 has no options"), "{e}");
    }

    #[test]
    fn meta_per_kind() {
        let item = |kind, n: usize| JointSchemaItem {
            kind,
            question: 0..1,
            options: (0..n)
                .map(|j| JointOption {
                    span: j + 1..j + 2,
                    ids: vec![j as u32],
                })
                .collect(),
        };
        assert_eq!(
            item(FieldKind::Noul, 2).meta(),
            QuestionMeta {
                kind: QuestionKind::Noul { yes: 0 },
                options: 2
            }
        );
        assert_eq!(item(FieldKind::Choice, 1).meta().kind, QuestionKind::Choice);
        assert_eq!(item(FieldKind::Score, 1).meta().kind, QuestionKind::Choice);
        assert_eq!(
            item(FieldKind::Score, 5).meta(),
            QuestionMeta {
                kind: QuestionKind::Score {
                    values: None,
                    isolated: false
                },
                options: 5
            }
        );
        assert_eq!(FieldKind::Score.type_id(), 2);
    }

    #[test]
    fn plan_reads_every_position() {
        let plan = full_sequence_plan(&[7, 8, 9]).unwrap();
        plan.validate().unwrap();
        assert_eq!(plan.prefix_positions(), vec![0, 1, 2]);
        assert!(plan.rows.is_empty());
        assert!(full_sequence_plan(&[]).is_err());
    }
}
