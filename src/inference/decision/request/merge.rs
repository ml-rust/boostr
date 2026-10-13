//! Several requests with no shared prefix run as one: [`merge_requests`]
//! puts their rows, gathers, head items, questions and temperatures into
//! one [`DecisionRequest`], and [`MergedPart`] records where each request
//! sits so `split_result` can cut the one result back apart.

use crate::error::{Error, Result};
use crate::inference::decision::run::{DecisionPlan, Gather, PrefixInput};
use crate::inference::decision::temperature::TemperaturePolicy;
use crate::inference::decision::types::QuestionMeta;
use std::ops::Range;

/// A head item that names gathered rows by their plan index. Merging
/// moves a request's gathers after the earlier requests' gathers, and its
/// items move with them.
pub trait GatherIndexed {
    /// The item with every gather index it names raised by `by`.
    fn offset_gathers(self, by: usize) -> Self;
}

/// One decision request: what `run_decision` takes besides the model, the
/// head and the config.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionRequest<I> {
    /// The prefix, rows and gathers.
    pub plan: DecisionPlan,
    /// The head item of each scoring row, as the head defines them.
    pub items: Vec<I>,
    /// The questions, in answer order.
    pub meta: Vec<QuestionMeta>,
    /// The temperature policy of the questions.
    pub temp: TemperaturePolicy,
}

/// Where one request's parts sit inside a merged request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergedPart {
    /// Its plan rows.
    pub rows: Range<usize>,
    /// Its gathers.
    pub gathers: Range<usize>,
    /// Its head items.
    pub items: Range<usize>,
    /// Its questions, and so its answers.
    pub questions: Range<usize>,
    /// Its scoring rows, and so its raw logits and temperatures.
    pub scoring_rows: Range<usize>,
}

/// Requests merged into one, and the part of each.
#[derive(Debug, Clone, PartialEq)]
pub struct MergedRequest<I> {
    /// The merged request. Run it as any other.
    pub request: DecisionRequest<I>,
    /// The part of each input request, in input order.
    pub parts: Vec<MergedPart>,
}

fn invalid(arg: &'static str, reason: String) -> Error {
    Error::InvalidArgument { arg, reason }
}

/// The temperature policy of the merge: the shared policy when every
/// request has the same one, else each question's own temperature.
fn merged_temperature<I>(requests: &[DecisionRequest<I>]) -> Result<TemperaturePolicy> {
    let first = &requests[0].temp;
    if requests.iter().all(|r| r.temp == *first) {
        return Ok(first.clone());
    }
    let mut ts = Vec::new();
    for (n, req) in requests.iter().enumerate() {
        req.temp
            .check_questions(req.meta.len())
            .map_err(|e| invalid("requests", format!("request {n}: {e}")))?;
        for (q, m) in req.meta.iter().enumerate() {
            let t = req
                .temp
                .for_question(q, m)
                .map_err(|e| invalid("requests", format!("request {n} question {q}: {e}")))?;
            ts.push(t);
        }
    }
    Ok(TemperaturePolicy::PerQuestion(ts))
}

/// Merges `requests` into one request whose rows run as full rows.
///
/// Rows, gathers, items and questions follow request order. Each gather
/// moves to its row's merged index and keeps its position. Each item's
/// gather indices rise by the gathers of the earlier requests. Requests
/// with one temperature policy keep it; mixed policies become
/// [`TemperaturePolicy::PerQuestion`] with each question's resolved
/// temperature.
///
/// # Errors
///
/// [`Error::InvalidArgument`] when `requests` is empty, when a request
/// has a prefix (its rows continue its own prefix, which no other request
/// shares), when a request's plan fails [`DecisionPlan::validate`], or
/// when a temperature does not resolve; each names the request.
pub fn merge_requests<I: GatherIndexed>(
    requests: Vec<DecisionRequest<I>>,
) -> Result<MergedRequest<I>> {
    if requests.is_empty() {
        return Err(invalid("requests", "no requests to merge".into()));
    }
    for (n, req) in requests.iter().enumerate() {
        if !req.plan.prefix.is_empty() {
            return Err(invalid(
                "requests",
                format!(
                    "request {n} has a prefix of {} tokens; merged rows share no prefix, so a \
                     request with a prefix runs alone",
                    req.plan.prefix_len()
                ),
            ));
        }
        req.plan
            .validate()
            .map_err(|e| invalid("requests", format!("request {n}: {e}")))?;
    }
    let temp = merged_temperature(&requests)?;

    let mut parts = Vec::with_capacity(requests.len());
    let mut rows = Vec::new();
    let mut gathers = Vec::new();
    let mut items = Vec::new();
    let mut meta = Vec::new();
    let mut scoring = 0;
    for req in requests {
        let row_base = rows.len();
        let gather_base = gathers.len();
        let scoring_rows: usize = req.meta.iter().map(QuestionMeta::scoring_rows).sum();
        let part = MergedPart {
            rows: row_base..row_base + req.plan.rows.len(),
            gathers: gather_base..gather_base + req.plan.gathers.len(),
            items: items.len()..items.len() + req.items.len(),
            questions: meta.len()..meta.len() + req.meta.len(),
            scoring_rows: scoring..scoring + scoring_rows,
        };
        scoring += scoring_rows;
        rows.extend(req.plan.rows);
        gathers.extend(req.plan.gathers.into_iter().map(|g| Gather {
            row: g.row.map(|r| r + row_base),
            pos: g.pos,
        }));
        items.extend(
            req.items
                .into_iter()
                .map(|it| it.offset_gathers(gather_base)),
        );
        meta.extend(req.meta);
        parts.push(part);
    }
    let plan = DecisionPlan {
        prefix: PrefixInput::Tokens(Vec::new()),
        rows,
        gathers,
    };
    Ok(MergedRequest {
        request: DecisionRequest {
            plan,
            items,
            meta,
            temp,
        },
        parts,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::inference::decision::types::QuestionKind;

    /// A test item naming one gather.
    #[derive(Debug, Clone, PartialEq)]
    pub(crate) struct Item(pub(crate) usize);

    impl GatherIndexed for Item {
        fn offset_gathers(self, by: usize) -> Self {
            Item(self.0 + by)
        }
    }

    fn at(row: usize, pos: usize) -> Gather {
        Gather {
            row: Some(row),
            pos,
        }
    }

    pub(crate) fn choice(k: usize) -> QuestionMeta {
        QuestionMeta {
            kind: QuestionKind::Choice,
            options: k,
        }
    }

    pub(crate) fn isolated(levels: usize) -> QuestionMeta {
        QuestionMeta {
            kind: QuestionKind::Score {
                values: None,
                isolated: true,
            },
            options: levels,
        }
    }

    fn request(
        rows: Vec<Vec<u32>>,
        gathers: Vec<Gather>,
        items: Vec<usize>,
        meta: Vec<QuestionMeta>,
        temp: TemperaturePolicy,
    ) -> DecisionRequest<Item> {
        DecisionRequest {
            plan: DecisionPlan {
                prefix: PrefixInput::Tokens(Vec::new()),
                rows,
                gathers,
            },
            items: items.into_iter().map(Item).collect(),
            meta,
            temp,
        }
    }

    /// Request A: two rows, a 3-way Choice read twice (two items on one
    /// gather) and a 2-way Choice. Request B: one row, an isolated Score
    /// of 3 levels.
    pub(crate) fn two() -> Vec<DecisionRequest<Item>> {
        vec![
            request(
                vec![vec![1, 2, 3], vec![4, 5]],
                vec![at(1, 1), at(0, 2)],
                vec![0, 0, 1],
                vec![choice(3), choice(3), choice(2)],
                TemperaturePolicy::Fixed(0.5),
            ),
            request(
                vec![vec![6, 7, 8, 9]],
                vec![at(0, 1), at(0, 2), at(0, 3)],
                vec![0, 1, 2],
                vec![isolated(3)],
                TemperaturePolicy::Fixed(2.0),
            ),
        ]
    }

    #[test]
    fn merge_moves_rows_gathers_and_items() {
        let merged = merge_requests(two()).unwrap();
        let req = &merged.request;
        assert!(req.plan.prefix.is_empty());
        assert_eq!(req.plan.rows.len(), 3);
        assert_eq!(req.plan.rows[2], vec![6, 7, 8, 9]);
        assert_eq!(
            req.plan.gathers,
            vec![at(1, 1), at(0, 2), at(2, 1), at(2, 2), at(2, 3)]
        );
        req.plan.validate().unwrap();
        let items: Vec<usize> = req.items.iter().map(|i| i.0).collect();
        assert_eq!(items, vec![0, 0, 1, 2, 3, 4]);
        assert_eq!(req.meta.len(), 4);
        assert_eq!(
            req.temp,
            TemperaturePolicy::PerQuestion(vec![0.5, 0.5, 0.5, 2.0])
        );
        assert_eq!(
            merged.parts,
            vec![
                MergedPart {
                    rows: 0..2,
                    gathers: 0..2,
                    items: 0..3,
                    questions: 0..3,
                    scoring_rows: 0..3,
                },
                MergedPart {
                    rows: 2..3,
                    gathers: 2..5,
                    items: 3..6,
                    questions: 3..4,
                    scoring_rows: 3..6,
                },
            ]
        );
    }

    #[test]
    fn one_policy_is_kept() {
        let mut reqs = two();
        reqs[1].temp = TemperaturePolicy::Fixed(0.5);
        let merged = merge_requests(reqs).unwrap();
        assert_eq!(merged.request.temp, TemperaturePolicy::Fixed(0.5));
    }

    #[test]
    fn prefixed_and_bad_requests_are_refused() {
        let mut reqs = two();
        reqs[1].plan.prefix = PrefixInput::Tokens(vec![9, 9]);
        reqs[1].plan.gathers = vec![at(0, 2)];
        let e = merge_requests(reqs).unwrap_err().to_string();
        assert!(e.contains("request 1 has a prefix of 2 tokens"), "{e}");
        let mut reqs = two();
        reqs[0].plan.gathers.push(at(5, 0));
        let e = merge_requests(reqs).unwrap_err().to_string();
        assert!(e.contains("request 0:") && e.contains("names row 5"), "{e}");
        let mut reqs = two();
        reqs[1].temp = TemperaturePolicy::Fixed(0.0);
        let e = merge_requests(reqs).unwrap_err().to_string();
        assert!(e.contains("request 1 question 0"), "{e}");
        let none: Vec<DecisionRequest<Item>> = Vec::new();
        assert!(merge_requests(none).is_err());
    }
}
