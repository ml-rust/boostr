//! [`LabelTokenHead`]: the System One label-token readout. Each option of
//! a question carries a label that is one token, and the option's logit is
//! that token's logit at the question's answer slot.
//!
//! A scoring row is one [`LabelTokenItem`]: the gathered hidden row of its
//! answer slot and the label token of each option. An isolated Score
//! question takes one item per level, labels `[no, yes]`.

use crate::error::{Error, Result};
use crate::inference::decision::{DecisionBackbone, DecisionHead, GatherIndexed, OptionLogits};
use crate::model::traits::ModelClient;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::collections::HashSet;

/// One scoring row of a label-token readout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabelTokenItem {
    /// The row of the gathered hidden rows that holds the answer slot: the
    /// index of its gather in plan order.
    pub slot: usize,
    /// The label token id of each option, in option order. Non-empty, no
    /// repeats, each below the backbone's vocabulary size.
    pub labels: Vec<u32>,
}

/// Merging requests moves `slot` past the earlier requests' gathers.
impl GatherIndexed for LabelTokenItem {
    fn offset_gathers(mut self, by: usize) -> Self {
        self.slot += by;
        self
    }
}

/// Option logits as label-token logits at answer slots.
///
/// All rows go through one `candidate_logits` call: the items' hidden rows
/// are gathered on the device, the label ids form one `[Q, K_max]` matrix
/// (a short row repeats its first label), and the valid entries are
/// selected on the device into [`OptionLogits`]. Only host-built index
/// vectors cross to the device.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LabelTokenHead;

fn invalid(arg: &'static str, reason: String) -> Error {
    Error::InvalidArgument { arg, reason }
}

/// `items` read rows of a `[rows, _]` hidden tensor with ids below `vocab`.
fn check_items(items: &[LabelTokenItem], rows: usize, vocab: usize) -> Result<()> {
    if items.is_empty() {
        return Err(invalid(
            "items",
            "no items: a label-token readout scores at least one row".into(),
        ));
    }
    for (n, item) in items.iter().enumerate() {
        if item.slot >= rows {
            return Err(invalid(
                "items",
                format!(
                    "item {n} reads hidden row {} but the plan gathered {rows} rows",
                    item.slot
                ),
            ));
        }
        if item.labels.is_empty() {
            return Err(invalid("items", format!("item {n} has no labels")));
        }
        let mut seen = HashSet::with_capacity(item.labels.len());
        for (k, &id) in item.labels.iter().enumerate() {
            if id as usize >= vocab {
                return Err(invalid(
                    "items",
                    format!(
                        "item {n} label {k} is token id {id}, outside the vocabulary [0, {vocab})"
                    ),
                ));
            }
            if !seen.insert(id) {
                return Err(invalid(
                    "items",
                    format!(
                        "item {n} label {k} repeats token id {id}; two options with one label \
                         cannot be told apart"
                    ),
                ));
            }
        }
    }
    Ok(())
}

impl<R: Runtime<DType = DType>> DecisionHead<R> for LabelTokenHead {
    type Item = LabelTokenItem;

    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when `hidden` is not 2-D, when `items` is
    /// empty, or when an item reads a row past `hidden`, has no labels,
    /// repeats a label, or names an id at or past
    /// [`DecisionBackbone::vocab_size`]; each names the item and the value.
    /// Backbone and device errors propagate.
    fn option_logits<C: ModelClient<R>, B: DecisionBackbone<R>>(
        &self,
        client: &C,
        backbone: &B,
        hidden: &Tensor<R>,
        items: &[LabelTokenItem],
    ) -> Result<OptionLogits<R>> {
        let shape = hidden.shape();
        if shape.len() != 2 {
            return Err(invalid(
                "hidden",
                format!("must be [rows, hidden_size], got {shape:?}"),
            ));
        }
        let rows = shape[0];
        check_items(items, rows, backbone.vocab_size())?;
        let counts: Vec<usize> = items.iter().map(|item| item.labels.len()).collect();
        let q = items.len();
        let k_max = counts.iter().copied().max().unwrap_or(1);
        let device = hidden.device();

        let in_order = q == rows && items.iter().enumerate().all(|(n, it)| it.slot == n);
        let picked = if in_order {
            hidden.clone()
        } else {
            let slots: Vec<i64> = items.iter().map(|item| item.slot as i64).collect();
            let slots = Tensor::<R>::from_slice(&slots, &[q], device)?;
            client.index_select(hidden, 0, &slots)?
        };

        let mut ids = Vec::with_capacity(q * k_max);
        for item in items {
            ids.extend(item.labels.iter().map(|&id| i64::from(id)));
            ids.extend(std::iter::repeat_n(
                i64::from(item.labels[0]),
                k_max - item.labels.len(),
            ));
        }
        let ids = Tensor::<R>::from_slice(&ids, &[q, k_max], device)?;
        let logits = backbone.candidate_logits(client, &picked, &ids)?;
        let all = logits.contiguous()?.reshape(&[q * k_max])?;
        if counts.iter().all(|&k| k == k_max) {
            return Ok(OptionLogits { flat: all, counts });
        }
        let keep: Vec<i64> = counts
            .iter()
            .enumerate()
            .flat_map(|(n, &k)| (0..k).map(move |j| (n * k_max + j) as i64))
            .collect();
        let keep = Tensor::<R>::from_slice(&keep, &[keep.len()], device)?;
        Ok(OptionLogits {
            flat: client.index_select(&all, 0, &keep)?,
            counts,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::decision::answer::{AnswerOptions, answer_from_probs};
    use crate::inference::decision::run::gather::gather_plan_hidden;
    use crate::inference::decision::run::runner::tests::{
        PREFIX, at, close, model, plan, sequence, tokens,
    };
    use crate::inference::decision::temperature::{TemperaturePolicy, tempered_softmax};
    use crate::inference::decision::types::{Answer, QuestionKind, QuestionMeta};
    use crate::inference::decision::{DecisionPlan, PrefixInput, RunnerConfig, run_decision};
    use crate::model::qwen35::Qwen35Model;
    use crate::test_utils::cpu_setup;
    use numr::runtime::RuntimeClient;
    use numr::runtime::cpu::{CpuClient, CpuRuntime};

    fn item(slot: usize, labels: &[u32]) -> LabelTokenItem {
        LabelTokenItem {
            slot,
            labels: labels.to_vec(),
        }
    }

    /// Ragged label sets, one per gather of `plan`: gather `g` scores
    /// `1 + g % 4` distinct ids.
    fn ragged(plan: &DecisionPlan, vocab: usize) -> Vec<LabelTokenItem> {
        (0..plan.gathers.len())
            .map(|g| {
                let labels = (0..1 + g % 4)
                    .map(|j| ((g * 3 + j * 5) % vocab) as u32)
                    .collect();
                LabelTokenItem { slot: g, labels }
            })
            .collect()
    }

    /// The full forward's logits at `pos` of `toks` run alone.
    fn full_logits(
        client: &CpuClient,
        model: &Qwen35Model<CpuRuntime>,
        toks: &[u32],
        pos: usize,
    ) -> Vec<f32> {
        let device = client.device();
        let vocab = model.config().vocab_size;
        let (mut kv, mut gdn) = model.lane_caches(1, toks.len(), device).unwrap();
        let ids: Vec<i64> = toks.iter().map(|&t| i64::from(t)).collect();
        let ids = Tensor::<CpuRuntime>::from_slice(&ids, &[1, toks.len()], device).unwrap();
        let all = model
            .forward_qwen35(client, &ids, &mut kv, &mut gdn, 0)
            .unwrap()
            .to_vec::<f32>();
        all[pos * vocab..(pos + 1) * vocab].to_vec()
    }

    fn err(
        client: &CpuClient,
        model: &Qwen35Model<CpuRuntime>,
        hidden: &Tensor<CpuRuntime>,
        items: &[LabelTokenItem],
    ) -> String {
        let Err(e) = LabelTokenHead.option_logits(client, model, hidden, items) else {
            panic!("items {items:?} must be refused");
        };
        e.to_string()
    }

    #[test]
    fn bad_items_are_refused_with_their_values() {
        let (client, device) = cpu_setup();
        let model = model(&device);
        let vocab = model.config().vocab_size;
        let hidden = Tensor::<CpuRuntime>::ones(&[3, 8], DType::F32, &device).unwrap();
        let e = err(&client, &model, &hidden, &[]);
        assert!(e.contains("no items"), "{e}");
        let e = err(&client, &model, &hidden, &[item(0, &[1]), item(3, &[2])]);
        assert!(
            e.contains("item 1 reads hidden row 3 but the plan gathered 3 rows"),
            "{e}"
        );
        let e = err(&client, &model, &hidden, &[item(1, &[])]);
        assert!(e.contains("item 0 has no labels"), "{e}");
        let past = vocab as u32;
        let e = err(&client, &model, &hidden, &[item(2, &[1, past])]);
        let want =
            format!("item 0 label 1 is token id {past}, outside the vocabulary [0, {vocab})");
        assert!(e.contains(&want), "{e}");
        let e = err(
            &client,
            &model,
            &hidden,
            &[item(0, &[1]), item(1, &[4, 7, 4])],
        );
        assert!(e.contains("item 1 label 2 repeats token id 4"), "{e}");
        let flat = Tensor::<CpuRuntime>::ones(&[8], DType::F32, &device).unwrap();
        let e = err(&client, &model, &flat, &[item(0, &[1])]);
        assert!(e.contains("must be [rows, hidden_size]"), "{e}");
    }

    /// Every label logit equals the full forward's logit of that token at
    /// the item's slot, and one call over ragged items equals one call per
    /// item, in plan order and in a permuted order that repeats a slot.
    #[test]
    fn label_logits_match_full_forward_and_single_items() {
        let (client, device) = cpu_setup();
        let model = model(&device);
        let vocab = model.config().vocab_size;
        let plan = plan(PREFIX);
        let cfg = RunnerConfig::default();
        let hidden = gather_plan_hidden(&client, &model, &plan, None, &cfg).unwrap();
        let items = ragged(&plan, vocab);
        let mut orders = vec![items.clone()];
        let mut shuffled: Vec<LabelTokenItem> = items.iter().rev().cloned().collect();
        shuffled.push(items[2].clone());
        orders.push(shuffled);

        for order in &orders {
            let out = LabelTokenHead
                .option_logits(&client, &model, &hidden, order)
                .unwrap();
            let counts: Vec<usize> = order.iter().map(|it| it.labels.len()).collect();
            assert_eq!(out.counts, counts);
            let flat = out.flat.to_vec::<f32>();
            assert_eq!(flat.len(), counts.iter().sum::<usize>());
            let mut start = 0;
            for it in order {
                let got = &flat[start..start + it.labels.len()];
                start += it.labels.len();
                let gather = &plan.gathers[it.slot];
                let full = full_logits(&client, &model, &sequence(&plan, gather), gather.pos);
                let want: Vec<f32> = it.labels.iter().map(|&id| full[id as usize]).collect();
                close(
                    got,
                    &want,
                    1e-4,
                    &format!("slot {} vs full forward", it.slot),
                );
                let alone = LabelTokenHead
                    .option_logits(&client, &model, &hidden, std::slice::from_ref(it))
                    .unwrap();
                assert_eq!(alone.counts, vec![it.labels.len()]);
                close(
                    got,
                    &alone.flat.to_vec::<f32>(),
                    1e-6,
                    &format!("slot {} vs alone", it.slot),
                );
            }
        }
    }

    /// The probabilities `answer` carries: the distribution, or `[p_yes]`.
    fn answer_probs(answer: &Answer) -> Vec<f64> {
        match answer {
            Answer::Choice { probs, .. } | Answer::Score { probs, .. } => probs.clone(),
            Answer::Noul { p_yes, .. } => vec![*p_yes],
        }
    }

    /// Choice, Noul, a listed Score, an isolated Score over three level
    /// rows and a Choice read in the prefix, through `run_decision`,
    /// against host answers from full-forward logits.
    #[test]
    fn run_decision_answers_every_kind() {
        let (client, device) = cpu_setup();
        let model = model(&device);
        let lens = [5usize, 3, 7, 4, 4, 5];
        let rows: Vec<Vec<u32>> = lens
            .iter()
            .enumerate()
            .map(|(r, &n)| tokens(n, r + 1))
            .collect();
        let mut gathers: Vec<_> = lens
            .iter()
            .enumerate()
            .map(|(r, &n)| at(Some(r), PREFIX + n - 1))
            .collect();
        gathers.push(at(None, PREFIX - 1));
        let plan = DecisionPlan {
            prefix: PrefixInput::Tokens(tokens(PREFIX, 0)),
            rows,
            gathers,
        };
        let labels: [&[u32]; 7] = [
            &[3, 7, 11],
            &[2, 9],
            &[1, 4, 6, 13],
            &[5, 8],
            &[5, 8],
            &[5, 8],
            &[0, 15],
        ];
        let items: Vec<LabelTokenItem> =
            labels.iter().enumerate().map(|(g, l)| item(g, l)).collect();
        let isolated = QuestionKind::Score {
            values: None,
            isolated: true,
        };
        let meta = vec![
            QuestionMeta {
                kind: QuestionKind::Choice,
                options: 3,
            },
            QuestionMeta {
                kind: QuestionKind::Noul { yes: 1 },
                options: 2,
            },
            QuestionMeta {
                kind: QuestionKind::Score {
                    values: None,
                    isolated: false,
                },
                options: 4,
            },
            QuestionMeta {
                kind: isolated,
                options: 3,
            },
            QuestionMeta {
                kind: QuestionKind::Choice,
                options: 2,
            },
        ];
        let temp = TemperaturePolicy::ByKind {
            choice: 0.7,
            noul: 1.4,
            score: 1.2,
        };
        let opts = AnswerOptions::default();
        let cfg = RunnerConfig::default();
        let out = run_decision(
            &client,
            &model,
            &LabelTokenHead,
            &plan,
            None,
            &items,
            &meta,
            &temp,
            &opts,
            &cfg,
        )
        .unwrap();
        assert_eq!(out.temperatures, vec![0.7, 1.4, 1.2, 1.2, 1.2, 1.2, 0.7]);

        let raw: Vec<Vec<f32>> = plan
            .gathers
            .iter()
            .zip(&items)
            .map(|(g, it)| {
                let full = full_logits(&client, &model, &sequence(&plan, g), g.pos);
                it.labels.iter().map(|&id| full[id as usize]).collect()
            })
            .collect();
        for (r, want) in raw.iter().enumerate() {
            close(&out.raw_logits[r], want, 1e-4, &format!("scoring row {r}"));
        }
        let soft = |r: usize, t: f64| tempered_softmax(&raw[r], t).unwrap();
        let level_yes: Vec<f64> = (3..6).map(|r| soft(r, 1.2)[1]).collect();
        let want = [
            answer_from_probs(&meta[0], soft(0, 0.7), &opts).unwrap(),
            answer_from_probs(&meta[1], soft(1, 1.4), &opts).unwrap(),
            answer_from_probs(&meta[2], soft(2, 1.2), &opts).unwrap(),
            answer_from_probs(&meta[3], level_yes, &opts).unwrap(),
            answer_from_probs(&meta[4], soft(6, 0.7), &opts).unwrap(),
        ];
        assert_eq!(out.answers.len(), want.len());
        for (q, (got, want)) in out.answers.iter().zip(&want).enumerate() {
            assert_eq!(
                std::mem::discriminant(got),
                std::mem::discriminant(want),
                "question {q}"
            );
            let (g, w) = (answer_probs(got), answer_probs(want));
            assert_eq!(g.len(), w.len(), "question {q}");
            for (i, (a, b)) in g.iter().zip(&w).enumerate() {
                assert!((a - b).abs() < 1e-5, "question {q} entry {i}: {a} vs {b}");
            }
        }
        if let (
            Answer::Score {
                fit_mass: Some(a), ..
            },
            Answer::Score {
                fit_mass: Some(b), ..
            },
        ) = (&out.answers[3], &want[3])
        {
            assert!((a - b).abs() < 1e-5, "fit mass {a} vs {b}");
        } else {
            panic!("question 3 must be an isolated Score answer with a fit mass");
        }
    }
}
