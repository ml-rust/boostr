//! The pointer readout as a [`DecisionHead`]: option logits from the
//! gathered hidden rows of each question's decide and option tokens.

use super::head::{PointerHead, PointerItem, invalid};
use crate::error::Result;
use crate::inference::decision::{DecisionBackbone, DecisionHead, OptionLogits};
use crate::model::traits::ModelClient;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::collections::HashSet;

/// `items` read rows of a `[rows, _]` hidden tensor.
fn check_items(items: &[PointerItem], rows: usize) -> Result<()> {
    if items.is_empty() {
        return Err(invalid(
            "items",
            "no items: a pointer readout scores at least one question".into(),
        ));
    }
    let past = |n: usize, what: String, row: usize| {
        invalid(
            "items",
            format!("item {n} {what} reads hidden row {row} but the plan gathered {rows} rows"),
        )
    };
    for (n, item) in items.iter().enumerate() {
        if item.decide >= rows {
            return Err(past(n, "decide".into(), item.decide));
        }
        if item.options.is_empty() {
            return Err(invalid("items", format!("item {n} has no options")));
        }
        let mut seen = HashSet::with_capacity(item.options.len());
        for (j, &row) in item.options.iter().enumerate() {
            if row >= rows {
                return Err(past(n, format!("option {j}"), row));
            }
            if row == item.decide {
                return Err(invalid(
                    "items",
                    format!(
                        "item {n} option {j} reads hidden row {row}, the item's decide row; an \
                         option is read at its own closing token"
                    ),
                ));
            }
            if !seen.insert(row) {
                return Err(invalid(
                    "items",
                    format!(
                        "item {n} option {j} repeats hidden row {row}; two options read at one \
                         row cannot be told apart"
                    ),
                ));
            }
        }
    }
    Ok(())
}

/// Option logits `z_j = (k(h_j) . q(h_d)) / sqrt(dp)` per question, in F32.
///
/// The decide rows and option rows are gathered on the device, cast to
/// F32, and mapped by `q` and `k` in one matmul each. Each option's key is
/// dotted with its question's query, picked by an owner index, so a ragged
/// option count needs no padding. Only host-built index vectors cross to
/// the device. The logits are raw: the runner applies the temperature.
impl<R: Runtime<DType = DType>> DecisionHead<R> for PointerHead<R> {
    type Item = PointerItem;

    /// # Errors
    ///
    /// [`crate::error::Error::InvalidArgument`] when `hidden` is not `[rows,
    /// hidden_size]`, when `items` is empty, or when an item reads a row
    /// past `hidden`, has no options, repeats an option row, or reads an
    /// option at its decide row; each names the item and the value. Device
    /// errors propagate.
    fn option_logits<C: ModelClient<R>, B: DecisionBackbone<R>>(
        &self,
        client: &C,
        _backbone: &B,
        hidden: &Tensor<R>,
        items: &[PointerItem],
    ) -> Result<OptionLogits<R>> {
        let shape = hidden.shape();
        let d = self.hidden_size();
        if shape.len() != 2 || shape[1] != d {
            return Err(invalid(
                "hidden",
                format!("must be [rows, {d}], got {shape:?}"),
            ));
        }
        check_items(items, shape[0])?;
        let counts: Vec<usize> = items.iter().map(|item| item.options.len()).collect();
        let total: usize = counts.iter().sum();
        let device = hidden.device();

        let hidden = if hidden.dtype() == DType::F32 {
            hidden.clone()
        } else {
            client.cast(hidden, DType::F32)?
        };
        let decide: Vec<i64> = items.iter().map(|item| item.decide as i64).collect();
        let options: Vec<i64> = items
            .iter()
            .flat_map(|item| item.options.iter().map(|&row| row as i64))
            .collect();
        let owner: Vec<i64> = counts
            .iter()
            .enumerate()
            .flat_map(|(n, &k)| std::iter::repeat_n(n as i64, k))
            .collect();
        let decide = Tensor::<R>::from_slice(&decide, &[items.len()], device)?;
        let options = Tensor::<R>::from_slice(&options, &[total], device)?;
        let owner = Tensor::<R>::from_slice(&owner, &[total], device)?;

        let h_decide = client.index_select(&hidden, 0, &decide)?;
        let h_options = client.index_select(&hidden, 0, &options)?;
        let queries = client.matmul_bias(&h_decide, &self.q_weight_t, &self.q_bias)?;
        let keys = client.matmul_bias(&h_options, &self.k_weight_t, &self.k_bias)?;
        let owned = client.index_select(&queries, 0, &owner)?;
        let dots = client.sum(&client.mul(&keys, &owned)?, &[1], false)?;
        Ok(OptionLogits {
            flat: client.mul_scalar(&dots, self.scale())?,
            counts,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::head::tests::built;
    use super::*;
    use crate::inference::decision::answer::{AnswerOptions, answer_from_probs};
    use crate::inference::decision::request::merge::tests::choice;
    use crate::inference::decision::run::gather::gather_plan_hidden;
    use crate::inference::decision::run::runner::tests::{
        PREFIX, alone, close, model, plan, tokens,
    };
    use crate::inference::decision::temperature::{TemperaturePolicy, tempered_softmax};
    use crate::inference::decision::types::{Answer, DecisionResult, QuestionKind, QuestionMeta};
    use crate::inference::decision::{
        DecisionRequest, RowSlot, RunnerConfig, merge_requests, plan_from_rows, run_decision,
        split_result,
    };
    use crate::model::qwen35::Qwen35Model;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuClient, CpuRuntime};

    const DP: usize = 5;

    /// Small weights that differ entry to entry.
    fn values(n: usize) -> f32 {
        ((n * 37 + 11) % 23) as f32 / 40.0 - 0.275
    }

    fn head(model: &Qwen35Model<CpuRuntime>, temperature: f64) -> PointerHead<CpuRuntime> {
        built(DP, model.config().hidden_size, temperature, values)
    }

    fn item(decide: usize, options: &[usize]) -> PointerItem {
        PointerItem {
            decide,
            options: options.to_vec(),
        }
    }

    /// The logits of `it` computed on the host from the hidden rows `h`
    /// `[rows, d]` row-major and the weights `built` gives from `values`.
    fn host_logits(h: &[f32], d: usize, it: &PointerItem) -> Vec<f32> {
        let affine = |row: usize, off: usize| -> Vec<f64> {
            (0..DP)
                .map(|i| {
                    let dot: f64 = (0..d)
                        .map(|j| f64::from(values(off + i * d + j)) * f64::from(h[row * d + j]))
                        .sum();
                    dot + f64::from(values(off + 1000 + i))
                })
                .collect()
        };
        let q = affine(it.decide, 0);
        let scale = 1.0 / (DP as f64).sqrt();
        it.options
            .iter()
            .map(|&row| {
                let k = affine(row, 2000);
                (k.iter().zip(&q).map(|(a, b)| a * b).sum::<f64>() * scale) as f32
            })
            .collect()
    }

    fn err(
        client: &CpuClient,
        model: &Qwen35Model<CpuRuntime>,
        hidden: &Tensor<CpuRuntime>,
        items: &[PointerItem],
    ) -> String {
        let Err(e) = head(model, 1.0).option_logits(client, model, hidden, items) else {
            panic!("items {items:?} must be refused");
        };
        e.to_string()
    }

    #[test]
    fn bad_items_are_refused_with_their_values() {
        let (client, device) = cpu_setup();
        let model = model(&device);
        let d = model.config().hidden_size;
        let hidden = Tensor::<CpuRuntime>::ones(&[3, d], DType::F32, &device).unwrap();
        let e = err(&client, &model, &hidden, &[]);
        assert!(e.contains("no items"), "{e}");
        let e = err(&client, &model, &hidden, &[item(3, &[0])]);
        assert!(
            e.contains("item 0 decide reads hidden row 3 but the plan gathered 3 rows"),
            "{e}"
        );
        let e = err(&client, &model, &hidden, &[item(2, &[0]), item(0, &[1, 3])]);
        assert!(
            e.contains("item 1 option 1 reads hidden row 3 but the plan gathered 3 rows"),
            "{e}"
        );
        let e = err(&client, &model, &hidden, &[item(0, &[])]);
        assert!(e.contains("item 0 has no options"), "{e}");
        let e = err(&client, &model, &hidden, &[item(1, &[0, 1])]);
        assert!(
            e.contains("item 0 option 1 reads hidden row 1, the item's decide row"),
            "{e}"
        );
        let e = err(&client, &model, &hidden, &[item(2, &[0, 1, 0])]);
        assert!(e.contains("item 0 option 2 repeats hidden row 0"), "{e}");
        let wide = Tensor::<CpuRuntime>::ones(&[3, d + 1], DType::F32, &device).unwrap();
        let e = err(&client, &model, &wide, &[item(0, &[1])]);
        let want = format!("must be [rows, {d}], got [3, {}]", d + 1);
        assert!(e.contains(&want), "{e}");
    }

    /// Ragged option counts in one call equal the host computation and
    /// one call per item, in plan order and in a permuted order.
    #[test]
    fn pointer_logits_match_host_and_single_items() {
        let (client, device) = cpu_setup();
        let model = model(&device);
        let d = model.config().hidden_size;
        let plan = plan(PREFIX);
        let hidden =
            gather_plan_hidden(&client, &model, &plan, None, &RunnerConfig::default()).unwrap();
        let rows = plan.gathers.len();
        let h = hidden.to_vec::<f32>();
        let items: Vec<PointerItem> = [1usize, 3, 2, 5, 4]
            .iter()
            .enumerate()
            .map(|(n, &k)| {
                let decide = (3 * n + 1) % rows;
                let options = (1..=k).map(|j| (decide + 2 * j) % rows).collect();
                PointerItem { decide, options }
            })
            .collect();
        let head = head(&model, 1.0);
        let reversed: Vec<PointerItem> = items.iter().rev().cloned().collect();
        for order in [&items, &reversed] {
            let out = head.option_logits(&client, &model, &hidden, order).unwrap();
            let counts: Vec<usize> = order.iter().map(|it| it.options.len()).collect();
            assert_eq!(out.counts, counts);
            let flat = out.flat.to_vec::<f32>();
            assert_eq!(flat.len(), counts.iter().sum::<usize>());
            let mut start = 0;
            for it in order {
                let got = &flat[start..start + it.options.len()];
                start += it.options.len();
                close(
                    got,
                    &host_logits(&h, d, it),
                    1e-4,
                    &format!("{it:?} vs host"),
                );
                let one = head
                    .option_logits(&client, &model, &hidden, std::slice::from_ref(it))
                    .unwrap();
                close(
                    got,
                    &one.flat.to_vec::<f32>(),
                    1e-6,
                    &format!("{it:?} alone"),
                );
            }
        }
    }

    fn noul() -> QuestionMeta {
        QuestionMeta {
            kind: QuestionKind::Noul { yes: 1 },
            options: 2,
        }
    }

    fn listed_score(levels: usize) -> QuestionMeta {
        QuestionMeta {
            kind: QuestionKind::Score {
                values: None,
                isolated: false,
            },
            options: levels,
        }
    }

    /// One request in Kev's row form: each question's row is the state
    /// then its branch, read at the decide token (the row's last) and at
    /// each option's closing position.
    fn request(
        state: usize,
        branches: &[(usize, &[usize])],
        meta: Vec<QuestionMeta>,
        temp: TemperaturePolicy,
        min_prefix: usize,
    ) -> DecisionRequest<PointerItem> {
        let mut rows = Vec::new();
        let mut slots = Vec::new();
        for (q, &(len, ends)) in branches.iter().enumerate() {
            let mut row = tokens(state, 0);
            row.extend(tokens(len, q + 1));
            let last = row.len() - 1;
            slots.push(RowSlot { row: q, pos: last });
            slots.extend(ends.iter().map(|&e| RowSlot {
                row: q,
                pos: state + e,
            }));
            rows.push(row);
        }
        let built = plan_from_rows(rows, &slots, min_prefix).unwrap();
        let mut next = built.slot_gathers.iter().copied();
        let items = branches
            .iter()
            .map(|&(_, ends)| PointerItem {
                decide: next.next().unwrap(),
                options: next.by_ref().take(ends.len()).collect(),
            })
            .collect();
        DecisionRequest {
            plan: built.plan,
            items,
            meta,
            temp,
        }
    }

    fn run(
        model: &Qwen35Model<CpuRuntime>,
        head: &PointerHead<CpuRuntime>,
        req: &DecisionRequest<PointerItem>,
    ) -> DecisionResult {
        let (client, _) = cpu_setup();
        run_decision(
            &client,
            model,
            head,
            &req.plan,
            None,
            &req.items,
            &req.meta,
            &req.temp,
            &AnswerOptions::default(),
            &RunnerConfig::default(),
        )
        .unwrap()
    }

    /// The distribution an answer carries, or `[p_yes]`.
    fn answer_probs(answer: &Answer) -> Vec<f32> {
        match answer {
            Answer::Choice { probs, .. } | Answer::Score { probs, .. } => {
                probs.iter().map(|&p| p as f32).collect()
            }
            Answer::Noul { p_yes, .. } => vec![*p_yes as f32],
        }
    }

    /// A Choice, a Noul and a listed Score sharing a state prefix, run at
    /// the head's temperature: raw logits equal the host computation from
    /// each row run alone, and the answers equal the host answers.
    #[test]
    fn run_decision_answers_every_kind_at_the_head_temperature() {
        let (client, device) = cpu_setup();
        let model = model(&device);
        let d = model.config().hidden_size;
        let head = head(&model, 1.6);
        let temp = TemperaturePolicy::Fixed(head.temperature());
        let meta = vec![choice(3), noul(), listed_score(4)];
        let req = request(
            7,
            &[(9, &[2, 4, 6]), (6, &[1, 3]), (10, &[1, 3, 5, 7])],
            meta.clone(),
            temp,
            0,
        );
        assert!(req.plan.prefix_len() >= 7, "the state is the shared prefix");
        let out = run(&model, &head, &req);
        assert_eq!(out.temperatures, vec![1.6; 3]);
        let opts = AnswerOptions::default();
        for (q, (it, m)) in req.items.iter().zip(&meta).enumerate() {
            let rows: Vec<usize> = std::iter::once(it.decide)
                .chain(it.options.clone())
                .collect();
            let mut h = Vec::new();
            for &g in &rows {
                let gather = &req.plan.gathers[g];
                let row = gather.row.unwrap();
                let mut seq = req.plan.prefix.tokens().to_vec();
                seq.extend(&req.plan.rows[row]);
                h.extend(alone(&client, &model, &seq, gather.pos).to_vec::<f32>());
            }
            let local = PointerItem {
                decide: 0,
                options: (1..rows.len()).collect(),
            };
            let want = host_logits(&h, d, &local);
            close(
                &out.raw_logits[q],
                &want,
                1e-4,
                &format!("question {q} raw"),
            );
            let probs = tempered_softmax(&out.raw_logits[q], 1.6).unwrap();
            let host = answer_from_probs(m, probs, &opts).unwrap();
            assert_eq!(
                std::mem::discriminant(&out.answers[q]),
                std::mem::discriminant(&host),
                "question {q}"
            );
            close(
                &answer_probs(&out.answers[q]),
                &answer_probs(&host),
                1e-6,
                &format!("question {q} answer"),
            );
        }
    }

    /// Two prefix-free requests with different temperatures and kinds run
    /// merged give each request the result it gets alone.
    #[test]
    fn merged_run_matches_runs_alone() {
        let (_, device) = cpu_setup();
        let model = model(&device);
        let head = head(&model, 1.0);
        let a = request(
            5,
            &[(6, &[1, 3]), (8, &[0, 2, 4, 6])],
            vec![noul(), choice(4)],
            TemperaturePolicy::Fixed(0.7),
            usize::MAX,
        );
        let b = request(
            4,
            &[(7, &[1, 3, 5])],
            vec![listed_score(3)],
            TemperaturePolicy::Fixed(1.4),
            usize::MAX,
        );
        let merged = merge_requests(vec![a.clone(), b.clone()]).unwrap();
        assert_eq!(merged.request.items[2].decide, a.plan.gathers.len());
        let split = split_result(run(&model, &head, &merged.request), &merged.parts).unwrap();
        for (alone, got) in [run(&model, &head, &a), run(&model, &head, &b)]
            .iter()
            .zip(&split)
        {
            assert_eq!(got.temperatures, alone.temperatures);
            assert_eq!(got.answers.len(), alone.answers.len());
            for (g, w) in got.raw_logits.iter().zip(&alone.raw_logits) {
                close(g, w, 1e-5, "raw logits");
            }
            for (g, w) in got.answers.iter().zip(&alone.answers) {
                close(&answer_probs(g), &answer_probs(w), 1e-5, "answers");
            }
        }
    }
}
