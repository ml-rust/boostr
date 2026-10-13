//! The decision runner: one prefix, many rows, one set of answers.
//!
//! The prefix runs once, batch 1 (`prefill_prefix`). The rows return in
//! plan order from `gather`, the head scores them, and `finish` turns the
//! scores into answers.

use super::backbone::{DecisionBackbone, PrefixState};
use super::finish::finish;
use super::gather::gather_plan_hidden;
use super::head::DecisionHead;
use super::packing::RunnerConfig;
use super::plan::DecisionPlan;
use crate::error::Result;
use crate::inference::decision::answer::AnswerOptions;
use crate::inference::decision::temperature::TemperaturePolicy;
use crate::inference::decision::types::{DecisionResult, QuestionMeta};
use crate::model::traits::ModelClient;
use numr::dtype::DType;
use numr::runtime::Runtime;

/// The prefix of `plan` run once, keeping the hidden rows its prefix
/// gathers read. A caller can keep the result and pass it to
/// [`run_decision`] for later plans with the same prefix tokens and prefix
/// gathers at kept positions.
///
/// # Errors
///
/// Plan errors as [`DecisionPlan::validate`]; backbone errors propagate.
pub fn prefill_prefix<R, C, B>(
    client: &C,
    backbone: &B,
    plan: &DecisionPlan,
) -> Result<PrefixState<R>>
where
    R: Runtime<DType = DType>,
    C: ModelClient<R>,
    B: DecisionBackbone<R>,
{
    plan.validate()?;
    backbone.prefill_prefix(client, plan.prefix.tokens(), &plan.prefix_positions())
}

/// Answers every question of `plan`.
///
/// Runs [`gather_plan_hidden`], hands the rows to `head` with `items`, and
/// answers `meta` from the head's option logits: per-question temperature
/// from `temp`, one device softmax, one read-back. See the `head` module
/// for the scoring-row layout an isolated Score question takes.
///
/// # Errors
///
/// As [`gather_plan_hidden`]; `Error::Cancelled` when `cfg.cancel` is set
/// before scoring; head errors propagate; `Error::InvalidArgument` when
/// the head's rows disagree with `meta`; temperature and answer errors
/// propagate.
#[allow(clippy::too_many_arguments)]
pub fn run_decision<R, C, B, H>(
    client: &C,
    backbone: &B,
    head: &H,
    plan: &DecisionPlan,
    prefix: Option<&PrefixState<R>>,
    items: &[H::Item],
    meta: &[QuestionMeta],
    temp: &TemperaturePolicy,
    opts: &AnswerOptions,
    cfg: &RunnerConfig,
) -> Result<DecisionResult>
where
    R: Runtime<DType = DType>,
    C: ModelClient<R>,
    B: DecisionBackbone<R>,
    H: DecisionHead<R>,
{
    let hidden = gather_plan_hidden(client, backbone, plan, prefix, cfg)?;
    cfg.check_cancel(|| "before scoring".into())?;
    let logits = head.option_logits(client, backbone, &hidden, items)?;
    finish(client, &logits, meta, temp, opts)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::head::OptionLogits;
    use super::super::packing::DEFAULT_PASS_TOKENS;
    use super::super::plan::{Gather, PrefixInput};
    use super::*;
    use crate::inference::decision::types::{Answer, QuestionKind};
    use crate::model::qwen35::Qwen35Model;
    use crate::model::qwen35::model::{Head, chunked_model};
    use crate::test_utils::cpu_setup;
    use numr::runtime::RuntimeClient;
    use numr::runtime::cpu::{CpuClient, CpuRuntime};
    use numr::tensor::Tensor;

    /// GDN chunk width, so prefixes and rows cross chunk boundaries.
    const CHUNK: usize = 4;
    pub(crate) const PREFIX: usize = 6;

    /// Scores gathered row `g` against its own candidate ids `items[g]`.
    struct TokenHead;

    impl DecisionHead<CpuRuntime> for TokenHead {
        type Item = Vec<i64>;

        fn option_logits<C: ModelClient<CpuRuntime>, B: DecisionBackbone<CpuRuntime>>(
            &self,
            client: &C,
            backbone: &B,
            hidden: &Tensor<CpuRuntime>,
            items: &[Vec<i64>],
        ) -> Result<OptionLogits<CpuRuntime>> {
            let mut parts = Vec::with_capacity(items.len());
            for (g, ids) in items.iter().enumerate() {
                let row = hidden.narrow(0, g, 1)?.contiguous()?;
                let k = ids.len();
                let ids = Tensor::<CpuRuntime>::from_slice(ids, &[1, k], hidden.device())?;
                parts.push(
                    backbone
                        .candidate_logits(client, &row, &ids)?
                        .reshape(&[k])?,
                );
            }
            let refs: Vec<&Tensor<CpuRuntime>> = parts.iter().collect();
            Ok(OptionLogits {
                flat: client.cat(&refs, 0)?,
                counts: items.iter().map(Vec::len).collect(),
            })
        }
    }

    pub(crate) fn model(device: &numr::runtime::cpu::CpuDevice) -> Qwen35Model<CpuRuntime> {
        chunked_model(device, 0x3535_0070, CHUNK, false, Head::Own)
    }

    pub(crate) fn tokens(n: usize, salt: usize) -> Vec<u32> {
        (0..n)
            .map(|t| ((salt * 7 + t * 5 + 2) % 15) as u32)
            .collect()
    }

    pub(crate) fn at(row: Option<usize>, pos: usize) -> Gather {
        Gather { row, pos }
    }

    /// Prefix 6; rows of 7, 3, 1, 5 and 9 tokens; gathers out of row
    /// order, two in the prefix, one at a row's first suffix token.
    pub(crate) fn plan(prefix: usize) -> DecisionPlan {
        let lens = [7usize, 3, 1, 5, 9];
        let p = prefix;
        let mut gathers = vec![
            at(Some(2), p),
            at(Some(0), p + 6),
            at(Some(4), p + 8),
            at(Some(1), p + 2),
            at(Some(0), p + 2),
            at(Some(3), p + 4),
            at(Some(4), p),
        ];
        if p > 0 {
            gathers.insert(1, at(None, p - 1));
            gathers.push(at(None, 0));
        }
        DecisionPlan {
            prefix: PrefixInput::Tokens(tokens(p, 0)),
            rows: lens
                .iter()
                .enumerate()
                .map(|(r, &n)| tokens(n, r + 1))
                .collect(),
            gathers,
        }
    }

    /// Ragged candidate ids: gather `g` scores `1 + g % 4` ids.
    pub(crate) fn items(plan: &DecisionPlan) -> Vec<Vec<i64>> {
        (0..plan.gathers.len())
            .map(|g| {
                (0..1 + g % 4)
                    .map(|j| ((g * 3 + j * 5) % 16) as i64)
                    .collect()
            })
            .collect()
    }

    fn metas(items: &[Vec<i64>]) -> Vec<QuestionMeta> {
        items
            .iter()
            .map(|i| QuestionMeta {
                kind: QuestionKind::Choice,
                options: i.len(),
            })
            .collect()
    }

    /// The hidden row at `pos` of `tokens` run alone: batch 1, no prefix
    /// sharing, no padding.
    pub(crate) fn alone(
        client: &CpuClient,
        model: &Qwen35Model<CpuRuntime>,
        toks: &[u32],
        pos: usize,
    ) -> Tensor<CpuRuntime> {
        let device = client.device();
        let (mut kv, mut gdn) = model.lane_caches(1, toks.len(), device).unwrap();
        let ids: Vec<i64> = toks.iter().map(|&t| i64::from(t)).collect();
        let ids = Tensor::<CpuRuntime>::from_slice(&ids, &[1, toks.len()], device).unwrap();
        let gather = Tensor::<CpuRuntime>::from_slice(&[pos as i64], &[1], device).unwrap();
        model
            .forward_qwen35_gather(client, &ids, &mut kv, &mut gdn, 0, &gather)
            .unwrap()
    }

    pub(crate) fn sequence(plan: &DecisionPlan, g: &Gather) -> Vec<u32> {
        let mut s = plan.prefix.tokens().to_vec();
        if let Some(r) = g.row {
            s.extend(&plan.rows[r]);
        }
        s
    }

    pub(crate) fn close(got: &[f32], want: &[f32], tol: f32, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() < tol, "{what} element {i}: {g} vs {w}");
        }
    }

    pub(crate) fn run<B: DecisionBackbone<CpuRuntime>>(
        client: &CpuClient,
        model: &B,
        plan: &DecisionPlan,
        prefix: Option<&PrefixState<CpuRuntime>>,
        cfg: &RunnerConfig,
    ) -> Result<DecisionResult> {
        let items = items(plan);
        let t = TemperaturePolicy::Fixed(1.3);
        let opts = AnswerOptions::default();
        run_decision(
            client,
            model,
            &TokenHead,
            plan,
            prefix,
            &items,
            &metas(&items),
            &t,
            &opts,
            cfg,
        )
    }

    /// One lane per pass, every row in one pass, a tight token budget and
    /// a byte budget of two of the widest lanes give the same logits and
    /// answers.
    #[test]
    fn packing_does_not_change_answers() {
        let (client, device) = cpu_setup();
        let model = model(&device);
        let plan = plan(PREFIX);
        let two_lanes = Some(2 * model.lane_bytes(9));
        let configs = [
            (1, DEFAULT_PASS_TOKENS, None),
            (64, DEFAULT_PASS_TOKENS, None),
            (64, 10, None),
            (64, DEFAULT_PASS_TOKENS, two_lanes),
        ];
        let results: Vec<DecisionResult> = configs
            .iter()
            .map(|&(max_lanes, pass_tokens, max_lane_bytes)| {
                let cfg = RunnerConfig {
                    max_lanes,
                    pass_tokens,
                    max_lane_bytes,
                    cancel: None,
                };
                run(&client, &model, &plan, None, &cfg).unwrap()
            })
            .collect();
        let probs = |a: &Answer| match a {
            Answer::Choice { probs, .. } => probs.iter().map(|&p| p as f32).collect::<Vec<_>>(),
            other => panic!("expected Choice, got {other:?}"),
        };
        for other in &results[1..] {
            for (g, raw) in results[0].raw_logits.iter().enumerate() {
                close(&other.raw_logits[g], raw, 1e-5, "raw logits");
                let a = probs(&results[0].answers[g]);
                close(&probs(&other.answers[g]), &a, 1e-5, "probabilities");
            }
        }
    }

    #[test]
    fn supplied_prefix_is_reused_or_refused() {
        let (client, device) = cpu_setup();
        let model = model(&device);
        let plan = plan(PREFIX);
        let cfg = RunnerConfig::default();
        let state = prefill_prefix(&client, &model, &plan).unwrap();
        let reused = run(&client, &model, &plan, Some(&state), &cfg).unwrap();
        let fresh = run(&client, &model, &plan, None, &cfg).unwrap();
        for (a, b) in reused.raw_logits.iter().zip(&fresh.raw_logits) {
            close(a, b, 1e-6, "reused prefix");
        }

        let mut shorter = plan.clone();
        shorter.prefix = PrefixInput::Tokens(tokens(PREFIX - 1, 0));
        shorter
            .gathers
            .retain(|g| g.row.is_some() && g.pos > PREFIX);
        for g in &mut shorter.gathers {
            g.pos -= 1;
        }
        let short_state = prefill_prefix(&client, &model, &shorter).unwrap();
        let err = run(&client, &model, &plan, Some(&short_state), &cfg)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("holds 5 positions, the plan's prefix has 6"),
            "{err}"
        );

        let mut rows_only = plan.clone();
        rows_only.gathers.retain(|g| g.row.is_some());
        let bare = prefill_prefix(&client, &model, &rows_only).unwrap();
        let err = run(&client, &model, &plan, Some(&bare), &cfg)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("prefix gather at position 0 is not among"),
            "{err}"
        );
    }

    #[test]
    fn plan_and_position_errors_surface() {
        let (client, device) = cpu_setup();
        let model = model(&device);
        let cfg = RunnerConfig::default();
        let mut pad = plan(PREFIX);
        pad.gathers.push(at(Some(2), PREFIX + 1));
        let err = run(&client, &model, &pad, None, &cfg)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("reads position 7 of row 2, past its last real token at 6"),
            "{err}"
        );

        let mut long = plan(PREFIX);
        let max = model.max_positions();
        long.rows.push(tokens(max - PREFIX + 1, 9));
        long.gathers.push(at(Some(5), PREFIX));
        let err = run(&client, &model, &long, None, &cfg)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains(&format!("the model addresses {max} positions")),
            "{err}"
        );
    }
}
