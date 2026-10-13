//! Gathering a plan's hidden rows: the prefix once, then the rows in
//! passes of right-padded lanes that each start from the prefix's GDN
//! state and attend to the one prefix KV cache.

use super::backbone::{DecisionBackbone, PrefixState};
use super::packing::{RunnerConfig, pack_passes};
use super::plan::DecisionPlan;
use super::runner::prefill_prefix;
use crate::error::{Error, Result};
use crate::model::traits::ModelClient;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::collections::HashMap;

fn invalid(arg: &'static str, reason: String) -> Error {
    Error::InvalidArgument { arg, reason }
}

/// The row of `state.hidden` that holds each kept prefix position.
fn kept_rows<R: Runtime>(state: &PrefixState<R>) -> HashMap<usize, usize> {
    let mut slots = HashMap::with_capacity(state.rows.len());
    for (row, &pos) in state.rows.iter().enumerate() {
        slots.entry(pos).or_insert(row);
    }
    slots
}

/// `state` fits `plan`: same length, and every prefix gather kept.
fn check_supplied<R: Runtime>(
    state: &PrefixState<R>,
    plan: &DecisionPlan,
    slots: &HashMap<usize, usize>,
) -> Result<()> {
    if state.len != plan.prefix_len() {
        return Err(invalid(
            "prefix",
            format!(
                "the supplied PrefixState holds {} positions, the plan's prefix has {}",
                state.len,
                plan.prefix_len()
            ),
        ));
    }
    if let Some(pos) = plan
        .prefix_positions()
        .into_iter()
        .find(|p| !slots.contains_key(p))
    {
        return Err(invalid(
            "prefix",
            format!(
                "prefix gather at position {pos} is not among the rows the supplied \
                 PrefixState kept ({:?}); build it with prefill_prefix over this plan",
                state.rows
            ),
        ));
    }
    Ok(())
}

/// The normed hidden rows `[G, hidden_size]` at every gather of `plan`,
/// in plan order.
///
/// `prefix` reuses a cached prefix; `None` runs it here. Each pass gets
/// fresh lane caches, so nothing a padded lane computed is ever read.
///
/// # Errors
///
/// Plan errors as [`DecisionPlan::validate`]. [`Error::InvalidArgument`]
/// when a supplied `prefix` has the wrong length or lacks a prefix
/// gather's row, when the config is invalid, when one lane passes
/// `cfg.max_lane_bytes`, or when the prefix plus the longest read row
/// passes [`DecisionBackbone::max_positions`], naming the row.
/// [`Error::Cancelled`] when `cfg.cancel` is set before the prefix or
/// before a pass. Backbone errors propagate.
pub fn gather_plan_hidden<R, C, B>(
    client: &C,
    backbone: &B,
    plan: &DecisionPlan,
    prefix: Option<&PrefixState<R>>,
    cfg: &RunnerConfig,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ModelClient<R>,
    B: DecisionBackbone<R>,
{
    plan.validate()?;
    let passes = pack_passes(plan, cfg, |width| backbone.lane_bytes(width))?;
    cfg.check_cancel(|| "before the prefix".into())?;
    let owned;
    let supplied = prefix.is_some();
    let prefix = match prefix {
        Some(state) => state,
        None => {
            owned = prefill_prefix(client, backbone, plan)?;
            &owned
        }
    };
    let slots = kept_rows(prefix);
    if supplied {
        check_supplied(prefix, plan, &slots)?;
    }
    let max_pos = backbone.max_positions();
    if let Some(pass) = passes.first()
        && prefix.rope_end + pass.width > max_pos
    {
        return Err(invalid(
            "plan",
            format!(
                "row {} has {} tokens after a prefix ending at position {}; the model \
                 addresses {max_pos} positions",
                pass.rows[0], pass.width, prefix.rope_end
            ),
        ));
    }

    let device = client.device();
    let mut pieces: Vec<Tensor<R>> = Vec::with_capacity(passes.len() + 1);
    let mut order: Vec<usize> = Vec::with_capacity(plan.gathers.len());
    let mut kept: Vec<i64> = Vec::new();
    for (i, g) in plan.gathers.iter().enumerate() {
        if g.row.is_none() {
            let row = slots.get(&g.pos).copied().ok_or_else(|| {
                invalid(
                    "prefix",
                    format!("no hidden row kept at position {}", g.pos),
                )
            })?;
            kept.push(row as i64);
            order.push(i);
        }
    }
    if !kept.is_empty() {
        let hidden = prefix.hidden.as_ref().ok_or_else(|| {
            invalid(
                "prefix",
                "the PrefixState kept rows but no hidden tensor".into(),
            )
        })?;
        let index = Tensor::<R>::from_slice(&kept, &[kept.len()], device)?;
        pieces.push(client.index_select(hidden, 0, &index)?);
    }

    for (i, pass) in passes.iter().enumerate() {
        cfg.check_cancel(|| format!("before pass {} of {}", i + 1, passes.len()))?;
        let lanes = pass.rows.len();
        let (mut kv, mut gdn) = backbone.lane_caches(lanes, pass.width, device)?;
        if prefix.len > 0 {
            for lane in 0..lanes {
                gdn.install_lane(client, lane, &prefix.gdn)?;
            }
        }
        let ids = Tensor::<R>::from_slice(&pass.ids(plan), &[lanes, pass.width], device)?;
        let gather = Tensor::<R>::from_slice(&pass.flat, &[pass.flat.len()], device)?;
        pieces.push(backbone.forward_rows(client, &ids, prefix, &mut gdn, &mut kv, &gather)?);
        order.extend(&pass.gathers);
    }

    let all = if pieces.len() == 1 {
        pieces.swap_remove(0)
    } else {
        let refs: Vec<&Tensor<R>> = pieces.iter().collect();
        client.cat(&refs, 0)?
    };
    if order.iter().enumerate().all(|(k, &g)| k == g) {
        return Ok(all);
    }
    let mut inverse = vec![0i64; order.len()];
    for (k, &g) in order.iter().enumerate() {
        inverse[g] = k as i64;
    }
    let index = Tensor::<R>::from_slice(&inverse, &[inverse.len()], device)?;
    Ok(client.index_select(&all, 0, &index)?)
}

#[cfg(test)]
mod tests {
    use super::super::plan::PrefixInput;
    use super::super::runner::tests::{
        PREFIX, alone, at, close, items, model, plan, run, sequence, tokens,
    };
    use super::*;
    use crate::model::qwen35::Qwen35Model;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    /// Every gathered row and every option logit equals its sequence run
    /// alone, for a prefix of 6 and for no prefix.
    #[test]
    fn rows_equal_sequences_run_alone() {
        let (client, device) = cpu_setup();
        let model = model(&device);
        for p in [PREFIX, 0] {
            let plan = plan(p);
            let cfg = RunnerConfig::default();
            let hidden = gather_plan_hidden(&client, &model, &plan, None, &cfg).unwrap();
            let width = model.config().hidden_size;
            let hidden = hidden.to_vec::<f32>();
            let out = run(&client, &model, &plan, None, &cfg).unwrap();
            let items = items(&plan);
            for (g, gather) in plan.gathers.iter().enumerate() {
                let want = alone(&client, &model, &sequence(&plan, gather), gather.pos);
                let what = format!("prefix {p}, gather {g}");
                close(
                    &hidden[g * width..(g + 1) * width],
                    &want.to_vec::<f32>(),
                    1e-4,
                    &what,
                );
                let k = items[g].len();
                let ids = Tensor::<CpuRuntime>::from_slice(&items[g], &[1, k], &device).unwrap();
                let logits = Qwen35Model::candidate_logits(&model, &client, &want, &ids).unwrap();
                close(&out.raw_logits[g], &logits.to_vec::<f32>(), 1e-4, &what);
            }
        }
    }

    /// A plan with no rows reads the prefix alone.
    #[test]
    fn prefix_only_plan() {
        let (client, device) = cpu_setup();
        let model = model(&device);
        let plan = DecisionPlan {
            prefix: PrefixInput::Tokens(tokens(PREFIX, 0)),
            rows: Vec::new(),
            gathers: vec![at(None, 5), at(None, 2), at(None, 5)],
        };
        let cfg = RunnerConfig::default();
        let got = gather_plan_hidden(&client, &model, &plan, None, &cfg).unwrap();
        let width = model.config().hidden_size;
        let got = got.to_vec::<f32>();
        for (g, gather) in plan.gathers.iter().enumerate() {
            let want = alone(&client, &model, plan.prefix.tokens(), gather.pos);
            close(
                &got[g * width..(g + 1) * width],
                &want.to_vec::<f32>(),
                1e-5,
                "prefix row",
            );
        }
    }
}
