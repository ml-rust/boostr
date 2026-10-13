//! [`JointSchemaHead`]: the Clef joint-schema decision head. It reads every
//! final-normed hidden row of a record and the output-projection rows of
//! every option's tokens, and scores all fields of the record jointly as a
//! [`DecisionHead`] over [`JointSchemaItem`]s.

use super::blocks::{Affine, FeedForward, Norm};
use super::config::JointSchemaConfig;
use super::decoder_layer::DecoderLayer;
use super::forward::Stages;
use super::item::JointSchemaItem;
use super::layout::cast_to;
use super::routing::RoutingLayer;
use crate::error::{Error, Result};
use crate::inference::decision::{DecisionBackbone, DecisionHead, OptionLogits};
use crate::model::traits::ModelClient;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// The Clef joint-schema head, in one compute dtype.
///
/// One record per call: `items` are the record's fields in request order,
/// and `hidden` holds every position of the record, as
/// [`full_sequence_plan`](super::item::full_sequence_plan) gathers it.
/// Merging records into one plan is not supported: the memory is the whole
/// sequence. The logits are raw and untempered; answer them with
/// `TemperaturePolicy::Fixed(1.0)` and
/// [`joint_answer_options`](super::item::joint_answer_options).
pub struct JointSchemaHead<R: Runtime> {
    pub(super) config: JointSchemaConfig,
    pub(super) dtype: DType,
    pub(super) hidden_norm: Norm<R>,
    pub(super) memory_projection: Affine<R>,
    pub(super) question_projection: Affine<R>,
    pub(super) option_question_projection: Affine<R>,
    pub(super) global_projection: Affine<R>,
    pub(super) option_context_projection: Affine<R>,
    pub(super) option_lexical_projection: Affine<R>,
    /// `[3, W]`: noul, choice, score.
    pub(super) type_embedding: Tensor<R>,
    pub(super) evidence_layers: Vec<RoutingLayer<R>>,
    pub(super) option_summary_norm: Norm<R>,
    pub(super) layers: Vec<DecoderLayer<R>>,
    pub(super) field_norm: Norm<R>,
    pub(super) option_norm: Norm<R>,
    pub(super) residual_scorer: FeedForward<R>,
    /// `exp(min(prior_logit_scale, ln 100))`, rounded to the dtype.
    pub(super) prior_scale: f64,
    /// `exp(min(joint_logit_scale, ln 100))`, rounded to the dtype.
    pub(super) joint_scale: f64,
    /// `sigmoid(residual_gate)`, rounded to the dtype.
    pub(super) gate: f64,
}

fn invalid(arg: &'static str, reason: String) -> Error {
    Error::InvalidArgument { arg, reason }
}

impl<R: Runtime<DType = DType>> JointSchemaHead<R> {
    /// The head's sizes.
    pub fn config(&self) -> &JointSchemaConfig {
        &self.config
    }

    /// The dtype every op of the head runs in.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The prior scale `exp(min(prior_logit_scale, ln 100))`.
    pub fn prior_scale(&self) -> f64 {
        self.prior_scale
    }

    /// The cosine scale `exp(min(joint_logit_scale, ln 100))`.
    pub fn joint_scale(&self) -> f64 {
        self.joint_scale
    }

    /// The joint-term weight `sigmoid(residual_gate)`.
    pub fn gate(&self) -> f64 {
        self.gate
    }

    /// The option ids of `items`, in field order, option order, token
    /// order: the rows `lexical` holds in [`Self::logits`].
    pub fn lexical_ids(items: &[JointSchemaItem]) -> Vec<u32> {
        items
            .iter()
            .flat_map(|item| item.options.iter().flat_map(|o| o.ids.iter().copied()))
            .collect()
    }

    /// `hidden` `[L, H]` and `lexical` `[ids, H]` fit `items`.
    fn check_inputs(
        &self,
        hidden: &Tensor<R>,
        lexical: &Tensor<R>,
        items: &[JointSchemaItem],
    ) -> Result<()> {
        let h = self.config.hidden_size;
        let shape = hidden.shape();
        if shape.len() != 2 || shape[0] == 0 || shape[1] != h {
            return Err(invalid(
                "hidden",
                format!("must be [L >= 1, {h}], every row of the record, got {shape:?}"),
            ));
        }
        if !hidden.dtype().is_float() {
            return Err(invalid(
                "hidden",
                format!("must be a float dtype, got {:?}", hidden.dtype()),
            ));
        }
        if items.is_empty() {
            return Err(invalid(
                "items",
                "no fields: a joint schema record has at least one".into(),
            ));
        }
        for (n, item) in items.iter().enumerate() {
            item.check(n, shape[0])?;
        }
        let ids: usize = items
            .iter()
            .flat_map(|item| item.options.iter().map(|o| o.ids.len()))
            .sum();
        if lexical.shape() != [ids, h].as_slice() || !lexical.dtype().is_float() {
            return Err(invalid(
                "lexical",
                format!(
                    "must be [{ids}, {h}] float, one output-projection row per option id, \
                     got {:?} {:?}",
                    lexical.shape(),
                    lexical.dtype()
                ),
            ));
        }
        Ok(())
    }

    /// The option logits of every field of one record, F32, one scoring row
    /// per field.
    ///
    /// - `hidden`: `[L, H]`, the final-normed hidden row of every position
    ///   of the record, in order
    /// - `lexical`: `[ids, H]`, the output-projection rows of
    ///   [`Self::lexical_ids`] of `items`
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when a shape disagrees with the above or
    /// an item fails [`JointSchemaItem::check`], naming it. Device errors
    /// propagate.
    pub fn logits<C: ModelClient<R>>(
        &self,
        client: &C,
        hidden: &Tensor<R>,
        lexical: &Tensor<R>,
        items: &[JointSchemaItem],
    ) -> Result<OptionLogits<R>> {
        self.check_inputs(hidden, lexical, items)?;
        let logits = self.forward(client, hidden, lexical, items, None)?;
        Ok(OptionLogits {
            flat: cast_to(client, &logits, DType::F32)?,
            counts: items.iter().map(|item| item.options.len()).collect(),
        })
    }

    /// Every intermediate of [`Self::logits`] under the reference's stage
    /// names, in stage order and in the head dtype: `hidden_normed`,
    /// `memory`, `global_vector`, `question_vectors`,
    /// `option_context_vectors`, `option_lexical_vectors`,
    /// `option_context_proj`, `option_lexical_proj`,
    /// `option_question_proj`, `option_queries`, `routed_<i>`,
    /// `base_fields`, `summary_weights`, `option_summaries`,
    /// `option_summaries_normed`, `global_projection`, `type_embedding`,
    /// `fields_in`, `decoder_<i>`, `fields_normed`, `options_normed`,
    /// `prior`, `cosine`, `residual`, `logits`.
    ///
    /// # Errors
    ///
    /// As [`Self::logits`].
    pub fn stages<C: ModelClient<R>>(
        &self,
        client: &C,
        hidden: &Tensor<R>,
        lexical: &Tensor<R>,
        items: &[JointSchemaItem],
    ) -> Result<Stages<R>> {
        self.check_inputs(hidden, lexical, items)?;
        let mut stages = Stages::new();
        self.forward(client, hidden, lexical, items, Some(&mut stages))?;
        Ok(stages)
    }
}

/// The head as a decision readout: the option ids' rows come from the
/// backbone's output projection, then [`JointSchemaHead::logits`].
impl<R: Runtime<DType = DType>> DecisionHead<R> for JointSchemaHead<R> {
    type Item = JointSchemaItem;

    /// # Errors
    ///
    /// As [`JointSchemaHead::logits`]; backbone errors of
    /// [`DecisionBackbone::output_rows`] propagate.
    fn option_logits<C: ModelClient<R>, B: DecisionBackbone<R>>(
        &self,
        client: &C,
        backbone: &B,
        hidden: &Tensor<R>,
        items: &[JointSchemaItem],
    ) -> Result<OptionLogits<R>> {
        let rows = hidden.shape().first().copied().unwrap_or(0);
        if items.is_empty() {
            return Err(invalid(
                "items",
                "no fields: a joint schema record has at least one".into(),
            ));
        }
        for (n, item) in items.iter().enumerate() {
            item.check(n, rows)?;
        }
        let lexical = backbone.output_rows(client, &Self::lexical_ids(items))?;
        self.logits(client, hidden, &lexical, items)
    }
}

#[cfg(test)]
mod tests {
    use super::super::item::{FieldKind, full_sequence_plan, joint_answer_options};
    use super::super::weights::tests::{RECORD, tiny_head, tiny_inputs, tiny_items};
    use super::*;
    use crate::inference::decision::run::gather::gather_plan_hidden;
    use crate::inference::decision::{Answer, RunnerConfig, TemperaturePolicy, run_decision};
    use crate::model::qwen35::model::{Head, chunked_model};
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    fn softmax(z: &[f32]) -> Vec<f64> {
        let top = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f64> = z.iter().map(|&v| f64::from(v - top).exp()).collect();
        let s: f64 = e.iter().sum();
        e.iter().map(|v| v / s).collect()
    }

    /// The head through `run_decision` over a tiny Qwen3.5: the plan of
    /// every position, the option rows from `output_rows`, and the answers
    /// of a Score, a Noul and a one-option Choice field.
    #[test]
    fn runs_as_a_decision_head() {
        let (client, device) = cpu_setup();
        let model = chunked_model(&device, 0x3535_00c1, 4, false, Head::Own);
        let hidden_size = model.config().hidden_size;
        let vocab = model.config().vocab_size as u32;
        let head = tiny_head(&client, &device, hidden_size);
        let tokens: Vec<u32> = (0..RECORD as u32).map(|t| (t * 7 + 3) % vocab).collect();
        let mut items = Vec::new();
        for (n, item) in tiny_items().into_iter().enumerate() {
            let kind = if n == 0 { FieldKind::Score } else { item.kind };
            let spans: Vec<_> = item.options.iter().map(|o| o.span.clone()).collect();
            items.push(JointSchemaItem::from_tokens(kind, item.question, &spans, &tokens).unwrap());
        }
        let plan = full_sequence_plan(&tokens).unwrap();
        let meta: Vec<_> = items.iter().map(JointSchemaItem::meta).collect();
        let cfg = RunnerConfig::default();
        let result = run_decision(
            &client,
            &model,
            &head,
            &plan,
            None,
            &items,
            &meta,
            &TemperaturePolicy::Fixed(1.0),
            &joint_answer_options(),
            &cfg,
        )
        .unwrap();

        let hidden = gather_plan_hidden(&client, &model, &plan, None, &cfg).unwrap();
        assert_eq!(hidden.shape(), &[RECORD, hidden_size]);
        let lexical = model
            .output_rows(&client, &JointSchemaHead::<CpuRuntime>::lexical_ids(&items))
            .unwrap();
        let want = head.logits(&client, &hidden, &lexical, &items).unwrap();
        let want = want.flat.to_vec::<f32>();
        let got: Vec<f32> = result.raw_logits.concat();
        assert_eq!(got, want);

        let p = softmax(&result.raw_logits[0]);
        match &result.answers[0] {
            Answer::Score {
                expected,
                confidence,
                ..
            } => {
                assert!((expected - (p[1] + 2.0 * p[2])).abs() < 1e-6);
                let top = p.iter().copied().fold(0.0, f64::max);
                assert!((confidence - top).abs() < 1e-6);
            }
            other => panic!("field 0 is a Score field, got {other:?}"),
        }
        match &result.answers[1] {
            Answer::Noul { p_yes, .. } => {
                assert!((p_yes - softmax(&result.raw_logits[1])[0]).abs() < 1e-6);
            }
            other => panic!("field 1 is a Noul field, got {other:?}"),
        }
        match &result.answers[2] {
            Answer::Choice { index, probs, .. } => {
                assert_eq!((*index, probs.as_slice()), (0, [1.0].as_slice()));
            }
            other => panic!("field 2 is a one-option Choice field, got {other:?}"),
        }
    }

    #[test]
    fn bad_inputs_are_refused_with_their_values() {
        let (client, device) = cpu_setup();
        let head = tiny_head(&client, &device, 16);
        let items = tiny_items();
        let (hidden, lexical) = tiny_inputs(&device, 16, &items);
        let err =
            |h: &Tensor<CpuRuntime>, l: &Tensor<CpuRuntime>, items: &[JointSchemaItem]| match head
                .logits(&client, h, l, items)
            {
                Ok(_) => panic!("the inputs must be refused"),
                Err(e) => e.to_string(),
            };
        let short = hidden.narrow(0, 0, 11).unwrap().contiguous().unwrap();
        let e = err(&short, &lexical, &items);
        assert!(e.contains("field 1 option 1 span 11..12"), "{e}");
        let fewer = lexical.narrow(0, 0, 3).unwrap().contiguous().unwrap();
        let e = err(&hidden, &fewer, &items);
        assert!(e.contains("must be [10, 16] float"), "{e}");
        let e = err(&hidden, &lexical, &[]);
        assert!(e.contains("no fields"), "{e}");
        let wide = Tensor::<CpuRuntime>::zeros(&[RECORD, 17], DType::F32, &device).unwrap();
        let e = err(&wide, &lexical, &items);
        assert!(e.contains("must be [L >= 1, 16]"), "{e}");
    }

    /// The head in BF16 tracks the F32 head of the same weights.
    #[cfg(feature = "f16")]
    #[test]
    fn bf16_head_tracks_f32() {
        use super::super::weights::tests::{SCALARS, tiny_config, tiny_tensors};
        let (client, device) = cpu_setup();
        let c = tiny_config(16);
        let f32_head = tiny_head(&client, &device, 16);
        let bf16_head = JointSchemaHead::from_tensors(
            &client,
            c,
            tiny_tensors(&device, &c),
            SCALARS,
            DType::BF16,
        )
        .unwrap();
        assert_eq!(bf16_head.dtype(), DType::BF16);
        assert_eq!(bf16_head.gate(), 0.376_953_125);
        let items = tiny_items();
        let (hidden, lexical) = tiny_inputs(&device, 16, &items);
        let want = f32_head.logits(&client, &hidden, &lexical, &items).unwrap();
        let got = bf16_head
            .logits(&client, &hidden, &lexical, &items)
            .unwrap();
        assert_eq!(got.flat.dtype(), DType::F32);
        let (got, want) = (got.flat.to_vec::<f32>(), want.flat.to_vec::<f32>());
        // Each BF16 rounding moves a value by up to 2^-9 relative, and the
        // cosine term is scaled by 100 times a gate near 0.38.
        for (j, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!((g - w).abs() < 1.0, "option {j}: {g} vs {w}");
        }
    }
}
