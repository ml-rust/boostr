//! The scoring side of a decision run: [`DecisionHead`] turns gathered
//! hidden rows into option logits, [`OptionLogits`].

use super::backbone::DecisionBackbone;
use crate::error::Result;
use crate::model::traits::ModelClient;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// The option logits of every scoring row, concatenated.
///
/// Scoring rows follow the questions in order. A question takes one row of
/// `options` logits, except an isolated Score question
/// (`QuestionKind::Score { isolated: true, .. }`), which takes one row per
/// level, each holding two logits: "no" at index 0, "yes" at index 1.
pub struct OptionLogits<R: Runtime> {
    /// `[sum(counts)]` F32 logits, row after row.
    pub flat: Tensor<R>,
    /// The logit count of each scoring row, each `>= 1`.
    pub counts: Vec<usize>,
}

/// Maps gathered hidden rows to option logits.
pub trait DecisionHead<R: Runtime<DType = DType>> {
    /// What the head knows about each question, such as its option ids.
    type Item;

    /// The option logits of every scoring row from `hidden` `[G,
    /// hidden_size]`, the plan's gathers in plan order.
    fn option_logits<C: ModelClient<R>, B: DecisionBackbone<R>>(
        &self,
        client: &C,
        backbone: &B,
        hidden: &Tensor<R>,
        items: &[Self::Item],
    ) -> Result<OptionLogits<R>>;
}
