//! What the generator drafts with: a model-free drafter or a draft model.

use super::accept::{greedy, rejection};
use crate::error::{Error, Result};
use crate::inference::speculative::client::SpeculativeClient;
use crate::inference::speculative::draft::{
    DraftModel, Drafter, ModelDrafter, ProbDraft, ProbDrafter,
};
use crate::inference::speculative::model::{SpeculativeModel, SpeculativeProbModel};
use crate::inference::speculative::types::{Draw, SampleRule, Verdict};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// The drafting seam of [`SpeculativeGenerator`](super::SpeculativeGenerator).
///
/// Every [`Drafter`] implements it with one-hot draft distributions. A
/// [`ModelDrafter`] implements it with the distributions its draft model
/// sampled from, for a target that implements [`SpeculativeProbModel`].
pub trait DraftSource<M: SpeculativeModel> {
    /// Draft up to `k` tokens after `context`, then run one verify forward.
    ///
    /// `feed` holds the last committed token on entry. The call appends the
    /// drafts it keeps, cut at `k` and at the first id outside the target
    /// vocabulary. `rule` is the verify rule, or `None` for plain greedy
    /// decoding. `draw` is the Philox position of the step.
    ///
    /// # Errors
    ///
    /// A drafter error, or any error of the verify forward.
    fn draft_and_verify(
        &mut self,
        model: &mut M,
        context: &[u32],
        k: usize,
        rule: Option<&SampleRule>,
        draw: Draw,
        feed: &mut Vec<u32>,
    ) -> Result<Verdict>;
}

/// The leading ids of `proposal` below `vocab`, at most `k` of them.
fn kept(proposal: &[u32], k: usize, vocab: usize) -> usize {
    proposal
        .iter()
        .take(k)
        .take_while(|&&id| (id as usize) < vocab)
        .count()
}

impl<M: SpeculativeModel, D: Drafter> DraftSource<M> for D {
    fn draft_and_verify(
        &mut self,
        model: &mut M,
        context: &[u32],
        k: usize,
        rule: Option<&SampleRule>,
        draw: Draw,
        feed: &mut Vec<u32>,
    ) -> Result<Verdict> {
        let len_before = model.len();
        if k > 0 {
            let proposal = self.propose(context, k);
            let n = kept(&proposal, k, model.vocab_size());
            feed.extend_from_slice(&proposal[..n]);
        }
        match rule {
            None => greedy::accept(model, feed, len_before),
            Some(rule) => rejection::accept(model, feed, rule, draw, len_before),
        }
    }
}

impl<R, M, DM, C> DraftSource<M> for ModelDrafter<R, DM, C>
where
    R: Runtime<DType = DType>,
    M: SpeculativeProbModel<R>,
    DM: DraftModel<R>,
    C: SpeculativeClient<R>,
{
    fn draft_and_verify(
        &mut self,
        model: &mut M,
        context: &[u32],
        k: usize,
        rule: Option<&SampleRule>,
        draw: Draw,
        feed: &mut Vec<u32>,
    ) -> Result<Verdict> {
        let len_before = model.len();
        let ProbDraft { tokens, probs } = if k > 0 {
            self.propose(context, k, rule, draw)?
        } else {
            ProbDraft::empty()
        };
        let n = kept(&tokens, k, model.vocab_size());
        feed.extend_from_slice(&tokens[..n]);
        let Some(rule) = rule else {
            return greedy::accept(model, feed, len_before);
        };
        let q = match probs {
            Some(q) if n > 0 && !rule.is_greedy() => Some(leading_rows(q, tokens.len(), n)?),
            _ => None,
        };
        rejection::accept_with_probs(model, feed, rule, draw, q.as_ref(), len_before)
    }
}

/// The first `n` rows of `q`, a `[1, proposed, V]` tensor.
///
/// # Errors
///
/// [`Error::InferenceError`] when `q` does not hold one row per proposed token.
fn leading_rows<R: Runtime>(q: Tensor<R>, proposed: usize, n: usize) -> Result<Tensor<R>> {
    let shape = q.shape();
    if shape.len() != 3 || shape[0] != 1 || shape[1] != proposed {
        return Err(Error::InferenceError {
            reason: format!(
                "the draft probs are {shape:?} for {proposed} draft tokens; expected \
                 [1, {proposed}, vocab]"
            ),
        });
    }
    if n == proposed {
        return Ok(q);
    }
    Ok(q.narrow(1, 0, n)?.contiguous()?)
}
