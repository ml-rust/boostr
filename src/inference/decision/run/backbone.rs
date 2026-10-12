//! The model side of a decision run: the [`DecisionBackbone`] trait and
//! the cached prefix it builds, [`PrefixState`].

use crate::error::Result;
use crate::inference::{LayeredGdnState, LayeredKvCache};
use crate::model::traits::ModelClient;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// A prefix run once, batch 1, never padded.
///
/// Valid only for the model that built it: the caches hold that model's
/// keys, values and GDN states. A caller may keep it across requests that
/// share the same prefix tokens. Nothing here records the tokens, so
/// handing it to a request with a different prefix of the same length is
/// the caller's error and goes undetected.
pub struct PrefixState<R: Runtime> {
    /// Every attention layer's keys and values over the prefix, batch 1.
    pub kv: LayeredKvCache<R>,
    /// Every GDN layer's state after the prefix, batch 1.
    pub gdn: LayeredGdnState<R>,
    /// Prefix length in positions.
    pub len: usize,
    /// The rope position the first suffix token takes.
    pub rope_end: usize,
    /// The prefix positions whose hidden rows `hidden` holds, in its row order.
    pub rows: Vec<usize>,
    /// Normed hidden rows `[rows.len(), hidden_size]` at `rows`. `None`
    /// when `rows` is empty.
    pub hidden: Option<Tensor<R>>,
}

impl<R: Runtime> PrefixState<R> {
    /// The row of `hidden` that holds prefix position `pos`.
    pub fn hidden_row(&self, pos: usize) -> Option<usize> {
        self.rows.iter().position(|&r| r == pos)
    }
}

/// A model a decision run can drive.
///
/// Rows run as lanes of one batch: each lane continues the shared prefix
/// with its own right-padded suffix. A lane is single-use: the state after
/// a padded row is not the state of its real tokens.
pub trait DecisionBackbone<R: Runtime<DType = DType>> {
    /// The number of positions, prefix included, the model can address.
    fn max_positions(&self) -> usize;

    /// The number of token ids the model scores: `candidate_logits` takes
    /// ids in `[0, vocab_size)`.
    fn vocab_size(&self) -> usize;

    /// Run `tokens` as one batch-1 sequence from position 0 and keep the
    /// normed hidden rows at `rows` (each `< tokens.len()`, any order,
    /// repeats allowed). An empty `tokens` gives an empty state.
    fn prefill_prefix<C: ModelClient<R>>(
        &self,
        client: &C,
        tokens: &[u32],
        rows: &[usize],
    ) -> Result<PrefixState<R>>;

    /// Fresh, empty caches for `lanes` lanes of up to `width` suffix
    /// positions each: a KV cache holding no keys and zero GDN states.
    ///
    /// Each lane holds its own copy of every GDN layer's convolution and
    /// recurrent state, the size of the prefix's GDN state, plus `width`
    /// keys and values per attention layer.
    fn lane_caches(
        &self,
        lanes: usize,
        width: usize,
        device: &R::Device,
    ) -> Result<(LayeredKvCache<R>, LayeredGdnState<R>)>;

    /// Run `ids` `[lanes, width]` (right-padded suffixes) as lanes that
    /// each continue `prefix`, and return the normed hidden rows `[G,
    /// hidden_size]` at `gather` `[G]` i64 flat indices `lane * width + t`.
    ///
    /// `gdn` holds the prefix's GDN state in every lane and `kv` holds no
    /// keys, both with `lanes` lanes, as `lane_caches` plus
    /// `LayeredGdnState::install_lane` leave them. The lanes attend to
    /// `prefix.kv` without copying it.
    fn forward_rows<C: ModelClient<R>>(
        &self,
        client: &C,
        ids: &Tensor<R>,
        prefix: &PrefixState<R>,
        gdn: &mut LayeredGdnState<R>,
        kv: &mut LayeredKvCache<R>,
        gather: &Tensor<R>,
    ) -> Result<Tensor<R>>;

    /// F32 logits `[Q, K]` of the token ids `ids` `[Q, K]` i64 at the
    /// normed hidden rows `hidden` `[Q, hidden_size]`.
    fn candidate_logits<C: ModelClient<R>>(
        &self,
        client: &C,
        hidden: &Tensor<R>,
        ids: &Tensor<R>,
    ) -> Result<Tensor<R>>;
}
