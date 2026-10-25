//! [`HybridModel::forward_rows`]: one decode step over rows that each keep
//! their own KV cache and SSM state.
//!
//! The rows of one step can sit at different positions. The embedding, the
//! norms, the feed-forward halves and the LM head run once over all rows.
//! Each attention or SSM mixer runs per row, over that row's own cache or
//! state at that row's own position.

use super::build::{HybridBlock, HybridModel};
use crate::error::{Error, Result};
use crate::inference::{KvCache, LayeredKvCache, LayeredSsmState};
use crate::model::traits::ModelClient;
use crate::nn::var_ops::var_contiguous;
use crate::quant::traits::DequantOps;
use numr::autograd::{Var, var_cat, var_narrow};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, ConvOps, IndexingOps, NormalizationOps,
    ReduceOps, ScalarOps, ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// One sequence of a [`HybridModel::forward_rows`] step.
pub struct HybridRow<'a, R: Runtime> {
    /// The row's KV cache, one layer per attention layer.
    pub kv_cache: &'a mut LayeredKvCache<R>,
    /// The row's SSM state, one layer per SSM layer. `None` only for a
    /// model with no SSM layer.
    pub ssm_state: Option<&'a mut LayeredSsmState<R>>,
    /// Absolute position of the row's token: the tokens its caches hold.
    pub position: usize,
}

impl<R: Runtime<DType = DType>> HybridModel<R>
where
    R::Client: IndexingOps<R>,
{
    /// One token for each row of `rows`. Returns the logits `[N, 1, vocab]`.
    ///
    /// `input_ids` is `[N, 1]` with `N = rows.len()`: row `i` feeds the
    /// token of `rows[i]`. Each row's caches advance by one token. Row `i`'s
    /// logits are the logits [`forward_hybrid`](Self::forward_hybrid) gives
    /// that row alone at its position.
    ///
    /// # Errors
    ///
    /// No row, `input_ids` not `[N, 1]`, a row with no SSM state on a model
    /// with SSM layers, a row position other than its cache length, or a
    /// cache of the wrong shape.
    pub fn forward_rows<C>(
        &self,
        client: &C,
        input_ids: &Tensor<R>,
        rows: &mut [HybridRow<'_, R>],
    ) -> Result<Tensor<R>>
    where
        C: ModelClient<R> + ConvOps<R> + NormalizationOps<R> + UnaryOps<R> + ActivationOps<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ActivationOps<R>
            + ConvOps<R>
            + ReduceOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + DequantOps<R>,
    {
        self.check_rows(input_ids, rows)?;
        let mut hidden = self.embed_tokens.forward(client, input_ids)?;
        let (mut attn_idx, mut ssm_idx) = (0usize, 0usize);
        for (layer, block) in self.blocks.iter().enumerate() {
            hidden = match block {
                HybridBlock::Attention(b) => {
                    let idx = attn_idx;
                    attn_idx += 1;
                    b.forward_with_mixer(client, &hidden, |normed| {
                        per_row(client, normed, rows, |row, x| {
                            let position = row.position;
                            let cache = row_kv_layer(row, idx, layer)?;
                            b.attention_forward(client, x, &self.rope, cache, position)
                        })
                    })?
                }
                HybridBlock::Mla(b) => {
                    let idx = attn_idx;
                    attn_idx += 1;
                    b.forward_with_mixer(client, &hidden, |normed| {
                        per_row(client, normed, rows, |row, x| {
                            let position = row.position;
                            let cache = row_kv_layer(row, idx, layer)?;
                            b.attn.forward_inference(client, x, cache, position)
                        })
                    })?
                }
                HybridBlock::Ssm(b) => {
                    let idx = ssm_idx;
                    ssm_idx += 1;
                    b.forward_with_mixer(client, &hidden, |normed| {
                        per_row(client, normed, rows, |row, x| {
                            let state = row
                                .ssm_state
                                .as_deref_mut()
                                .and_then(|s| s.layer_mut(idx))
                                .ok_or_else(|| Error::ModelError {
                                    reason: format!(
                                        "a row's SSM state has no layer {idx} for model layer \
                                         {layer}; build the state with num_ssm_layers() layers"
                                    ),
                                })?;
                            let out = b.mamba.forward_inference(client, x.tensor(), state)?;
                            Ok(Var::new(out, false))
                        })
                    })?
                }
            };
        }
        let hidden = self.norm.forward(client, &hidden)?;
        Ok(self.lm_head.forward(client, &hidden)?.tensor().clone())
    }

    /// Reject a step `forward_rows` cannot run.
    fn check_rows(&self, input_ids: &Tensor<R>, rows: &[HybridRow<'_, R>]) -> Result<()> {
        let n = rows.len();
        if n == 0 || input_ids.shape() != [n, 1].as_slice() {
            return Err(Error::InvalidArgument {
                arg: "input_ids",
                reason: format!(
                    "forward_rows takes [N, 1] ids for N >= 1 rows, got shape {:?} for {n} rows",
                    input_ids.shape()
                ),
            });
        }
        if self.num_ssm_layers() > 0 && rows.iter().any(|r| r.ssm_state.is_none()) {
            return Err(Error::ModelError {
                reason: format!(
                    "this hybrid model has {} SSM layers, and a row carries no SSM state; give \
                     every row a LayeredSsmState",
                    self.num_ssm_layers()
                ),
            });
        }
        Ok(())
    }
}

/// Run `mixer` on each row's slice of `x` (`[N, S, hidden]`) and stack the
/// outputs back along the batch axis. One row runs on `x` itself.
fn per_row<'r, R, C, F>(
    client: &C,
    x: &Var<R>,
    rows: &mut [HybridRow<'r, R>],
    mut mixer: F,
) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: ModelClient<R>,
    R::Client: TensorOps<R> + ShapeOps<R>,
    F: FnMut(&mut HybridRow<'r, R>, &Var<R>) -> Result<Var<R>>,
{
    if let [row] = &mut *rows {
        return mixer(row, x);
    }
    let mut outs = Vec::with_capacity(rows.len());
    for (i, row) in rows.iter_mut().enumerate() {
        let slice = var_narrow(x, 0, i, 1).map_err(Error::Numr)?;
        outs.push(mixer(row, &var_contiguous(&slice)?)?);
    }
    let refs: Vec<&Var<R>> = outs.iter().collect();
    var_cat(&refs, 0, client).map_err(Error::Numr)
}

/// Attention layer `idx` of `row`'s KV cache, model layer `layer`.
fn row_kv_layer<'a, R: Runtime<DType = DType>>(
    row: &'a mut HybridRow<'_, R>,
    idx: usize,
    layer: usize,
) -> Result<&'a mut KvCache<R>>
where
    R::Client: IndexingOps<R>,
{
    row.kv_cache
        .layer_mut(idx)
        .ok_or_else(|| Error::ModelError {
            reason: format!(
                "a row's KV cache has no layer {idx} for model layer {layer}; build the cache \
                 with num_attention_layers() layers"
            ),
        })
}

#[cfg(test)]
mod tests {
    use super::super::forward::tests::{checkpoint, config, map};
    use super::*;
    use crate::inference::kv_cache::layered::LayeredKvCacheConfig;
    use crate::nn::VarBuilder;
    use crate::test_utils::{cpu_setup, max_abs_diff};
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    type T = Tensor<CpuRuntime>;
    const TOL: f32 = 1e-4;

    /// One sequence's caches, fed one call at a time.
    struct Seq {
        kv: LayeredKvCache<CpuRuntime>,
        ssm: Option<LayeredSsmState<CpuRuntime>>,
        position: usize,
    }

    fn seq(m: &HybridModel<CpuRuntime>, dev: &CpuDevice) -> Seq {
        let (heads, width) = m.kv_cache_dims().unwrap();
        let kv_config = LayeredKvCacheConfig {
            batch_size: 1,
            num_kv_heads: heads,
            initial_capacity: 4,
            max_seq_len: 16,
            head_dim: width,
            dtype: DType::F32,
        };
        Seq {
            kv: LayeredKvCache::new(m.num_attention_layers(), &kv_config, dev).unwrap(),
            ssm: m
                .mamba_config()
                .map(|c| LayeredSsmState::new(m.num_ssm_layers(), 1, c, DType::F32, dev).unwrap()),
            position: 0,
        }
    }

    /// Logits of `tokens` fed to `s` alone.
    fn alone(m: &HybridModel<CpuRuntime>, s: &mut Seq, tokens: &[i64]) -> Vec<f32> {
        let (client, dev) = cpu_setup();
        let input = T::from_slice(tokens, &[1, tokens.len()], &dev).unwrap();
        let logits = match s.ssm.as_mut() {
            Some(state) => m.forward_hybrid(&client, &input, &mut s.kv, state, s.position),
            None => m.forward_with_kv_cache(&client, &input, &mut s.kv, s.position),
        };
        s.position += tokens.len();
        logits.unwrap().to_vec::<f32>()
    }

    /// Logits of one `forward_rows` step, `tokens[i]` fed to `seqs[i]`.
    fn rows(m: &HybridModel<CpuRuntime>, seqs: &mut [Seq], tokens: &[i64]) -> Vec<f32> {
        let (client, dev) = cpu_setup();
        let input = T::from_slice(tokens, &[tokens.len(), 1], &dev).unwrap();
        let mut rows: Vec<HybridRow<'_, CpuRuntime>> = seqs
            .iter_mut()
            .map(|s| HybridRow {
                kv_cache: &mut s.kv,
                ssm_state: s.ssm.as_mut(),
                position: s.position,
            })
            .collect();
        let logits = m.forward_rows(&client, &input, &mut rows).unwrap();
        for s in seqs.iter_mut() {
            s.position += 1;
        }
        logits.to_vec::<f32>()
    }

    fn model(ssm: &[bool], moe: bool) -> HybridModel<CpuRuntime> {
        let (_, dev) = cpu_setup();
        let config = config(ssm, moe);
        let mut map = map(&checkpoint(&config, &dev));
        HybridModel::from_varbuilder(&mut VarBuilder::new(&mut map, &dev), &config).unwrap()
    }

    /// Three prompts of different lengths prefill alone. Each step then
    /// feeds every row one token in one `forward_rows`, and every row's
    /// logits match the same token fed to a copy of that row alone.
    fn rows_match_alone(ssm: &[bool], moe: bool) {
        let (_, dev) = cpu_setup();
        let m = model(ssm, moe);
        let prompts: [&[i64]; 3] = [&[3, 10, 17, 1, 8], &[5, 2], &[11, 4, 19]];
        let mut batched: Vec<Seq> = prompts.iter().map(|_| seq(&m, &dev)).collect();
        let mut single: Vec<Seq> = prompts.iter().map(|_| seq(&m, &dev)).collect();
        for (i, prompt) in prompts.iter().enumerate() {
            alone(&m, &mut batched[i], prompt);
            alone(&m, &mut single[i], prompt);
        }
        for step in 0..4i64 {
            let tokens: Vec<i64> = (0..3).map(|i| (step * 5 + i * 7 + 2) % 24).collect();
            let got = rows(&m, &mut batched, &tokens);
            let vocab = got.len() / 3;
            for (i, s) in single.iter_mut().enumerate() {
                let want = alone(&m, s, &tokens[i..=i]);
                let diff = max_abs_diff(&got[i * vocab..(i + 1) * vocab], &want);
                assert!(
                    diff <= TOL,
                    "ssm {ssm:?} step {step} row {i}: max diff {diff}"
                );
            }
        }
    }

    #[test]
    fn mla_moe_rows_at_different_positions_match_each_row_alone() {
        rows_match_alone(&[false, false], true);
    }

    #[test]
    fn mamba2_mla_moe_rows_match_each_row_alone() {
        rows_match_alone(&[true, true, false], true);
    }

    #[test]
    fn one_row_matches_the_row_alone() {
        let (_, dev) = cpu_setup();
        let m = model(&[true, false], false);
        let (mut a, mut b) = (seq(&m, &dev), seq(&m, &dev));
        alone(&m, &mut a, &[1, 2, 3]);
        alone(&m, &mut b, &[1, 2, 3]);
        let got = rows(&m, std::slice::from_mut(&mut a), &[9]);
        let want = alone(&m, &mut b, &[9]);
        assert!(max_abs_diff(&got, &want) <= TOL);
    }

    #[test]
    fn a_row_without_ssm_state_and_a_misshapen_input_are_refused() {
        let (client, dev) = cpu_setup();
        let m = model(&[true, false], false);
        let mut s = seq(&m, &dev);
        let input = T::from_slice(&[1i64], &[1, 1], &dev).unwrap();
        let mut bare = [HybridRow {
            kv_cache: &mut s.kv,
            ssm_state: None,
            position: 0,
        }];
        let err = m.forward_rows(&client, &input, &mut bare).unwrap_err();
        assert!(err.to_string().contains("SSM state"), "{err}");

        let two = T::from_slice(&[1i64, 2], &[2, 1], &dev).unwrap();
        let mut one = [HybridRow {
            kv_cache: &mut s.kv,
            ssm_state: s.ssm.as_mut(),
            position: 0,
        }];
        let err = m.forward_rows(&client, &two, &mut one).unwrap_err();
        assert!(err.to_string().contains("[N, 1]"), "{err}");
    }
}
