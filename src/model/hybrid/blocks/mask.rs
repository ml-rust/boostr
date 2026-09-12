//! [`AttentionBlock::attention_mask`]: the additive mask one attention step
//! adds to its scores — causal, sliding-window, or ALiBi bias.

use super::attention::AttentionBlock;
use crate::error::Result;
use crate::model::attention_mask::causal_window_mask;
use crate::model::traits::ModelClient;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> AttentionBlock<R> {
    /// Additive attention mask for one attention step.
    ///
    /// Always `Some`: the ALiBi branch returns the bias, and every other
    /// configuration returns a causal mask (windowed when `sliding_window > 0`).
    /// The `Option` is the caller's argument type, not a signal that masking is
    /// optional — an unmasked prefill lets every position attend to FUTURE
    /// tokens, which stays invisible to shape checks and still emits fluent
    /// text.
    ///
    /// `dtype` is the dtype of the attention scores this mask is added to. The
    /// additive-mask sites do not reconcile dtypes, so a stack running in
    /// BF16/F16 must state its dtype here.
    ///
    /// Row `i` is absolute position `position + i` and the cache holds
    /// `sk = position + sq` keys, so [`causal_window_mask`] derives the key
    /// offset from `sk - sq` and needs no extra argument.
    // Seven independent scalars, none derivable from another: `position` is the
    // ALiBi branch's own key offset, and `dtype`/`device` describe the scores
    // this mask is added to, not each other. Bundling them into a struct would
    // add a type whose only job is to be destructured back at the one call site.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention_mask<C>(
        &self,
        client: &C,
        batch: usize,
        sq: usize,
        sk: usize,
        position: usize,
        dtype: DType,
        device: &R::Device,
    ) -> Result<Option<Var<R>>>
    where
        C: ModelClient<R>,
        R::Client: numr::ops::TypeConversionOps<R>,
    {
        if self.use_alibi {
            // ALiBi's own kernel writes the causal structure along with the
            // distance bias, so the sliding window does not apply here.
            let bias = Tensor::<R>::zeros(&[batch, self.num_heads, sq, sk], DType::F32, device)?;
            client.alibi_add_bias_causal(&bias, batch, self.num_heads, sq, sk, position)?;
            // The ALiBi kernel writes F32 slopes; cast once so the bias
            // carries the dtype of the scores it is added to.
            let bias = bias.to_dtype(dtype)?;
            Ok(Some(Var::new(bias, false)))
        } else {
            // ALWAYS masked, even with no sliding window. This branch is the
            // prefill/training path: without a causal mask every position
            // attends to FUTURE tokens, which makes the next-token objective
            // trivially cheatable and corrupts every prompt position at
            // inference. It stays invisible to shape checks and still emits
            // fluent text — the same failure that survived in the LLaMA
            // decoder until parity testing caught it.
            //
            // `window_size == 0` yields a pure causal mask, so this covers both
            // the windowed and unwindowed cases. On the decode path
            // (`sq == 1`, `sk == position + 1`) the shared builder's key offset
            // makes every cached key visible, as it must be.
            //
            // The window predicate alone does not mask the future — the shared
            // builder always applies causality alongside it.
            let mask = causal_window_mask(client, sq, sk, self.sliding_window, dtype, device)?;
            Ok(Some(Var::new(mask, false)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{Linear, RmsNorm};
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    const NEG: f32 = f32::MIN;

    /// Reference mask: `0.0` where query `i` may attend key `j`, `f32::MIN` else.
    ///
    /// Query row `i` sits at absolute position `sk - sq + i`; `window_size == 0`
    /// means unlimited, matching the kernel contract.
    fn expected_mask(sq: usize, sk: usize, window_size: usize) -> Vec<f32> {
        let offset = sk - sq;
        let mut out = Vec::with_capacity(sq * sk);
        for i in 0..sq {
            let pos = offset + i;
            for j in 0..sk {
                let future = j > pos;
                let too_old = window_size > 0 && j + window_size <= pos;
                out.push(if future || too_old { NEG } else { 0.0 });
            }
        }
        out
    }

    /// An attention block carrying only the two flags under test. The projections
    /// are never touched by `attention_mask`, so they are left minimal.
    fn flag_block(use_alibi: bool, sliding_window: usize) -> AttentionBlock<CpuRuntime> {
        let (_, device) = cpu_setup();
        let w = || Tensor::<CpuRuntime>::zeros(&[4, 4], DType::F32, &device).unwrap();
        let n = || Tensor::<CpuRuntime>::zeros(&[4], DType::F32, &device).unwrap();
        AttentionBlock {
            input_layernorm: RmsNorm::new(n(), 1e-5, false),
            q_proj: Linear::new(w(), None, false),
            k_proj: Linear::new(w(), None, false),
            v_proj: Linear::new(w(), None, false),
            o_proj: Linear::new(w(), None, false),
            post_attention_layernorm: RmsNorm::new(n(), 1e-5, false),
            gate_proj: Linear::new(w(), None, false),
            up_proj: Linear::new(w(), None, false),
            down_proj: Linear::new(w(), None, false),
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 2,
            use_alibi,
            sliding_window,
        }
    }

    fn mask_values(window_size: usize, sq: usize, sk: usize, position: usize) -> Vec<f32> {
        let (client, device) = cpu_setup();
        let block = flag_block(false, window_size);
        let mask = block
            .attention_mask(&client, 1, sq, sk, position, DType::F32, &device)
            .unwrap()
            .expect("a positive window must produce a mask");
        assert_eq!(mask.shape(), &[1, 1, sq, sk]);
        mask.tensor().to_vec::<f32>()
    }

    #[test]
    fn windowed_prefill_mask_has_the_exact_expected_pattern() {
        let actual = mask_values(3, 6, 6, 0);
        assert_eq!(actual, expected_mask(6, 6, 3));

        // Spelled out for the record: row `i` keeps exactly `j ∈ (i - 3, i]`.
        #[rustfmt::skip]
        let literal = vec![
            0.0, NEG, NEG, NEG, NEG, NEG,
            0.0, 0.0, NEG, NEG, NEG, NEG,
            0.0, 0.0, 0.0, NEG, NEG, NEG,
            NEG, 0.0, 0.0, 0.0, NEG, NEG,
            NEG, NEG, 0.0, 0.0, 0.0, NEG,
            NEG, NEG, NEG, 0.0, 0.0, 0.0,
        ];
        assert_eq!(actual, literal);
    }

    #[test]
    fn windowed_mask_never_overflows_to_negative_infinity() {
        // `triu` and `tril` regions must stay disjoint: an overlap would sum
        // `f32::MIN + f32::MIN` and overflow to `-inf`.
        for window_size in 1..=5 {
            for value in mask_values(window_size, 6, 6, 0) {
                assert!(value.is_finite(), "window {window_size} produced {value}");
            }
        }
    }

    #[test]
    fn window_wider_than_the_sequence_is_pure_causal() {
        assert_eq!(mask_values(10, 4, 4, 0), expected_mask(4, 4, 0));
        assert_eq!(mask_values(4, 4, 4, 0), expected_mask(4, 4, 0));
    }

    #[test]
    fn decode_mask_rows_are_offset_by_the_cached_key_count() {
        // One new query against four cached keys: the query is at absolute
        // position 3, so a window of 2 keeps keys 2 and 3 only.
        assert_eq!(mask_values(2, 1, 4, 3), vec![NEG, NEG, 0.0, 0.0]);
        assert_eq!(mask_values(2, 1, 4, 3), expected_mask(1, 4, 2));

        // Two new queries against five cached keys, window 3.
        assert_eq!(mask_values(3, 2, 5, 3), expected_mask(2, 5, 3));
    }

    #[test]
    fn alibi_mask_covers_every_batch_and_head() {
        let (client, device) = cpu_setup();
        let block = flag_block(true, 0);
        let mask = block
            .attention_mask(&client, 2, 3, 3, 0, DType::F32, &device)
            .unwrap()
            .expect("ALiBi always produces a bias mask");
        assert_eq!(mask.shape(), &[2, 2, 3, 3]);
    }

    #[test]
    fn alibi_ignores_the_sliding_window() {
        // ALiBi's kernel writes causality together with the distance bias; the two
        // mechanisms do not compose, so the window must not change the result.
        let (client, device) = cpu_setup();
        let unwindowed = flag_block(true, 0)
            .attention_mask(&client, 1, 4, 4, 0, DType::F32, &device)
            .unwrap()
            .expect("ALiBi always produces a bias mask");
        let windowed = flag_block(true, 2)
            .attention_mask(&client, 1, 4, 4, 0, DType::F32, &device)
            .unwrap()
            .expect("ALiBi always produces a bias mask");
        assert_eq!(
            unwindowed.tensor().to_vec::<f32>(),
            windowed.tensor().to_vec::<f32>()
        );
    }

    #[test]
    fn plain_attention_is_still_causal() {
        // Regression guard for a REAL bug this replaced: the unwindowed, non-ALiBi
        // branch used to return `None`, so hybrid prefill attended to FUTURE
        // tokens. Shape checks cannot see it and the model still emits fluent
        // text. Prefill must be causal; decode must see every cached key.
        let (client, device) = cpu_setup();
        let block = flag_block(false, 0);

        // Prefill: strictly-upper triangle masked, diagonal and below visible.
        let mask = block
            .attention_mask(&client, 1, 4, 4, 0, DType::F32, &device)
            .unwrap()
            .expect("prefill must be masked, or attention sees the future");
        let values = mask.tensor().to_vec::<f32>();
        for i in 0..4 {
            for j in 0..4 {
                let got = values[i * 4 + j];
                if j > i {
                    assert_eq!(got, f32::MIN, "future key ({i},{j}) must be masked");
                } else {
                    assert_eq!(got, 0.0, "past/self key ({i},{j}) must be visible");
                }
            }
        }

        // Decode: one query against five cached keys — all are in the past.
        let mask = block
            .attention_mask(&client, 1, 1, 6, 5, DType::F32, &device)
            .unwrap()
            .expect("decode is masked too");
        assert!(
            mask.tensor().to_vec::<f32>().iter().all(|v| *v == 0.0),
            "every cached key precedes the query, so none may be masked"
        );
    }
}
