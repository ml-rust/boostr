//! [`Qwen35Model`] prefill forwards that skip the `lm_head` rows nothing reads.
//!
//! A prefill samples from the logits of its last row only. Over a 248320-token
//! vocabulary, a 512-token chunk of full logits takes 508 MB. These forwards
//! leave the caches exactly as [`forward_qwen35`](Qwen35Model::forward_qwen35)
//! leaves them, and run `output_norm` and `lm_head` on the last row, or skip
//! them.

use super::build::Qwen35Model;
use super::forward_layers::AttentionStep;
use crate::error::{Error, Result};
use crate::inference::{LayeredGdnState, LayeredKvCache};
use crate::model::traits::ModelClient;
use crate::quant::traits::DequantOps;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, ConvOps, IndexingOps, MatmulOps,
    ReduceOps, ScalarOps, ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> Qwen35Model<R>
where
    R::Client: IndexingOps<R>,
{
    /// [`forward_qwen35`](Self::forward_qwen35), returning the logits of the
    /// last token only: `[batch, 1, vocab_size]`.
    ///
    /// # Errors
    ///
    /// As `forward_qwen35`, and [`Error::ModelError`] when `input_ids` holds
    /// no token.
    pub fn forward_qwen35_last<C>(
        &self,
        client: &C,
        input_ids: &Tensor<R>,
        kv_cache: &mut LayeredKvCache<R>,
        gdn_state: &mut LayeredGdnState<R>,
        rope_pos: usize,
    ) -> Result<Tensor<R>>
    where
        C: ModelClient<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + ConvOps<R>
            + DequantOps<R>
            + MatmulOps<R>,
    {
        let (embeds, positions) = self.token_inputs(client, input_ids, rope_pos)?;
        self.forward_qwen35_embeds_last(client, &embeds, &positions, kv_cache, gdn_state)
    }

    /// [`forward_qwen35_embeds`](Self::forward_qwen35_embeds), returning the
    /// logits of the last row only: `[batch, 1, vocab_size]`.
    ///
    /// # Errors
    ///
    /// As `forward_qwen35_embeds`, and [`Error::ModelError`] when `embeds`
    /// holds no row.
    pub fn forward_qwen35_embeds_last<C>(
        &self,
        client: &C,
        embeds: &Tensor<R>,
        positions: &Tensor<R>,
        kv_cache: &mut LayeredKvCache<R>,
        gdn_state: &mut LayeredGdnState<R>,
    ) -> Result<Tensor<R>>
    where
        C: ModelClient<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + ConvOps<R>
            + DequantOps<R>
            + MatmulOps<R>,
    {
        self.check_embeds(embeds, positions)?;
        let seq = embeds.shape()[1];
        let Some(last) = seq.checked_sub(1) else {
            return Err(Error::ModelError {
                reason: "qwen35: embeds holds 0 rows. Pass at least one row.".to_string(),
            });
        };
        let hidden = self.forward_blocks(
            client,
            Var::new(embeds.clone(), false),
            positions,
            kv_cache,
            gdn_state,
            &AttentionStep::Shared,
        )?;
        let row = hidden.tensor().narrow(1, last, 1)?.contiguous()?;
        drop(hidden);
        let normed = self.norm.forward(client, &Var::new(row, false))?;
        self.logits_from_hidden(client, &normed)
    }

    /// Run `input_ids` through every block to advance `kv_cache` and
    /// `gdn_state`, computing no logits. A chunked prefill runs its earlier
    /// chunks through this.
    ///
    /// # Errors
    ///
    /// As [`forward_qwen35`](Self::forward_qwen35), and [`Error::ModelError`]
    /// when `input_ids` holds no token.
    pub fn advance_qwen35<C>(
        &self,
        client: &C,
        input_ids: &Tensor<R>,
        kv_cache: &mut LayeredKvCache<R>,
        gdn_state: &mut LayeredGdnState<R>,
        rope_pos: usize,
    ) -> Result<()>
    where
        C: ModelClient<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + ConvOps<R>
            + DequantOps<R>
            + MatmulOps<R>,
    {
        let (embeds, positions) = self.token_inputs(client, input_ids, rope_pos)?;
        self.advance_qwen35_embeds(client, &embeds, &positions, kv_cache, gdn_state)
    }

    /// [`advance_qwen35`](Self::advance_qwen35) over input embeddings.
    ///
    /// # Errors
    ///
    /// As [`forward_qwen35_embeds`](Self::forward_qwen35_embeds).
    pub fn advance_qwen35_embeds<C>(
        &self,
        client: &C,
        embeds: &Tensor<R>,
        positions: &Tensor<R>,
        kv_cache: &mut LayeredKvCache<R>,
        gdn_state: &mut LayeredGdnState<R>,
    ) -> Result<()>
    where
        C: ModelClient<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + ConvOps<R>
            + DequantOps<R>
            + MatmulOps<R>,
    {
        self.check_embeds(embeds, positions)?;
        self.forward_blocks(
            client,
            Var::new(embeds.clone(), false),
            positions,
            kv_cache,
            gdn_state,
            &AttentionStep::Shared,
        )?;
        Ok(())
    }

    /// Input embeddings and text positions of `input_ids` `[batch, seq]`.
    ///
    /// Refuses an empty `seq` before embedding, because an empty index buffer
    /// can give a WebGPU backend error in place of this message.
    fn token_inputs<C>(
        &self,
        client: &C,
        input_ids: &Tensor<R>,
        rope_pos: usize,
    ) -> Result<(Tensor<R>, Tensor<R>)>
    where
        C: ModelClient<R>,
    {
        if input_ids.shape().get(1) == Some(&0) {
            return Err(Error::ModelError {
                reason: "qwen35: input_ids holds 0 tokens. Pass at least one token.".to_string(),
            });
        }
        let embeds = self.embed_tokens(client, input_ids)?;
        let positions = self.text_positions(input_ids.shape()[1], rope_pos, input_ids.device())?;
        Ok((embeds, positions))
    }
}

#[cfg(test)]
mod tests {
    use super::super::build::tests::{
        H_KV, HD, Head, MAX_POS, SEQ, VOCAB, chunked_model, rotated_model, tiny_model,
    };
    use super::*;
    use crate::inference::kv_cache::layered::LayeredKvCacheConfig;
    use crate::test_utils::{cpu_setup, tensor_max_abs_diff as max_abs_diff};
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    type Caches = (LayeredKvCache<CpuRuntime>, LayeredGdnState<CpuRuntime>);

    fn caches(device: &CpuDevice, model: &Qwen35Model<CpuRuntime>, batch: usize) -> Caches {
        let kv_config = LayeredKvCacheConfig {
            batch_size: batch,
            num_kv_heads: H_KV,
            initial_capacity: MAX_POS,
            max_seq_len: MAX_POS,
            head_dim: HD,
            dtype: DType::F32,
        };
        let kv =
            LayeredKvCache::<CpuRuntime>::new(model.num_attention_layers(), &kv_config, device)
                .unwrap();
        let gdn = LayeredGdnState::<CpuRuntime>::zeros(
            model.num_gdn_layers(),
            model.gdn_config(),
            batch,
            DType::F32,
            device,
        )
        .unwrap();
        (kv, gdn)
    }

    fn tokens(device: &CpuDevice, batch: usize) -> Tensor<CpuRuntime> {
        let ids: Vec<i64> = (0..batch * SEQ)
            .map(|i| ((i * 5 + 3) % VOCAB) as i64)
            .collect();
        Tensor::<CpuRuntime>::from_slice(&ids, &[batch, SEQ], device).unwrap()
    }

    /// The last row of the full forward, as `[batch, 1, vocab]`.
    fn full_last_row(
        client: &CpuClient,
        model: &Qwen35Model<CpuRuntime>,
        ids: &Tensor<CpuRuntime>,
        device: &CpuDevice,
    ) -> Tensor<CpuRuntime> {
        let (mut kv, mut gdn) = caches(device, model, ids.shape()[0]);
        let full = model
            .forward_qwen35(client, ids, &mut kv, &mut gdn, 0)
            .unwrap();
        assert_eq!(full.shape()[1], SEQ);
        full.narrow(1, SEQ - 1, 1).unwrap().contiguous().unwrap()
    }

    fn assert_same(a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) {
        assert_eq!(a.shape(), b.shape());
        let diff = max_abs_diff(a, b);
        assert!(diff <= 1e-6, "logits diff {diff}");
    }

    #[test]
    fn last_row_equals_the_full_forward() {
        let (client, device) = cpu_setup();
        for (name, model) in [
            ("plain", tiny_model(&device, 0x3535_0030)),
            ("rotated", rotated_model(&device, 0x3535_0031)),
            (
                "tied",
                chunked_model(&device, 0x3535_0034, 2, false, Head::Tied),
            ),
        ] {
            for batch in [1, 2] {
                let ids = tokens(&device, batch);
                let expected = full_last_row(&client, &model, &ids, &device);
                let (mut kv, mut gdn) = caches(&device, &model, batch);
                let last = model
                    .forward_qwen35_last(&client, &ids, &mut kv, &mut gdn, 0)
                    .unwrap();
                assert_eq!(last.shape(), &[batch, 1, VOCAB], "{name}");
                assert_same(&last, &expected);
                assert_eq!(kv.seq_len(), SEQ, "{name}");
            }
        }
    }

    /// Chunks advanced with no logits, then a last-row chunk, leave the state
    /// and the logits of one forward over all the tokens. The next decode step
    /// reads that state.
    #[test]
    fn advance_then_last_equals_one_forward_and_decodes_alike() {
        let (client, device) = cpu_setup();
        let model = chunked_model(&device, 0x3535_0032, 2, false, Head::Own);
        let ids = tokens(&device, 1);
        let expected = full_last_row(&client, &model, &ids, &device);

        let (mut kv, mut gdn) = caches(&device, &model, 1);
        let head = ids.narrow(1, 0, 3).unwrap().contiguous().unwrap();
        let tail = ids.narrow(1, 3, SEQ - 3).unwrap().contiguous().unwrap();
        model
            .advance_qwen35(&client, &head, &mut kv, &mut gdn, 0)
            .unwrap();
        assert_eq!(kv.seq_len(), 3);
        let last = model
            .forward_qwen35_last(&client, &tail, &mut kv, &mut gdn, 3)
            .unwrap();
        assert_same(&last, &expected);

        let (mut full_kv, mut full_gdn) = caches(&device, &model, 1);
        model
            .forward_qwen35(&client, &ids, &mut full_kv, &mut full_gdn, 0)
            .unwrap();
        let step = ids.narrow(1, 0, 1).unwrap().contiguous().unwrap();
        let a = model
            .forward_qwen35(&client, &step, &mut kv, &mut gdn, SEQ)
            .unwrap();
        let b = model
            .forward_qwen35(&client, &step, &mut full_kv, &mut full_gdn, SEQ)
            .unwrap();
        assert_same(&a, &b);
    }

    #[test]
    fn an_empty_prefill_is_an_error_and_writes_nothing() {
        let (client, device) = cpu_setup();
        let model = tiny_model(&device, 0x3535_0033);
        let (mut kv, mut gdn) = caches(&device, &model, 1);
        let ids = Tensor::<CpuRuntime>::from_slice::<i64>(&[], &[1, 0], &device).unwrap();
        let expected = "qwen35: input_ids holds 0 tokens. Pass at least one token.";
        let last = model
            .forward_qwen35_last(&client, &ids, &mut kv, &mut gdn, 0)
            .unwrap_err()
            .to_string();
        assert!(last.contains(expected), "{last}");
        let advance = model
            .advance_qwen35(&client, &ids, &mut kv, &mut gdn, 0)
            .unwrap_err()
            .to_string();
        assert!(advance.contains(expected), "{advance}");
        assert_eq!(kv.seq_len(), 0);
    }
}
