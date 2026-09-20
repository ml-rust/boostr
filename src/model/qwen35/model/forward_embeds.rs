//! [`Qwen35Model`] forward over pre-built input embeddings.
//!
//! The token forward is the embedding lookup followed by this path. Image
//! inputs enter here: the caller splices vision-tower rows between text
//! embeddings and supplies the IMROPE positions of every row, so nothing
//! below the first block knows whether a row came from a token or an
//! image.

use super::build::Qwen35Model;
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
    /// Input embeddings of `input_ids` `[batch, seq]`: the table lookup
    /// plus the inverse Hadamard rotation when the table is stored
    /// rotated. Returns `[batch, seq, hidden_size]`, the exact rows the
    /// token forward feeds to the first block.
    pub fn embed_tokens<C>(&self, client: &C, input_ids: &Tensor<R>) -> Result<Tensor<R>>
    where
        C: ModelClient<R>,
    {
        let shape = input_ids.shape();
        if shape.len() != 2 {
            return Err(Error::ModelError {
                reason: format!("qwen35: input_ids must be [batch, seq], got {shape:?}"),
            });
        }
        Ok(self
            .embed_tokens
            .forward(client, input_ids)?
            .tensor()
            .clone())
    }

    /// `[4, seq]` i32 text positions: `t = h = w = rope_pos + i`, `e = 0`.
    ///
    /// # Errors
    ///
    /// [`Error::ModelError`] when `rope_pos + seq` exceeds the rope table.
    pub fn text_positions(
        &self,
        seq: usize,
        rope_pos: usize,
        device: &R::Device,
    ) -> Result<Tensor<R>> {
        let max_pos = self.rope.cos_cache().shape()[0];
        if rope_pos + seq > max_pos {
            return Err(Error::ModelError {
                reason: format!(
                    "qwen35: rope positions {rope_pos}..{} exceed the rope table of {max_pos}",
                    rope_pos + seq
                ),
            });
        }
        let mut data = Vec::with_capacity(4 * seq);
        for _ in 0..3 {
            data.extend((0..seq).map(|i| (rope_pos + i) as i32));
        }
        data.resize(4 * seq, 0);
        Ok(Tensor::<R>::from_slice(&data, &[4, seq], device)?)
    }

    /// Inference forward over input embeddings: logits
    /// `[batch, seq, vocab_size]`.
    ///
    /// - `embeds`: `[batch, seq, hidden_size]`, text rows from
    ///   [`embed_tokens`](Self::embed_tokens) and image rows from the vision
    ///   tower as they are
    /// - `positions`: `[4, seq]` i32 IMROPE streams `t, h, w, e` for every
    ///   row
    ///
    /// The KV slot of row `i` is `kv_cache.seq_len() + i`; the rope
    /// position is whatever `positions` holds.
    ///
    /// # Errors
    ///
    /// [`Error::ModelError`] when `embeds` is not `[batch, seq, hidden_size]`,
    /// `positions` is not `[4, seq]` i32, or `kv_cache` / `gdn_state` hold
    /// fewer layers than the model has of that kind. Block errors propagate.
    pub fn forward_qwen35_embeds<C>(
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
        let hidden_size = self.config.hidden_size;
        let shape = embeds.shape();
        if shape.len() != 3 || shape[2] != hidden_size {
            return Err(Error::ModelError {
                reason: format!(
                    "qwen35: embeds must be [batch, seq, {hidden_size}], got {shape:?}"
                ),
            });
        }
        let seq = shape[1];
        let pos_shape = positions.shape();
        if pos_shape != [4, seq].as_slice() || positions.dtype() != DType::I32 {
            return Err(Error::ModelError {
                reason: format!(
                    "qwen35: positions must be [4, {seq}] i32, got {pos_shape:?} {:?}",
                    positions.dtype()
                ),
            });
        }
        let hidden = Var::new(embeds.clone(), false);
        let hidden = self.forward_layers(client, hidden, positions, kv_cache, gdn_state)?;
        let logits = self.lm_head.forward(client, &hidden)?;
        Ok(logits.tensor().clone())
    }
}

#[cfg(test)]
mod tests {
    use super::super::build::tests::{H_KV, HD, HIDDEN, MAX_POS, SEQ, VOCAB, tiny_model};
    use super::*;
    use crate::inference::kv_cache::layered::LayeredKvCacheConfig;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    fn caches(
        device: &CpuDevice,
        model: &Qwen35Model<CpuRuntime>,
    ) -> (LayeredKvCache<CpuRuntime>, LayeredGdnState<CpuRuntime>) {
        let kv_config = LayeredKvCacheConfig {
            batch_size: 1,
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
            1,
            DType::F32,
            device,
        )
        .unwrap();
        (kv, gdn)
    }

    fn tokens(device: &CpuDevice) -> Tensor<CpuRuntime> {
        let ids: Vec<i64> = (0..SEQ).map(|i| ((i * 7 + 1) % VOCAB) as i64).collect();
        Tensor::<CpuRuntime>::from_slice(&ids, &[1, SEQ], device).unwrap()
    }

    /// The embeds path with text positions is the token path.
    #[test]
    fn embeds_path_matches_token_path() {
        let (client, device) = cpu_setup();
        let model = tiny_model(&device, 0x3535_0020);
        let ids = tokens(&device);

        let (mut kv, mut gdn) = caches(&device, &model);
        let from_ids = model
            .forward_qwen35(&client, &ids, &mut kv, &mut gdn, 0)
            .unwrap();

        let (mut kv, mut gdn) = caches(&device, &model);
        let embeds = model.embed_tokens(&client, &ids).unwrap();
        assert_eq!(embeds.shape(), &[1, SEQ, HIDDEN]);
        let positions = model.text_positions(SEQ, 0, &device).unwrap();
        let from_embeds = model
            .forward_qwen35_embeds(&client, &embeds, &positions, &mut kv, &mut gdn)
            .unwrap();

        assert_eq!(from_ids.shape(), &[1, SEQ, VOCAB]);
        assert_eq!(from_ids.to_vec::<f32>(), from_embeds.to_vec::<f32>());
    }

    #[test]
    fn text_positions_layout() {
        let (_client, device) = cpu_setup();
        let model = tiny_model(&device, 0x3535_0021);
        let p = model.text_positions(3, 5, &device).unwrap();
        assert_eq!(p.shape(), &[4, 3]);
        assert_eq!(p.to_vec::<i32>(), vec![5, 6, 7, 5, 6, 7, 5, 6, 7, 0, 0, 0]);
        assert!(model.text_positions(3, MAX_POS - 2, &device).is_err());
    }

    #[test]
    fn rejects_bad_shapes() {
        let (client, device) = cpu_setup();
        let model = tiny_model(&device, 0x3535_0022);
        let (mut kv, mut gdn) = caches(&device, &model);
        let embeds = model.embed_tokens(&client, &tokens(&device)).unwrap();

        let short = model.text_positions(SEQ - 1, 0, &device).unwrap();
        assert!(
            model
                .forward_qwen35_embeds(&client, &embeds, &short, &mut kv, &mut gdn)
                .is_err()
        );

        let positions = model.text_positions(SEQ, 0, &device).unwrap();
        let narrow = embeds
            .narrow(2, 0, HIDDEN - 1)
            .unwrap()
            .contiguous()
            .unwrap();
        assert!(
            model
                .forward_qwen35_embeds(&client, &narrow, &positions, &mut kv, &mut gdn)
                .is_err()
        );
    }
}
