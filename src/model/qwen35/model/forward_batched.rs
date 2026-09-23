//! [`Qwen35Model`] decode step over a batch of sequences.
//!
//! One lane per sequence, all lanes stepping together. The KV lanes are
//! left-padded so every sequence ends at the shared cursor, and `kv_start`
//! names each lane's own first position, so a lane reads only its own
//! tokens. The GDN state carries a row per lane and each row evolves on its
//! own. A lane therefore produces the tokens it would produce alone,
//! whatever else is in flight.

use super::build::Qwen35Model;
use super::forward::AttentionStep;
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
    /// `[4, batch, 1]` i32 text positions: lane `b` sits at `rope_pos[b]`,
    /// with `t = h = w = rope_pos[b]` and `e = 0`.
    ///
    /// # Errors
    ///
    /// [`Error::ModelError`] when a position is past the rope table.
    pub fn text_positions_batched(
        &self,
        rope_pos: &[usize],
        device: &R::Device,
    ) -> Result<Tensor<R>> {
        let max_pos = self.rope.cos_cache().shape()[0];
        if let Some(pos) = rope_pos.iter().find(|p| **p >= max_pos) {
            return Err(Error::ModelError {
                reason: format!("qwen35: rope position {pos} is past the table of {max_pos}"),
            });
        }
        let batch = rope_pos.len();
        let mut data = Vec::with_capacity(4 * batch);
        for _ in 0..3 {
            data.extend(rope_pos.iter().map(|p| *p as i32));
        }
        data.resize(4 * batch, 0);
        Ok(Tensor::<R>::from_slice(&data, &[4, batch, 1], device)?)
    }

    /// One decode step for `batch` sequences: logits `[batch, 1, vocab]`.
    ///
    /// - `input_ids`: `[batch, 1]`, the token each lane decodes from
    /// - `positions`: `[4, batch, 1]` i32, from
    ///   [`text_positions_batched`](Self::text_positions_batched)
    /// - `kv_cache`, `gdn_state`: allocated with `batch` lanes, every lane
    ///   already prefilled and left-padded to the shared cursor
    /// - `kv_start`: `[batch]` i32, lane `b`'s first real cache position
    ///
    /// # Errors
    ///
    /// [`Error::ModelError`] when a shape disagrees with the contract or a
    /// cache holds fewer layers than the model has of that kind. Block
    /// errors propagate.
    pub fn forward_qwen35_batched_decode<C>(
        &self,
        client: &C,
        input_ids: &Tensor<R>,
        positions: &Tensor<R>,
        kv_cache: &mut LayeredKvCache<R>,
        gdn_state: &mut LayeredGdnState<R>,
        kv_start: &Tensor<R>,
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
        let ids_shape = input_ids.shape();
        if ids_shape.len() != 2 || ids_shape[1] != 1 {
            return Err(Error::ModelError {
                reason: format!("qwen35: batched decode takes [batch, 1] ids, got {ids_shape:?}"),
            });
        }
        let batch = ids_shape[0];
        let pos_shape = positions.shape();
        if pos_shape != [4, batch, 1].as_slice() || positions.dtype() != DType::I32 {
            return Err(Error::ModelError {
                reason: format!(
                    "qwen35: positions must be [4, {batch}, 1] i32, got {pos_shape:?} {:?}",
                    positions.dtype()
                ),
            });
        }
        if kv_start.shape() != [batch].as_slice() || kv_start.dtype() != DType::I32 {
            return Err(Error::ModelError {
                reason: format!(
                    "qwen35: kv_start must be [{batch}] i32, got {:?} {:?}",
                    kv_start.shape(),
                    kv_start.dtype()
                ),
            });
        }

        let embeds = self.embed_tokens(client, input_ids)?;
        let hidden = Var::new(embeds, false);
        let hidden = self.forward_layers(
            client,
            hidden,
            positions,
            kv_cache,
            gdn_state,
            &AttentionStep::PerLane { kv_start },
        )?;
        self.logits_from_hidden(client, &hidden)
    }
}
