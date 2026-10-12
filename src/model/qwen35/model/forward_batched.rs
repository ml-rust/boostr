//! [`Qwen35Model`] decode step over a batch of sequences.
//!
//! One lane per sequence, all lanes stepping together. Each KV lane holds
//! its own sequence with its own start and end, so a lane reads only its
//! own tokens and writes at its own next position. The GDN state carries a
//! row per lane and each row evolves on its own. A lane therefore produces
//! the tokens it would produce alone, whatever else is in flight.

use super::build::Qwen35Model;
use super::forward_layers::AttentionStep;
use crate::error::{Error, Result};
use crate::inference::{LaneKvStep, LayeredGdnState, LayeredKvCache};
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
    /// One decode step for `batch` sequences: logits `[batch, 1, vocab]`.
    ///
    /// - `input_ids`: `[batch, 1]`, the token each lane decodes from
    /// - `positions`: `[4, batch, 1]` i32, from
    ///   [`LaneTable::positions`](crate::inference::LaneTable::positions)
    /// - `kv_cache`, `gdn_state`: allocated with `batch` lanes, every lane
    ///   already prefilled from its own start
    /// - `lanes`: each lane's key range and write position, as a
    ///   [`LaneTable`](crate::inference::LaneTable) hands them out
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
        lanes: &LaneKvStep<'_, R>,
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
        for (name, bound) in [("kv_start", lanes.kv_start), ("kv_end", lanes.kv_end)] {
            if bound.shape() != [batch].as_slice() || bound.dtype() != DType::I32 {
                return Err(Error::ModelError {
                    reason: format!(
                        "qwen35: {name} must be [{batch}] i32, got {:?} {:?}",
                        bound.shape(),
                        bound.dtype()
                    ),
                });
            }
        }
        if lanes.write_pos.len() != batch {
            return Err(Error::ModelError {
                reason: format!(
                    "qwen35: write_pos must carry one position per lane, got {} for {batch} lanes",
                    lanes.write_pos.len()
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
            &AttentionStep::PerLane { lanes },
        )?;
        self.logits_from_hidden(client, &hidden)
    }
}
