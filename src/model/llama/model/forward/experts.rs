//! Per-expert weight access for MoE LLaMA layers.

use super::types::Llama;
use crate::model::llama::model::blocks::{ExpertWeights, LlamaFfn};
use numr::dtype::DType;
use numr::ops::ShapeOps;
use numr::runtime::Runtime;

impl<R: Runtime<DType = DType>> Llama<R> {
    /// Extract the weight tensors (gate, up, down projections) for a single MoE expert.
    ///
    /// Returns `None` if `layer_idx` is out of range, the layer is dense (non-MoE),
    /// or `expert_id` is out of range.
    pub fn get_expert_weights(
        &self,
        layer_idx: usize,
        expert_id: usize,
    ) -> Option<ExpertWeights<R>> {
        let block = self.layers.get(layer_idx)?;
        match &block.mlp {
            LlamaFfn::Moe(moe) => moe.get_expert_weights(expert_id),
            LlamaFfn::Dense(_) => None,
        }
    }

    /// Replace the weight tensors for a single MoE expert in-place.
    ///
    /// Returns an error if `layer_idx` is out of range, the layer is dense,
    /// `expert_id` is out of range, or an internal error occurs.
    pub fn set_expert_weights(
        &self,
        layer_idx: usize,
        expert_id: usize,
        weights: ExpertWeights<R>,
    ) -> crate::error::Result<()>
    where
        R::Client: ShapeOps<R>,
    {
        use crate::error::Error;
        let block = self
            .layers
            .get(layer_idx)
            .ok_or_else(|| Error::ModelError {
                reason: format!(
                    "layer_idx {} out of range (num_layers={})",
                    layer_idx,
                    self.layers.len()
                ),
            })?;
        match &block.mlp {
            LlamaFfn::Moe(moe) => moe.set_expert_weights(expert_id, weights),
            LlamaFfn::Dense(_) => Err(Error::ModelError {
                reason: format!("layer {} is a dense FFN layer, not MoE", layer_idx),
            }),
        }
    }
}
