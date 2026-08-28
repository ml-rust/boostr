//! [`VoxCpm2Model::set_activation_checkpointing`] — split out of `loader.rs`
//! to keep it under the crate's 500-line hard limit for model-architecture
//! files.

use super::VoxCpm2Model;
use numr::dtype::DType;
use numr::runtime::Runtime;

impl<R: Runtime<DType = DType>> VoxCpm2Model<R> {
    /// Turn activation checkpointing on or off across every transformer
    /// stack a training pass runs: `feat_encoder`, `base_lm`,
    /// `residual_lm`, and `feat_decoder`.
    ///
    /// `on` drops each layer's intermediates during the forward pass and
    /// recomputes them during backward, which is what caps training VRAM.
    /// It costs ~33% extra compute.
    ///
    /// `off` is the default and the only correct setting for inference.
    /// The KV-cached decode path never reads the flag: a recomputed segment
    /// must not touch the cache, so `forward_cached` is never checkpointed.
    ///
    /// The AudioVAE (`vae_encoder`/`vae_decoder`) and the `fsq`/`aux`
    /// projections carry no layer stack and are unaffected.
    pub fn set_activation_checkpointing(&mut self, on: bool) {
        self.feat_encoder.set_activation_checkpointing(on);
        self.base_lm.set_activation_checkpointing(on);
        self.residual_lm.set_activation_checkpointing(on);
        self.feat_decoder.set_activation_checkpointing(on);
    }

    /// Whether every stack [`Self::set_activation_checkpointing`] touches
    /// currently runs with activation checkpointing.
    pub fn activation_checkpointing(&self) -> bool {
        self.feat_encoder.activation_checkpointing()
            && self.base_lm.activation_checkpointing()
            && self.residual_lm.activation_checkpointing()
            && self.feat_decoder.activation_checkpointing()
    }
}
