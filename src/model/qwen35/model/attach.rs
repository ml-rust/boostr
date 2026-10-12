//! [`Attach`]: wraps a built `qwen35` module in its Hadamard rotation when
//! the contract names its GGUF tensor, and [`gguf_layer_name`].
//!
//! The Hadamard contract is keyed on the ORIGINAL GGUF names
//! (`blk.N.attn_q.weight`), so the loader rebuilds those from the layer
//! index and suffix before asking `HadamardContract::rotates`.

use crate::error::{Error, Result};
use crate::format::gguf::HadamardContract;
use crate::nn::{
    HadamardRotation, MaybeQuantEmbedding, MaybeQuantLinear, MaybeRotatedEmbedding,
    MaybeRotatedLinear, RotatedEmbedding, RotatedLinear,
};
use numr::dtype::DType;
use numr::runtime::Runtime;
use std::cell::RefCell;
use std::collections::HashMap;

/// Wraps a built module in its Hadamard rotation when the contract names
/// its GGUF tensor.
///
/// `by_width` caches one `HadamardRotation` per sign width so the handful
/// of distinct widths (one per tensor shape, not one per layer) are each
/// uploaded to `device` once and cheaply cloned for every tensor that
/// shares the width — `Tensor::clone` shares storage rather than copying
/// device memory.
pub(super) struct Attach<'a, R: Runtime> {
    hadamard: Option<&'a HadamardContract>,
    device: &'a R::Device,
    by_width: RefCell<HashMap<usize, HadamardRotation<R>>>,
}

impl<'a, R: Runtime<DType = DType>> Attach<'a, R> {
    pub(super) fn new(hadamard: Option<&'a HadamardContract>, device: &'a R::Device) -> Self {
        Self {
            hadamard,
            device,
            by_width: RefCell::new(HashMap::new()),
        }
    }

    /// The rotation for an input of `width` features. Only valid to call
    /// when `self.hadamard` is `Some`; the caller checks `rotates` /
    /// `inverts` first, which fail closed on `None`. Cached per `width` in
    /// `self.by_width`.
    fn rotation(&self, hadamard: &HadamardContract, width: usize) -> Result<HadamardRotation<R>> {
        if let Some(cached) = self.by_width.borrow().get(&width) {
            return Ok(cached.clone());
        }
        let signs = hadamard.signs_for_width(width)?;
        let built = HadamardRotation::new(hadamard.block_size, signs, DType::F32, self.device)?;
        self.by_width.borrow_mut().insert(width, built.clone());
        Ok(built)
    }

    /// `Rotated` when `hadamard.rotates(gguf_name)`, else `Plain`. The sign
    /// width is `inner.shape()[1]`, the linear's `in_features`.
    pub(super) fn linear(
        &self,
        gguf_name: &str,
        inner: MaybeQuantLinear<R>,
    ) -> Result<MaybeRotatedLinear<R>> {
        let Some(hadamard) = self.hadamard.filter(|h| h.rotates(gguf_name)) else {
            return Ok(MaybeRotatedLinear::Plain(inner));
        };
        let shape = inner.shape();
        if shape.len() != 2 {
            return Err(Error::ModelError {
                reason: format!("qwen35: '{gguf_name}' must be 2-D to rotate, got {shape:?}"),
            });
        }
        let rotation = self.rotation(hadamard, shape[1])?;
        Ok(MaybeRotatedLinear::Rotated(Box::new(RotatedLinear::new(
            inner, rotation,
        )?)))
    }

    /// `Rotated` when `hadamard.inverts(gguf_name)`, else `Plain`. The sign
    /// width is the embedding dim.
    pub(super) fn embedding(
        &self,
        gguf_name: &str,
        inner: MaybeQuantEmbedding<R>,
        embed_dim: usize,
    ) -> Result<MaybeRotatedEmbedding<R>> {
        let Some(hadamard) = self.hadamard.filter(|h| h.inverts(gguf_name)) else {
            return Ok(MaybeRotatedEmbedding::Plain(inner));
        };
        let rotation = self.rotation(hadamard, embed_dim)?;
        Ok(MaybeRotatedEmbedding::Rotated(RotatedEmbedding::new(
            inner, rotation,
        )?))
    }
}

/// `blk.{i}.{suffix}`: the GGUF name a layer tensor was stored under.
pub(super) fn gguf_layer_name(layer: usize, suffix: &str) -> String {
    format!("blk.{layer}.{suffix}")
}
