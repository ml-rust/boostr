//! Mixed-radix index packing/unpacking and straight-through/identity helpers
//! for [`Fsq`].
//!
//! Split out of `quantizer.rs` to keep that file readable — these are the
//! mechanically separable pieces (index codec + STE/passthrough), not part
//! of the `bound`/`quantize_codes` core
//! that stays in `quantizer.rs`. See that module's docs for the math.

use super::quantizer::Fsq;
use crate::error::{Error, Result};
use numr::autograd::{Var, var_add};
use numr::dtype::DType;
use numr::ops::TensorOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Identity view of `v` that preserves its autograd id and grad_fn.
///
/// `Var::clone()` mints a *fresh* `TensorId` for the clone (see
/// `numr::autograd::Var::clone`), which is correct when the clone becomes an
/// independent graph node but WRONG when a Var is meant to pass straight
/// through unchanged: if `v` is itself a leaf (`grad_fn = None`), the clone
/// becomes a second, disconnected leaf, and any gradient computed for it never
/// reaches `v.id()` — silently orphaning whichever caller is holding onto the
/// original id (exactly the "severs autograd" landmine, via `Clone` instead of
/// `Var::new`). This reproduces `numr`'s own `var_identity` helper (used in
/// `var_contiguous`) since that one is private to `numr`.
pub(crate) fn var_passthrough<R: Runtime>(v: &Var<R>) -> Var<R> {
    match (v.requires_grad(), v.grad_fn().cloned()) {
        (true, Some(grad_fn)) => {
            Var::with_id_and_grad_fn(v.tensor().clone(), v.id(), Some(grad_fn))
        }
        (true, None) => Var::with_id(v.tensor().clone(), v.id(), true),
        (false, _) => Var::with_id(v.tensor().clone(), v.id(), false),
    }
}

impl<R: Runtime<DType = DType>> Fsq<R> {
    /// `z + (round(z) - z).detach()`: forward = `round(z)`, backward = identity.
    ///
    /// The `(round(z) - z)` residual is computed on the raw `Tensor` (not
    /// through `var_*`) and wrapped in `Var::new(_, false)` — this is the
    /// legitimate "detach" use of `Var::new`: it is a genuine constant offset
    /// with no gradient path of its own, exactly mirroring PyTorch's
    /// `(zhat - z).detach()`. `z` itself stays on the tracked graph the whole
    /// time, so `var_add` passes the incoming gradient straight through to it.
    pub(super) fn round_ste<C>(&self, z: &Var<R>, client: &C) -> Result<Var<R>>
    where
        C: RuntimeClient<R> + TensorOps<R>,
        R::Client: TensorOps<R>,
    {
        let z_tensor = z.tensor();
        let rounded = client.round(z_tensor).map_err(Error::Numr)?;
        let residual = client.sub(&rounded, z_tensor).map_err(Error::Numr)?;
        let residual = Var::new(residual, false);
        var_add(z, &residual, client).map_err(Error::Numr)
    }

    /// Mixed-radix pack: `round(sum((code * half_width + half_width) * basis))`
    /// cast to `DType::I32`.
    ///
    /// Not on the gradient path (indices are discrete) — operates on the raw
    /// `Tensor`, not `Var`.
    pub(super) fn codes_to_indices<C>(&self, client: &C, zhat: &Tensor<R>) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R> + TensorOps<R>,
    {
        let scaled = client.mul(zhat, &self.half_width).map_err(Error::Numr)?;
        let scaled = client.add(&scaled, &self.half_width).map_err(Error::Numr)?;
        let weighted = client.mul(&scaled, &self.basis).map_err(Error::Numr)?;

        let last_dim = weighted.shape().len().saturating_sub(1);
        let summed = client
            .sum(&weighted, &[last_dim], false)
            .map_err(Error::Numr)?;
        let rounded = client.round(&summed).map_err(Error::Numr)?;
        client.cast(&rounded, DType::I32).map_err(Error::Numr)
    }

    /// Inverse of [`codes_to_indices`](Self::codes_to_indices): mixed-radix
    /// unpack into normalized grid codes, shape `[..., codebook_dim]`.
    ///
    /// `level_indices = (indices // basis) % levels`, then
    /// `codes = (level_indices - half_width) / half_width`. Implemented with
    /// `floor`/`div`/`mul`/`sub` (no integer floor-div/mod op exists in numr;
    /// this is exactly how the PyTorch reference does it too — float tensor
    /// ops, not a host-side loop — so it stays backend-generic).
    pub(super) fn decode_indices<C>(&self, client: &C, indices: &Tensor<R>) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R> + TensorOps<R>,
    {
        let indices_f32 = client.cast(indices, DType::F32).map_err(Error::Numr)?;
        let expanded = indices_f32.unsqueeze(-1).map_err(Error::Numr)?;

        let div_basis = client.div(&expanded, &self.basis).map_err(Error::Numr)?;
        let floor_div = client.floor(&div_basis).map_err(Error::Numr)?;
        let div_levels = client
            .div(&floor_div, &self.levels_f32)
            .map_err(Error::Numr)?;
        let floor_levels = client.floor(&div_levels).map_err(Error::Numr)?;
        let mod_part = client
            .mul(&floor_levels, &self.levels_f32)
            .map_err(Error::Numr)?;
        let level_indices = client.sub(&floor_div, &mod_part).map_err(Error::Numr)?;

        let shifted = client
            .sub(&level_indices, &self.half_width)
            .map_err(Error::Numr)?;
        client.div(&shifted, &self.half_width).map_err(Error::Numr)
    }
}

#[cfg(test)]
mod tests;
