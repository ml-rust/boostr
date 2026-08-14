//! Finite Scalar Quantizer — CPU/CUDA/WebGPU generic.
//!
//! Ports `FSQ` from lucidrains/vector-quantize-pytorch
//! (`vector_quantize_pytorch/finite_scalar_quantization.py`, revision as of
//! 2026-08, single-codebook case — `num_codebooks = 1`, `preserve_symmetry =
//! false`, `bound_hard_clamp = false`, the defaults `ResidualFSQ(num_quantizers
//! = 1)` degenerates to, which is how NeuCodec/WideCodec use it).
//!
//! # Math (mirrors the reference exactly)
//!
//! For each of the `d = levels.len()` scalar dimensions, with `eps = 1e-3`:
//!
//! ```text
//! half_l     = (level - 1) * (1 + eps) / 2
//! offset     = 0.5 if level is even else 0.0
//! shift      = atanh(offset / half_l)          // 0 for odd levels
//! bounded(z) = tanh(z + shift) * half_l - offset
//! half_width = level // 2
//! code       = round_ste(bounded(z)) / half_width
//! ```
//!
//! `round_ste(x) = x + (round(x) - x).detach()`: forward value is `round(x)`,
//! backward gradient is the identity (straight-through estimator).
//!
//! Indices use a mixed-radix (cumulative-product) basis: `basis[0] = 1`,
//! `basis[i] = basis[i-1] * levels[i-1]`.
//!
//! ```text
//! codes_to_indices(code)   = round(sum((code * half_width + half_width) * basis))
//! indices_to_level_indices = (indices // basis) % levels
//! indices_to_codes(index)  = (level_indices - half_width) / half_width
//! ```

use super::config::FsqConfig;
use crate::error::{Error, Result};
use crate::nn::linear::Linear;
use crate::nn::module::Module;
use numr::autograd::{Var, var_add, var_div, var_mul, var_sub, var_tanh};
use numr::dtype::DType;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::{Tensor, TensorId};

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
fn var_passthrough<R: Runtime>(v: &Var<R>) -> Var<R> {
    match (v.requires_grad(), v.grad_fn().cloned()) {
        (true, Some(grad_fn)) => {
            Var::with_id_and_grad_fn(v.tensor().clone(), v.id(), Some(grad_fn))
        }
        (true, None) => Var::with_id(v.tensor().clone(), v.id(), true),
        (false, _) => Var::with_id(v.tensor().clone(), v.id(), false),
    }
}

/// Finite Scalar Quantizer.
///
/// Quantizes each of `levels.len()` scalar dimensions onto a fixed, evenly
/// spaced grid (`levels[i]` points per dimension), then packs the per-dimension
/// grid coordinates into a single mixed-radix index. Unlike VQ-VAE-style vector
/// quantizers, the "codebook" is implicit (defined entirely by `levels`) — there
/// are no learned codebook vectors and no commitment loss.
///
/// When `input_dim != levels.len()` (e.g. NeuCodec: `dim = 2048` against 8
/// levels), `project_in`/`project_out` `Linear` layers map between the model's
/// feature dimension and the quantized `codebook_dim`.
pub struct Fsq<R: Runtime> {
    config: FsqConfig,
    /// `(level - 1) * (1 + eps) / 2`, shape `[codebook_dim]`.
    half_l: Tensor<R>,
    /// `atanh(offset / half_l)`, shape `[codebook_dim]`.
    shift: Tensor<R>,
    /// `0.5` for even levels, `0.0` for odd levels, shape `[codebook_dim]`.
    offset: Tensor<R>,
    /// `level // 2`, shape `[codebook_dim]`.
    half_width: Tensor<R>,
    /// Mixed-radix basis (cumulative product of levels), shape `[codebook_dim]`.
    basis: Tensor<R>,
    /// `levels` as f32, shape `[codebook_dim]`.
    levels_f32: Tensor<R>,
    project_in: Option<Linear<R>>,
    project_out: Option<Linear<R>>,
}

impl<R: Runtime<DType = DType>> Fsq<R> {
    /// Bound-widening epsilon, matching the reference's default.
    const EPS: f32 = 1e-3;

    /// Build an `Fsq` for `config`, precomputing the per-dimension bound/scale
    /// constants on `device`.
    ///
    /// `project_in`/`project_out` are required exactly when
    /// `config.needs_projection()` is true, and must be absent otherwise.
    pub fn new(
        config: FsqConfig,
        device: &R::Device,
        project_in: Option<Linear<R>>,
        project_out: Option<Linear<R>>,
    ) -> Result<Self> {
        config.validate()?;

        let needs_projection = config.needs_projection();
        if needs_projection && (project_in.is_none() || project_out.is_none()) {
            return Err(Error::InvalidArgument {
                arg: "project_in/project_out",
                reason: format!(
                    "input_dim ({}) != codebook_dim ({}); both projections are required",
                    config.input_dim,
                    config.codebook_dim()
                ),
            });
        }
        if !needs_projection && (project_in.is_some() || project_out.is_some()) {
            return Err(Error::InvalidArgument {
                arg: "project_in/project_out",
                reason: "input_dim == codebook_dim; no projection should be supplied".to_string(),
            });
        }

        let dim = config.codebook_dim();
        let mut half_l = Vec::with_capacity(dim);
        let mut shift = Vec::with_capacity(dim);
        let mut offset = Vec::with_capacity(dim);
        let mut half_width = Vec::with_capacity(dim);
        let mut basis = Vec::with_capacity(dim);
        let mut levels_f32 = Vec::with_capacity(dim);

        let mut running_basis = 1.0f32;
        for &level in &config.levels {
            let level_f = level as f32;
            let hl = (level_f - 1.0) * (1.0 + Self::EPS) / 2.0;
            let off = if level % 2 == 0 { 0.5f32 } else { 0.0f32 };
            let sh = if off == 0.0 {
                0.0f32
            } else {
                (off / hl).atanh()
            };
            let hw = (level / 2) as f32;

            half_l.push(hl);
            shift.push(sh);
            offset.push(off);
            half_width.push(hw);
            basis.push(running_basis);
            levels_f32.push(level_f);

            running_basis *= level_f;
        }

        Ok(Self {
            half_l: Tensor::from_slice(&half_l, &[dim], device),
            shift: Tensor::from_slice(&shift, &[dim], device),
            offset: Tensor::from_slice(&offset, &[dim], device),
            half_width: Tensor::from_slice(&half_width, &[dim], device),
            basis: Tensor::from_slice(&basis, &[dim], device),
            levels_f32: Tensor::from_slice(&levels_f32, &[dim], device),
            config,
            project_in,
            project_out,
        })
    }

    /// The configuration this quantizer was built from.
    pub fn config(&self) -> &FsqConfig {
        &self.config
    }

    /// Bound `z` onto the FSQ grid, normalized to `[-1, 1]`-ish per dimension.
    ///
    /// Straight-through: forward value is the rounded grid point, backward
    /// gradient is `d(tanh(z + shift) * half_l - offset) / dz / half_width` —
    /// every step here is a tracked `var_*` op, so gradients reach `z`.
    fn bound<C>(&self, z: &Var<R>, client: &C) -> Result<Var<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R> + TensorOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        // Genuine constants (precomputed from `levels`, independent of any
        // trainable parameter) — `Var::new(_, false)` is correct here.
        let shift = Var::new(self.shift.clone(), false);
        let half_l = Var::new(self.half_l.clone(), false);
        let offset = Var::new(self.offset.clone(), false);
        let half_width = Var::new(self.half_width.clone(), false);

        let shifted = var_add(z, &shift, client).map_err(Error::Numr)?;
        let tanh_val = var_tanh(&shifted, client).map_err(Error::Numr)?;
        let scaled = var_mul(&tanh_val, &half_l, client).map_err(Error::Numr)?;
        let bounded = var_sub(&scaled, &offset, client).map_err(Error::Numr)?;
        let rounded = self.round_ste(&bounded, client)?;
        var_div(&rounded, &half_width, client).map_err(Error::Numr)
    }

    /// `z + (round(z) - z).detach()`: forward = `round(z)`, backward = identity.
    ///
    /// The `(round(z) - z)` residual is computed on the raw `Tensor` (not
    /// through `var_*`) and wrapped in `Var::new(_, false)` — this is the
    /// legitimate "detach" use of `Var::new`: it is a genuine constant offset
    /// with no gradient path of its own, exactly mirroring PyTorch's
    /// `(zhat - z).detach()`. `z` itself stays on the tracked graph the whole
    /// time, so `var_add` passes the incoming gradient straight through to it.
    fn round_ste<C>(&self, z: &Var<R>, client: &C) -> Result<Var<R>>
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
    fn codes_to_indices<C>(&self, client: &C, zhat: &Tensor<R>) -> Result<Tensor<R>>
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
    fn decode_indices<C>(&self, client: &C, indices: &Tensor<R>) -> Result<Tensor<R>>
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

    /// Encode: quantize `z` and return `(codes, indices)`.
    ///
    /// `z`: `[..., input_dim]`. `codes`: `[..., input_dim]` (post `project_out`
    /// if projection is configured, else `[..., codebook_dim]`). `indices`:
    /// `[...]`, `DType::I32`.
    ///
    /// The straight-through estimator makes `codes` differentiable w.r.t. `z`
    /// (and w.r.t. `project_in`/`project_out` weights, if trainable); `indices`
    /// is a discrete byproduct and carries no gradient.
    pub fn quantize<C>(&self, client: &C, z: &Var<R>) -> Result<(Var<R>, Tensor<R>)>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R> + TensorOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        match z.shape().last().copied() {
            Some(last) if last == self.config.input_dim => {}
            _ => {
                return Err(Error::InvalidArgument {
                    arg: "z",
                    reason: format!(
                        "expected last dimension {}, got shape {:?}",
                        self.config.input_dim,
                        z.shape()
                    ),
                });
            }
        }

        let projected = match &self.project_in {
            Some(linear) => linear.forward(client, z)?,
            // `Var::clone()` mints a fresh autograd id and would silently
            // disconnect this leaf from the caller's `z.id()` (the id the
            // caller looks gradients up by after `backward`). Use the
            // identity-preserving passthrough instead.
            None => var_passthrough(z),
        };

        let bounded = self.bound(&projected, client)?;
        let indices = self.codes_to_indices(client, bounded.tensor())?;

        let codes = match &self.project_out {
            Some(linear) => linear.forward(client, &bounded)?,
            None => bounded,
        };

        Ok((codes, indices))
    }

    /// Decode: `indices` (`[...]`, integer dtype) -> `codes` (`[...,
    /// input_dim]`).
    ///
    /// This is the decode path a decoder-only pipeline needs: reconstructs the
    /// normalized grid codes via mixed-radix unpack, then applies
    /// `project_out` if configured. `indices` carries no gradient (discrete),
    /// so the decoded codes are wrapped as a non-differentiable leaf before any
    /// (potentially trainable) `project_out` is applied.
    pub fn indices_to_codes<C>(&self, client: &C, indices: &Tensor<R>) -> Result<Var<R>>
    where
        C: RuntimeClient<R> + TensorOps<R>,
        R::Client: TensorOps<R>,
    {
        let codes = self.decode_indices(client, indices)?;
        let codes = Var::new(codes, false);
        match &self.project_out {
            Some(linear) => linear.forward(client, &codes),
            None => Ok(codes),
        }
    }

    /// All parameters with their stable autograd IDs (projections only — FSQ
    /// itself has no learned codebook).
    pub fn parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        let mut params = Vec::new();
        if let Some(linear) = &self.project_in {
            params.extend(linear.parameters());
        }
        if let Some(linear) = &self.project_out {
            params.extend(linear.parameters());
        }
        params
    }

    /// Trainable parameters with their stable autograd IDs.
    pub fn trainable_parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        self.parameters()
            .into_iter()
            .filter(|param| param.1.requires_grad())
            .collect()
    }
}

impl<R: Runtime<DType = DType>> Module<R> for Fsq<R> {
    fn parameters(&self) -> Vec<&Var<R>> {
        Fsq::parameters(self).into_iter().map(|p| p.1).collect()
    }

    fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        let mut params = Vec::new();
        if let Some(linear) = &self.project_in {
            for (name, var) in linear.named_parameters() {
                params.push((format!("project_in.{name}"), var));
            }
        }
        if let Some(linear) = &self.project_out {
            for (name, var) in linear.named_parameters() {
                params.push((format!("project_out.{name}"), var));
            }
        }
        params
    }
}

#[cfg(test)]
#[path = "quantizer_tests.rs"]
mod tests;
