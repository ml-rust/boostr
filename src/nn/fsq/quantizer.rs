//! Finite Scalar Quantizer — CPU/CUDA/WebGPU generic.
//!
//! Ports `FSQ` from lucidrains/vector-quantize-pytorch
//! (`vector_quantize_pytorch/finite_scalar_quantization.py`, revision as of
//! 2026-08, single-codebook case — `num_codebooks = 1`, `preserve_symmetry =
//! false`, `bound_hard_clamp = false`, which is how NeuCodec/WideCodec use it).
//!
//! This type is upstream's `FSQ` and *only* `FSQ`. The residual wrapper —
//! upstream's `ResidualFSQ`, which owns the projections, the per-quantizer
//! `scales`, and the extra pre-`bound` on the encode path — lives in
//! [`super::residual::ResidualFsq`]. Conflating the two is a real numerical
//! trap; see that module's docs.
//!
//! `Fsq` keeps optional `project_in`/`project_out` of its own because callers
//! configure them through [`FsqConfig`] (`input_dim != levels.len()`); upstream
//! `FSQ`'s equivalents are `nn.Identity` in that case, which is exactly what
//! `None` means here.
//!
//! # Math (mirrors the reference exactly)
//!
//! For each of the `d = levels.len()` scalar dimensions, with `eps = 1e-3`:
//!
//! ```text
//! half_l     = (level - 1) * (1 + eps) / 2
//! offset     = 0.5 if level is even else 0.0
//! shift      = atanh(offset / half_l)          // 0 for odd levels
//! bound(z)   = tanh(z + shift) * half_l - offset          // FSQ.bound
//! half_width = level // 2
//! quantize(z) = round_ste(bound(z)) / half_width          // FSQ.quantize
//! ```
//!
//! Note the two are SEPARATE upstream functions (`Fsq::bound` and
//! `Fsq::quantize_codes`) and `bound` is NOT idempotent — its output range is
//! asymmetric, `(-half_l - offset, half_l - offset)`. `ResidualFsq` applies it
//! twice on purpose. Do not fuse them back together.
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

use super::codes::var_passthrough;
use super::config::FsqConfig;
use crate::error::{Error, Result};
use crate::nn::linear::Linear;
use crate::nn::module::Module;
use numr::autograd::{Var, var_add, var_div, var_mul, var_sub, var_tanh};
use numr::dtype::DType;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::{Tensor, TensorId};

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
    ///
    /// `pub(super)`: read by [`super::codes`]'s `codes_to_indices` /
    /// `decode_indices`, which live in a sibling module.
    pub(super) half_width: Tensor<R>,
    /// Mixed-radix basis (cumulative product of levels), shape `[codebook_dim]`.
    ///
    /// `pub(super)`: read by [`super::codes`]'s `codes_to_indices` /
    /// `decode_indices`, which live in a sibling module.
    pub(super) basis: Tensor<R>,
    /// `levels` as f32, shape `[codebook_dim]`.
    ///
    /// `pub(super)`: read by [`super::codes`]'s `decode_indices`, which lives
    /// in a sibling module.
    pub(super) levels_f32: Tensor<R>,
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
            half_l: Tensor::try_from_slice(&half_l, &[dim], device)?,
            shift: Tensor::try_from_slice(&shift, &[dim], device)?,
            offset: Tensor::try_from_slice(&offset, &[dim], device)?,
            half_width: Tensor::try_from_slice(&half_width, &[dim], device)?,
            basis: Tensor::try_from_slice(&basis, &[dim], device)?,
            levels_f32: Tensor::try_from_slice(&levels_f32, &[dim], device)?,
            config,
            project_in,
            project_out,
        })
    }

    /// The configuration this quantizer was built from.
    pub fn config(&self) -> &FsqConfig {
        &self.config
    }

    /// Upstream `FSQ.bound`: `tanh(z + shift) * half_l - offset`.
    ///
    /// Squashes `z` into the (asymmetric) interval
    /// `(-half_l - offset, half_l - offset)` per dimension. NO rounding, NO
    /// division by `half_width` — that is `quantize_codes`, a strictly
    /// different function.
    ///
    /// The asymmetry is why this is not idempotent: `bound(bound(z)) !=
    /// bound(z)`. [`ResidualFsq`](super::residual::ResidualFsq) relies on
    /// applying it twice (once to seed the residual, once inside
    /// `quantize_codes`) exactly as upstream `ResidualFSQ.forward` does.
    ///
    /// Every step is a tracked `var_*` op, so gradients reach `z`.
    pub(crate) fn bound<C>(&self, z: &Var<R>, client: &C) -> Result<Var<R>>
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

        let shifted = var_add(z, &shift, client).map_err(Error::Numr)?;
        let tanh_val = var_tanh(&shifted, client).map_err(Error::Numr)?;
        let scaled = var_mul(&tanh_val, &half_l, client).map_err(Error::Numr)?;
        var_sub(&scaled, &offset, client).map_err(Error::Numr)
    }

    /// Upstream `FSQ.quantize`: `round_ste(bound(z)) / half_width`.
    ///
    /// Snaps `z` onto the FSQ grid, normalized to `[-1, 1]`-ish per dimension.
    /// Straight-through: forward value is the rounded grid point, backward
    /// gradient is `d(bound(z)) / dz / half_width` — every step here is a
    /// tracked `var_*` op, so gradients reach `z`.
    ///
    /// Named `quantize_codes` rather than `quantize` because
    /// [`Fsq::quantize`](Self::quantize) is the public encode entry point
    /// (projections + index packing) that wraps it.
    fn quantize_codes<C>(&self, z: &Var<R>, client: &C) -> Result<Var<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R> + TensorOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        let half_width = Var::new(self.half_width.clone(), false);
        let bounded = self.bound(z, client)?;
        let rounded = self.round_ste(&bounded, client)?;
        var_div(&rounded, &half_width, client).map_err(Error::Numr)
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

        let bounded = self.quantize_codes(&projected, client)?;
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
mod tests;
