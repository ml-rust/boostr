//! Residual Finite Scalar Quantizer — CPU/CUDA/WebGPU generic.
//!
//! Ports `ResidualFSQ` from lucidrains/vector-quantize-pytorch
//! (`vector_quantize_pytorch/residual_fsq.py`, revision as of 2026-08).
//!
//! `ResidualFSQ` is a DIFFERENT class from `FSQ` ([`Fsq`]), and the difference
//! is not cosmetic. `ResidualFSQ` owns:
//!
//! * `project_in: Linear(dim -> codebook_dim)` / `project_out: Linear(codebook_dim -> dim)`
//!   (`nn.Identity` when `dim == codebook_dim`),
//! * `num_quantizers` inner `FSQ` layers, whose OWN projections are always
//!   `nn.Identity` — the residual wrapper does all the projecting,
//! * per-quantizer `scales[i] = (levels - 1) ** -i` (so `scales[0]` is all-ones).
//!
//! # The double-bound trap — do NOT "simplify" this away
//!
//! Upstream's forward is:
//!
//! ```python
//! x = self.project_in(x)
//! residual = first(self.layers).bound(x)          # PURE bound: no round, no divide
//! for layer, scale in zip(self.layers, self.scales):
//!     quantized, indices = layer(residual / scale)  # FSQ.quantize -> round_ste(bound(z)) / half_width
//!     quantized = quantized * scale
//!     residual = residual - quantized.detach()
//!     quantized_out = quantized_out + quantized
//! quantized_out = self.project_out(quantized_out)
//! ```
//!
//! `bound` is therefore applied TWICE before the first rounding: once to seed
//! `residual`, and again inside `FSQ.quantize`. It looks redundant. It is not:
//! `bound(z) = tanh(z + shift) * half_l - offset` has the ASYMMETRIC output
//! range `(-half_l - offset, half_l - offset)`, so it is not idempotent and
//! `bound(bound(z)) != bound(z)` wherever `offset != 0` (i.e. every even level,
//! which includes NeuCodec's `levels = [4; 8]`).
//!
//! Measured against the real NeuCodec checkpoint, collapsing this to a single
//! `bound` changes **43.75% of the emitted indices**. Anyone tempted to delete
//! the pre-`bound` on line "seed the residual" below is introducing that bug.
//!
//! The decode path ([`ResidualFsq::decode`], upstream `get_output_from_indices`)
//! has no such subtlety: per-quantizer codebook lookup, scale, sum, `project_out`.

use super::codes::var_passthrough;
use super::config::ResidualFsqConfig;
use super::quantizer::Fsq;
use crate::error::{Error, Result};
use crate::nn::linear::Linear;
use crate::nn::module::Module;
use numr::autograd::{Var, var_add, var_div, var_mul, var_sub};
use numr::dtype::DType;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::{Tensor, TensorId};

/// Already-built parts for [`ResidualFsq`], following the `*Weights` convention
/// used throughout `model/audio/neucodec/`.
pub struct ResidualFsqWeights<R: Runtime> {
    /// `Linear(dim -> codebook_dim)`; `None` iff `dim == codebook_dim`.
    pub project_in: Option<Linear<R>>,
    /// `Linear(codebook_dim -> dim)`; `None` iff `dim == codebook_dim`.
    pub project_out: Option<Linear<R>>,
    /// Inner FSQ layers — exactly `num_quantizers` of them, each WITHOUT
    /// projections (this wrapper owns the projections).
    pub layers: Vec<Fsq<R>>,
}

/// Residual Finite Scalar Quantizer: a stack of [`Fsq`] layers, each quantizing
/// what the previous ones could not represent.
pub struct ResidualFsq<R: Runtime> {
    config: ResidualFsqConfig,
    project_in: Option<Linear<R>>,
    project_out: Option<Linear<R>>,
    layers: Vec<Fsq<R>>,
    /// `scales[i][j] = (levels[j] - 1) ^ -i`, shape `[codebook_dim]` each.
    /// `scales[0]` is all-ones.
    scales: Vec<Tensor<R>>,
}

impl<R: Runtime<DType = DType>> ResidualFsq<R> {
    /// Assemble from already-built parts, validating layer count, layer grids,
    /// and projection shapes against `config`.
    pub fn new(
        config: ResidualFsqConfig,
        weights: ResidualFsqWeights<R>,
        device: &R::Device,
    ) -> Result<Self> {
        config.validate()?;
        let codebook_dim = config.codebook_dim();

        if weights.layers.len() != config.num_quantizers {
            return Err(Error::ModelError {
                reason: format!(
                    "expected {} FSQ layers (num_quantizers), got {}",
                    config.num_quantizers,
                    weights.layers.len()
                ),
            });
        }
        for (index, layer) in weights.layers.iter().enumerate() {
            let layer_config = layer.config();
            if layer_config.levels != config.levels {
                return Err(Error::ModelError {
                    reason: format!(
                        "layer {index} levels {:?} do not match residual levels {:?}",
                        layer_config.levels, config.levels
                    ),
                });
            }
            // Upstream's inner FSQ projections are nn.Identity; a projecting
            // inner layer would double-project.
            if layer_config.input_dim != codebook_dim {
                return Err(Error::ModelError {
                    reason: format!(
                        "layer {index} input_dim {} must equal codebook_dim {codebook_dim} \
                         (inner FSQ layers must not project)",
                        layer_config.input_dim
                    ),
                });
            }
        }

        Self::check_projections(&config, &weights, codebook_dim)?;

        let mut scales = Vec::with_capacity(config.num_quantizers);
        for index in 0..config.num_quantizers {
            let values: Vec<f32> = config
                .levels
                .iter()
                .map(|&level| ((level as f32) - 1.0).powi(-(index as i32)))
                .collect();
            scales.push(Tensor::from_slice(&values, &[codebook_dim], device));
        }

        Ok(Self {
            config,
            project_in: weights.project_in,
            project_out: weights.project_out,
            layers: weights.layers,
            scales,
        })
    }

    /// Presence + shape validation for `project_in`/`project_out`.
    fn check_projections(
        config: &ResidualFsqConfig,
        weights: &ResidualFsqWeights<R>,
        codebook_dim: usize,
    ) -> Result<()> {
        let needs = config.needs_projection();
        let present = weights.project_in.is_some() || weights.project_out.is_some();
        if needs && (weights.project_in.is_none() || weights.project_out.is_none()) {
            return Err(Error::InvalidArgument {
                arg: "project_in/project_out",
                reason: format!(
                    "dim ({}) != codebook_dim ({codebook_dim}); both projections are required",
                    config.dim
                ),
            });
        }
        if !needs && present {
            return Err(Error::InvalidArgument {
                arg: "project_in/project_out",
                reason: "dim == codebook_dim; no projection should be supplied".to_string(),
            });
        }

        if let Some(linear) = &weights.project_in {
            expect_weight_shape(linear, &[codebook_dim, config.dim], "project_in")?;
        }
        if let Some(linear) = &weights.project_out {
            expect_weight_shape(linear, &[config.dim, codebook_dim], "project_out")?;
        }
        Ok(())
    }

    /// The configuration this quantizer was built from.
    pub fn config(&self) -> &ResidualFsqConfig {
        &self.config
    }

    /// Encode `x` (`[..., dim]`) into `(codes, indices)`.
    ///
    /// `codes`: `[..., dim]` (the summed, projected reconstruction).
    /// `indices`: `[..., num_quantizers]`, `DType::I32`.
    ///
    /// Follows upstream `ResidualFSQ.forward` step for step, including the
    /// pre-`bound` that seeds `residual` — see this module's docs for why that
    /// second bound is load-bearing rather than redundant.
    pub fn encode<C>(&self, client: &C, x: &Var<R>) -> Result<(Var<R>, Tensor<R>)>
    where
        C: RuntimeClient<R> + TensorOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        match x.shape().last().copied() {
            Some(last) if last == self.config.dim => {}
            _ => {
                return Err(Error::InvalidArgument {
                    arg: "x",
                    reason: format!(
                        "expected last dimension {}, got shape {:?}",
                        self.config.dim,
                        x.shape()
                    ),
                });
            }
        }

        let projected = match &self.project_in {
            Some(linear) => linear.forward(client, x)?,
            // Not `Var::clone()`: that mints a fresh TensorId and would orphan
            // the caller's `x.id()` from the gradient graph.
            None => var_passthrough(x),
        };

        let first = self.layers.first().ok_or_else(|| Error::ModelError {
            reason: "residual FSQ has no layers".to_string(),
        })?;
        // Seed the residual with the PURE bound. Deleting this line (or reusing
        // the already-bounded value from inside the loop) silently changes 43.75%
        // of NeuCodec's indices — `bound` is not idempotent.
        let mut residual = first.bound(&projected, client)?;

        let mut quantized_out: Option<Var<R>> = None;
        let mut all_indices: Vec<Tensor<R>> = Vec::with_capacity(self.layers.len());

        for (index, layer) in self.layers.iter().enumerate() {
            let scale_tensor = self.scales.get(index).ok_or_else(|| Error::ModelError {
                reason: format!("missing scale for quantizer {index}"),
            })?;
            // Constant derived from `levels` alone — a legitimate non-tracked leaf.
            let scale = Var::new(scale_tensor.clone(), false);

            let scaled = var_div(&residual, &scale, client).map_err(Error::Numr)?;
            let (quantized, indices) = layer.quantize(client, &scaled)?;
            let quantized = var_mul(&quantized, &scale, client).map_err(Error::Numr)?;

            // `quantized.detach()`: same pattern `round_ste` uses — wrap the raw
            // tensor as a fresh non-tracked leaf. `Var::clone()` would keep the
            // value but mint a new id while leaving the grad_fn semantics wrong.
            let detached = Var::new(quantized.tensor().clone(), false);
            residual = var_sub(&residual, &detached, client).map_err(Error::Numr)?;

            quantized_out = Some(match quantized_out {
                Some(acc) => var_add(&acc, &quantized, client).map_err(Error::Numr)?,
                None => quantized,
            });
            all_indices.push(indices);
        }

        let quantized_out = quantized_out.ok_or_else(|| Error::ModelError {
            reason: "residual FSQ produced no quantized output".to_string(),
        })?;

        let stack_dim = all_indices
            .first()
            .ok_or_else(|| Error::ModelError {
                reason: "residual FSQ produced no indices".to_string(),
            })?
            .shape()
            .len() as isize;
        let index_refs: Vec<&Tensor<R>> = all_indices.iter().collect();
        let indices = client.stack(&index_refs, stack_dim).map_err(Error::Numr)?;

        let codes = match &self.project_out {
            Some(linear) => linear.forward(client, &quantized_out)?,
            None => quantized_out,
        };
        Ok((codes, indices))
    }

    /// Decode `indices` (`[..., num_quantizers]`, integer dtype) into codes
    /// (`[..., dim]`) — upstream `get_output_from_indices`.
    ///
    /// Per-quantizer codebook lookup, multiply by `scales[i]`, sum over
    /// quantizers, then `project_out`. Indices are discrete, so the summed codes
    /// are a non-differentiable leaf before the (possibly trainable) projection.
    pub fn decode<C>(&self, client: &C, indices: &Tensor<R>) -> Result<Var<R>>
    where
        C: RuntimeClient<R> + TensorOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        let rank = indices.shape().len();
        let last_axis = rank.checked_sub(1).ok_or_else(|| Error::InvalidArgument {
            arg: "indices",
            reason: "expected at least one dimension (trailing quantizer axis)".to_string(),
        })?;
        if indices.shape().get(last_axis).copied() != Some(self.config.num_quantizers) {
            return Err(Error::InvalidArgument {
                arg: "indices",
                reason: format!(
                    "expected trailing dimension {} (num_quantizers), got shape {:?}",
                    self.config.num_quantizers,
                    indices.shape()
                ),
            });
        }

        let mut summed: Option<Var<R>> = None;
        for (index, layer) in self.layers.iter().enumerate() {
            let slice = indices
                .narrow(last_axis as isize, index, 1)
                .map_err(Error::Numr)?
                .squeeze(Some(last_axis as isize))
                .contiguous()
                .map_err(Error::Numr)?;
            let codes = layer.indices_to_codes(client, &slice)?;

            let scale_tensor = self.scales.get(index).ok_or_else(|| Error::ModelError {
                reason: format!("missing scale for quantizer {index}"),
            })?;
            let scale = Var::new(scale_tensor.clone(), false);
            let scaled = var_mul(&codes, &scale, client).map_err(Error::Numr)?;

            summed = Some(match summed {
                Some(acc) => var_add(&acc, &scaled, client).map_err(Error::Numr)?,
                None => scaled,
            });
        }

        let summed = summed.ok_or_else(|| Error::ModelError {
            reason: "residual FSQ has no layers".to_string(),
        })?;
        match &self.project_out {
            Some(linear) => linear.forward(client, &summed),
            None => Ok(summed),
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

/// Check a projection's `[out, in]` weight shape, erroring rather than panicking.
fn expect_weight_shape<R: Runtime>(
    linear: &Linear<R>,
    expected: &[usize],
    name: &'static str,
) -> Result<()> {
    let shape = linear.weight().tensor().shape();
    if shape != expected {
        return Err(Error::ModelError {
            reason: format!("{name} weight shape {shape:?} does not match expected {expected:?}"),
        });
    }
    Ok(())
}

impl<R: Runtime<DType = DType>> Module<R> for ResidualFsq<R> {
    fn parameters(&self) -> Vec<&Var<R>> {
        ResidualFsq::parameters(self)
            .into_iter()
            .map(|param| param.1)
            .collect()
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
#[path = "residual_tests.rs"]
mod tests;
