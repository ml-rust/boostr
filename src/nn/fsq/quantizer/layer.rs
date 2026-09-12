//! The [`Fsq`] type: constant precomputation from `levels`, projection
//! wiring, and parameter enumeration.
//!
//! Ports `FSQ` from lucidrains/vector-quantize-pytorch
//! (`vector_quantize_pytorch/finite_scalar_quantization.py`, revision as of
//! 2026-08, single-codebook case — `num_codebooks = 1`, `preserve_symmetry =
//! false`, `bound_hard_clamp = false`, which is how NeuCodec/WideCodec use it).
//!
//! This type is lucidrains/vector-quantize-pytorch's `FSQ` and *only* `FSQ`.
//! The residual wrapper — its `ResidualFSQ`, which owns the projections, the per-quantizer
//! `scales`, and the extra pre-`bound` on the encode path — lives in
//! [`ResidualFsq`](crate::nn::fsq::residual::ResidualFsq). Conflating the two
//! is a real numerical trap; see that module's docs.
//!
//! `Fsq` keeps optional `project_in`/`project_out` of its own because callers
//! configure them through [`FsqConfig`] (`input_dim != levels.len()`);
//! lucidrains/vector-quantize-pytorch's `FSQ`'s equivalents are `nn.Identity`
//! in that case, which is exactly what `None` means here.
//!
//! The quantization math (`bound`, `quantize`, index packing) is documented in
//! the sibling `codec` file.

use crate::error::{Error, Result};
use crate::nn::fsq::config::FsqConfig;
use crate::nn::linear::Linear;
use crate::nn::module::Module;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::runtime::Runtime;
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
    pub(super) config: FsqConfig,
    /// `(level - 1) * (1 + eps) / 2`, shape `[codebook_dim]`.
    pub(super) half_l: Tensor<R>,
    /// `atanh(offset / half_l)`, shape `[codebook_dim]`.
    pub(super) shift: Tensor<R>,
    /// `0.5` for even levels, `0.0` for odd levels, shape `[codebook_dim]`.
    pub(super) offset: Tensor<R>,
    /// `level // 2`, shape `[codebook_dim]`.
    ///
    /// Visible to the `fsq` module: read by `fsq::codes`'s
    /// `codes_to_indices` / `decode_indices`, which live in a sibling module.
    pub(in crate::nn::fsq) half_width: Tensor<R>,
    /// Mixed-radix basis (cumulative product of levels), shape `[codebook_dim]`.
    ///
    /// Visible to the `fsq` module: read by `fsq::codes`'s
    /// `codes_to_indices` / `decode_indices`, which live in a sibling module.
    pub(in crate::nn::fsq) basis: Tensor<R>,
    /// `levels` as f32, shape `[codebook_dim]`.
    ///
    /// Visible to the `fsq` module: read by `fsq::codes`'s `decode_indices`,
    /// which lives in a sibling module.
    pub(in crate::nn::fsq) levels_f32: Tensor<R>,
    pub(super) project_in: Option<Linear<R>>,
    pub(super) project_out: Option<Linear<R>>,
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
            half_l: Tensor::from_slice(&half_l, &[dim], device)?,
            shift: Tensor::from_slice(&shift, &[dim], device)?,
            offset: Tensor::from_slice(&offset, &[dim], device)?,
            half_width: Tensor::from_slice(&half_width, &[dim], device)?,
            basis: Tensor::from_slice(&basis, &[dim], device)?,
            levels_f32: Tensor::from_slice(&levels_f32, &[dim], device)?,
            config,
            project_in,
            project_out,
        })
    }

    /// The configuration this quantizer was built from.
    pub fn config(&self) -> &FsqConfig {
        &self.config
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
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    // --- projection wiring ---------------------------------------------------

    #[test]
    fn test_projection_required_when_dims_differ() {
        let (_, device) = cpu_setup();
        let config = FsqConfig::new(vec![4, 4], 5).unwrap();
        let err = Fsq::<CpuRuntime>::new(config, &device, None, None)
            .err()
            .unwrap();
        assert!(matches!(
            err,
            Error::InvalidArgument {
                arg: "project_in/project_out",
                ..
            }
        ));
    }

    #[test]
    fn test_projection_rejected_when_dims_match() {
        let (_, device) = cpu_setup();
        let config = FsqConfig::new(vec![4, 4], 2).unwrap();
        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 4], &[2, 2], &device).unwrap();
        let project_in = Some(Linear::new(weight, None, false));
        let err = Fsq::<CpuRuntime>::new(config, &device, project_in, None)
            .err()
            .unwrap();
        assert!(matches!(
            err,
            Error::InvalidArgument {
                arg: "project_in/project_out",
                ..
            }
        ));
    }
}
