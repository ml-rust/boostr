//! The [`ResidualFsq`] type: assembly from [`ResidualFsqWeights`], validation,
//! per-quantizer scales, and parameter enumeration.
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
//! The encode path has a load-bearing double `bound`; see the sibling `codec`
//! file's docs before touching it.

use crate::error::{Error, Result};
use crate::nn::fsq::config::ResidualFsqConfig;
use crate::nn::fsq::quantizer::Fsq;
use crate::nn::linear::Linear;
use crate::nn::module::Module;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::runtime::Runtime;
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
    pub(super) config: ResidualFsqConfig,
    pub(super) project_in: Option<Linear<R>>,
    pub(super) project_out: Option<Linear<R>>,
    pub(super) layers: Vec<Fsq<R>>,
    /// `scales[i][j] = (levels[j] - 1) ^ -i`, shape `[codebook_dim]` each.
    /// `scales[0]` is all-ones.
    pub(super) scales: Vec<Tensor<R>>,
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
            // lucidrains/vector-quantize-pytorch's inner FSQ projections are nn.Identity; a projecting
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
            scales.push(Tensor::from_slice(&values, &[codebook_dim], device)?);
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
mod tests {
    use super::*;
    use crate::nn::fsq::config::FsqConfig;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    // --- validation -----------------------------------------------------------

    #[test]
    fn test_wrong_layer_count_rejected() {
        let (_, device) = cpu_setup();
        let config = ResidualFsqConfig::new(vec![4, 4], 2, 3).unwrap();
        let layer_config = config.layer_config().unwrap();
        let layers = vec![
            Fsq::<CpuRuntime>::new(layer_config.clone(), &device, None, None).unwrap(),
            Fsq::<CpuRuntime>::new(layer_config, &device, None, None).unwrap(),
        ];

        let err = ResidualFsq::new(
            config,
            ResidualFsqWeights {
                project_in: None,
                project_out: None,
                layers,
            },
            &device,
        )
        .err()
        .unwrap();
        assert!(matches!(err, Error::ModelError { .. }), "got {err:?}");
    }

    #[test]
    fn test_mismatched_projection_dims_rejected() {
        let (_, device) = cpu_setup();
        // dim = 5, codebook_dim = 2 -> project_in must be [2, 5], project_out [5, 2].
        let config = ResidualFsqConfig::new(vec![4, 4], 5, 1).unwrap();
        let layer =
            Fsq::<CpuRuntime>::new(config.layer_config().unwrap(), &device, None, None).unwrap();

        // Wrong out-features on project_in: [3, 5] instead of [2, 5].
        let bad_in = Tensor::<CpuRuntime>::from_slice(&[0.1f32; 15], &[3, 5], &device).unwrap();
        let good_out = Tensor::<CpuRuntime>::from_slice(&[0.2f32; 10], &[5, 2], &device).unwrap();

        let err = ResidualFsq::new(
            config,
            ResidualFsqWeights {
                project_in: Some(Linear::new(bad_in, None, false)),
                project_out: Some(Linear::new(good_out, None, false)),
                layers: vec![layer],
            },
            &device,
        )
        .err()
        .unwrap();
        assert!(matches!(err, Error::ModelError { .. }), "got {err:?}");
    }

    #[test]
    fn test_missing_projection_rejected_when_dims_differ() {
        let (_, device) = cpu_setup();
        let config = ResidualFsqConfig::new(vec![4, 4], 5, 1).unwrap();
        let layer =
            Fsq::<CpuRuntime>::new(config.layer_config().unwrap(), &device, None, None).unwrap();

        let err = ResidualFsq::new(
            config,
            ResidualFsqWeights {
                project_in: None,
                project_out: None,
                layers: vec![layer],
            },
            &device,
        )
        .err()
        .unwrap();
        assert!(
            matches!(
                err,
                Error::InvalidArgument {
                    arg: "project_in/project_out",
                    ..
                }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn test_projecting_inner_layer_rejected() {
        let (_, device) = cpu_setup();
        // Inner layers must be plain FSQ (lucidrains/vector-quantize-pytorch's are nn.Identity-projected).
        let config = ResidualFsqConfig::new(vec![4, 4], 2, 1).unwrap();
        let w_in = Tensor::<CpuRuntime>::from_slice(&[0.1f32; 10], &[2, 5], &device).unwrap();
        let w_out = Tensor::<CpuRuntime>::from_slice(&[0.2f32; 10], &[5, 2], &device).unwrap();
        let projecting_layer = Fsq::<CpuRuntime>::new(
            FsqConfig::new(vec![4, 4], 5).unwrap(),
            &device,
            Some(Linear::new(w_in, None, false)),
            Some(Linear::new(w_out, None, false)),
        )
        .unwrap();

        let err = ResidualFsq::new(
            config,
            ResidualFsqWeights {
                project_in: None,
                project_out: None,
                layers: vec![projecting_layer],
            },
            &device,
        )
        .err()
        .unwrap();
        assert!(matches!(err, Error::ModelError { .. }), "got {err:?}");
    }

    #[test]
    fn test_zero_num_quantizers_rejected() {
        let err = ResidualFsqConfig::new(vec![4, 4], 2, 0).err().unwrap();
        assert!(matches!(
            err,
            Error::InvalidArgument {
                arg: "num_quantizers",
                ..
            }
        ));
    }
}
