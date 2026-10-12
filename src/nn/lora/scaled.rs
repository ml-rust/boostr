//! [`LoraLinear`] construction from loaded factors with a scaling given per
//! module, and the shape and dtype checks every loaded adapter passes.

use crate::error::{Error, Result};
use crate::nn::MaybeQuantLinear;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::LoraLinear;

impl<R: Runtime<DType = DType>> LoraLinear<R> {
    /// Wrap `base` `[out_features, in_features]` with loaded factors
    /// `lora_a` `[rank, in_features]` and `lora_b` `[out_features, rank]`.
    ///
    /// `scaling` is stored as given. The caller computes it per module:
    /// `alpha / rank`, `alpha / sqrt(rank)` under rsLoRA, or a per-module
    /// alpha or rank override. [`Self::from_weights`] derives
    /// `alpha / rank` instead.
    ///
    /// `trainable` sets `requires_grad` on both factors. Pass `false` for an
    /// adapter served at inference; see [`Self::set_trainable`].
    ///
    /// # Errors
    ///
    /// [`Error::ModelError`] when a shape or dtype check of
    /// [`check_factors`] fails, or when `scaling` is not finite.
    pub fn from_scaled_weights(
        base: impl Into<MaybeQuantLinear<R>>,
        lora_a: Tensor<R>,
        lora_b: Tensor<R>,
        scaling: f32,
        trainable: bool,
    ) -> Result<Self> {
        let base = base.into();
        check_factors(&base, &lora_a, &lora_b)?;
        if !scaling.is_finite() {
            return Err(Error::ModelError {
                reason: format!("LoRA scaling {scaling} is not finite"),
            });
        }
        Ok(Self {
            base,
            lora_a: Var::new(lora_a, trainable),
            lora_b: Var::new(lora_b, trainable),
            scaling,
        })
    }
}

/// Check that `lora_a` is `[rank, in_features]` and `lora_b` is
/// `[out_features, rank]` for `base` `[out_features, in_features]`, with
/// `rank > 0`. Both factors share one dtype, and a dense base has that
/// dtype too.
///
/// # Errors
///
/// [`Error::ModelError`] naming the failed check and the shapes involved.
pub(super) fn check_factors<R: Runtime<DType = DType>>(
    base: &MaybeQuantLinear<R>,
    lora_a: &Tensor<R>,
    lora_b: &Tensor<R>,
) -> Result<()> {
    let (out_features, in_features) = base_dims(base)?;
    check_a(lora_a, in_features)?;
    let rank = lora_a.shape()[0];
    if lora_b.shape() != [out_features, rank] {
        return Err(Error::ModelError {
            reason: format!(
                "LoRA lora_b must be [out_features, rank] = [{out_features}, {rank}] for base \
                 [{out_features}, {in_features}] and lora_a {:?}, got {:?}",
                lora_a.shape(),
                lora_b.shape()
            ),
        });
    }
    if lora_a.dtype() != lora_b.dtype() {
        return Err(Error::ModelError {
            reason: format!(
                "LoRA factors disagree on dtype: lora_a {:?}, lora_b {:?}",
                lora_a.dtype(),
                lora_b.dtype()
            ),
        });
    }
    if let Some(weight) = base.weight() {
        let base_dtype = weight.tensor().dtype();
        if base_dtype != lora_a.dtype() {
            return Err(Error::ModelError {
                reason: format!(
                    "LoRA factors are {:?} but the dense base weight is {base_dtype:?}; cast the \
                     adapter to the base dtype before attaching it",
                    lora_a.dtype()
                ),
            });
        }
    }
    Ok(())
}

/// The `(out_features, in_features)` of a 2-D `base`.
///
/// # Errors
///
/// [`Error::ModelError`] when the base weight is not 2-D.
pub(super) fn base_dims<R: Runtime<DType = DType>>(
    base: &MaybeQuantLinear<R>,
) -> Result<(usize, usize)> {
    match base.shape() {
        [out, inp] => Ok((*out, *inp)),
        other => Err(Error::ModelError {
            reason: format!("LoRA base weight must be 2-D [out, in], got {other:?}"),
        }),
    }
}

/// Check that `lora_a` is `[rank, in_features]` with `rank > 0`.
///
/// # Errors
///
/// [`Error::ModelError`] naming the expected and actual shapes.
pub(super) fn check_a<R: Runtime>(lora_a: &Tensor<R>, in_features: usize) -> Result<()> {
    match lora_a.shape() {
        [rank, inp] if *rank > 0 && *inp == in_features => Ok(()),
        other => Err(Error::ModelError {
            reason: format!(
                "LoRA lora_a must be [rank, in_features] = [rank > 0, {in_features}], got {other:?}"
            ),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Linear;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    fn tensor(device: &CpuDevice, shape: &[usize]) -> Tensor<CpuRuntime> {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin() * 0.1).collect();
        Tensor::<CpuRuntime>::from_slice(&data, shape, device).unwrap()
    }

    fn base(device: &CpuDevice) -> Linear<CpuRuntime> {
        Linear::new(tensor(device, &[6, 4]), None, false)
    }

    #[test]
    fn scaling_is_stored_as_given() {
        let device = CpuDevice::new();
        let lora = LoraLinear::from_scaled_weights(
            base(&device),
            tensor(&device, &[2, 4]),
            tensor(&device, &[6, 2]),
            0.375,
            false,
        )
        .unwrap();
        assert_eq!(lora.scaling(), 0.375);
        assert_eq!(lora.rank(), 2);
        assert!(!lora.is_trainable());
    }

    #[test]
    fn mismatched_shapes_are_rejected() {
        let device = CpuDevice::new();
        let wrong_in = LoraLinear::from_scaled_weights(
            base(&device),
            tensor(&device, &[2, 5]),
            tensor(&device, &[6, 2]),
            1.0,
            false,
        );
        assert!(wrong_in.is_err());
        let wrong_out = LoraLinear::from_scaled_weights(
            base(&device),
            tensor(&device, &[2, 4]),
            tensor(&device, &[5, 2]),
            1.0,
            false,
        );
        assert!(wrong_out.is_err());
        let wrong_rank = LoraLinear::from_scaled_weights(
            base(&device),
            tensor(&device, &[2, 4]),
            tensor(&device, &[6, 3]),
            1.0,
            false,
        );
        assert!(wrong_rank.is_err());
    }

    #[test]
    fn non_finite_scaling_is_rejected() {
        let device = CpuDevice::new();
        let lora = LoraLinear::from_scaled_weights(
            base(&device),
            tensor(&device, &[2, 4]),
            tensor(&device, &[6, 2]),
            f32::NAN,
            false,
        );
        assert!(lora.is_err());
    }
}
