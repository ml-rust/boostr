//! [`NeuCodecEncoder`]: the parts bundle, construction, checkpoint loading,
//! and accessors.

use crate::error::{Error, Result};
use crate::model::audio::neucodec::acoustic_encoder::AcousticEncoder;
use crate::model::audio::neucodec::loader;
use crate::model::audio::neucodec::semantic_adapter::{SEMANTIC_ADAPTER_CHANNELS, SemanticAdapter};
use crate::model::audio::neucodec::semantic_encoder::SemanticEncoder;
use crate::nn::Linear;
use crate::nn::fsq::ResidualFsq;
use numr::dtype::DType;
use numr::runtime::Runtime;
use std::path::Path;

/// Width of the concatenated semantic+acoustic prior: one 1024-wide branch
/// each, and also the FSQ quantizer's `dim`.
pub const PRIOR_DIM: usize = 2 * SEMANTIC_ADAPTER_CHANNELS;

/// Already-built parts for [`NeuCodecEncoder`], following the `*Weights`
/// convention used across this module.
pub struct NeuCodecEncoderWeights<R: Runtime> {
    pub acoustic_encoder: AcousticEncoder<R>,
    pub semantic_encoder: SemanticEncoder<R>,
    pub semantic_adapter: SemanticAdapter<R>,
    /// The reference implementation's `fc_prior`, stored as `fc_encoder.*` in the checkpoint.
    pub fc_prior: Linear<R>,
    pub quantizer: ResidualFsq<R>,
}

/// The full NeuCodec encoder: both branches, the prior projection, and the
/// residual FSQ quantizer.
pub struct NeuCodecEncoder<R: Runtime> {
    pub(super) acoustic_encoder: AcousticEncoder<R>,
    pub(super) semantic_encoder: SemanticEncoder<R>,
    pub(super) semantic_adapter: SemanticAdapter<R>,
    pub(super) fc_prior: Linear<R>,
    pub(super) quantizer: ResidualFsq<R>,
}

impl<R: Runtime<DType = DType>> NeuCodecEncoder<R> {
    /// Assemble from already-built parts, validating that the prior projection
    /// and the quantizer agree on [`PRIOR_DIM`].
    pub fn new(weights: NeuCodecEncoderWeights<R>) -> Result<Self> {
        check_fc_prior(&weights.fc_prior)?;

        let dim = weights.quantizer.config().dim;
        if dim != PRIOR_DIM {
            return Err(Error::ModelError {
                reason: format!("quantizer dim {dim} does not match prior width {PRIOR_DIM}"),
            });
        }

        Ok(Self {
            acoustic_encoder: weights.acoustic_encoder,
            semantic_encoder: weights.semantic_encoder,
            semantic_adapter: weights.semantic_adapter,
            fc_prior: weights.fc_prior,
            quantizer: weights.quantizer,
        })
    }

    /// Load every part from a `neuphonic/neucodec` checkpoint (file, or the
    /// directory containing `model.safetensors`).
    pub fn from_safetensors<P: AsRef<Path>>(path: P, device: &R::Device) -> Result<Self> {
        let path = path.as_ref();
        Self::new(NeuCodecEncoderWeights {
            acoustic_encoder: loader::load_acoustic_encoder::<R, _>(path, device)?,
            semantic_encoder: loader::load_semantic_encoder::<R, _>(path, device)?,
            semantic_adapter: loader::load_semantic_adapter::<R, _>(path, device)?,
            fc_prior: loader::load_fc_prior::<R, _>(path, device)?,
            quantizer: loader::load_residual_fsq::<R, _>(path, device)?,
        })
    }

    pub fn acoustic_encoder(&self) -> &AcousticEncoder<R> {
        &self.acoustic_encoder
    }

    pub fn semantic_encoder(&self) -> &SemanticEncoder<R> {
        &self.semantic_encoder
    }

    pub fn semantic_adapter(&self) -> &SemanticAdapter<R> {
        &self.semantic_adapter
    }

    pub fn quantizer(&self) -> &ResidualFsq<R> {
        &self.quantizer
    }
}

/// `fc_prior` must be `Linear(2048 -> 2048)` WITH bias.
fn check_fc_prior<R: Runtime>(linear: &Linear<R>) -> Result<()> {
    let shape = linear.weight().tensor().shape();
    if shape != [PRIOR_DIM, PRIOR_DIM].as_slice() {
        return Err(Error::ModelError {
            reason: format!("fc_prior weight shape {shape:?} != [{PRIOR_DIM}, {PRIOR_DIM}]"),
        });
    }
    if linear.bias().is_none() {
        return Err(Error::ModelError {
            reason: "fc_prior must have a bias".to_string(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

    #[test]
    fn fc_prior_must_be_2048_square_with_bias() {
        let (_client, device) = cpu_setup();

        let good = Linear::<CpuRuntime>::new(
            Tensor::from_slice(
                &vec![0.0f32; PRIOR_DIM * PRIOR_DIM],
                &[PRIOR_DIM, PRIOR_DIM],
                &device,
            )
            .unwrap(),
            Some(Tensor::from_slice(&vec![0.0f32; PRIOR_DIM], &[PRIOR_DIM], &device).unwrap()),
            false,
        );
        assert!(check_fc_prior(&good).is_ok());

        let no_bias = Linear::<CpuRuntime>::new(
            Tensor::from_slice(
                &vec![0.0f32; PRIOR_DIM * PRIOR_DIM],
                &[PRIOR_DIM, PRIOR_DIM],
                &device,
            )
            .unwrap(),
            None,
            false,
        );
        assert!(check_fc_prior(&no_bias).is_err());

        let wrong_shape = Linear::<CpuRuntime>::new(
            Tensor::from_slice(&[0.0f32; 4 * 8], &[4, 8], &device).unwrap(),
            Some(Tensor::from_slice(&[0.0f32; 4], &[4], &device).unwrap()),
            false,
        );
        assert!(check_fc_prior(&wrong_shape).is_err());
    }
}
