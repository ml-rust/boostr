//! Loads the FSQ quantizer (`quantizer.project_in`/`project_out`) from a
//! `neuphonic/neucodec` checkpoint.

use super::support::checked_tensor;
use crate::error::Result;
use crate::format::safetensors_loader::SafeTensorsLoader;
use crate::nn::Linear;
use crate::nn::fsq::{Fsq, FsqConfig};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::path::Path;

/// Top-level prefix for the FSQ quantizer's projections.
pub const DEFAULT_QUANTIZER_PREFIX: &str = "quantizer";

/// Per-dimension FSQ levels for the released NeuCodec checkpoint:
/// `4^8 = 65_536` codes at 50 Hz.
pub const NEUCODEC_FSQ_LEVELS: [u32; 8] = [4; 8];

/// Width of the FSQ latent that `project_out` produces and the decoder's `fc`
/// consumes (checkpoint: 2048).
const FSQ_INPUT_DIM: usize = 2048;

/// Load the FSQ quantizer (`quantizer.project_in`/`project_out`) from a
/// checkpoint.
///
/// Upstream builds it as `ResidualFSQ(dim=2048, levels=[4]*8,
/// num_quantizers=1)`, which stores exactly these two projections under
/// `quantizer.*` — the same names this reads.
///
/// Only `project_out` is needed to decode, but `project_in` is loaded too so
/// the returned quantizer can also encode (and so a missing/mis-shaped tensor
/// is caught at load time rather than at first use).
pub fn load_fsq_quantizer<R: Runtime<DType = DType>, P: AsRef<Path>>(
    path: P,
    device: &R::Device,
) -> Result<Fsq<R>> {
    let config = FsqConfig::new(NEUCODEC_FSQ_LEVELS.to_vec(), FSQ_INPUT_DIM)?;
    let codebook_dim = config.codebook_dim();

    let mut loader = SafeTensorsLoader::open(path)?;
    let mut take = |name: &str, expected: &[usize]| -> Result<Tensor<R>> {
        checked_tensor::<R>(
            &mut loader,
            device,
            DEFAULT_QUANTIZER_PREFIX,
            name,
            expected,
        )
    };

    let project_in = Linear::new(
        take("project_in.weight", &[codebook_dim, FSQ_INPUT_DIM])?,
        Some(take("project_in.bias", &[codebook_dim])?),
        false,
    );
    let project_out = Linear::new(
        take("project_out.weight", &[FSQ_INPUT_DIM, codebook_dim])?,
        Some(take("project_out.bias", &[FSQ_INPUT_DIM])?),
        false,
    );

    Fsq::new(config, device, Some(project_in), Some(project_out))
}
