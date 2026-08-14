//! Loads `fc_encoder.*` — the encoder-side prior projection that upstream
//! `NeuCodec.encode_code` calls `fc_prior`.
//!
//! The checkpoint name (`fc_encoder`) and the upstream attribute name
//! (`fc_prior`) disagree; this is the single place that mapping lives.

use super::support::checked_tensor;
use crate::error::Result;
use crate::format::safetensors_loader::SafeTensorsLoader;
use crate::model::audio::neucodec::encoder::PRIOR_DIM;
use crate::nn::Linear;
use numr::dtype::DType;
use numr::runtime::Runtime;
use std::path::Path;

/// Top-level prefix of the prior projection in the HF checkpoint.
pub const DEFAULT_FC_PRIOR_PREFIX: &str = "fc_encoder";

/// Load `fc_encoder.{weight,bias}` as `Linear(2048 -> 2048)` (bias present).
pub fn load_fc_prior<R: Runtime<DType = DType>, P: AsRef<Path>>(
    path: P,
    device: &R::Device,
) -> Result<Linear<R>> {
    let mut loader = SafeTensorsLoader::open(path)?;
    let prefix = DEFAULT_FC_PRIOR_PREFIX;
    let weight = checked_tensor::<R>(
        &mut loader,
        device,
        prefix,
        "weight",
        &[PRIOR_DIM, PRIOR_DIM],
    )?;
    let bias = checked_tensor::<R>(&mut loader, device, prefix, "bias", &[PRIOR_DIM])?;
    Ok(Linear::new(weight, Some(bias), false))
}
