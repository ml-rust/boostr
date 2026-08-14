//! Shared tensor-fetch helper for NeuCodec's per-component loaders.
//!
//! Every component loader reads a checkpoint tensor by `{prefix}.{name}` and
//! shape-checks it before handing it to the model builder; this is that one
//! check, factored out so each loader's `tensor` method is a one-line
//! delegate instead of a repeated copy.

use crate::error::{Error, Result};
use crate::format::safetensors_loader::SafeTensorsLoader;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Load `{prefix}.{name}` and verify its shape matches `expected`.
pub(super) fn checked_tensor<R: Runtime<DType = DType>>(
    loader: &mut SafeTensorsLoader,
    device: &R::Device,
    prefix: &str,
    name: &str,
    expected: &[usize],
) -> Result<Tensor<R>> {
    let full = format!("{prefix}.{name}");
    let t = loader.load_tensor::<R>(&full, device)?;
    if t.shape() != expected {
        return Err(Error::ModelError {
            reason: format!(
                "{full}: expected shape {expected:?}, checkpoint has {:?}",
                t.shape()
            ),
        });
    }
    Ok(t)
}
