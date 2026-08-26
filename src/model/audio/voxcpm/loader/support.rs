//! Shared tensor-fetch helper for the VoxCPM2 decoder loader.
//!
//! Same idiom as `neucodec/loader/support.rs` (not reused directly: that
//! module's helper is private to its own `loader` submodule).

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
