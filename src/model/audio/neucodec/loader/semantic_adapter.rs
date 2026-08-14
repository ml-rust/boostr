//! Loads `semantic_adapter.*` and assembles a [`SemanticAdapter`].

use super::support::checked_tensor;
use crate::error::Result;
use crate::format::safetensors_loader::SafeTensorsLoader;
use crate::model::audio::neucodec::semantic_adapter::{
    SEMANTIC_ADAPTER_CHANNELS, SEMANTIC_ADAPTER_KERNEL_SIZE, SemanticAdapter,
    SemanticAdapterWeights,
};
use crate::nn::Conv1d;
use numr::dtype::DType;
use numr::ops::PaddingMode;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::path::Path;

/// Top-level prefix for the semantic adapter (`neucodec.module.SemanticEncoder`).
pub const DEFAULT_SEMANTIC_ADAPTER_PREFIX: &str = "semantic_adapter";

/// Reads `semantic_adapter.*` and assembles a [`SemanticAdapter`].
///
/// Checkpoint tensors, confirmed from the safetensors header:
/// `conv1.weight` (no bias), `conv2.{weight,bias}`, `conv3.{weight,bias}`,
/// `conv4.weight` (no bias) — all `[1024, 1024, 3]` / `[1024]`. See
/// [`semantic_adapter`](crate::model::audio::neucodec::semantic_adapter) for
/// the upstream name mapping.
struct SemanticAdapterLoader<'a, R: Runtime<DType = DType>> {
    loader: &'a mut SafeTensorsLoader,
    device: &'a R::Device,
    prefix: String,
}

impl<R: Runtime<DType = DType>> SemanticAdapterLoader<'_, R> {
    fn tensor(&mut self, name: &str, expected: &[usize]) -> Result<Tensor<R>> {
        checked_tensor::<R>(self.loader, self.device, &self.prefix, name, expected)
    }

    fn conv(&mut self, name: &str, bias: bool) -> Result<Conv1d<R>> {
        let c = SEMANTIC_ADAPTER_CHANNELS;
        let k = SEMANTIC_ADAPTER_KERNEL_SIZE;
        let weight = self.tensor(&format!("{name}.weight"), &[c, c, k])?;
        let bias = if bias {
            Some(self.tensor(&format!("{name}.bias"), &[c])?)
        } else {
            None
        };
        Ok(Conv1d::new(
            weight,
            bias,
            1,
            PaddingMode::Custom(1, 1, 0, 0),
            1,
            1,
            false,
        ))
    }

    fn build(&mut self) -> Result<SemanticAdapterWeights<R>> {
        Ok(SemanticAdapterWeights {
            conv1: self.conv("conv1", false)?,
            conv2: self.conv("conv2", true)?,
            conv3: self.conv("conv3", true)?,
            conv4: self.conv("conv4", false)?,
        })
    }
}

/// Load the semantic adapter from a `neuphonic/neucodec` checkpoint.
pub fn load_semantic_adapter<R: Runtime<DType = DType>, P: AsRef<Path>>(
    path: P,
    device: &R::Device,
) -> Result<SemanticAdapter<R>> {
    let mut loader = SafeTensorsLoader::open(path)?;
    let weights = SemanticAdapterLoader::<R> {
        loader: &mut loader,
        device,
        prefix: DEFAULT_SEMANTIC_ADAPTER_PREFIX.to_string(),
    }
    .build()?;
    Ok(SemanticAdapter::new(weights))
}
