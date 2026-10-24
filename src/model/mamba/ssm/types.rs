//! SSM input bundle types and helpers.

use crate::error::Result;
use crate::model::mamba::mamba2::Mamba2Config;
use crate::nn::segment_masks::SegmentMasks;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Bundled SSM inputs to reduce parameter count (training / autograd path).
pub struct SsmInput<'a, R: Runtime> {
    pub x: &'a Var<R>,
    pub a: &'a Var<R>,
    pub b: &'a Var<R>,
    pub c: &'a Var<R>,
    pub d_param: Option<&'a Var<R>>,
    pub dt: &'a Var<R>,
    pub config: &'a Mamba2Config,
    /// Optional clamp applied to the recurrent hidden state after each sequential update.
    pub hidden_state_clamp: Option<f64>,
    /// Constant `[batch, seq, 1, 1]` mask: 0 at a document start, else 1.
    /// The decay term is multiplied by it, so a document start reads no
    /// state from the previous token. `None` keeps one document per row.
    pub keep: Option<&'a Var<R>>,
}

/// The scan's `keep` mask for `segments`, as a constant `Var`.
///
/// # Errors
///
/// [`crate::error::Error::Numr`] when the mask reshape fails.
pub fn keep_var<R: Runtime<DType = DType>>(
    segments: Option<&SegmentMasks<R>>,
) -> Result<Option<Var<R>>> {
    segments
        .map(|masks| masks.keep_4d().map(|keep| Var::new(keep, false)))
        .transpose()
}

/// Bundled SSM inputs for inference (raw tensors, no Var).
pub struct SsmInferenceInput<'a, R: Runtime> {
    pub x: &'a Tensor<R>,
    pub a: &'a Tensor<R>,
    pub b: &'a Tensor<R>,
    pub c: &'a Tensor<R>,
    pub d_param: Option<&'a Tensor<R>>,
    pub dt: &'a Tensor<R>,
    pub config: &'a Mamba2Config,
}

pub use crate::nn::var_ops::var_contiguous;
