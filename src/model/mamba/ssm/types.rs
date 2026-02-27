//! SSM input bundle types and helpers.

use crate::model::mamba::mamba2::Mamba2Config;
use numr::autograd::Var;
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

/// Make a Var contiguous (copies data if non-contiguous).
/// Preserves the requires_grad flag but not the grad_fn (treated as a leaf).
pub fn var_contiguous<R: Runtime>(v: &Var<R>) -> Var<R> {
    if v.tensor().is_contiguous() {
        v.clone()
    } else {
        Var::new(v.tensor().contiguous(), v.requires_grad())
    }
}
