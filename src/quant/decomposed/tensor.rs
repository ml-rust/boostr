//! Decomposed quantized tensor for AWQ/GPTQ formats

use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::method::DecomposedQuantMethod;

/// A decomposed quantized tensor storing separate component tensors.
///
/// Unlike `QuantTensor` (monolithic GGUF blocks), AWQ and GPTQ store
/// weights as separate qweight/scales/qzeros tensors. This type holds
/// all components together with the method descriptor.
///
/// # Component shapes (AWQ, group_size=128)
///
/// For a logical weight [out_features, in_features]:
/// - `qweight`: `[in_features, out_features/8]` packed u32 (stored as F32 for tensor compat)
/// - `scales`: `[in_features/group_size, out_features]` F32
/// - `qzeros`: `[in_features/group_size, out_features]` F32 (unpacked at load time)
pub struct DecomposedQuantTensor<R: Runtime> {
    /// Packed quantized weights (u32 reinterpreted as tensor element type)
    pub qweight: Tensor<R>,
    /// Per-group scale factors
    pub scales: Tensor<R>,
    /// Per-group zero points (unpacked to full precision at load time)
    pub qzeros: Tensor<R>,
    /// GPTQ column-to-group mapping (None for AWQ)
    pub g_idx: Option<Tensor<R>>,
    /// Quantization method and parameters
    pub method: DecomposedQuantMethod,
    /// Logical shape of the full-precision weight [out_features, in_features]
    pub logical_shape: Vec<usize>,
}

impl<R: Runtime> DecomposedQuantTensor<R> {
    /// Create a new decomposed quantized tensor
    pub fn new(
        qweight: Tensor<R>,
        scales: Tensor<R>,
        qzeros: Tensor<R>,
        g_idx: Option<Tensor<R>>,
        method: DecomposedQuantMethod,
        logical_shape: Vec<usize>,
    ) -> Self {
        Self {
            qweight,
            scales,
            qzeros,
            g_idx,
            method,
            logical_shape,
        }
    }

    /// Logical shape of the weight (as if it were full precision)
    pub fn shape(&self) -> &[usize] {
        &self.logical_shape
    }

    /// Group size for this tensor's quantization method
    pub fn group_size(&self) -> usize {
        self.method.group_size()
    }
}
