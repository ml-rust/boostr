//! The Hadamard rotation a quantized matmul can fold into its activation
//! quantization.

use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// A Hadamard rotation of the activation, applied before the matmul:
/// multiply by `signs` when present, then the normalized Walsh-Hadamard
/// transform per `block_size` segment. The same operation as
/// `FwhtOps::fwht(x, block_size, signs)`.
///
/// `block_size` is a power of two that divides the activation's last dim;
/// `signs`, when present, is 1-D with that dim's width in the activation's
/// dtype.
pub struct Rotation<'a, R: Runtime> {
    pub block_size: usize,
    pub signs: Option<&'a Tensor<R>>,
}
