//! Calibration operations for quantization (AWQ, GPTQ, Fisher Information)

use crate::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Calibration operations for quantization.
///
/// Provides the core kernels needed by AWQ and GPTQ quantization pipelines:
/// channel importance scoring, Fisher information, Hessian accumulation,
/// and column-wise quantization with error compensation.
pub trait CalibrationOps<R: Runtime> {
    /// AWQ channel importance scoring.
    ///
    /// Computes per-channel importance scores for Activation-aware Weight Quantization.
    ///
    /// Algorithm:
    /// ```text
    /// act_scale[j] = max_batch(|activations[:, j]|)
    /// score[j] = mean_i(act_scale[j] * |weights[i, j]|)
    /// ```
    ///
    /// # Layout contract
    ///
    /// - `activations`: `[N, K]` — calibration activations (N samples, K channels)
    /// - `weights`: `[M, K]` — weight matrix (M output features, K input channels)
    /// - Output: `[K]` — per-channel importance scores
    fn awq_channel_scores(
        &self,
        activations: &Tensor<R>,
        weights: &Tensor<R>,
    ) -> Result<Tensor<R>>;

    /// Diagonal Fisher Information Matrix.
    ///
    /// Computes the diagonal FIM from a batch of gradients.
    ///
    /// Algorithm:
    /// ```text
    /// fisher[i] = mean_n(gradients[n, i]^2)
    /// ```
    ///
    /// # Layout contract
    ///
    /// - `gradients`: `[N, P]` — gradient samples (N samples, P parameters)
    /// - Output: `[P]` — diagonal Fisher information
    fn fisher_information(&self, gradients: &Tensor<R>) -> Result<Tensor<R>>;

    /// GPTQ Hessian accumulation.
    ///
    /// Accumulates the Hessian approximation from a block of calibration data.
    ///
    /// Algorithm:
    /// ```text
    /// H_new = H + (2 / batch) * X^T @ X
    /// ```
    ///
    /// # Layout contract
    ///
    /// - `hessian`: `[K, K]` — current Hessian estimate
    /// - `x_block`: `[B, K]` — block of calibration inputs
    /// - Output: `[K, K]` — updated Hessian
    fn gptq_hessian_update(
        &self,
        hessian: &Tensor<R>,
        x_block: &Tensor<R>,
    ) -> Result<Tensor<R>>;

    /// GPTQ column-wise quantization with error compensation.
    ///
    /// Quantizes a weight matrix column by column using the inverse Hessian
    /// to optimally distribute quantization error.
    ///
    /// # Layout contract
    ///
    /// - `weight`: `[M, K]` — weight matrix to quantize
    /// - `h_inv`: `[K, K]` — inverse Hessian (from Cholesky of H + diag)
    /// - `num_bits`: quantization bit width (e.g. 4)
    /// - `group_size`: number of columns per quantization group
    /// - `symmetric`: if true, use symmetric quantization (no zero point)
    ///
    /// # Returns
    ///
    /// `(quantized_weight, scales, zeros)`:
    /// - `quantized_weight`: `[M, K]` — quantized weight (dequantized to float)
    /// - `scales`: `[M, K/group_size]` — per-group scale factors
    /// - `zeros`: `[M, K/group_size]` — per-group zero points (all zero if symmetric)
    fn gptq_quantize_column(
        &self,
        weight: &Tensor<R>,
        h_inv: &Tensor<R>,
        num_bits: u32,
        group_size: u32,
        symmetric: bool,
    ) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>)>;
}
