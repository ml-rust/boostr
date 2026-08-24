//! Initialization strategies for new tensors.

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// PyTorch's `fan_in` for a weight tensor, matching
/// `torch.nn.init._calculate_fan_in_and_fan_out`.
///
/// PyTorch computes `fan_in = tensor.size(1) * prod(tensor.shape[2:])`, i.e.
/// the product of every dimension EXCEPT the leading output dimension.
///
/// - `Linear` weight `[out_features, in_features]` → `in_features`. This layout
///   is fixed by [`crate::nn::Linear::forward`], which computes
///   `input @ weight^T` and reads `out_features` from `shape[0]`.
/// - `Conv1d` weight `[out_channels, in_channels / groups, kernel]` →
///   `in_channels / groups * kernel`. The division by `groups` is already
///   baked into the stored `shape[1]`, so a depthwise conv `[C, 1, K]`
///   correctly yields `K`, not `C` and not `1`.
///
/// PyTorch rejects tensors with fewer than 2 dimensions. A 1-D tensor has no
/// separate input dimension, so its own length is the only defensible fan_in.
fn pytorch_fan_in(shape: &[usize]) -> usize {
    if shape.len() < 2 {
        return shape.first().copied().unwrap_or(1);
    }
    shape[1..].iter().product()
}

/// Initialization strategy for new tensors.
#[derive(Debug, Clone, Copy)]
pub enum Init {
    /// All zeros
    Zeros,
    /// All ones
    Ones,
    /// Constant value
    Const(f32),
    /// Uniform random in `[-bound, bound]`
    Uniform(f32),
    /// Kaiming uniform (PyTorch `Linear`/`Conv` default):
    /// U(-1/sqrt(fan_in), 1/sqrt(fan_in)).
    ///
    /// `fan_in` follows PyTorch exactly — see [`pytorch_fan_in`]. For a
    /// `[out_features, in_features]` weight that is `in_features`, NOT
    /// `out_features`: the leading dimension is the output side.
    PyTorchLinear,
    /// PyTorch Embedding default: N(0, 1)
    PyTorchEmbedding,
    /// Kaiming (He) normal: N(0, sqrt(2 / fan_in))
    ///
    /// Standard initialization for ReLU networks. fan_in is the product of
    /// all dimensions except the last (output) dimension.
    ///
    /// NOTE: this assumes an `[..., in, out]` layout — the OPPOSITE of the
    /// `[out_features, in_features]` layout `Linear` stores and
    /// [`Init::PyTorchLinear`] assumes. No model in this workspace uses
    /// `Kaiming`; the first one that does must transpose or switch variant.
    Kaiming,
    /// Xavier (Glorot) normal: N(0, sqrt(2 / (fan_in + fan_out)))
    ///
    /// Standard initialization for Sigmoid/Tanh networks. Used in some
    /// attention weight initializations.
    ///
    /// NOTE: carries the same `[..., in, out]` layout assumption as
    /// [`Init::Kaiming`], opposite to [`Init::PyTorchLinear`]. Xavier is
    /// symmetric in `fan_in + fan_out`, so a 2-D weight is unaffected by the
    /// mix-up; a 3-D or higher weight is not.
    Xavier,
    /// Normal distribution with given mean and standard deviation.
    Randn { mean: f64, stdev: f64 },
    /// Truncated normal: N(mean, stdev) clamped to [mean - 2*stdev, mean + 2*stdev]
    ///
    /// Used by GPT-2, BERT, and most modern LLMs for training stability.
    TruncatedNormal { mean: f64, stdev: f64 },
}

impl Init {
    /// Create a tensor initialized according to this strategy.
    ///
    /// # Arguments
    /// * `shape` - Shape of the tensor to create
    /// * `dtype` - Data type
    /// * `device` - Device to create on
    /// * `client` - Runtime client (needed for random ops)
    pub fn init_tensor<R, C>(
        &self,
        shape: &[usize],
        dtype: DType,
        device: &R::Device,
        client: &C,
    ) -> Result<Tensor<R>>
    where
        R: Runtime<DType = DType>,
        C: numr::runtime::RuntimeClient<R>
            + numr::ops::RandomOps<R>
            + numr::ops::ScalarOps<R>
            + numr::ops::BinaryOps<R>
            + numr::ops::CompareOps<R>
            + numr::ops::TensorOps<R>,
    {
        // Trait bounds on the function provide the methods

        match *self {
            Init::Zeros => Ok(Tensor::zeros(shape, dtype, device)?),
            Init::Ones => Ok(Tensor::ones(shape, dtype, device)?),
            Init::Const(val) => {
                let t = Tensor::zeros(shape, dtype, device)?;
                client.add_scalar(&t, val as f64).map_err(Error::Numr)
            }
            Init::Uniform(bound) => {
                // U(-bound, bound) = rand() * 2*bound - bound
                let r = client.rand(shape, dtype).map_err(Error::Numr)?;
                let scaled = client
                    .mul_scalar(&r, 2.0 * bound as f64)
                    .map_err(Error::Numr)?;
                client
                    .add_scalar(&scaled, -(bound as f64))
                    .map_err(Error::Numr)
            }
            Init::PyTorchLinear => {
                // Kaiming uniform: U(-1/sqrt(fan_in), 1/sqrt(fan_in))
                let fan_in = pytorch_fan_in(shape);
                let bound = 1.0 / (fan_in as f64).sqrt();
                let r = client.rand(shape, dtype).map_err(Error::Numr)?;
                let scaled = client.mul_scalar(&r, 2.0 * bound).map_err(Error::Numr)?;
                client.add_scalar(&scaled, -bound).map_err(Error::Numr)
            }
            Init::PyTorchEmbedding => {
                // PyTorch `nn.Embedding` default initializes weights with N(0, 1).
                client.randn(shape, dtype).map_err(Error::Numr)
            }
            Init::Kaiming => {
                // Kaiming/He normal: N(0, sqrt(2 / fan_in))
                let fan_in = if shape.len() >= 2 {
                    shape[..shape.len() - 1].iter().product::<usize>()
                } else {
                    shape[0]
                };
                let std = (2.0 / fan_in as f64).sqrt();
                let r = client.randn(shape, dtype).map_err(Error::Numr)?;
                client.mul_scalar(&r, std).map_err(Error::Numr)
            }
            Init::Xavier => {
                // Xavier/Glorot normal: N(0, sqrt(2 / (fan_in + fan_out)))
                let (fan_in, fan_out) = if shape.len() >= 2 {
                    let fi = shape[..shape.len() - 1].iter().product::<usize>();
                    let fo = shape[shape.len() - 1];
                    (fi, fo)
                } else {
                    (shape[0], shape[0])
                };
                let std = (2.0 / (fan_in + fan_out) as f64).sqrt();
                let r = client.randn(shape, dtype).map_err(Error::Numr)?;
                client.mul_scalar(&r, std).map_err(Error::Numr)
            }
            Init::Randn { mean, stdev } => {
                let r = client.randn(shape, dtype).map_err(Error::Numr)?;
                let scaled = client.mul_scalar(&r, stdev).map_err(Error::Numr)?;
                if mean != 0.0 {
                    client.add_scalar(&scaled, mean).map_err(Error::Numr)
                } else {
                    Ok(scaled)
                }
            }
            Init::TruncatedNormal { mean, stdev } => {
                // Generate N(0, 1), clamp to [-2, 2], then scale by stdev and shift by mean
                let r = client.randn(shape, dtype).map_err(Error::Numr)?;
                let clamped = client.clamp(&r, -2.0, 2.0).map_err(Error::Numr)?;
                let scaled = client.mul_scalar(&clamped, stdev).map_err(Error::Numr)?;
                if mean != 0.0 {
                    client.add_scalar(&scaled, mean).map_err(Error::Numr)
                } else {
                    Ok(scaled)
                }
            }
        }
    }

    /// Create a tensor initialized according to this strategy, with a
    /// deterministic seed for every random draw.
    ///
    /// Mirrors `init_tensor` exactly: same argument order, same per-variant
    /// math, same number and order of random draws. The only difference is
    /// that every draw is routed through `rand_seeded`/`randn_seeded` instead
    /// of `rand`/`randn`, so the same seed always reproduces the same tensor.
    /// `Zeros`, `Ones`, and `Const` are already deterministic and behave
    /// identically to `init_tensor`.
    ///
    /// # Arguments
    /// * `shape` - Shape of the tensor to create
    /// * `dtype` - Data type
    /// * `device` - Device to create on
    /// * `client` - Runtime client (needed for random ops)
    /// * `seed` - Deterministic seed for the PRNG
    pub fn init_tensor_seeded<R, C>(
        &self,
        shape: &[usize],
        dtype: DType,
        device: &R::Device,
        client: &C,
        seed: u64,
    ) -> Result<Tensor<R>>
    where
        R: Runtime<DType = DType>,
        C: numr::runtime::RuntimeClient<R>
            + numr::ops::RandomOps<R>
            + numr::ops::ScalarOps<R>
            + numr::ops::BinaryOps<R>
            + numr::ops::CompareOps<R>
            + numr::ops::TensorOps<R>,
    {
        match *self {
            Init::Zeros => Ok(Tensor::zeros(shape, dtype, device)?),
            Init::Ones => Ok(Tensor::ones(shape, dtype, device)?),
            Init::Const(val) => {
                let t = Tensor::zeros(shape, dtype, device)?;
                client.add_scalar(&t, val as f64).map_err(Error::Numr)
            }
            Init::Uniform(bound) => {
                // U(-bound, bound) = rand() * 2*bound - bound
                let r = client
                    .rand_seeded(shape, dtype, seed)
                    .map_err(Error::Numr)?;
                let scaled = client
                    .mul_scalar(&r, 2.0 * bound as f64)
                    .map_err(Error::Numr)?;
                client
                    .add_scalar(&scaled, -(bound as f64))
                    .map_err(Error::Numr)
            }
            Init::PyTorchLinear => {
                // Kaiming uniform: U(-1/sqrt(fan_in), 1/sqrt(fan_in))
                let fan_in = pytorch_fan_in(shape);
                let bound = 1.0 / (fan_in as f64).sqrt();
                let r = client
                    .rand_seeded(shape, dtype, seed)
                    .map_err(Error::Numr)?;
                let scaled = client.mul_scalar(&r, 2.0 * bound).map_err(Error::Numr)?;
                client.add_scalar(&scaled, -bound).map_err(Error::Numr)
            }
            Init::PyTorchEmbedding => {
                // PyTorch `nn.Embedding` default initializes weights with N(0, 1).
                client.randn_seeded(shape, dtype, seed).map_err(Error::Numr)
            }
            Init::Kaiming => {
                // Kaiming/He normal: N(0, sqrt(2 / fan_in))
                let fan_in = if shape.len() >= 2 {
                    shape[..shape.len() - 1].iter().product::<usize>()
                } else {
                    shape[0]
                };
                let std = (2.0 / fan_in as f64).sqrt();
                let r = client
                    .randn_seeded(shape, dtype, seed)
                    .map_err(Error::Numr)?;
                client.mul_scalar(&r, std).map_err(Error::Numr)
            }
            Init::Xavier => {
                // Xavier/Glorot normal: N(0, sqrt(2 / (fan_in + fan_out)))
                let (fan_in, fan_out) = if shape.len() >= 2 {
                    let fi = shape[..shape.len() - 1].iter().product::<usize>();
                    let fo = shape[shape.len() - 1];
                    (fi, fo)
                } else {
                    (shape[0], shape[0])
                };
                let std = (2.0 / (fan_in + fan_out) as f64).sqrt();
                let r = client
                    .randn_seeded(shape, dtype, seed)
                    .map_err(Error::Numr)?;
                client.mul_scalar(&r, std).map_err(Error::Numr)
            }
            Init::Randn { mean, stdev } => {
                let r = client
                    .randn_seeded(shape, dtype, seed)
                    .map_err(Error::Numr)?;
                let scaled = client.mul_scalar(&r, stdev).map_err(Error::Numr)?;
                if mean != 0.0 {
                    client.add_scalar(&scaled, mean).map_err(Error::Numr)
                } else {
                    Ok(scaled)
                }
            }
            Init::TruncatedNormal { mean, stdev } => {
                // Generate N(0, 1), clamp to [-2, 2], then scale by stdev and shift by mean.
                // This clamps rather than rejection-samples, so there is no resample
                // loop to worry about falling back to an unseeded draw: exactly one
                // seeded `randn_seeded` call feeds the whole tensor, same as the
                // unseeded variant's single `randn` call.
                let r = client
                    .randn_seeded(shape, dtype, seed)
                    .map_err(Error::Numr)?;
                let clamped = client.clamp(&r, -2.0, 2.0).map_err(Error::Numr)?;
                let scaled = client.mul_scalar(&clamped, stdev).map_err(Error::Numr)?;
                if mean != 0.0 {
                    client.add_scalar(&scaled, mean).map_err(Error::Numr)
                } else {
                    Ok(scaled)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
