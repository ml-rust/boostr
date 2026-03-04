//! Initialization strategies for new tensors.

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

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
    /// Kaiming uniform (PyTorch Linear default): U(-1/sqrt(in), 1/sqrt(in))
    PyTorchLinear,
    /// PyTorch Embedding default: N(0, 1) approximated as U(-1, 1)
    PyTorchEmbedding,
    /// Kaiming (He) normal: N(0, sqrt(2 / fan_in))
    ///
    /// Standard initialization for ReLU networks. fan_in is the product of
    /// all dimensions except the last (output) dimension.
    Kaiming,
    /// Xavier (Glorot) normal: N(0, sqrt(2 / (fan_in + fan_out)))
    ///
    /// Standard initialization for Sigmoid/Tanh networks. Used in some
    /// attention weight initializations.
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
            Init::Zeros => Ok(Tensor::zeros(shape, dtype, device)),
            Init::Ones => Ok(Tensor::ones(shape, dtype, device)),
            Init::Const(val) => {
                let t = Tensor::zeros(shape, dtype, device);
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
                let fan_in = shape[0];
                let bound = 1.0 / (fan_in as f64).sqrt();
                let r = client.rand(shape, dtype).map_err(Error::Numr)?;
                let scaled = client.mul_scalar(&r, 2.0 * bound).map_err(Error::Numr)?;
                client.add_scalar(&scaled, -bound).map_err(Error::Numr)
            }
            Init::PyTorchEmbedding => {
                // N(0, 1) approximated as U(-1, 1)
                let r = client.rand(shape, dtype).map_err(Error::Numr)?;
                let scaled = client.mul_scalar(&r, 2.0).map_err(Error::Numr)?;
                client.add_scalar(&scaled, -1.0).map_err(Error::Numr)
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    fn device() -> CpuDevice {
        CpuDevice::new()
    }

    fn client() -> numr::runtime::cpu::CpuClient {
        let d = device();
        CpuRuntime::default_client(&d)
    }

    #[test]
    fn test_init_zeros() {
        let d = device();
        let c = client();
        let t = Init::Zeros
            .init_tensor(&[2, 3], DType::F32, &d, &c)
            .unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        let data: Vec<f32> = t.to_vec();
        assert!(data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_init_kaiming() {
        let d = device();
        let c = client();
        // [out=64, in=128] → fan_in=128, std=sqrt(2/128)≈0.125
        let t = Init::Kaiming
            .init_tensor(&[64, 128], DType::F32, &d, &c)
            .unwrap();
        assert_eq!(t.shape(), &[64, 128]);
        let data: Vec<f32> = t.to_vec();
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        // Mean should be close to 0
        assert!(mean.abs() < 0.1, "Kaiming mean too large: {mean}");
        // Std should be close to sqrt(2/128) ≈ 0.125
        let var: f32 = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
        let std = var.sqrt();
        // fan_in = product of all dims except last = 64
        let expected_std = (2.0f32 / 64.0).sqrt();
        assert!(
            (std - expected_std).abs() < 0.05,
            "Kaiming std {std} vs expected {expected_std}"
        );
    }

    #[test]
    fn test_init_xavier() {
        let d = device();
        let c = client();
        // [256, 512] → fan_in=256, fan_out=512, std=sqrt(2/768)≈0.051
        let t = Init::Xavier
            .init_tensor(&[256, 512], DType::F32, &d, &c)
            .unwrap();
        assert_eq!(t.shape(), &[256, 512]);
        let data: Vec<f32> = t.to_vec();
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        assert!(mean.abs() < 0.05, "Xavier mean too large: {mean}");
    }

    #[test]
    fn test_init_randn() {
        let d = device();
        let c = client();
        let t = Init::Randn {
            mean: 5.0,
            stdev: 0.1,
        }
        .init_tensor(&[1000], DType::F32, &d, &c)
        .unwrap();
        let data: Vec<f32> = t.to_vec();
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        assert!((mean - 5.0).abs() < 0.1, "Randn mean {mean} should be ~5.0");
    }

    #[test]
    fn test_init_truncated_normal() {
        let d = device();
        let c = client();
        let t = Init::TruncatedNormal {
            mean: 0.0,
            stdev: 0.02,
        }
        .init_tensor(&[10000], DType::F32, &d, &c)
        .unwrap();
        let data: Vec<f32> = t.to_vec();
        // All values should be within [-0.04, 0.04] (2*stdev)
        for &v in &data {
            assert!(
                (-0.04..=0.04).contains(&v),
                "Truncated normal value {v} out of range [-0.04, 0.04]"
            );
        }
    }
}
