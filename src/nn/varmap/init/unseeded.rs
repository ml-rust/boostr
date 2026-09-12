//! `Init::init_tensor`: unseeded initialization.

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::Init;
use super::strategy::{pytorch_fan_in, xavier_fans};

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
                let fan_in = pytorch_fan_in(shape);
                let std = (2.0 / fan_in as f64).sqrt();
                let r = client.randn(shape, dtype).map_err(Error::Numr)?;
                client.mul_scalar(&r, std).map_err(Error::Numr)
            }
            Init::Xavier => {
                // Xavier/Glorot normal: N(0, sqrt(2 / (fan_in + fan_out))).
                // Leading dim is the output side, matching `pytorch_fan_in`.
                let (fan_in, fan_out) = xavier_fans(shape);
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
        // fan_in is in_features = 128, matching `pytorch_fan_in` and the
        // [out, in] layout every Linear here stores. This assertion previously read
        // 64 — the leading dim — which is fan_OUT, and contradicted the comment
        // three lines above it.
        let expected_std = (2.0f32 / 128.0).sqrt();
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

    // Shape is [4096] (16384 bytes of f32): at that size, two independent
    // unseeded draws colliding bit-for-bit by chance is not a real
    // possibility, so an equality assertion there is a genuine discriminator,
    // not a coincidence.
    const BIG: &[usize] = &[4096];

    /// The assertion that proves the seeded path does something the unseeded
    /// path does not: two unseeded draws of the same `Init` variant must NOT
    /// collide. `BIG` is large enough (4096 f32 values) that a coincidental
    /// bit-for-bit match is not a real possibility, so a failure here means the
    /// unseeded RNG stream is broken (e.g. accidentally reusing a fixed seed),
    /// not bad luck.
    #[test]
    fn test_unseeded_init_tensor_twice_differs() {
        let d = device();
        let c = client();
        let a = Init::Randn {
            mean: 0.0,
            stdev: 1.0,
        }
        .init_tensor(BIG, DType::F32, &d, &c)
        .unwrap();
        let b = Init::Randn {
            mean: 0.0,
            stdev: 1.0,
        }
        .init_tensor(BIG, DType::F32, &d, &c)
        .unwrap();
        assert_ne!(a.to_vec::<f32>(), b.to_vec::<f32>());
    }
}
