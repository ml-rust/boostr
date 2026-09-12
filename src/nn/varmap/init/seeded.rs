//! `Init::init_tensor_seeded`: deterministic initialization.

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::Init;
use super::strategy::{pytorch_fan_in, xavier_fans};

impl Init {
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
                // Kaiming/He normal: N(0, sqrt(2 / fan_in)). Same
                // `pytorch_fan_in` convention as the unseeded path.
                let fan_in = pytorch_fan_in(shape);
                let std = (2.0 / fan_in as f64).sqrt();
                let r = client
                    .randn_seeded(shape, dtype, seed)
                    .map_err(Error::Numr)?;
                client.mul_scalar(&r, std).map_err(Error::Numr)
            }
            Init::Xavier => {
                // Xavier/Glorot normal: N(0, sqrt(2 / (fan_in + fan_out))).
                // Same `xavier_fans` split as the unseeded path.
                let (fan_in, fan_out) = xavier_fans(shape);
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

    // ===== Seeded-path tests =====
    //
    // Shape is [4096] (16384 bytes of f32) for every random variant below: at
    // that size, two independent unseeded draws colliding bit-for-bit by chance
    // is not a real possibility, so an equality assertion there is a genuine
    // discriminator, not a coincidence.
    const BIG: &[usize] = &[4096];

    #[test]
    fn test_seeded_uniform_same_seed_bit_identical() {
        let d = device();
        let c = client();
        let a = Init::Uniform(0.5)
            .init_tensor_seeded(BIG, DType::F32, &d, &c, 42)
            .unwrap();
        let b = Init::Uniform(0.5)
            .init_tensor_seeded(BIG, DType::F32, &d, &c, 42)
            .unwrap();
        assert_eq!(a.to_vec::<f32>(), b.to_vec::<f32>());
    }

    #[test]
    fn test_seeded_uniform_different_seed_differs() {
        let d = device();
        let c = client();
        let a = Init::Uniform(0.5)
            .init_tensor_seeded(BIG, DType::F32, &d, &c, 1)
            .unwrap();
        let b = Init::Uniform(0.5)
            .init_tensor_seeded(BIG, DType::F32, &d, &c, 2)
            .unwrap();
        assert_ne!(a.to_vec::<f32>(), b.to_vec::<f32>());
    }

    #[test]
    fn test_seeded_pytorch_linear_same_seed_bit_identical() {
        let d = device();
        let c = client();
        let shape = &[128, 64];
        let a = Init::PyTorchLinear
            .init_tensor_seeded(shape, DType::F32, &d, &c, 7)
            .unwrap();
        let b = Init::PyTorchLinear
            .init_tensor_seeded(shape, DType::F32, &d, &c, 7)
            .unwrap();
        assert_eq!(a.to_vec::<f32>(), b.to_vec::<f32>());
    }

    #[test]
    fn test_seeded_pytorch_linear_different_seed_differs() {
        let d = device();
        let c = client();
        let shape = &[128, 64];
        let a = Init::PyTorchLinear
            .init_tensor_seeded(shape, DType::F32, &d, &c, 7)
            .unwrap();
        let b = Init::PyTorchLinear
            .init_tensor_seeded(shape, DType::F32, &d, &c, 8)
            .unwrap();
        assert_ne!(a.to_vec::<f32>(), b.to_vec::<f32>());
    }

    #[test]
    fn test_seeded_pytorch_embedding_same_seed_bit_identical() {
        let d = device();
        let c = client();
        let a = Init::PyTorchEmbedding
            .init_tensor_seeded(BIG, DType::F32, &d, &c, 3)
            .unwrap();
        let b = Init::PyTorchEmbedding
            .init_tensor_seeded(BIG, DType::F32, &d, &c, 3)
            .unwrap();
        assert_eq!(a.to_vec::<f32>(), b.to_vec::<f32>());
    }

    #[test]
    fn test_seeded_pytorch_embedding_different_seed_differs() {
        let d = device();
        let c = client();
        let a = Init::PyTorchEmbedding
            .init_tensor_seeded(BIG, DType::F32, &d, &c, 3)
            .unwrap();
        let b = Init::PyTorchEmbedding
            .init_tensor_seeded(BIG, DType::F32, &d, &c, 4)
            .unwrap();
        assert_ne!(a.to_vec::<f32>(), b.to_vec::<f32>());
    }

    #[test]
    fn test_seeded_kaiming_same_seed_bit_identical() {
        let d = device();
        let c = client();
        let shape = &[64, 128];
        let a = Init::Kaiming
            .init_tensor_seeded(shape, DType::F32, &d, &c, 11)
            .unwrap();
        let b = Init::Kaiming
            .init_tensor_seeded(shape, DType::F32, &d, &c, 11)
            .unwrap();
        assert_eq!(a.to_vec::<f32>(), b.to_vec::<f32>());
    }

    #[test]
    fn test_seeded_kaiming_different_seed_differs() {
        let d = device();
        let c = client();
        let shape = &[64, 128];
        let a = Init::Kaiming
            .init_tensor_seeded(shape, DType::F32, &d, &c, 11)
            .unwrap();
        let b = Init::Kaiming
            .init_tensor_seeded(shape, DType::F32, &d, &c, 12)
            .unwrap();
        assert_ne!(a.to_vec::<f32>(), b.to_vec::<f32>());
    }

    #[test]
    fn test_seeded_xavier_same_seed_bit_identical() {
        let d = device();
        let c = client();
        let shape = &[256, 512];
        let a = Init::Xavier
            .init_tensor_seeded(shape, DType::F32, &d, &c, 13)
            .unwrap();
        let b = Init::Xavier
            .init_tensor_seeded(shape, DType::F32, &d, &c, 13)
            .unwrap();
        assert_eq!(a.to_vec::<f32>(), b.to_vec::<f32>());
    }

    #[test]
    fn test_seeded_xavier_different_seed_differs() {
        let d = device();
        let c = client();
        let shape = &[256, 512];
        let a = Init::Xavier
            .init_tensor_seeded(shape, DType::F32, &d, &c, 13)
            .unwrap();
        let b = Init::Xavier
            .init_tensor_seeded(shape, DType::F32, &d, &c, 14)
            .unwrap();
        assert_ne!(a.to_vec::<f32>(), b.to_vec::<f32>());
    }

    #[test]
    fn test_seeded_randn_same_seed_bit_identical() {
        let d = device();
        let c = client();
        let init = Init::Randn {
            mean: 5.0,
            stdev: 0.1,
        };
        let a = init
            .init_tensor_seeded(BIG, DType::F32, &d, &c, 21)
            .unwrap();
        let b = init
            .init_tensor_seeded(BIG, DType::F32, &d, &c, 21)
            .unwrap();
        assert_eq!(a.to_vec::<f32>(), b.to_vec::<f32>());
    }

    #[test]
    fn test_seeded_randn_different_seed_differs() {
        let d = device();
        let c = client();
        let init = Init::Randn {
            mean: 5.0,
            stdev: 0.1,
        };
        let a = init
            .init_tensor_seeded(BIG, DType::F32, &d, &c, 21)
            .unwrap();
        let b = init
            .init_tensor_seeded(BIG, DType::F32, &d, &c, 22)
            .unwrap();
        assert_ne!(a.to_vec::<f32>(), b.to_vec::<f32>());
    }

    #[test]
    fn test_seeded_truncated_normal_same_seed_bit_identical() {
        let d = device();
        let c = client();
        let init = Init::TruncatedNormal {
            mean: 0.0,
            stdev: 0.02,
        };
        let a = init
            .init_tensor_seeded(BIG, DType::F32, &d, &c, 31)
            .unwrap();
        let b = init
            .init_tensor_seeded(BIG, DType::F32, &d, &c, 31)
            .unwrap();
        assert_eq!(a.to_vec::<f32>(), b.to_vec::<f32>());
    }

    #[test]
    fn test_seeded_truncated_normal_different_seed_differs() {
        let d = device();
        let c = client();
        let init = Init::TruncatedNormal {
            mean: 0.0,
            stdev: 0.02,
        };
        let a = init
            .init_tensor_seeded(BIG, DType::F32, &d, &c, 31)
            .unwrap();
        let b = init
            .init_tensor_seeded(BIG, DType::F32, &d, &c, 32)
            .unwrap();
        assert_ne!(a.to_vec::<f32>(), b.to_vec::<f32>());
    }

    /// The largest absolute value in a uniform draw of `n` samples over
    /// `[-b, b]` lands within `b * (1 - 2/n)` of `b` in expectation, so at
    /// 8192 samples the observed maximum pins `b` to well under a percent.
    fn max_abs(data: &[f32]) -> f32 {
        data.iter().fold(0.0f32, |m, v| m.max(v.abs()))
    }

    /// The seeded twin must agree with the unseeded path on the distribution.
    /// They draw different numbers, so this compares the bound each respects,
    /// not the values. If only one path is fixed the bounds differ by 2x and
    /// this fails.
    #[test]
    fn test_pytorch_linear_seeded_and_unseeded_share_a_distribution() {
        let d = device();
        let c = client();

        let shape = &[128, 32];
        let correct_bound = 1.0f32 / 32.0f32.sqrt();

        let unseeded = Init::PyTorchLinear
            .init_tensor(shape, DType::F32, &d, &c)
            .unwrap();
        let seeded = Init::PyTorchLinear
            .init_tensor_seeded(shape, DType::F32, &d, &c, 4242)
            .unwrap();

        let m_unseeded = max_abs(&unseeded.to_vec::<f32>());
        let m_seeded = max_abs(&seeded.to_vec::<f32>());

        for (label, observed) in [("unseeded", m_unseeded), ("seeded", m_seeded)] {
            assert!(
                observed <= correct_bound && observed > 0.95 * correct_bound,
                "{label} max {observed} does not match bound {correct_bound}"
            );
        }
    }

    fn sample_sd(v: &[f32]) -> f64 {
        let n = v.len() as f64;
        let mean = v.iter().map(|&x| x as f64).sum::<f64>() / n;
        (v.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n).sqrt()
    }

    /// The seeded Kaiming/Xavier arms must use the same `pytorch_fan_in` /
    /// `xavier_fans` convention as the unseeded path: the leading dim is the
    /// output side. They previously read `fan_in` off the OPPOSITE end of the
    /// shape (`shape[..len-1].product()`), so a seeded `[out, in]` Kaiming init
    /// had its std scaled by `sqrt(in/out)`.
    ///
    /// Kaiming uses a 2-D `[256, 64]` (4x aspect: correct `sqrt(2/64)`, old
    /// `sqrt(2/256)`, 2x apart). Xavier is symmetric in `fan_in + fan_out` on
    /// 2-D, so it uses `[4, 8, 16]`: correct `fan_in = 8*16 = 128, fan_out = 4`,
    /// old split `fan_in = 4*8 = 32, fan_out = 16`.
    #[test]
    fn seeded_kaiming_and_xavier_use_the_fan_in_convention() {
        let d = device();
        let c = client();

        // --- Kaiming ---
        let shape = [256usize, 64];
        let expected = (2.0 / pytorch_fan_in(&shape) as f64).sqrt();
        let wrong = (2.0 / 256.0f64).sqrt();
        assert!((expected - (2.0f64 / 64.0).sqrt()).abs() < 1e-12);

        let seeded = Init::Kaiming
            .init_tensor_seeded(&shape, DType::F32, &d, &c, 77)
            .unwrap();
        let unseeded = Init::Kaiming
            .init_tensor(&shape, DType::F32, &d, &c)
            .unwrap();
        let sd_seeded = sample_sd(&seeded.to_vec::<f32>());
        let sd_unseeded = sample_sd(&unseeded.to_vec::<f32>());
        // 16384 draws: sample sd sits well inside 10% of the true sd, and the
        // two candidates differ by 2x.
        assert!(
            (sd_seeded - expected).abs() < 0.1 * expected,
            "seeded Kaiming sd {sd_seeded:.4} should be near {expected:.4}, not {wrong:.4}"
        );
        assert!(
            (sd_seeded - sd_unseeded).abs() < 0.1 * expected,
            "seeded Kaiming sd {sd_seeded:.4} should match unseeded {sd_unseeded:.4}"
        );

        // --- Xavier ---
        let shape = [4usize, 8, 16];
        let (fan_in, fan_out) = xavier_fans(&shape);
        assert_eq!((fan_in, fan_out), (128, 4));
        let expected = (2.0 / (fan_in + fan_out) as f64).sqrt();
        let swapped = (2.0f64 / (32.0 + 16.0)).sqrt();

        let seeded = Init::Xavier
            .init_tensor_seeded(&shape, DType::F32, &d, &c, 78)
            .unwrap();
        let unseeded = Init::Xavier
            .init_tensor(&shape, DType::F32, &d, &c)
            .unwrap();
        let sd_seeded = sample_sd(&seeded.to_vec::<f32>());
        let sd_unseeded = sample_sd(&unseeded.to_vec::<f32>());
        // 512 draws: sample sd sits inside 25% of the true sd; the candidates
        // differ by sqrt(132/48) ~= 1.66x.
        assert!(
            (sd_seeded - expected).abs() < 0.25 * expected,
            "seeded Xavier sd {sd_seeded:.4} should be near {expected:.4}, not the swapped {swapped:.4}"
        );
        assert!(
            (sd_seeded - sd_unseeded).abs() < 0.25 * expected,
            "seeded Xavier sd {sd_seeded:.4} should match unseeded {sd_unseeded:.4}"
        );
    }

    /// Deterministic variants ignore the seed entirely: seeded and unseeded
    /// paths must agree bit-for-bit.
    #[test]
    fn test_deterministic_variants_seeded_matches_unseeded() {
        let d = device();
        let c = client();
        for init in [Init::Zeros, Init::Ones, Init::Const(3.5)] {
            let unseeded = init.init_tensor(&[8, 8], DType::F32, &d, &c).unwrap();
            let seeded = init
                .init_tensor_seeded(&[8, 8], DType::F32, &d, &c, 999)
                .unwrap();
            assert_eq!(
                unseeded.to_vec::<f32>(),
                seeded.to_vec::<f32>(),
                "{init:?} must be identical whether seeded or not"
            );
        }
    }
}
