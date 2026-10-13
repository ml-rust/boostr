//! [`SourceModuleHnNSF`]: [`SineGen`] → Linear → tanh.
//!
//! Flow per f0 sample:
//!
//! 1. For each harmonic `h ∈ {1, 2, …, harmonic_num+1}`, generate a sine wave
//!    at `h · f0 / sample_rate` cycles per sample, as upstream's `_f02sine`
//!    does: the per-sample increments are linearly downsampled by the
//!    upsample scale, summed at that rate, scaled back up, and linearly
//!    interpolated to the sample rate.
//! 2. Stack harmonics along a new last axis → `[B, T, harmonic_num+1]`.
//! 3. Voiced/unvoiced mask `uv = (f0 > threshold)`; voiced frames carry the
//!    harmonics plus a small amount of Gaussian noise, while unvoiced frames
//!    are driven entirely by Gaussian noise (`sine_amp / 3`). Each harmonic
//!    also receives a random initial phase.
//! 4. `l_linear` + `tanh` collapse the `harmonic_num+1` channels to a single
//!    time-domain excitation signal.
//!
//! The random initial phase and additive noise match the reference NSF
//! implementation. A seed makes both draws reproducible on one backend. Without
//! one, they come from the runtime's unseeded RNG.

use super::SineGen;
use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, MatmulOps, ScalarOps, ShapeOps, TensorOps, TypeConversionOps, UnaryOps,
    UtilityOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// `SourceModuleHnNSF` — SineGen → Linear → tanh.
pub struct SourceModuleHnNSF<R: Runtime> {
    sine_gen: SineGen,
    /// Linear weight `[1, harmonic_num + 1]`.
    weight: Tensor<R>,
    /// Linear bias `[1]`.
    bias: Tensor<R>,
}

impl<R: Runtime> SourceModuleHnNSF<R> {
    pub fn new(sine_gen: SineGen, weight: Tensor<R>, bias: Tensor<R>) -> Result<Self> {
        let expected_in = sine_gen.harmonic_num + 1;
        if weight.shape() != [1, expected_in] {
            return Err(Error::InvalidArgument {
                arg: "weight",
                reason: format!("expected [1, {expected_in}], got {:?}", weight.shape()),
            });
        }
        if bias.shape() != [1] {
            return Err(Error::InvalidArgument {
                arg: "bias",
                reason: format!("expected [1], got {:?}", bias.shape()),
            });
        }
        Ok(Self {
            sine_gen,
            weight,
            bias,
        })
    }

    /// Forward: f0 contour `[B, T, 1]` → excitation `[B, T, 1]`. `seed`
    /// reads as [`SineGen::forward`] reads it.
    pub fn forward<C>(&self, client: &C, f0: &Tensor<R>, seed: Option<u64>) -> Result<Tensor<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R>
            + UnaryOps<R>
            + BinaryOps<R>
            + ScalarOps<R>
            + MatmulOps<R>
            + TensorOps<R>
            + ShapeOps<R>
            + CompareOps<R>
            + TypeConversionOps<R>
            + numr::ops::RandomOps<R>
            + numr::ops::ReduceOps<R>
            + numr::ops::IndexingOps<R>
            + UtilityOps<R>,
    {
        let sines = self.sine_gen.forward(client, f0, seed)?; // [B, T, h]
        let shape = sines.shape();
        let (b, t, h) = (shape[0], shape[1], shape[2]);
        // Flatten to [B*T, h], linear, reshape to [B, T, 1], tanh.
        let flat = sines.reshape(&[b * t, h]).map_err(Error::Numr)?;
        let w_t = self.weight.transpose(0, 1).map_err(Error::Numr)?;
        let projected = client
            .matmul_bias(&flat, &w_t, &self.bias)
            .map_err(Error::Numr)?;
        let reshaped = projected.reshape(&[b, t, 1]).map_err(Error::Numr)?;
        client.tanh(&reshaped).map_err(Error::Numr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    fn zeros(shape: &[usize], device: &<CpuRuntime as Runtime>::Device) -> Tensor<CpuRuntime> {
        let n: usize = shape.iter().product();
        Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; n], shape, device).unwrap()
    }

    #[test]
    fn source_module_output_shape_is_b_t_1() {
        let (client, device) = cpu_setup();
        let sg = SineGen::new(24_000.0, 4);
        let module =
            SourceModuleHnNSF::new(sg, zeros(&[1, 5], &device), zeros(&[1], &device)).unwrap();
        let f0 = Tensor::<CpuRuntime>::from_slice(&[200.0f32, 200.0, 200.0], &[1, 3, 1], &device)
            .unwrap();
        let out = module.forward(&client, &f0, None).unwrap();
        assert_eq!(out.shape(), &[1, 3, 1]);
    }

    #[test]
    fn source_module_zero_weight_gives_zero_output() {
        // With weight=0, bias=0, Linear output is 0, tanh(0)=0.
        let (client, device) = cpu_setup();
        let sg = SineGen::new(24_000.0, 2);
        let module =
            SourceModuleHnNSF::new(sg, zeros(&[1, 3], &device), zeros(&[1], &device)).unwrap();
        let f0 = Tensor::<CpuRuntime>::from_slice(&[200.0f32; 4], &[1, 4, 1], &device).unwrap();
        let out = module.forward(&client, &f0, None).unwrap();
        for v in out.to_vec::<f32>() {
            assert!(v.abs() < 1e-5, "got {v}");
        }
    }

    #[test]
    fn source_module_rejects_wrong_weight_shape() {
        let (_client, device) = cpu_setup();
        let sg = SineGen::new(24_000.0, 2);
        let bad = SourceModuleHnNSF::new(sg, zeros(&[1, 99], &device), zeros(&[1], &device));
        assert!(bad.is_err());
    }
}
