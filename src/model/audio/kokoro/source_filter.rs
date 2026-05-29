//! Harmonic-plus-noise source-filter excitation for the ISTFTNet generator.
//!
//! Upstream `decoder.generator.m_source` = `SourceModuleHnNSF`:
//!
//! ```text
//! m_source
//! ├── l_sin_gen: SineGen(sampling_rate, harmonic_num=8)   # no parameters
//! ├── l_linear:  Linear(harmonic_num + 1 = 9, 1)           # fc.weight, fc.bias
//! └── l_tanh:    nn.Tanh                                    # no parameters
//! ```
//!
//! Flow per f0 sample:
//!
//! 1. For each harmonic `h ∈ {1, 2, …, harmonic_num+1}`, generate a sine wave
//!    at `h · f0 / sample_rate` cycles per sample. Phase is the cumulative
//!    sum of per-sample angular increments, wrapped to `[0, 1)` to stay
//!    numerically stable over long sequences.
//! 2. Stack harmonics along a new last axis → `[B, T, harmonic_num+1]`.
//! 3. Voiced/unvoiced mask `uv = (f0 > threshold)`; unvoiced regions get
//!    small Gaussian noise instead of harmonic content.
//! 4. `l_linear` + `tanh` collapse the `harmonic_num+1` channels to a single
//!    time-domain excitation signal.
//!
//! This implementation is **deterministic**: the random initial phase and
//! Gaussian noise in the upstream reference are skipped. That matches how
//! production TTS systems typically run inference — reproducibility matters
//! more than phase randomness for human listeners. If subjective quality
//! differs vs upstream, the noise path can be added back with a provided RNG.

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, MatmulOps, ScalarOps, ShapeOps, TensorOps, TypeConversionOps, UnaryOps,
    UtilityOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Harmonic-aware sine generator driven by an f0 contour.
#[derive(Debug, Clone, Copy)]
pub struct SineGen {
    pub sample_rate: f32,
    pub harmonic_num: usize,
    pub sine_amp: f32,
    pub voiced_threshold: f32,
}

impl SineGen {
    pub fn new(sample_rate: f32, harmonic_num: usize) -> Self {
        Self {
            sample_rate,
            harmonic_num,
            sine_amp: 0.1,
            voiced_threshold: 0.0,
        }
    }

    /// Generate `[B, T, harmonic_num + 1]` sine harmonics from `f0 [B, T, 1]`.
    ///
    /// `f0` is in Hz; entries `≤ voiced_threshold` are unvoiced and produce
    /// zero output (pre-`tanh` mixing).
    pub fn forward<R, C>(&self, client: &C, f0: &Tensor<R>) -> Result<Tensor<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R>
            + UnaryOps<R>
            + BinaryOps<R>
            + ScalarOps<R>
            + TensorOps<R>
            + ShapeOps<R>
            + CompareOps<R>
            + TypeConversionOps<R>
            + numr::ops::ReduceOps<R>
            + UtilityOps<R>,
    {
        let shape = f0.shape();
        if shape.len() != 3 || shape[2] != 1 {
            return Err(Error::InvalidArgument {
                arg: "f0",
                reason: format!("expected [B, T, 1], got {shape:?}"),
            });
        }
        let (b, t) = (shape[0], shape[1]);
        let h = self.harmonic_num + 1;

        // Build the [B, T, h] frequency buffer by multiplying f0 by each
        // harmonic index. Start with an empty output, fill harmonic by harmonic.
        let f0_flat = f0.reshape(&[b, t, 1]).map_err(Error::Numr)?;
        let mut harmonics: Vec<Tensor<R>> = Vec::with_capacity(h);
        for i in 0..h {
            // harmonic index is `i + 1` (first entry = fundamental).
            let scale = (i as f64 + 1.0) / self.sample_rate as f64;
            // rad_values[:, :, i] = f0 * (i+1) / sr — cycles per sample.
            let scaled = client.mul_scalar(&f0_flat, scale).map_err(Error::Numr)?;
            harmonics.push(scaled);
        }
        let rad_values = client
            .cat(&harmonics.iter().collect::<Vec<_>>(), 2)
            .map_err(Error::Numr)?; // [B, T, h]

        // Cumulative sum along time axis → per-sample phase in cycles.
        // Wrap to [0, 1) via `x - floor(x)` to avoid precision loss.
        let phase = client.cumsum(&rad_values, 1).map_err(Error::Numr)?;
        let phase_floor = client.floor(&phase).map_err(Error::Numr)?;
        let phase_frac = client.sub(&phase, &phase_floor).map_err(Error::Numr)?;

        // sin(2π · phase_frac)
        let two_pi_phase = client
            .mul_scalar(&phase_frac, 2.0 * std::f64::consts::PI)
            .map_err(Error::Numr)?;
        let sine = client.sin(&two_pi_phase).map_err(Error::Numr)?;

        // Voiced/unvoiced mask from the fundamental. (f0 > threshold) as f32.
        let threshold_tensor = client
            .fill(&[b, t, 1], self.voiced_threshold as f64, f0.dtype())
            .map_err(Error::Numr)?;
        let uv = client
            .gt(&f0_flat, &threshold_tensor)
            .map_err(Error::Numr)?;
        let uv_f = client.cast(&uv, f0.dtype()).map_err(Error::Numr)?;

        // Amplitude: sine_amp where voiced, else 0.
        let amp = client
            .mul_scalar(&uv_f, self.sine_amp as f64)
            .map_err(Error::Numr)?;
        // Broadcast amp [B, T, 1] × sine [B, T, h]
        client.mul(&sine, &amp).map_err(Error::Numr)
    }
}

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

    /// Forward: f0 contour `[B, T, 1]` → excitation `[B, T, 1]`.
    pub fn forward<C>(&self, client: &C, f0: &Tensor<R>) -> Result<Tensor<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R>
            + UnaryOps<R>
            + BinaryOps<R>
            + ScalarOps<R>
            + MatmulOps<R>
            + TensorOps<R>
            + CompareOps<R>
            + numr::ops::ReduceOps<R>
            + UtilityOps<R>,
    {
        let sines = self.sine_gen.forward(client, f0)?; // [B, T, h]
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
        Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; n], shape, device)
    }

    #[test]
    fn sine_gen_output_shape_has_harmonic_plus_one_channels() {
        let (client, device) = cpu_setup();
        let sg = SineGen::new(24_000.0, 8);
        let f0 = Tensor::<CpuRuntime>::from_slice(
            &[100.0f32, 100.0, 100.0, 100.0, 100.0],
            &[1, 5, 1],
            &device,
        );
        let sines = sg.forward(&client, &f0).unwrap();
        assert_eq!(sines.shape(), &[1, 5, 9]); // 8 harmonics + fundamental
    }

    #[test]
    fn unvoiced_f0_yields_zero_sines() {
        // f0 ≤ threshold (0.0 by default) → uv=0 → sines multiplied by 0.
        let (client, device) = cpu_setup();
        let sg = SineGen::new(24_000.0, 2);
        let f0 = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 4], &[1, 4, 1], &device);
        let sines = sg.forward(&client, &f0).unwrap();
        for v in sines.to_vec::<f32>() {
            assert!(v.abs() < 1e-5, "expected zero, got {v}");
        }
    }

    #[test]
    fn sine_gen_rejects_wrong_f0_rank() {
        let (client, device) = cpu_setup();
        let sg = SineGen::new(24_000.0, 2);
        let f0 = zeros(&[1, 4], &device);
        assert!(sg.forward(&client, &f0).is_err());
    }

    #[test]
    fn source_module_output_shape_is_b_t_1() {
        let (client, device) = cpu_setup();
        let sg = SineGen::new(24_000.0, 4);
        let module =
            SourceModuleHnNSF::new(sg, zeros(&[1, 5], &device), zeros(&[1], &device)).unwrap();
        let f0 = Tensor::<CpuRuntime>::from_slice(&[200.0f32, 200.0, 200.0], &[1, 3, 1], &device);
        let out = module.forward(&client, &f0).unwrap();
        assert_eq!(out.shape(), &[1, 3, 1]);
    }

    #[test]
    fn source_module_zero_weight_gives_zero_output() {
        // With weight=0, bias=0, Linear output is 0, tanh(0)=0.
        let (client, device) = cpu_setup();
        let sg = SineGen::new(24_000.0, 2);
        let module =
            SourceModuleHnNSF::new(sg, zeros(&[1, 3], &device), zeros(&[1], &device)).unwrap();
        let f0 = Tensor::<CpuRuntime>::from_slice(&[200.0f32; 4], &[1, 4, 1], &device);
        let out = module.forward(&client, &f0).unwrap();
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
