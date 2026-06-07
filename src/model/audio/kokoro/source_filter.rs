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
//! 3. Voiced/unvoiced mask `uv = (f0 > threshold)`; voiced frames carry the
//!    harmonics plus a small amount of Gaussian noise, while unvoiced frames
//!    are driven entirely by Gaussian noise (`sine_amp / 3`). Each harmonic
//!    also receives a random initial phase.
//! 4. `l_linear` + `tanh` collapse the `harmonic_num+1` channels to a single
//!    time-domain excitation signal.
//!
//! The random initial phase and additive noise use the runtime's RNG, matching
//! the upstream reference. Seed numr's global RNG before synthesis if you need
//! bit-reproducible output.

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
    /// Std-dev of the additive Gaussian noise (NSF default `0.003`). Voiced
    /// frames get this much noise on top of the harmonics; unvoiced frames are
    /// driven by `sine_amp / 3` noise instead of harmonics.
    pub noise_std: f32,
    pub voiced_threshold: f32,
}

impl SineGen {
    pub fn new(sample_rate: f32, harmonic_num: usize) -> Self {
        Self {
            sample_rate,
            harmonic_num,
            sine_amp: 0.1,
            noise_std: 0.003,
            voiced_threshold: 0.0,
        }
    }

    /// Generate `[B, T, harmonic_num + 1]` harmonic-plus-noise excitation from
    /// `f0 [B, T, 1]`, matching the upstream NSF `SineGen`.
    ///
    /// `f0` is in Hz. Voiced frames (`f0 > voiced_threshold`) emit the sine
    /// harmonics scaled by `sine_amp` plus `noise_std` Gaussian noise; unvoiced
    /// frames emit `sine_amp / 3` Gaussian noise only. Each harmonic also gets a
    /// random initial phase, as in the reference. The Gaussian/uniform draws use
    /// the runtime's RNG — seed it (numr global RNG) for reproducible synthesis.
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
            + numr::ops::RandomOps<R>
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
        let dtype = f0.dtype();

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

        // Random initial phase per (batch, harmonic), fundamental fixed at 0.
        // Upstream adds it to `rad_values[:, 0, :]` before the cumsum; since a
        // constant added to the first step propagates to every later step, this
        // is equivalent to adding the offset (broadcast over time) to the
        // cumulative phase.
        let rand_phase = client.rand(&[b, 1, h - 1], dtype).map_err(Error::Numr)?;
        let zero_col = client.fill(&[b, 1, 1], 0.0, dtype).map_err(Error::Numr)?;
        let rand_ini = client
            .cat(&[&zero_col, &rand_phase], 2)
            .map_err(Error::Numr)?; // [B, 1, h]

        // Cumulative sum along time axis → per-sample phase in cycles, plus the
        // random initial phase. Wrap to [0, 1) via `x - floor(x)` for precision.
        let phase = client.cumsum(&rad_values, 1).map_err(Error::Numr)?;
        let phase = client.add(&phase, &rand_ini).map_err(Error::Numr)?;
        let phase_floor = client.floor(&phase).map_err(Error::Numr)?;
        let phase_frac = client.sub(&phase, &phase_floor).map_err(Error::Numr)?;

        // sin(2π · phase_frac)
        let two_pi_phase = client
            .mul_scalar(&phase_frac, 2.0 * std::f64::consts::PI)
            .map_err(Error::Numr)?;
        let sine = client.sin(&two_pi_phase).map_err(Error::Numr)?;

        // Voiced/unvoiced mask from the fundamental. (f0 > threshold) as f32.
        let threshold_tensor = client
            .fill(&[b, t, 1], self.voiced_threshold as f64, dtype)
            .map_err(Error::Numr)?;
        let uv = client
            .gt(&f0_flat, &threshold_tensor)
            .map_err(Error::Numr)?;
        let uv_f = client.cast(&uv, dtype).map_err(Error::Numr)?;

        // Voiced harmonics: sine · sine_amp · uv  (silenced on unvoiced frames).
        let amp = client
            .mul_scalar(&uv_f, self.sine_amp as f64)
            .map_err(Error::Numr)?;
        let sine_waves = client.mul(&sine, &amp).map_err(Error::Numr)?;

        // Additive noise: noise_amp = uv·noise_std + (1 − uv)·(sine_amp / 3).
        // Voiced frames get a small amount of noise; unvoiced frames are noise-
        // driven (no harmonics). noise = noise_amp · N(0, 1).
        let one_minus_uv = {
            let neg = client.mul_scalar(&uv_f, -1.0).map_err(Error::Numr)?;
            client.add_scalar(&neg, 1.0).map_err(Error::Numr)?
        };
        let voiced_noise = client
            .mul_scalar(&uv_f, self.noise_std as f64)
            .map_err(Error::Numr)?;
        let unvoiced_noise = client
            .mul_scalar(&one_minus_uv, (self.sine_amp / 3.0) as f64)
            .map_err(Error::Numr)?;
        let noise_amp = client
            .add(&voiced_noise, &unvoiced_noise)
            .map_err(Error::Numr)?; // [B, T, 1]
        let gauss = client.randn(&[b, t, h], dtype).map_err(Error::Numr)?;
        let noise = client.mul(&gauss, &noise_amp).map_err(Error::Numr)?;

        client.add(&sine_waves, &noise).map_err(Error::Numr)
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
            + ShapeOps<R>
            + CompareOps<R>
            + TypeConversionOps<R>
            + numr::ops::RandomOps<R>
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
    fn unvoiced_f0_is_noise_driven() {
        // f0 ≤ threshold (0.0) → uv=0 → no harmonics, output is noise with
        // amplitude sine_amp/3 ≈ 0.0333 (matches upstream NSF). Verify the
        // noise path is active (non-zero) and its RMS sits near sine_amp/3.
        let (client, device) = cpu_setup();
        let sg = SineGen::new(24_000.0, 2);
        let t = 512;
        let f0 = Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; t], &[1, t, 1], &device);
        let out = sg.forward(&client, &f0).unwrap();
        let data: Vec<f32> = out.to_vec();
        assert!(data.iter().all(|v| v.is_finite()));
        assert!(
            data.iter().any(|&v| v != 0.0),
            "noise path produced all zeros"
        );
        let rms = (data.iter().map(|v| v * v).sum::<f32>() / data.len() as f32).sqrt();
        let expected = sg.sine_amp / 3.0; // ≈ 0.0333
        assert!(
            rms > expected * 0.4 && rms < expected * 2.0,
            "unvoiced noise RMS {rms} far from expected {expected}"
        );
    }

    #[test]
    fn voiced_f0_carries_harmonics() {
        // Voiced frames should have substantially more energy than the small
        // unvoiced noise floor (sine harmonics at amplitude sine_amp = 0.1).
        let (client, device) = cpu_setup();
        let sg = SineGen::new(24_000.0, 4);
        let t = 512;
        let f0 = Tensor::<CpuRuntime>::from_slice(&vec![220.0f32; t], &[1, t, 1], &device);
        let out = sg.forward(&client, &f0).unwrap();
        let data: Vec<f32> = out.to_vec();
        let rms = (data.iter().map(|v| v * v).sum::<f32>() / data.len() as f32).sqrt();
        // Harmonic RMS (~sine_amp/√2 ≈ 0.07) dwarfs the voiced noise_std=0.003.
        assert!(rms > 0.02, "voiced harmonic energy too low: rms={rms}");
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
