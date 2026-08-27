//! VoxCPM2's `fsq_layer`: a finite-scalar-quantization bottleneck between the
//! `base_lm` decoder and `feat_decoder`'s DiT, plus the `stop` classifier
//! chain that shares its input width.
//!
//! Reference: `ScalarQuantizationLayer.forward`, EVAL mode only. The
//! reference's training branch runs a straight-through estimator around
//! `torch.round`; boostr is inference-only, so that branch is dead code and
//! is deliberately NOT ported here.

use crate::error::{Error, Result};
use crate::nn::MaybeQuantLinear;
use crate::quant::traits::QuantMatmulOps;
use numr::autograd::{Var, var_div_scalar, var_mul_scalar, var_silu, var_tanh};
use numr::dtype::DType;
use numr::ops::{ActivationOps, BinaryOps, ScalarOps, TensorOps, TypeConversionOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};

/// `fsq_layer`: `out_proj(round_ties_even(tanh(in_proj(hidden)) * scale) /
/// scale)`.
///
/// `in_proj` narrows `lm_hidden -> latent_dim` (2048 -> 512), `out_proj`
/// widens back (512 -> 2048). The intermediate `tanh` output lives in
/// `(-1, 1)`; scaling by `scale` (9) and rounding to the nearest integer
/// before dividing back snaps each of the 512 channels onto one of 19 evenly
/// spaced levels, `k / 9` for `k` in `-9..=9`.
///
/// Both projections are [`MaybeQuantLinear`], not plain `Linear`: a GGUF
/// stores them block-quantized, and the quantized variant multiplies the
/// weight PACKED through `quant_matmul` instead of expanding it to dense F32
/// at load. A safetensors checkpoint yields the `Standard` variant and runs
/// exactly the dense path it always did.
pub struct ScalarQuantization<R: Runtime> {
    in_proj: MaybeQuantLinear<R>,
    out_proj: MaybeQuantLinear<R>,
    /// Rounding-grid divisor (`FsqConfig::scale`, 9 on the verified
    /// checkpoint).
    scale: f32,
}

impl<R: Runtime<DType = DType>> ScalarQuantization<R> {
    /// Wrap already-loaded `in_proj`/`out_proj` weights with `scale`. Use
    /// [`crate::model::audio::voxcpm::fsq::loader`] to build one from a
    /// checkpoint.
    pub fn new(in_proj: MaybeQuantLinear<R>, out_proj: MaybeQuantLinear<R>, scale: f32) -> Self {
        Self {
            in_proj,
            out_proj,
            scale,
        }
    }

    /// `fsq_layer.forward` (eval mode). Accepts either rank-3 `[b, T,
    /// hidden]` (batched sequence input) or rank-2 `[b, hidden]` (the
    /// per-step decode path). Any other rank is `Error::InvalidArgument`.
    ///
    /// # Ties-to-even rounding
    ///
    /// `torch.round` is IEEE round-half-to-EVEN, which disagrees with Rust's
    /// `f32::round`/numr's [`numr::tensor::Tensor::round`]
    /// (round-half-away-from-zero) at every exact tie — e.g. `-0.5 -> -0.0`
    /// under ties-even vs `-1.0` under ties-away. This uses
    /// [`numr::ops::UnaryOps::round_ties_even`] (exposed here as
    /// `Tensor::round_ties_even`) to match the reference. Do NOT swap this
    /// for `round`: the two only agree away from `.5` boundaries.
    pub fn forward<C>(&self, client: &C, hidden: &Var<R>) -> Result<Var<R>>
    where
        // `QuantMatmulOps` + `BinaryOps` + `TypeConversionOps` are what
        // `MaybeQuantLinear::forward` needs over a dense `Linear::forward`:
        // the packed multiply, its bias add, and the decomposed-quant arm's
        // cast of activations to F32.
        C: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + QuantMatmulOps<R>
            + BinaryOps<R>
            + TypeConversionOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + UnaryOps<R>,
    {
        match hidden.shape().len() {
            2 | 3 => {}
            rank => {
                return Err(Error::InvalidArgument {
                    arg: "hidden",
                    reason: format!(
                        "expected rank 2 or 3, got rank {rank} ({:?})",
                        hidden.shape()
                    ),
                });
            }
        }

        let projected = self.in_proj.forward(client, hidden)?;
        let squashed = var_tanh(&projected, client).map_err(Error::Numr)?;
        let scaled = var_mul_scalar(&squashed, self.scale as f64, client).map_err(Error::Numr)?;

        // Ties-to-even, NOT ties-away — see the doc comment above.
        let rounded_tensor = scaled.tensor().round_ties_even().map_err(Error::Numr)?;
        let rounded = Var::new(rounded_tensor, false);

        let levels = var_div_scalar(&rounded, self.scale as f64, client).map_err(Error::Numr)?;
        self.out_proj.forward(client, &levels)
    }
}

/// The six auxiliary projections around `fsq_layer` that a future
/// `VoxCpm2Model` orchestrator will own: encoder/DiT bridges and the stop
/// classifier. See [`crate::model::audio::voxcpm::fsq::loader`] for the
/// checkpoint key layout each field is loaded from.
///
/// All six are [`MaybeQuantLinear`] for the same reason
/// [`ScalarQuantization`]'s pair is: a GGUF stores them block-quantized and
/// they multiply PACKED, while a safetensors checkpoint yields the
/// `Standard` variant and the dense path is unchanged.
pub struct AuxProjections<R: Runtime> {
    pub enc_to_lm_proj: MaybeQuantLinear<R>,
    pub lm_to_dit_proj: MaybeQuantLinear<R>,
    pub res_to_dit_proj: MaybeQuantLinear<R>,
    pub fusion_concat_proj: MaybeQuantLinear<R>,
    pub stop_proj: MaybeQuantLinear<R>,
    /// Bias-free: the checkpoint carries no `stop_head.bias` tensor. See
    /// [`crate::model::audio::voxcpm::fsq::loader`] for how this is loaded.
    pub stop_head: MaybeQuantLinear<R>,
}

impl<R: Runtime<DType = DType>> AuxProjections<R> {
    /// `stop_head(silu(stop_proj(hidden)))`: the fixed composition the
    /// reference always runs together to produce stop-token logits.
    pub fn stop<C>(&self, client: &C, hidden: &Var<R>) -> Result<Var<R>>
    where
        // The extra three bounds over a dense `Linear::forward` — see
        // [`ScalarQuantization::forward`].
        C: RuntimeClient<R>
            + TensorOps<R>
            + ActivationOps<R>
            + ScalarOps<R>
            + QuantMatmulOps<R>
            + BinaryOps<R>
            + TypeConversionOps<R>,
        R::Client: TensorOps<R> + ActivationOps<R> + ScalarOps<R>,
    {
        let projected = self.stop_proj.forward(client, hidden)?;
        let activated = var_silu(&projected, client).map_err(Error::Numr)?;
        self.stop_head.forward(client, &activated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Weight;
    use crate::test_utils::cpu_setup;
    use numr::tensor::Tensor;

    /// Builds a `ScalarQuantization` whose `in_proj`/`out_proj` are both
    /// identity-shaped `[hidden, hidden]` so the quantized-level math is
    /// directly observable through the layer's public `forward`.
    fn identity_quantizer(
        hidden: usize,
        scale: f32,
        device: &numr::runtime::cpu::CpuDevice,
    ) -> ScalarQuantization<numr::runtime::cpu::CpuRuntime> {
        let identity: Vec<f32> = (0..hidden * hidden)
            .map(|i| if i / hidden == i % hidden { 1.0 } else { 0.0 })
            .collect();
        let zeros = vec![0.0f32; hidden];
        let in_proj = MaybeQuantLinear::from_weight(
            Weight::Standard(Tensor::from_slice(&identity, &[hidden, hidden], device).unwrap()),
            Some(Tensor::from_slice(&zeros, &[hidden], device).unwrap()),
        );
        let out_proj = MaybeQuantLinear::from_weight(
            Weight::Standard(Tensor::from_slice(&identity, &[hidden, hidden], device).unwrap()),
            Some(Tensor::from_slice(&zeros, &[hidden], device).unwrap()),
        );
        ScalarQuantization::new(in_proj, out_proj, scale)
    }

    /// Builds a quantizer whose `in_proj` is bias-only (zero weight), so
    /// `tanh(in_proj(hidden))` depends only on `bias` and NOT on `hidden`'s
    /// value — used to drive tanh into its saturation regime, where the
    /// output is exactly `+-1.0f32` regardless of tanh's concrete
    /// implementation (scalar libm or the SIMD exp-ratio kernels — both
    /// round to +-1.0 well before `|x| = 40`, since the true error is
    /// `~exp(-80)`, far below `f32`'s ~1.2e-7 ULP). This is what makes the
    /// resulting `.5` ties below EXACT, not merely close.
    fn saturating_quantizer(
        scale: f32,
        device: &numr::runtime::cpu::CpuDevice,
    ) -> ScalarQuantization<numr::runtime::cpu::CpuRuntime> {
        let zero_weight = vec![0.0f32; 4]; // [2, 2], all zero
        let bias = vec![-40.0f32, 40.0]; // saturates tanh to exactly [-1.0, 1.0]
        let in_proj = MaybeQuantLinear::from_weight(
            Weight::Standard(Tensor::from_slice(&zero_weight, &[2, 2], device).unwrap()),
            Some(Tensor::from_slice(&bias, &[2], device).unwrap()),
        );
        let identity = vec![1.0f32, 0.0, 0.0, 1.0];
        let zeros = vec![0.0f32, 0.0];
        let out_proj = MaybeQuantLinear::from_weight(
            Weight::Standard(Tensor::from_slice(&identity, &[2, 2], device).unwrap()),
            Some(Tensor::from_slice(&zeros, &[2], device).unwrap()),
        );
        ScalarQuantization::new(in_proj, out_proj, scale)
    }

    /// Exact `.5`-tie regression test, at the two `scale` values (from the
    /// task's own verified reference table) where ties-to-even and
    /// ties-away-from-zero actually disagree by a whole unit:
    ///
    /// ```text
    /// scale=0.5: tanh -> +-1.0, raw = +-0.5
    ///   ties-to-even:        round(-0.5)=-0, round(0.5)=0   -> levels [0.0, 0.0]
    ///   ties-away (WRONG):   round(-0.5)=-1, round(0.5)=1   -> levels [-2.0, 2.0]
    /// scale=2.5: tanh -> +-1.0, raw = +-2.5
    ///   ties-to-even:        round(-2.5)=-2, round(2.5)=2   -> levels [-0.8, 0.8]
    ///   ties-away (WRONG):   round(-2.5)=-3, round(2.5)=3   -> levels [-1.2, 1.2]
    /// ```
    ///
    /// (The task's other example ties, `-1.5/1.5` and `3.5`, round to the
    /// SAME value under both rules — the "even" neighbor happens to equal
    /// the "away from zero" neighbor there — so they carry no discriminating
    /// power and are intentionally not used here.)
    #[test]
    fn quantization_matches_ties_to_even_not_ties_away() {
        let (client, device) = cpu_setup();
        let input = Var::new(
            Tensor::from_slice(&[0.0f32, 0.0], &[1, 2], &device).unwrap(),
            false,
        );

        let cases: [(f32, [f32; 2]); 2] = [(0.5, [0.0, 0.0]), (2.5, [-0.8, 0.8])];
        for (scale, expected) in cases {
            let quantizer = saturating_quantizer(scale, &device);
            let out = quantizer.forward(&client, &input).unwrap();
            let data: Vec<f32> = out.tensor().to_vec();
            for (got, want) in data.iter().zip(expected.iter()) {
                assert!(
                    (got - want).abs() < 1e-5,
                    "scale={scale}: ties-to-even mismatch, got {got}, want {want} \
                     (a switch to ties-away-from-zero rounding gives a different \
                     value here)"
                );
            }
        }
    }

    #[test]
    fn quantized_levels_are_bounded_to_k_over_scale() {
        let (client, device) = cpu_setup();
        let hidden = 4usize;
        let scale = 9.0f32;
        let quantizer = identity_quantizer(hidden, scale, &device);

        // Large-magnitude pre-tanh inputs saturate tanh toward +-1, so the
        // quantized level must land on +-9/9 = +-1.0, never beyond it.
        let input = Var::new(
            Tensor::from_slice(&[-50.0f32, -1.0, 1.0, 50.0], &[1, hidden], &device).unwrap(),
            false,
        );
        let out = quantizer.forward(&client, &input).unwrap();
        let data: Vec<f32> = out.tensor().to_vec();
        for &v in &data {
            assert!(
                (-1.0..=1.0).contains(&v),
                "quantized level {v} outside the (-1, 1) tanh-derived bound"
            );
            // Every level is k/9 for integer k in -9..=9.
            let k = v * scale;
            assert!(
                (k - k.round()).abs() < 1e-4,
                "level {v} is not a multiple of 1/{scale}"
            );
        }
    }

    #[test]
    fn accepts_rank_2_and_rank_3() {
        let (client, device) = cpu_setup();
        let hidden = 4usize;
        let quantizer = identity_quantizer(hidden, 9.0, &device);

        let rank2 = Var::new(
            Tensor::from_slice(&[0.1f32; 8], &[2, hidden], &device).unwrap(),
            false,
        );
        assert_eq!(
            quantizer.forward(&client, &rank2).unwrap().shape(),
            &[2, hidden]
        );

        let rank3 = Var::new(
            Tensor::from_slice(&[0.1f32; 24], &[2, 3, hidden], &device).unwrap(),
            false,
        );
        assert_eq!(
            quantizer.forward(&client, &rank3).unwrap().shape(),
            &[2, 3, hidden]
        );
    }

    #[test]
    fn rejects_rank_1_and_rank_4() {
        let (client, device) = cpu_setup();
        let hidden = 4usize;
        let quantizer = identity_quantizer(hidden, 9.0, &device);

        let rank1 = Var::new(
            Tensor::from_slice(&[0.1f32; 4], &[hidden], &device).unwrap(),
            false,
        );
        assert!(quantizer.forward(&client, &rank1).is_err());

        let rank4 = Var::new(
            Tensor::from_slice(&[0.1f32; 16], &[1, 1, 4, hidden], &device).unwrap(),
            false,
        );
        assert!(quantizer.forward(&client, &rank4).is_err());
    }
}
