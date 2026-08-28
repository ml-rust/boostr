//! Split out of `layer.rs` to keep that file under the crate's 500-line hard
//! limit for model-architecture files after `ScalarQuantization::apply_lora`
//! and `AuxProjections::apply_lora` were added. `use super::*;` below
//! reaches every item `layer.rs` itself imported, exactly as if this module
//! were still inline.

use super::*;
// Only the tests construct a base projection directly; the library builds
// them through `TensorLoader::linear`, which returns `MaybeLoraLinear`.
use crate::nn::MaybeQuantLinear;
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
    let in_proj: MaybeLoraLinear<_> = MaybeQuantLinear::from_weight(
        Weight::Standard(Tensor::from_slice(&identity, &[hidden, hidden], device).unwrap()),
        Some(Tensor::from_slice(&zeros, &[hidden], device).unwrap()),
    )
    .into();
    let out_proj: MaybeLoraLinear<_> = MaybeQuantLinear::from_weight(
        Weight::Standard(Tensor::from_slice(&identity, &[hidden, hidden], device).unwrap()),
        Some(Tensor::from_slice(&zeros, &[hidden], device).unwrap()),
    )
    .into();
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
    let in_proj: MaybeLoraLinear<_> = MaybeQuantLinear::from_weight(
        Weight::Standard(Tensor::from_slice(&zero_weight, &[2, 2], device).unwrap()),
        Some(Tensor::from_slice(&bias, &[2], device).unwrap()),
    )
    .into();
    let identity = vec![1.0f32, 0.0, 0.0, 1.0];
    let zeros = vec![0.0f32, 0.0];
    let out_proj: MaybeLoraLinear<_> = MaybeQuantLinear::from_weight(
        Weight::Standard(Tensor::from_slice(&identity, &[2, 2], device).unwrap()),
        Some(Tensor::from_slice(&zeros, &[2], device).unwrap()),
    )
    .into();
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
