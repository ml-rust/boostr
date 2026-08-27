//! Fixtures and helpers for the VoxCPM2 per-patch generation loop tests.
//!
//! The tiny estimator, its dimension constants and the weight fillers come
//! from `local_dit::tests`; the two LMs from `minicpm4::model::tests`. Only
//! the pieces those do not cover (`feat_encoder`, `fsq_layer`, the six
//! auxiliary projections) are built here.
//!
//! The stop classifier is CONSTRUCTED to answer a fixed class regardless of
//! the hidden state it is handed ([`stop_chain`]), because every stop-guard
//! assertion in `tests/mod.rs` has to know the answer without depending on
//! the tiny weights' arithmetic.

use super::super::*;
use crate::model::audio::voxcpm::local_dit::tests::{
    FEAT_DIM, HEAD_DIM, HIDDEN_DIM, PATCH_SIZE, layer, linear, model as dit_model, norm, t,
};
use crate::model::audio::voxcpm::minicpm4::model::tests::{HIDDEN, tiny_model, tiny_nope_model};
use crate::nn::{MaybeQuantLinear, RoPE, Weight};
use numr::runtime::cpu::{CpuDevice, CpuRuntime};

/// `base_lm`/`residual_lm` hidden width and `feat_encoder`'s pooled width are
/// both 8 in the tiny fixtures, so every auxiliary projection below is
/// square. A mismatch would be a shape error, not a silent one.
const _: () = assert!(HIDDEN == HIDDEN_DIM);

/// Cache capacity. `tiny_model`'s RoPE table is 16 positions long, so
/// `new_kv_cache` refuses anything longer.
const MAX_LENGTH: usize = 16;

/// Two Euler steps: enough that the estimator actually runs (step 1 is the
/// CFG-zero-star warmup), cheap enough to run the loop many times.
const N_TIMESTEPS: usize = 2;

/// Every sub-model the loop touches, owned so the borrowed
/// [`PatchGenerator`] can point at it.
pub(super) struct Fixture {
    feat_encoder: LocalEncoder<CpuRuntime>,
    feat_decoder: LocalDit<CpuRuntime>,
    base_lm: MiniCpm4Model<CpuRuntime>,
    residual_lm: MiniCpm4Model<CpuRuntime>,
    fsq: ScalarQuantization<CpuRuntime>,
    pub(super) aux: AuxProjections<CpuRuntime>,
}

impl Fixture {
    pub(super) fn generator(&self) -> PatchGenerator<'_, CpuRuntime> {
        PatchGenerator {
            feat_encoder: &self.feat_encoder,
            feat_decoder: &self.feat_decoder,
            base_lm: &self.base_lm,
            residual_lm: &self.residual_lm,
            fsq: &self.fsq,
            aux: &self.aux,
            config: VoxCpm2Config {
                patch_size: PATCH_SIZE,
                feat_dim: FEAT_DIM,
            },
        }
    }
}

/// `feat_encoder` over `[1, 1, PATCH_SIZE, FEAT_DIM]`. One transformer layer,
/// not zero: with no layers the CLS pool returns `norm(special_token)`, which
/// is constant in the input and would hide any wiring bug in step 3.
fn feat_encoder(device: &CpuDevice) -> LocalEncoder<CpuRuntime> {
    let rope = RoPE::<CpuRuntime>::precompute_freqs(32, HEAD_DIM, 10000.0, None, device)
        .expect("rope")
        .narrow_positions(PATCH_SIZE + 1)
        .expect("narrow");
    LocalEncoder {
        in_proj: linear(HIDDEN_DIM, FEAT_DIM, 1.7, true, device),
        special_token: Var::new(t(&[1, 1, 1, HIDDEN_DIM], 2.3, device), false),
        layers: vec![layer(3.1, device)],
        norm: norm(device),
        rope,
        hidden_dim: HIDDEN_DIM,
    }
}

/// A stop chain whose argmax is a FIXED class for any hidden state.
///
/// `stop_proj` has a zero weight and an all-ones bias, so its output is the
/// constant `silu(1) > 0` in every channel whatever the hidden state was.
/// `stop_head` then reads class 0 off an all-zero row (logit exactly 0) and
/// class 1 off a row of `sign`, so class 1 wins iff `sign` is positive.
fn stop_chain(
    stop: bool,
    device: &CpuDevice,
) -> (MaybeQuantLinear<CpuRuntime>, MaybeQuantLinear<CpuRuntime>) {
    let sign = if stop { 1.0f32 } else { -1.0 };
    let stop_proj = MaybeQuantLinear::from_weight(
        Weight::Standard(
            Tensor::<CpuRuntime>::zeros(&[HIDDEN, HIDDEN], DType::F32, device).expect("zeros"),
        ),
        Some(Tensor::<CpuRuntime>::from_slice(&[1.0f32; HIDDEN], &[HIDDEN], device).expect("bias")),
    );
    let mut head = vec![0.0f32; 2 * HIDDEN];
    head[HIDDEN..].fill(sign);
    let stop_head = MaybeQuantLinear::from_weight(
        Weight::Standard(
            Tensor::<CpuRuntime>::from_slice(&head, &[2, HIDDEN], device).expect("head"),
        ),
        None,
    );
    (stop_proj, stop_head)
}

pub(super) fn fixture(stop: bool, device: &CpuDevice) -> Fixture {
    let (stop_proj, stop_head) = stop_chain(stop, device);
    Fixture {
        feat_encoder: feat_encoder(device),
        feat_decoder: dit_model(1, device),
        base_lm: tiny_model(device),
        residual_lm: tiny_nope_model(device),
        fsq: ScalarQuantization::new(
            linear(4, HIDDEN, 4.2, true, device),
            linear(HIDDEN, 4, 5.3, true, device),
            9.0,
        ),
        aux: AuxProjections {
            enc_to_lm_proj: linear(HIDDEN, HIDDEN_DIM, 6.1, true, device),
            lm_to_dit_proj: linear(HIDDEN_DIM, HIDDEN, 7.2, true, device),
            res_to_dit_proj: linear(HIDDEN_DIM, HIDDEN, 8.3, true, device),
            fusion_concat_proj: linear(HIDDEN, 2 * HIDDEN, 9.4, true, device),
            stop_proj,
            stop_head,
        },
    }
}

/// A [`PrefillState`] with both caches empty, so `position` starts at 0 and
/// `decode_step`'s write-order rule is satisfied without running a prefill.
pub(super) fn state(fx: &Fixture, device: &CpuDevice) -> GenerateState<CpuRuntime> {
    let prefill = PrefillState {
        lm_hidden: Var::new(t(&[1, HIDDEN], 0.9, device), false),
        residual_hidden: Var::new(t(&[1, HIDDEN], 1.3, device), false),
        base_cache: fx.base_lm.new_kv_cache(1, MAX_LENGTH).expect("base cache"),
        residual_cache: fx
            .residual_lm
            .new_kv_cache(1, MAX_LENGTH)
            .expect("residual cache"),
        position: 0,
        intermediates: None,
    };
    GenerateState::start(
        prefill,
        VoxCpm2Config {
            patch_size: PATCH_SIZE,
            feat_dim: FEAT_DIM,
        },
    )
    .expect("start")
}

pub(super) fn options(min_len: usize, max_len: usize) -> GenerateOptions {
    GenerateOptions {
        cfm: CfmOptions {
            n_timesteps: N_TIMESTEPS,
            ..CfmOptions::default()
        },
        min_len,
        max_len,
        seed: 7,
    }
}

pub(super) fn values(v: &Var<CpuRuntime>) -> Vec<f32> {
    v.tensor().contiguous().expect("contiguous").to_vec::<f32>()
}

pub(super) fn noise(seed: f32, device: &CpuDevice) -> Var<CpuRuntime> {
    Var::new(t(&[1, FEAT_DIM, PATCH_SIZE], seed, device), false)
}
