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
use crate::model::audio::voxcpm::model::config::REF_AUDIO_END_ID;
use crate::model::audio::voxcpm::vae::{
    AudioVaeDecoder, AudioVaeDecoderWeights, AudioVaeEncoder, AudioVaeEncoderWeights, CausalConv1d,
    CausalTransposeConv1d, DecoderBlock, DecoderBlockWeights, EncoderBlock, EncoderBlockWeights,
    ResUnit, Snake,
};
use crate::nn::{MaybeLoraLinear, MaybeQuantEmbedding, MaybeQuantLinear, RoPE, Weight};
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
///
/// Fields are `pub(crate)`, not `pub(super)`: `train/tests.rs` (a sibling
/// of `generate`, not a descendant) reuses this exact fixture and needs
/// direct field access to `apply_lora` individual sub-models — see that
/// module's doc comment.
pub(crate) struct Fixture {
    pub(crate) feat_encoder: LocalEncoder<CpuRuntime>,
    pub(crate) feat_decoder: LocalDit<CpuRuntime>,
    pub(crate) base_lm: MiniCpm4Model<CpuRuntime>,
    pub(crate) residual_lm: MiniCpm4Model<CpuRuntime>,
    pub(crate) fsq: ScalarQuantization<CpuRuntime>,
    pub(crate) aux: AuxProjections<CpuRuntime>,
}

impl Fixture {
    pub(crate) fn generator(&self) -> PatchGenerator<'_, CpuRuntime> {
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
        activation_checkpointing: false,
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
) -> (MaybeLoraLinear<CpuRuntime>, MaybeLoraLinear<CpuRuntime>) {
    let sign = if stop { 1.0f32 } else { -1.0 };
    let stop_proj: MaybeLoraLinear<CpuRuntime> = MaybeQuantLinear::from_weight(
        Weight::Standard(
            Tensor::<CpuRuntime>::zeros(&[HIDDEN, HIDDEN], DType::F32, device).expect("zeros"),
        ),
        Some(Tensor::<CpuRuntime>::from_slice(&[1.0f32; HIDDEN], &[HIDDEN], device).expect("bias")),
    )
    .into();
    let mut head = vec![0.0f32; 2 * HIDDEN];
    head[HIDDEN..].fill(sign);
    let stop_head: MaybeLoraLinear<CpuRuntime> = MaybeQuantLinear::from_weight(
        Weight::Standard(
            Tensor::<CpuRuntime>::from_slice(&head, &[2, HIDDEN], device).expect("head"),
        ),
        None,
    )
    .into();
    (stop_proj, stop_head)
}

pub(crate) fn fixture(stop: bool, device: &CpuDevice) -> Fixture {
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
pub(crate) fn state(fx: &Fixture, device: &CpuDevice) -> GenerateState<CpuRuntime> {
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

pub(crate) fn options(min_len: usize, max_len: usize) -> GenerateOptions {
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

pub(crate) fn values(v: &Var<CpuRuntime>) -> Vec<f32> {
    v.tensor().contiguous().expect("contiguous").to_vec::<f32>()
}

pub(crate) fn noise(seed: f32, device: &CpuDevice) -> Var<CpuRuntime> {
    Var::new(t(&[1, FEAT_DIM, PATCH_SIZE], seed, device), false)
}

// --- dimensionally-trivial AudioVAE, for tests that need a whole
// `VoxCpm2Model` rather than the bare `PatchGenerator` -----------------------
//
// `VoxCpm2Model::prefill`/`prefill_capturing` touch neither `vae_encoder` nor
// `vae_decoder` at all, but the struct still requires both fields. Every
// tensor below is the smallest shape each constructor accepts (channel = 1,
// kernel = 1 or 2), so building these costs nothing and is never meant to run
// a real forward pass.

fn tiny_causal_conv(device: &CpuDevice) -> CausalConv1d<CpuRuntime> {
    let weight = Tensor::<CpuRuntime>::zeros(&[1, 1, 1], DType::F32, device).expect("zeros");
    CausalConv1d::new(weight, None, 1, 1, 1).expect("kernel_size 1 always fits")
}

fn tiny_strided_conv(device: &CpuDevice) -> CausalConv1d<CpuRuntime> {
    // `new_strided` requires kernel_size == 2 * stride; stride 1 needs a
    // kernel of 2.
    let weight = Tensor::<CpuRuntime>::zeros(&[1, 1, 2], DType::F32, device).expect("zeros");
    CausalConv1d::new_strided(weight, None, 1, 1).expect("kernel_size 2 matches stride 1")
}

fn tiny_transpose_conv(device: &CpuDevice) -> CausalTransposeConv1d<CpuRuntime> {
    let weight = Tensor::<CpuRuntime>::zeros(&[1, 1, 2], DType::F32, device).expect("zeros");
    CausalTransposeConv1d::new(weight, None, 1).expect("kernel_size 2 matches stride 1")
}

fn tiny_snake(device: &CpuDevice) -> Snake<CpuRuntime> {
    let alpha = Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1, 1, 1], device).expect("alpha");
    Snake::new(alpha).expect("[1, 1, 1] is a valid Snake shape")
}

fn tiny_res_unit(device: &CpuDevice) -> ResUnit<CpuRuntime> {
    ResUnit::new(
        tiny_snake(device),
        tiny_causal_conv(device),
        tiny_snake(device),
        tiny_causal_conv(device),
    )
}

fn tiny_encoder_block(device: &CpuDevice) -> EncoderBlock<CpuRuntime> {
    EncoderBlock::new(EncoderBlockWeights {
        res1: tiny_res_unit(device),
        res3: tiny_res_unit(device),
        res9: tiny_res_unit(device),
        snake: tiny_snake(device),
        downsample: tiny_strided_conv(device),
    })
}

fn tiny_vae_encoder(device: &CpuDevice) -> AudioVaeEncoder<CpuRuntime> {
    AudioVaeEncoder::new(AudioVaeEncoderWeights {
        front: tiny_causal_conv(device),
        blocks: std::array::from_fn(|_| tiny_encoder_block(device)),
        fc_mu: tiny_causal_conv(device),
    })
}

fn tiny_decoder_block(device: &CpuDevice) -> DecoderBlock<CpuRuntime> {
    // `scale_embed`/`bias_embed` must be `[num_sr_buckets, input_dim]`;
    // `input_dim` is 1 here, and one bucket is enough since `forward` (which
    // reads `sr_bucket`) is never called.
    let embed = Tensor::<CpuRuntime>::zeros(&[1, 1], DType::F32, device).expect("zeros");
    DecoderBlock::new(DecoderBlockWeights {
        snake: tiny_snake(device),
        upsample: tiny_transpose_conv(device),
        res1: tiny_res_unit(device),
        res3: tiny_res_unit(device),
        res9: tiny_res_unit(device),
        scale_embed: embed.clone(),
        bias_embed: embed,
    })
    .expect("input_dim 1 matches scale_embed/bias_embed width 1")
}

fn tiny_vae_decoder(device: &CpuDevice) -> AudioVaeDecoder<CpuRuntime> {
    AudioVaeDecoder::new(AudioVaeDecoderWeights {
        front_dw: tiny_causal_conv(device),
        front_pw: tiny_causal_conv(device),
        blocks: std::array::from_fn(|_| tiny_decoder_block(device)),
        final_snake: tiny_snake(device),
        final_conv: tiny_causal_conv(device),
    })
}

/// Large enough to hold every token id a `model(...)`-based test embeds:
/// the layout's control ids (`REF_AUDIO_FILLER_ID` 0, `AUDIO_START_ID` 101,
/// `REF_AUDIO_START_ID` 103, `REF_AUDIO_END_ID` 104 — see `sequence.rs`)
/// plus every small literal id a test uses as ordinary text.
const TEXT_VOCAB_SIZE: usize = REF_AUDIO_END_ID as usize + 1;

/// A real `embed_tokens` table for `model(...)`'s `base_lm`.
///
/// `tiny_model` builds `base_lm` with `embed_tokens: None`, matching
/// `residual_lm` (which genuinely has none) — every OTHER test in this
/// module drives the loop with pre-computed `inputs_embeds` and never calls
/// `MiniCpm4Model::embed`. Only `model(...)`'s prefill path calls it (via
/// `base_lm.embed` on the tokenized text), so the table is added HERE rather
/// than in `tiny_model` itself, which every other fixture user still gets
/// unchanged. Filled the same deterministic way as the fixture's other
/// weights, via `t`.
fn text_embed_tokens(device: &CpuDevice) -> MaybeQuantEmbedding<CpuRuntime> {
    MaybeQuantEmbedding::from_weight(
        Weight::Standard(t(&[TEXT_VOCAB_SIZE, HIDDEN], 12.5, device)),
        false,
    )
    .expect("Weight::Standard never errors")
}

/// A full [`VoxCpm2Model`] for tests that call `prefill`/`prefill_capturing`
/// directly rather than driving a bare [`PatchGenerator`]. Reuses `fx`'s
/// sub-models verbatim except `base_lm`, which gains the `embed_tokens`
/// table above; `vae_encoder`/`vae_decoder` are new, trivial stand-ins since
/// prefill never reads them.
pub(crate) fn model(fx: Fixture, device: &CpuDevice) -> VoxCpm2Model<CpuRuntime> {
    let mut base_lm = fx.base_lm;
    base_lm.embed_tokens = Some(text_embed_tokens(device));
    VoxCpm2Model {
        vae_encoder: tiny_vae_encoder(device),
        vae_decoder: tiny_vae_decoder(device),
        feat_encoder: fx.feat_encoder,
        base_lm,
        residual_lm: fx.residual_lm,
        feat_decoder: fx.feat_decoder,
        fsq: fx.fsq,
        aux: fx.aux,
        config: VoxCpm2Config {
            patch_size: PATCH_SIZE,
            feat_dim: FEAT_DIM,
        },
    }
}
