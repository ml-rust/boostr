//! Unit B of the VoxCPM2 end-to-end orchestrator: the per-patch generation
//! loop that drives a finished [`PrefillState`] to a stop token or a cap.
//!
//! ```text
//! per iteration i:
//!   1. mu         = cat(lm_to_dit_proj(lm_hidden), res_to_dit_proj(residual_hidden))
//!   2. pred_feat  = feat_decoder(mu, cond = prefix_feat_cond^T)
//!   3. curr_embed = enc_to_lm_proj(feat_encoder(pred_feat.unsqueeze(1)))
//!   4. patches.push(pred_feat); prefix_feat_cond = pred_feat
//!   5. stop?      = argmax(stop(lm_hidden)) == 1   <- the CURRENT lm_hidden
//!   6. lm_hidden  = fsq(base_lm.decode_step(curr_embed, base_cache, position))
//!   7. residual_hidden = residual_lm.decode_step(
//!          fusion_concat_proj(cat(lm_hidden, curr_embed)), residual_cache, position)
//!   8. position  += 1
//! ```
//!
//! The VAE decode and the wav wrapper are a LATER unit: this file returns
//! patches and nothing else.
//!
//! # Traps this file exists to get right
//!
//! - **`prefix_feat_cond` starts as the ZERO text-pad patch**, `[1,
//!   patch_size, feat_dim]` of zeros — NOT the last patch of the reference
//!   audio. [`GenerateState::start`] builds it; [`PrefillState`] never
//!   feeds it.
//! - **The initial `lm_hidden` is UN-fsq'd.** It is the post-blend
//!   `enc_outputs` row at a TEXT position, which the prefill's blend left
//!   alone. From step 6 onward every `lm_hidden` IS fsq'd. Do not normalise
//!   the two: re-fsq'ing the prefill's row, and dropping the fsq in step 6,
//!   both stay shape-valid.
//! - **Step 7 consumes the POST-fsq `lm_hidden`**, and the concat order is
//!   `(lm_hidden, curr_embed)`. That concat is `2 * lm_hidden` wide, so
//!   swapping the halves is shape-valid and computes a different model.
//! - **ONE shared `position` counter drives BOTH caches.** The prefill
//!   primed both to `S` and both advance by exactly one per iteration.
//!   [`MiniCpm4Model::decode_step`] rejects `position != cache.seq_len()`, so
//!   a drift errors rather than corrupts — that check is why this counter is
//!   not duplicated per cache.
//! - **The stop guard is STRICTLY greater**: `i > min_len`, so a stop token
//!   at `i <= min_len` is ignored and at least `min_len + 2` patches are
//!   always emitted (`i = 0..=min_len` push unconditionally, and `i = min_len
//!   + 1` pushes before its check can fire).
//! - **The `max_len` exit is silent in the reference.** Here it is
//!   [`GenerateOutcome::MaxLen`], distinct from
//!   [`GenerateOutcome::StopToken`], so a caller can tell a finished
//!   utterance from a truncated one.
//! - **`step_with_noise` ignores `temperature`.** Scaling the noise draw is
//!   the drawing wrapper's job, the same split `solve_euler`/`sample` uses.
//!   A caller injecting `z` owns its scale.
//!
//! # The one device read
//!
//! Step 5 turns two logits into control flow, so it cannot stay on device.
//! The read is `argmax` ON DEVICE plus
//! [`Tensor::item`](numr::tensor::Tensor::item) on the resulting
//! single-element index: EIGHT bytes per patch, never the logits themselves
//! and never a whole tensor. See [`stop_predicted`].

use crate::error::{Error, Result};
use crate::model::audio::voxcpm::fsq::{AuxProjections, ScalarQuantization};
use crate::model::audio::voxcpm::local_dit::{CfmOptions, LocalDit};
use crate::model::audio::voxcpm::local_encoder::LocalEncoder;
use crate::model::audio::voxcpm::minicpm4::MiniCpm4Model;
use crate::model::audio::voxcpm::model::config::VoxCpm2Config;
use crate::model::audio::voxcpm::model::loader::VoxCpm2Model;
use crate::model::audio::voxcpm::model::prefill::PrefillState;
use crate::model::traits::ModelClient;
use numr::autograd::{Var, var_mul_scalar};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, RandomOps, ReduceOps,
    ScalarOps, ShapeOps, TensorOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

mod capture;
mod teacher_forced;
pub use capture::StepIntermediates;
pub use teacher_forced::TeacherForcedConditioning;

/// Stop-token class index in `stop_head`'s 2-wide output. Class 0 is
/// "continue", class 1 is "stop".
const STOP_CLASS: i64 = 1;

/// Knobs for the per-patch loop.
///
/// Everything the CFM sampler already names lives in [`CfmOptions`], never
/// duplicated here. Build one with [`GenerateOptions::new`]; there is no
/// `Default`, because `max_len` has no defensible default — the KV caches
/// the prefill allocated bound it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenerateOptions {
    /// Sampler settings for every [`LocalDit`] call: `n_timesteps`,
    /// `cfg_value`, `temperature`, `sway_sampling_coef`,
    /// `use_cfg_zero_star`.
    pub cfm: CfmOptions,
    /// Iterations whose stop token is IGNORED. The guard is `i > min_len`,
    /// strictly greater, so `min_len + 2` patches is the floor.
    pub min_len: usize,
    /// Hard cap on emitted patches. Reaching it yields
    /// [`GenerateOutcome::MaxLen`].
    pub max_len: usize,
    /// Base seed for the self-drawing [`PatchGenerator::step`]. Patch `i`
    /// draws with `seed + i`, so each patch gets fresh noise and a whole run
    /// is reproducible from this one number.
    pub seed: u64,
}

impl GenerateOptions {
    /// The clone script's verified settings, with `max_len` and `seed` from
    /// the caller. `cfg_value` 2.0, `temperature` 1.0, `sway_sampling_coef`
    /// 1.0, `use_cfg_zero_star` on, `min_len` 2.
    ///
    /// `n_timesteps` is [`CfmOptions::default`]'s 10, NOT the reference clone
    /// script's 32. This deviates from the reference deliberately, on measured
    /// evidence: 32 costs 4x the compute of 10 (RTF 4.00 vs 1.30 on an RTX
    /// 3060) and sounds WORSE — flatter, less prosodic variation. Whisper
    /// transcribes 10, 16, 24 and 32 word-perfect, so intelligibility does not
    /// separate them; the difference is naturalness, judged by ear.
    ///
    /// The direction is the opposite of the usual intuition, and the reason is
    /// that more solver steps converge harder toward the mode of the flow,
    /// which smooths away exactly the prosodic variation that makes speech
    /// sound alive. More steps is not more quality here.
    ///
    /// If generation variance ever becomes a problem at 10, `--best-of` is the
    /// lever to reach for, not a higher step count.
    pub fn new(max_len: usize, seed: u64) -> Self {
        Self {
            cfm: CfmOptions::default(),
            min_len: 2,
            max_len,
            seed,
        }
    }
}

/// Why [`PatchGenerator::generate`] stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerateOutcome {
    /// The stop classifier predicted class 1 past `min_len`. The utterance
    /// is complete.
    StopToken,
    /// `max_len` patches were emitted without a stop token. The reference
    /// exits silently here; the utterance is TRUNCATED.
    MaxLen,
}

/// Why one [`PatchGenerator::step_with_noise`] returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepOutcome {
    /// The patch was emitted and the loop state advanced through step 8.
    Continued,
    /// The patch was emitted and the stop guard fired, so steps 6-8 were
    /// SKIPPED — the caches and `position` are untouched, exactly as the
    /// reference's `break` leaves them.
    Stopped,
}

/// Live state of the per-patch loop. Its `prefill` field is the one-shot
/// handoff, mutated in place: `lm_hidden`, `residual_hidden`, both caches
/// and `position` all advance per iteration.
pub struct GenerateState<R: Runtime> {
    /// The prefill handoff, driven forward. `position` here is the ONE
    /// counter both caches follow.
    pub prefill: PrefillState<R>,
    /// `[1, patch_size, feat_dim]` — the DiT's prefix condition for the NEXT
    /// iteration. Zeros on entry to iteration 0 (the text-pad patch); the
    /// previous `pred_feat` on entry to every later iteration.
    pub prefix_feat_cond: Var<R>,
    /// Patches emitted so far, each `[1, patch_size, feat_dim]`, in order.
    /// Its length is the loop's iteration index `i`.
    pub patches: Vec<Var<R>>,
}

/// The sub-models the per-patch loop touches, borrowed from a
/// [`VoxCpm2Model`] by [`VoxCpm2Model::patch_generator`]. The AudioVAE is
/// deliberately absent: nothing in this unit encodes or decodes audio, so
/// nothing here can accidentally depend on it.
pub struct PatchGenerator<'a, R: Runtime> {
    /// `feat_encoder`, run on ONE patch per iteration (`[1, 1, patch_size,
    /// feat_dim]`).
    pub feat_encoder: &'a LocalEncoder<R>,
    /// `feat_decoder`, the CFM estimator this loop integrates.
    pub feat_decoder: &'a LocalDit<R>,
    /// `base_lm`, stepped once per iteration.
    pub base_lm: &'a MiniCpm4Model<R>,
    /// `residual_lm`, stepped once per iteration at the SAME position.
    pub residual_lm: &'a MiniCpm4Model<R>,
    /// `fsq_layer`, applied to every `base_lm` step output.
    pub fsq: &'a ScalarQuantization<R>,
    /// The auxiliary projections and the stop chain.
    pub aux: &'a AuxProjections<R>,
    /// Patch geometry.
    pub config: VoxCpm2Config,
}

impl<R: Runtime<DType = DType>> VoxCpm2Model<R> {
    /// Borrow the sub-models the per-patch loop needs.
    pub fn patch_generator(&self) -> PatchGenerator<'_, R> {
        PatchGenerator {
            feat_encoder: &self.feat_encoder,
            feat_decoder: &self.feat_decoder,
            base_lm: &self.base_lm,
            residual_lm: &self.residual_lm,
            fsq: &self.fsq,
            aux: &self.aux,
            config: self.config,
        }
    }
}

impl<R: Runtime<DType = DType>> GenerateState<R> {
    /// Open the loop over a finished prefill.
    ///
    /// `prefix_feat_cond` is set to the ZERO text-pad patch — the reference
    /// conditions patch 0 on zeros, not on the reference audio's tail. Dtype
    /// and device follow `prefill.lm_hidden`, so the condition matches the
    /// tensors it will meet. Errors when either handoff row is not `[1,
    /// hidden]`.
    pub fn start(prefill: PrefillState<R>, config: VoxCpm2Config) -> Result<Self> {
        check_row("prefill.lm_hidden", &prefill.lm_hidden)?;
        check_row("prefill.residual_hidden", &prefill.residual_hidden)?;
        let hidden = prefill.lm_hidden.tensor();
        let zeros = Tensor::<R>::zeros(
            &[1, config.patch_size, config.feat_dim],
            hidden.dtype(),
            hidden.device(),
        )
        .map_err(Error::Numr)?;
        Ok(Self {
            prefill,
            prefix_feat_cond: Var::new(zeros, false),
            patches: Vec::new(),
        })
    }
}

impl<R: Runtime<DType = DType>> PatchGenerator<'_, R> {
    /// One iteration, with the CFM noise supplied by the caller.
    ///
    /// This is the primitive: it draws nothing, so a caller (the CFM gate)
    /// can pin `z` per step and reproduce a run exactly.
    /// [`step`](Self::step) is the thin drawing wrapper over it. `z` is `[1,
    /// feat_dim, patch_size]` and is used AS GIVEN —
    /// `options.cfm.temperature` is not applied here; see the module docs.
    /// Runs steps 1-8 in order. Returns [`StepOutcome::Stopped`] when the
    /// stop guard fires, in which case steps 6-8 did not run and the caches
    /// and `position` are unchanged.
    pub fn step_with_noise<C>(
        &self,
        client: &C,
        state: &mut GenerateState<R>,
        z: &Var<R>,
        options: &GenerateOptions,
    ) -> Result<StepOutcome>
    where
        C: ModelClient<R> + TypeConversionOps<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + TypeConversionOps<R>,
    {
        self.step_with_noise_inner(client, state, z, options, false)
            .map(|(outcome, _)| outcome)
    }

    /// One iteration, drawing the CFM noise itself.
    ///
    /// `z` is `randn_seeded(options.seed + i) * options.cfm.temperature` over
    /// `[1, feat_dim, patch_size]`, where `i` is the index of the patch about
    /// to be emitted — so consecutive patches never share noise and the run
    /// is reproducible from `options.seed`. Everything after the draw is
    /// [`step_with_noise`](Self::step_with_noise). `randn_seeded` is
    /// reproducible per backend, so one seed draws differently on CPU and
    /// CUDA.
    pub fn step<C>(
        &self,
        client: &C,
        state: &mut GenerateState<R>,
        options: &GenerateOptions,
    ) -> Result<StepOutcome>
    where
        C: ModelClient<R> + TypeConversionOps<R> + RandomOps<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + TypeConversionOps<R>,
    {
        let hidden = state.prefill.lm_hidden.tensor();
        let noise = client
            .randn_seeded(
                &[1, self.config.feat_dim, self.config.patch_size],
                hidden.dtype(),
                options.seed.wrapping_add(state.patches.len() as u64),
            )
            .map_err(Error::Numr)?;
        let z = var_mul_scalar(
            &Var::new(noise, false),
            options.cfm.temperature as f64,
            client,
        )
        .map_err(Error::Numr)?;
        self.step_with_noise(client, state, &z, options)
    }

    /// Run the loop to a stop token or `max_len`.
    ///
    /// Steps with [`step`](Self::step), so the noise comes from
    /// `options.seed`. The emitted patches stay in `state.patches`, each `[1,
    /// patch_size, feat_dim]`; this returns only WHY the loop ended, so the
    /// caller can tell a finished utterance ([`GenerateOutcome::StopToken`])
    /// from a truncated one ([`GenerateOutcome::MaxLen`]). Does NOT
    /// VAE-decode and does NOT write audio — that is a later unit. Errors
    /// when `max_len` is 0, and propagates the first step error (a
    /// `position`/cache drift included) rather than continuing.
    pub fn generate<C>(
        &self,
        client: &C,
        state: &mut GenerateState<R>,
        options: &GenerateOptions,
    ) -> Result<GenerateOutcome>
    where
        C: ModelClient<R> + TypeConversionOps<R> + RandomOps<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + TypeConversionOps<R>,
    {
        if options.max_len == 0 {
            return Err(Error::InvalidArgument {
                arg: "options.max_len",
                reason: "expected at least 1, got 0".to_string(),
            });
        }
        while state.patches.len() < options.max_len {
            if self.step(client, state, options)? == StepOutcome::Stopped {
                return Ok(GenerateOutcome::StopToken);
            }
        }
        Ok(GenerateOutcome::MaxLen)
    }
}

/// Did the stop classifier pick class 1? `logits` is `[1, 2]`. `argmax` runs
/// ON DEVICE and yields a single I64 index;
/// [`Tensor::item`](numr::tensor::Tensor::item) then copies THAT ONE value
/// back — 8 bytes per patch. The read is unavoidable (the answer drives
/// control flow) and deliberately the narrowest form: the logits never leave
/// the device, and nothing here calls `to_vec`.
fn stop_predicted<R, C>(client: &C, logits: &Var<R>) -> Result<bool>
where
    R: Runtime<DType = DType>,
    C: ModelClient<R>,
{
    let shape = logits.shape();
    if shape.len() != 2 || shape[0] != 1 || shape[1] != 2 {
        return Err(Error::InvalidArgument {
            arg: "logits",
            reason: format!("expected stop logits [1, 2], got {shape:?}"),
        });
    }
    let index = client
        .argmax(logits.tensor(), 1, false)
        .map_err(Error::Numr)?;
    Ok(index.item::<i64>().map_err(Error::Numr)? == STOP_CLASS)
}

/// Validate a `[1, hidden]` per-step hidden state, returning `hidden`.
fn check_row<R: Runtime<DType = DType>>(arg: &'static str, v: &Var<R>) -> Result<usize> {
    let shape = v.shape();
    if shape.len() != 2 || shape[0] != 1 {
        return Err(Error::InvalidArgument {
            arg,
            reason: format!("expected [1, hidden] (batch 1, one position), got {shape:?}"),
        });
    }
    Ok(shape[1])
}

/// Validate a rank-3 patch tensor against an exact expected shape.
fn check_patch<R: Runtime<DType = DType>>(
    arg: &'static str,
    v: &Var<R>,
    expected: &[usize; 3],
) -> Result<()> {
    let shape = v.shape();
    if shape != expected.as_slice() {
        return Err(Error::InvalidArgument {
            arg,
            reason: format!("expected {expected:?}, got {shape:?}"),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests;
