//! The per-patch loop's public types: [`GenerateOptions`], [`GenerateOutcome`],
//! [`StepOutcome`], [`GenerateState`] and [`PatchGenerator`] — construction
//! only. The loop itself lives in `super::step`.

use super::*;

/// Stop-token class index in `stop_head`'s 2-wide output. Class 0 is
/// "continue", class 1 is "stop". `pub(crate)` so
/// [`super::super::train::stop_targets`](crate::model::audio::voxcpm::model::train)
/// can build the SAME class index as a training target instead of
/// duplicating the magic number.
pub(crate) const STOP_CLASS: i64 = 1;

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

#[cfg(test)]
mod tests {
    use super::*;

    /// The options constructor's real settings, not the test's cheap ones.
    ///
    /// `n_timesteps` is 10, DELIBERATELY not the reference clone script's 32.
    /// Measured on an RTX 3060: 32 costs 4x the compute of 10 (RTF 4.00 vs 1.30)
    /// and sounds flatter — more solver steps converge harder toward the mode of
    /// the flow and smooth away prosodic variation. Whisper transcribes 10, 16,
    /// 24 and 32 word-perfect, so this was decided by listening, not by WER.
    ///
    /// If this assertion ever fails because someone "restored" 32 to match the
    /// reference, that is a regression in both speed and quality — see
    /// `GenerateOptions::new`'s doc comment. The parity gate
    /// `examples/voxcpm_step_check.rs` pins 32 separately and on purpose,
    /// because its Python fixtures were generated at that step count.
    #[test]
    fn default_options_carry_the_clone_scripts_values() {
        let opts = GenerateOptions::new(600, 0);
        assert_eq!(
            opts.cfm.n_timesteps, 10,
            "10 is chosen over the reference 32"
        );
        assert_eq!(opts.cfm.cfg_value, 2.0);
        assert_eq!(opts.min_len, 2);
        assert_eq!(opts.max_len, 600);
    }
}
