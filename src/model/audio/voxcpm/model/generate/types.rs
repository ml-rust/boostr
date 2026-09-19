//! The per-patch loop's public types: [`GenerateOptions`], [`RowOptions`],
//! [`GenerateOutcome`], [`StepOutcome`], [`GenerateState`] and
//! [`PatchGenerator`] — construction and row bookkeeping only. The loop
//! itself lives in `super::step`.

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
#[derive(Debug, Clone, PartialEq)]
pub struct GenerateOptions {
    /// Sampler settings for every [`LocalDit`] call: `n_timesteps`,
    /// `cfg_value`, `temperature`, `sway_sampling_coef`,
    /// `use_cfg_zero_star`.
    pub cfm: CfmOptions,
    /// Iterations whose stop token is IGNORED. The guard is `i > min_len`,
    /// strictly greater, so `min_len + 2` patches is the floor.
    pub min_len: usize,
    /// Hard cap on emitted patches, per row and on the loop as a whole.
    /// Reaching it yields [`GenerateOutcome::MaxLen`].
    pub max_len: usize,
    /// Base seed for the self-drawing [`PatchGenerator::step`]. Patch `i`
    /// draws with `seed + i`, so each patch gets fresh noise and a whole run
    /// is reproducible from this one number.
    pub seed: u64,
    /// Per-row settings for a batch of more than one row. Empty (what
    /// [`new`](Self::new) builds) means every row caps at `max_len` and
    /// draws from `seed`. Non-empty, its length must equal the state's
    /// batch, and row `b` caps at `rows[b].max_len` (at most `max_len`) and
    /// draws from `rows[b].seed`.
    pub rows: Vec<RowOptions>,
}

/// One row's cap and seed inside a batch — see [`GenerateOptions::rows`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowOptions {
    /// This row's patch cap, `1..=GenerateOptions::max_len`.
    pub max_len: usize,
    /// This row's base seed; patch `i` draws with `seed + i`.
    pub seed: u64,
}

impl GenerateOptions {
    /// The clone script's verified settings, with `max_len` and `seed` from
    /// the caller. `cfg_value` 2.0, `temperature` 1.0, `sway_sampling_coef`
    /// 1.0, `use_cfg_zero_star` on, `min_len` 2.
    ///
    /// `n_timesteps` is [`CfmOptions::default`]'s 10, NOT the reference clone
    /// script's 32. This deviates from the reference deliberately, on measured
    /// evidence: 32 costs proportionally more compute than 10 and sounds
    /// WORSE — flatter, less prosodic variation. Whisper transcribes 10, 16,
    /// 24 and 32 word-perfect, so intelligibility does not separate them;
    /// the difference is naturalness, judged by ear.
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
            rows: Vec::new(),
        }
    }

    /// Row `b`'s cap and seed: its [`rows`](Self::rows) entry, or the
    /// shared `max_len`/`seed` when `rows` is empty.
    pub fn row(&self, row: usize) -> RowOptions {
        self.rows.get(row).copied().unwrap_or(RowOptions {
            max_len: self.max_len,
            seed: self.seed,
        })
    }

    /// Reject a zero cap, a `rows` list that does not match `batch`, or a
    /// row cap outside `1..=max_len`.
    pub(super) fn check(&self, batch: usize) -> Result<()> {
        if self.max_len == 0 {
            return Err(Error::InvalidArgument {
                arg: "options.max_len",
                reason: "expected at least 1, got 0".to_string(),
            });
        }
        if !self.rows.is_empty() && self.rows.len() != batch {
            return Err(Error::InvalidArgument {
                arg: "options.rows",
                reason: format!(
                    "expected one entry per batch row ({batch}) or none, got {}",
                    self.rows.len()
                ),
            });
        }
        for (b, row) in self.rows.iter().enumerate() {
            if row.max_len == 0 || row.max_len > self.max_len {
                return Err(Error::InvalidArgument {
                    arg: "options.rows",
                    reason: format!(
                        "row {b}: expected max_len in 1..={}, got {}",
                        self.max_len, row.max_len
                    ),
                });
            }
        }
        Ok(())
    }
}

/// Why a row, or [`PatchGenerator::generate`] as a whole, stopped.
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
    /// Some rows can have finished on this step; the caller asks
    /// [`GenerateState::all_finished`].
    Continued,
    /// The patch was emitted, a stop token fired, and with it EVERY row is
    /// finished, so steps 6-8 were SKIPPED — the caches and `position` are
    /// untouched, exactly as the reference's `break` leaves them.
    Stopped,
}

/// Live state of the per-patch loop over `batch` rows. Its `prefill` field
/// is the one-shot handoff, mutated in place: `lm_hidden`,
/// `residual_hidden`, both caches and `position` all advance per iteration.
///
/// Every row steps on every iteration, a finished row included — its new
/// patches are computed and discarded, which is what keeps the shared
/// caches aligned without a compaction pass. [`patch_len`](Self::patch_len)
/// is the per-row count that stops advancing at the finish.
pub struct GenerateState<R: Runtime> {
    /// The prefill handoff, driven forward. `position` here is the ONE
    /// counter both caches follow.
    pub prefill: PrefillState<R>,
    /// `[batch, patch_size, feat_dim]` — the DiT's prefix condition for the
    /// NEXT iteration. Zeros on entry to iteration 0 (the text-pad patch);
    /// the previous `pred_feat` on entry to every later iteration.
    pub prefix_feat_cond: Var<R>,
    /// Patches emitted so far, each `[batch, patch_size, feat_dim]`, in
    /// order. Its length is the loop's iteration index `i`; row `b`'s own
    /// patches are the first `patch_len[b]` of them.
    pub patches: Vec<Var<R>>,
    /// Rows in the batch.
    pub batch: usize,
    /// Per-row finish, `None` while the row is still generating.
    pub outcomes: Vec<Option<GenerateOutcome>>,
    /// Per-row emitted patch count; frozen once the row finishes.
    pub patch_len: Vec<usize>,
}

/// The sub-models the per-patch loop touches, borrowed from a
/// [`VoxCpm2Model`] by [`VoxCpm2Model::patch_generator`]. The AudioVAE is
/// deliberately absent: nothing in this unit encodes or decodes audio, so
/// nothing here can accidentally depend on it.
pub struct PatchGenerator<'a, R: Runtime> {
    /// `feat_encoder`, run on ONE patch per row per iteration (`[batch, 1,
    /// patch_size, feat_dim]`).
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
    /// tensors it will meet. Errors when either handoff row is not
    /// `[prefill.batch, hidden]`.
    pub fn start(prefill: PrefillState<R>, config: VoxCpm2Config) -> Result<Self> {
        let batch = prefill.batch;
        if batch == 0 {
            return Err(Error::InvalidArgument {
                arg: "prefill.batch",
                reason: "expected at least 1 row, got 0".to_string(),
            });
        }
        check_row("prefill.lm_hidden", &prefill.lm_hidden, batch)?;
        check_row("prefill.residual_hidden", &prefill.residual_hidden, batch)?;
        let hidden = prefill.lm_hidden.tensor();
        let zeros = Tensor::<R>::zeros(
            &[batch, config.patch_size, config.feat_dim],
            hidden.dtype(),
            hidden.device(),
        )
        .map_err(Error::Numr)?;
        Ok(Self {
            prefill,
            prefix_feat_cond: Var::new(zeros, false),
            patches: Vec::new(),
            batch,
            outcomes: vec![None; batch],
            patch_len: vec![0; batch],
        })
    }

    /// Whether row `row` has finished (stop token or cap).
    pub fn finished(&self, row: usize) -> bool {
        self.outcomes.get(row).is_some_and(Option::is_some)
    }

    /// Whether every row has finished, so the loop has nothing left to do.
    pub fn all_finished(&self) -> bool {
        self.outcomes.iter().all(Option::is_some)
    }

    /// Row `row`'s own patches: the first `patch_len[row]` entries of
    /// [`patches`](Self::patches), each still `[batch, patch_size,
    /// feat_dim]`. Errors on a row outside the batch.
    pub fn row_patches(&self, row: usize) -> Result<&[Var<R>]> {
        let len = *self
            .patch_len
            .get(row)
            .ok_or_else(|| Error::InvalidArgument {
                arg: "row",
                reason: format!("expected a row below {}, got {row}", self.batch),
            })?;
        Ok(&self.patches[..len])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The options constructor's real settings, not the test's cheap ones.
    ///
    /// `n_timesteps` is 10, DELIBERATELY not the reference clone script's 32.
    /// Measured: 32 costs proportionally more compute than 10 and sounds
    /// flatter — more solver steps converge harder toward the mode of
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
        assert!(opts.rows.is_empty());
        assert_eq!(
            opts.row(3),
            RowOptions {
                max_len: 600,
                seed: 0
            }
        );
    }

    /// Per-row entries override the shared cap and seed, and are checked
    /// against the batch and the shared cap.
    #[test]
    fn row_options_override_and_are_checked() {
        let mut opts = GenerateOptions::new(10, 5);
        opts.rows = vec![
            RowOptions {
                max_len: 4,
                seed: 1,
            },
            RowOptions {
                max_len: 10,
                seed: 2,
            },
        ];
        assert_eq!(
            opts.row(0),
            RowOptions {
                max_len: 4,
                seed: 1
            }
        );
        assert_eq!(opts.row(1).seed, 2);
        assert!(opts.check(2).is_ok());
        assert!(opts.check(1).is_err(), "rows must match the batch");
        opts.rows[0].max_len = 11;
        assert!(opts.check(2).is_err(), "a row cap above max_len");
        opts.rows[0].max_len = 0;
        assert!(opts.check(2).is_err(), "a zero row cap");
        assert!(GenerateOptions::new(0, 0).check(1).is_err());
    }
}
