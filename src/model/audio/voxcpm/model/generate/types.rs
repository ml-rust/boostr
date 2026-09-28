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
/// duplicated here. Build one with [`GenerateOptions::new`]. There is no
/// `Default`: the KV caches the prefill allocated bound `max_len`.
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
    /// script's 32. The deviation is deliberate. More solver steps converge
    /// harder toward the mode of the flow and smooth away prosodic variation,
    /// so 32 costs more compute and sounds flatter.
    ///
    /// For generation variance at 10, reach for `--best-of`, not a higher step
    /// count.
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
/// Every STEPPED row runs on every iteration, a finished one included. Its
/// new patches are computed and discarded, which keeps the shared caches
/// aligned without a compaction pass. [`patch_len`](Self::patch_len) is the
/// per-row count that stops advancing at the finish.
///
/// Three widths, none interchangeable.
///
/// | Width | Counts | Moves |
/// | --- | --- | --- |
/// | `batch` | CEILING: slots this state can ever hold | never |
/// | `alloc_width` | ALLOCATION: rows every device tensor and both caches hold | up only |
/// | `active_width` | STEPPED: the prefix one iteration runs | up and down |
///
/// `batch` is what a slot index and the options' `rows` list are checked
/// against. It is the length of every HOST per-row vector: `patches`,
/// `context`, `outcomes`, `patch_len` and `slot_options`. They are sized at the ceiling
/// once, so a slot index always addresses them.
///
/// `alloc_width` is what the DEVICE tensors hold: both KV caches,
/// `lm_hidden`, `residual_hidden`, `prefix_feat_cond` and the lane table. It
/// starts at one row and grows with the stepped width, never past the ceiling
/// and never back down. Never read a device tensor's row count off `batch`.
///
/// `active_width` is the prefix of slots holding a request, rounded up to a
/// width the DiT's Euler graph cache already has a capture for. No step
/// touches the rows at or above it.
pub struct GenerateState<R: Runtime> {
    /// The prefill handoff, driven forward. `position` here is the ONE
    /// counter both caches follow.
    pub prefill: PrefillState<R>,
    /// `[batch, patch_size, feat_dim]` — the DiT's prefix condition for the
    /// NEXT iteration. On entry to iteration 0 it is the row's last raw
    /// prefill patch: zeros (the text-pad patch) without a continuation
    /// tail, the prompt's last patch with one. On entry to every later
    /// iteration it is the previous `pred_feat`.
    pub prefix_feat_cond: Var<R>,
    /// Per row, the continuation decode context
    /// ([`PrefillState::context`]): audio patches the decode prepends to the
    /// row's own patches and trims back off the waveform. Empty for a row
    /// without a continuation tail. Never counted in `patch_len`, never
    /// seen by the stop guard or the noise index.
    pub context: Vec<Vec<Var<R>>>,
    /// Patches emitted so far, per row: `patches[b]` holds row `b`'s own
    /// patches in order, each `[1, patch_size, feat_dim]`. Every row is
    /// pushed a patch on every iteration, so each inner vector is as long as
    /// the loop has run; the first `patch_len[b]` of them are the ones row
    /// `b` counts. Read a row's own patches with
    /// [`row_patches`](Self::row_patches).
    pub patches: Vec<Vec<Var<R>>>,
    /// Slots this state can ever hold — the CEILING, fixed for its life.
    ///
    /// A slot index, [`GenerateOptions::rows`] and every HOST per-row
    /// vector on this state are sized and checked against it. The DEVICE
    /// tensors are NOT: they hold [`alloc_width`](Self::alloc_width()) rows,
    /// which is this or less.
    pub batch: usize,
    /// Rows ALLOCATED: what both KV caches, both handoff rows,
    /// `prefix_feat_cond` and the lane table were built at.
    ///
    /// A state opens at one row whatever its ceiling, and grows to cover a
    /// slot before anything is seated into it. Growth rounds up to the same
    /// allowlist the stepped width uses and stops at the ceiling. It NEVER
    /// shrinks.
    ///
    /// Read it from outside this module through the `alloc_width` method.
    pub(crate) alloc_width: usize,
    /// Slots STEPPED: the prefix `0..active_width` one iteration runs over.
    ///
    /// Slots are seated lowest-free-first, so the occupied slots are a prefix
    /// and the rows at or above this width hold no request. Those rows are
    /// not solved, encoded or decoded. The allocation is what the state COULD
    /// hold; this is what it holds now, rounded up to a width the DiT's Euler
    /// graph cache already has a capture for.
    ///
    /// Recomputed by [`admit`](Self::admit) and [`release`](Self::release).
    /// Read it from outside this module through the `active_width` method.
    pub(crate) active_width: usize,
    /// Per-row finish, `None` while the row is still generating.
    pub outcomes: Vec<Option<GenerateOutcome>>,
    /// Per-row emitted patch count; frozen once the row finishes.
    pub patch_len: Vec<usize>,
    /// Row `b`'s cap and seed as [`admit`](Self::admit) seated them, `None`
    /// for a slot no request has been seated into.
    pub slot_options: Vec<Option<RowOptions>>,
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
    /// `prefix_feat_cond` and `context` are TAKEN from the prefill. Its
    /// condition is each row's last raw prefill patch: zeros without a
    /// continuation tail (the reference conditions patch 0 on zeros, not on
    /// the reference audio's tail), the prompt's last patch with one. A
    /// prefill carrying no condition conditions every row on zeros, with
    /// dtype and device from `prefill.lm_hidden`.
    ///
    /// Errors when either handoff row is not `[prefill.batch, hidden]`, when
    /// the condition is not `[prefill.batch, patch_size, feat_dim]`, or when
    /// `context` is neither empty nor one entry per row.
    pub fn start(mut prefill: PrefillState<R>, config: VoxCpm2Config) -> Result<Self> {
        let batch = prefill.batch;
        if batch == 0 {
            return Err(Error::InvalidArgument {
                arg: "prefill.batch",
                reason: "expected at least 1 row, got 0".to_string(),
            });
        }
        check_row("prefill.lm_hidden", &prefill.lm_hidden, batch)?;
        check_row("prefill.residual_hidden", &prefill.residual_hidden, batch)?;
        let prefix_feat_cond = match prefill.prefix_feat_cond.take() {
            Some(cond) => {
                check_patch(
                    "prefill.prefix_feat_cond",
                    &cond,
                    &[batch, config.patch_size, config.feat_dim],
                )?;
                cond
            }
            None => {
                let hidden = prefill.lm_hidden.tensor();
                Var::new(
                    zero_patches(batch, config, hidden.dtype(), hidden.device())?,
                    false,
                )
            }
        };
        let mut context = std::mem::take(&mut prefill.context);
        if context.is_empty() {
            context = (0..batch).map(|_| Vec::new()).collect();
        } else if context.len() != batch {
            return Err(Error::InvalidArgument {
                arg: "prefill.context",
                reason: format!(
                    "expected one entry per prefilled row ({batch}) or none, got {}; \
                     pass the PrefillState a prefill entry point returned",
                    context.len()
                ),
            });
        }
        // Every row of a prefilled batch holds a request, so the stepped
        // width IS the allocation and there is nothing to narrow. The slot
        // path recomputes the width on every seat and release.
        Ok(Self {
            prefill,
            prefix_feat_cond,
            context,
            patches: (0..batch).map(|_| Vec::new()).collect(),
            batch,
            alloc_width: batch,
            active_width: batch,
            outcomes: vec![None; batch],
            patch_len: vec![0; batch],
            slot_options: vec![None; batch],
        })
    }

    /// Slots the loop STEPS: the prefix `0..active_width()` one iteration
    /// runs over. [`batch`](Self::batch) is the ceiling instead, and is what
    /// a slot index is checked against.
    pub fn active_width(&self) -> usize {
        self.active_width
    }

    /// Rows the device tensors hold — see [`alloc_width`](Self::alloc_width()).
    /// At least [`active_width`](Self::active_width()), at most
    /// [`batch`](Self::batch).
    pub fn alloc_width(&self) -> usize {
        self.alloc_width
    }

    /// Whether slot `row` holds a request.
    ///
    /// The lane table is the one record of that. It decides the slot's key
    /// range, so nothing here disagrees with what the kernels read.
    ///
    /// A slot at or above the allocation holds nothing by construction.
    /// Seating one grows the allocation to cover it first.
    pub fn occupied(&self, row: usize) -> bool {
        row < self.alloc_width && !self.prefill.lanes.is_free(row)
    }

    /// Whether row `row` has finished (stop token or cap). A free slot has
    /// nothing to generate and always counts as finished.
    pub fn finished(&self, row: usize) -> bool {
        !self.occupied(row) || self.outcomes.get(row).is_some_and(Option::is_some)
    }

    /// Whether every row has finished, so the loop has nothing left to do.
    pub fn all_finished(&self) -> bool {
        (0..self.batch).all(|row| self.finished(row))
    }

    /// Row `row`'s own patches: the first `patch_len[row]` entries of
    /// `patches[row]`, each `[1, patch_size, feat_dim]`. Errors on a row
    /// outside the batch.
    pub fn row_patches(&self, row: usize) -> Result<&[Var<R>]> {
        let bad_row = || Error::InvalidArgument {
            arg: "row",
            reason: format!("expected a row below {}, got {row}", self.batch),
        };
        let len = *self.patch_len.get(row).ok_or_else(bad_row)?;
        let own = self.patches.get(row).ok_or_else(bad_row)?;
        Ok(&own[..len])
    }

    /// Iterations the loop has run. Every row is pushed a patch on every
    /// iteration, finished rows included, so this is any row's stored patch
    /// count and is at least every `patch_len` entry.
    pub fn emitted(&self) -> usize {
        self.patches.first().map_or(0, Vec::len)
    }
}

/// `[rows, patch_size, feat_dim]` of zeros: the text-pad patch, the
/// iteration-0 condition of a row without a continuation tail, and the
/// fallback when a handoff carries no condition.
///
/// The ONE place that shape is built, so a slot seated by
/// [`GenerateState::admit`] is conditioned on exactly what a row of
/// [`GenerateState::start`] is.
pub(super) fn zero_patches<R: Runtime<DType = DType>>(
    rows: usize,
    config: VoxCpm2Config,
    dtype: DType,
    device: &R::Device,
) -> Result<Tensor<R>> {
    Tensor::<R>::zeros(&[rows, config.patch_size, config.feat_dim], dtype, device)
        .map_err(Error::Numr)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The options constructor's real settings, not the test's cheap ones.
    ///
    /// `n_timesteps` is 10, DELIBERATELY not the reference clone script's 32.
    /// See `GenerateOptions::new`. Restoring 32 here regresses both speed and
    /// quality. The parity gate `examples/voxcpm_step_check.rs` pins 32 on
    /// purpose, because its Python fixtures were generated at that count.
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
