//! What the owner thread does with one batch: resolve each row's reference
//! voice, prefill the group at once, drive the per-patch loop, and hand each
//! row its audio.
//!
//! Every row of a batch steps on every iteration; a finished row's patches
//! are computed and its bookkeeping stands still. That is what lets a row be
//! retired mid-batch — a cancelled streaming row is marked finished and runs
//! dead to the end, exactly as a row that stopped early does.
//!
//! A row's noise draw depends on neither the batch width nor its neighbours,
//! so a batched row is bit for bit the render it would have had alone.

use std::sync::atomic::Ordering;

use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use boostr::model::audio::voxcpm::model::config::AUDIO_START_ID;
use boostr::model::audio::voxcpm::model::{
    GenerateOptions, GenerateOutcome, GenerateState, PrefillRow, StepOutcome, VoxCpm2Model,
};

use crate::error::{Error, Result};
use crate::voxcpm::engine::batch::{Job, batch_width, row_options};
use crate::voxcpm::engine::types::{EngineInner, VoxCpm2RenderClient};

/// The prompt a pad row prefills: the audio-start token alone, zero-shot.
/// The shortest sequence `prefill_batch` accepts, so a pad never widens
/// `S_max`, and capped at one patch so it retires after the first step.
const PAD_TOKENS: [u32; 1] = [AUDIO_START_ID];

/// One row of a running batch: the request, its reference tensor, and
/// whether the caller is still taking output.
struct Row<'a, R: Runtime> {
    job: Job,
    ref_feat: Option<&'a Tensor<R>>,
    /// Cleared once the row's caller went away or its own decode failed.
    /// A dead row keeps stepping; nothing is decoded or sent for it again.
    live: bool,
    /// Patches already handed to a streaming caller.
    emitted: usize,
}

impl<R> EngineInner<R>
where
    R: Runtime<DType = DType> + 'static,
    R::Client: VoxCpm2RenderClient<R>,
    VoxCpm2Model<R>: Send + Sync,
    Tensor<R>: Send + Sync,
{
    /// Generate every job of one batch together and reply to each.
    ///
    /// Never propagates: this is the owner thread's top level, so every
    /// failure leaves through the rows' reply channels.
    pub(super) fn run_batch(&self, jobs: Vec<Job>) {
        let mut rows = Vec::with_capacity(jobs.len());
        for job in jobs {
            match self.reference(job.voice.as_deref()) {
                Ok(ref_feat) => rows.push(Row {
                    job,
                    ref_feat,
                    live: true,
                    emitted: 0,
                }),
                Err(err) => job.reply.fail(err),
            }
        }
        if rows.is_empty() {
            return;
        }
        if let Err(err) = self.generate_batch(&mut rows) {
            fail_all(&rows, err);
        }
    }

    /// A voice ID's encoded reference. `None` in, `None` out: that is
    /// zero-shot. The voice table is fixed at load, so an unknown ID here
    /// means the caller's own check and this one disagree.
    fn reference(&self, voice: Option<&str>) -> Result<Option<&Tensor<R>>> {
        let Some(voice_id) = voice else {
            return Ok(None);
        };
        let voice = self
            .voices
            .get(voice_id)
            .ok_or_else(|| Error::InvalidArgument {
                arg: "voice",
                reason: format!("unknown voice {voice_id:?}"),
            })?;
        Ok(Some(&voice.ref_feat))
    }

    /// Prefill, generate and decode the whole batch. An error here belongs
    /// to every row; a per-row failure is reported on that row and does not
    /// return.
    fn generate_batch(&self, rows: &mut [Row<'_, R>]) -> Result<()> {
        let width = batch_width(rows.len());
        let mut prefill_rows: Vec<PrefillRow<'_, R>> = rows
            .iter()
            .map(|row| PrefillRow {
                ref_feat: row.ref_feat,
                text_token_ids: &row.job.text_token_ids,
            })
            .collect();
        prefill_rows.resize(
            width,
            PrefillRow {
                ref_feat: None,
                text_token_ids: &PAD_TOKENS,
            },
        );

        let real_max_lens: Vec<usize> = rows.iter().map(|row| row.job.max_len).collect();
        let max_len = real_max_lens.iter().copied().max().unwrap_or(1);
        let s_max = rows
            .iter()
            .map(|row| row.job.seq_len)
            .chain(std::iter::once(PAD_TOKENS.len()))
            .max()
            .unwrap_or(PAD_TOKENS.len());

        let mut options = GenerateOptions::new(max_len, self.options.seed);
        options.cfm.n_timesteps = self.options.n_timesteps;
        options.cfm.cfg_value = self.options.cfg_value;
        options.min_len = self.options.min_len;
        options.rows = row_options(&real_max_lens, width, self.options.seed);

        let client = self.client.as_ref();
        let prefill = self
            .model
            .prefill_batch(client, &prefill_rows, s_max + max_len)?;
        let mut state = GenerateState::start(prefill, self.model.config)?;

        self.drive(&mut state, &options, rows)?;
        self.deliver(&state, rows);
        Ok(())
    }

    /// Step the batch to the end, emitting streaming chunks as they become
    /// decodable and retiring rows whose callers went away.
    fn drive(
        &self,
        state: &mut GenerateState<R>,
        options: &GenerateOptions,
        rows: &mut [Row<'_, R>],
    ) -> Result<()> {
        let chunk = self.options.stream_chunk_patches.max(1);
        let client = self.client.as_ref();
        let generator = self.model.patch_generator();
        while !state.all_finished() {
            let outcome = generator.step(client, state, options)?;
            for (b, row) in rows.iter_mut().enumerate() {
                if !row.live {
                    continue;
                }
                let Some((_, cancelled)) = row.job.reply.stream() else {
                    continue;
                };
                if cancelled.load(Ordering::Relaxed) {
                    retire(state, b);
                    row.live = false;
                    continue;
                }
                if state.patch_len[b] - row.emitted < chunk {
                    continue;
                }
                match host_samples(self.model.decode_row_from(client, state, b, row.emitted)) {
                    Ok(samples) => {
                        row.emitted = state.patch_len[b];
                        if !row.job.reply.send_chunk(samples) {
                            retire(state, b);
                            row.live = false;
                        }
                    }
                    Err(err) => fail_row(state, b, row, err),
                }
            }
            if outcome == StepOutcome::Stopped {
                break;
            }
        }
        Ok(())
    }

    /// Final reply per row: the whole waveform for a buffered caller, the
    /// tail chunk for a streaming one. A streaming caller sees the end when
    /// its sender drops with the batch.
    fn deliver(&self, state: &GenerateState<R>, rows: &mut [Row<'_, R>]) {
        let client = self.client.as_ref();
        for (b, row) in rows.iter_mut().enumerate() {
            if !row.live {
                continue;
            }
            row.live = false;
            let streaming = row.job.reply.stream().is_some();
            let decoded = if streaming {
                if row.emitted >= state.patch_len[b] {
                    continue;
                }
                self.model.decode_row_from(client, state, b, row.emitted)
            } else {
                self.model.decode_row(client, state, b)
            };
            match host_samples(decoded) {
                Ok(samples) if streaming => {
                    row.emitted = state.patch_len[b];
                    row.job.reply.send_chunk(samples);
                }
                Ok(samples) => row.job.reply.send_samples(samples),
                Err(err) => row.job.reply.fail(err),
            }
        }
    }
}

/// A decoded `[1, 1, samples]` tensor as host floats.
fn host_samples<R: Runtime<DType = DType>>(
    decoded: std::result::Result<Tensor<R>, boostr::error::Error>,
) -> Result<Vec<f32>> {
    let decoded = decoded?;
    Ok(decoded.contiguous()?.to_vec())
}

/// Mark row `b` finished so the loop stops advancing its bookkeeping. The
/// row's patches are still computed on every remaining step — nothing skips
/// a row — but nothing reads them again.
fn retire<R: Runtime>(state: &mut GenerateState<R>, b: usize) {
    if let Some(outcome) = state.outcomes.get_mut(b) {
        outcome.get_or_insert(GenerateOutcome::MaxLen);
    }
}

/// Report a per-row failure and retire the row; the rest of the batch runs on.
fn fail_row<R: Runtime>(state: &mut GenerateState<R>, b: usize, row: &mut Row<'_, R>, err: Error) {
    row.job.reply.fail(err);
    retire(state, b);
    row.live = false;
}

/// Report one batch-wide failure to every row still waiting.
///
/// The error moves to the first live row unchanged, so a batch of one keeps
/// the exact error a single render returns today; the rest carry its text.
fn fail_all<R: Runtime>(rows: &[Row<'_, R>], err: Error) {
    let text = err.to_string();
    let mut original = Some(err);
    for row in rows.iter().filter(|row| row.live) {
        match original.take() {
            Some(err) => row.job.reply.fail(err),
            None => row.job.reply.fail(Error::ModelError {
                reason: text.clone(),
            }),
        }
    }
}
