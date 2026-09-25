//! The owner thread's continuous-batching loop: one persistent
//! [`GenerateState`], its slot table, and the admit/step/release cycle.

use std::sync::mpsc::Receiver;

use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use boostr::model::audio::voxcpm::model::{
    GenerateOptions, GenerateState, PrefillRow, RowOptions, VoxCpm2Model,
};

use crate::error::{Error, Result};
use crate::voxcpm::engine::batch::{Job, clamp_max_batch, collect_batch};
use crate::voxcpm::engine::continuous::slots::{Slots, drain};
use crate::voxcpm::engine::run::Row;
use crate::voxcpm::engine::types::{EngineInner, MAX_LEN_CAP, VoxCpm2RenderClient};

/// Run one persistent batch until the queue closes.
///
/// The cycle is: take whatever is queued into free slots, step the batch
/// once, then reply for and release every slot that finished. Only the
/// cold start blocks — when no slot holds a request there is nothing to
/// step, so the thread waits for one.
///
/// A failure to open the batch at all, or a failed step, is not survivable:
/// the state behind it is what every row reads. Every waiting caller is
/// failed and every later request is refused, so no caller is left blocked
/// on a thread that has stopped generating.
pub(in crate::voxcpm::engine) fn run_continuous<R>(inner: &EngineInner<R>, jobs: &Receiver<Job>)
where
    R: Runtime<DType = DType> + 'static,
    R::Client: VoxCpm2RenderClient<R>,
    VoxCpm2Model<R>: Send + Sync,
    Tensor<R>: Send + Sync,
{
    let width = clamp_max_batch(inner.options.max_batch);
    let capacity = match slot_capacity(inner) {
        Ok(capacity) => capacity,
        Err(err) => return refuse_queue(jobs, &err),
    };
    let mut batch = match Batch::open(inner, width, capacity) {
        Ok(batch) => batch,
        Err(err) => return refuse_queue(jobs, &err),
    };

    let mut closed = false;
    loop {
        if !closed {
            closed = batch.admit_queued(inner, jobs);
        }
        if !batch.slots.any_occupied() {
            if closed {
                return;
            }
            match jobs.recv() {
                Ok(first) => batch.cold_start(inner, jobs, first),
                Err(_) => return,
            }
            // Every job of the cold start can have been refused; going
            // round again blocks for the next one rather than stepping an
            // empty batch.
            continue;
        }
        if let Err(err) = batch.step(inner) {
            batch.fail_rows(&err);
            drop(batch);
            return refuse_queue(jobs, &err);
        }
    }
}

/// Cache positions one slot is opened with.
///
/// The budget is
/// [`VoxCpm2SynthOptions::continuous_capacity`](crate::voxcpm::VoxCpm2SynthOptions::continuous_capacity),
/// which a caller sizes to the requests it serves. The model's rope table
/// caps it: a position the table does not hold cannot be rotated, so a
/// longer cache could be filled and never read.
///
/// This is deliberately NOT derived from [`MAX_LEN_CAP`]. That is the
/// ceiling ONE request is allowed to reach, and a persistent batch holds
/// `max_batch` slots of this size in two caches for its whole life, so
/// sizing every slot for the longest request any of them could ever make
/// reserves far more than a working engine uses.
fn slot_capacity<R: Runtime<DType = DType>>(inner: &EngineInner<R>) -> Result<usize> {
    let want = inner.options.continuous_capacity;
    Ok(match inner.model.lane_rope_positions()? {
        Some(positions) => want.min(positions),
        None => want,
    })
}

/// One persistent batch: the state every slot steps in, the rows seated in
/// it, and the per-row options the loop reads.
struct Batch<'a, R: Runtime<DType = DType>> {
    state: GenerateState<R>,
    slots: Slots<Row<'a, R>>,
    options: GenerateOptions,
    /// Cache positions one slot holds — see [`slot_capacity`].
    capacity: usize,
    /// The engine seed every row draws from.
    seed: u64,
}

impl<'a, R> Batch<'a, R>
where
    R: Runtime<DType = DType> + 'static,
    R::Client: VoxCpm2RenderClient<R>,
    VoxCpm2Model<R>: Send + Sync,
    Tensor<R>: Send + Sync,
{
    /// Open an empty batch of `width` slots.
    ///
    /// `width` is the CEILING and is fixed for the batch's whole life: it
    /// bounds the slots, and nothing is seated above it. The device tensors
    /// start at one row and [`GenerateState`] widens them as slots above the
    /// allocation are seated, so an idle batch holds one row rather than
    /// `width` of them. What one step runs over is the prefix of slots that
    /// hold a request, which [`GenerateState`] recomputes as slots are
    /// seated and released. That stepped width is the shape key of the DiT's
    /// Euler graph cache, and it only ever takes one of a short list of
    /// values, so the captures are shared rather than churned.
    fn open(inner: &'a EngineInner<R>, width: usize, capacity: usize) -> Result<Self> {
        let mut options = GenerateOptions::new(MAX_LEN_CAP, inner.options.seed);
        options.cfm.n_timesteps = inner.options.n_timesteps;
        options.cfm.cfg_value = inner.options.cfg_value;
        options.min_len = inner.options.min_len;
        let mut batch = Self {
            state: GenerateState::open(&inner.model, width, capacity)?,
            slots: Slots::new(width),
            options,
            capacity,
            seed: inner.options.seed,
        };
        batch.refresh_options();
        Ok(batch)
    }

    /// Take whatever is queued into the free slots. Returns `true` once the
    /// queue has closed.
    fn admit_queued(&mut self, inner: &'a EngineInner<R>, jobs: &Receiver<Job>) -> bool {
        let (taken, closed) = drain(jobs, self.slots.free_count());
        for job in taken {
            self.seat_next(inner, job);
        }
        closed
    }

    /// Fill an empty batch, starting from the job the thread just blocked
    /// for.
    ///
    /// This is the ONE place the drain window applies. In the steady state a
    /// joiner is taken between two steps and waits for neither the window
    /// nor the batch, so the window costs a request nothing once anything is
    /// running — and a request that arrives alone pays it exactly once,
    /// here.
    fn cold_start(&mut self, inner: &'a EngineInner<R>, jobs: &Receiver<Job>, first: Job) {
        let free = self.slots.free_count();
        for job in collect_batch(jobs, first, inner.options.batch_window, free) {
            self.seat_next(inner, job);
        }
    }

    /// Seat `job` in the lowest free slot.
    fn seat_next(&mut self, inner: &'a EngineInner<R>, job: Job) {
        match self.slots.free() {
            Some(slot) => self.seat(inner, slot, job),
            None => job.reply.fail(Error::ModelError {
                reason: format!(
                    "VoxCPM2 continuous batch has no free slot among its {}; this request \
                     should have stayed queued",
                    self.slots.width()
                ),
            }),
        }
    }

    /// Prefill `job` on its own and seat it in slot `slot`.
    ///
    /// The prefill runs here, on the owner thread, between two steps: it is
    /// a different sequence length from a decode step, and a lane is
    /// installed from a batch of one whose keys start at cache position 0.
    ///
    /// Anything that refuses the request — an unknown voice, a request too
    /// long for a slot, a prefill error — is reported on the job's own reply
    /// and leaves the slot free and the batch untouched.
    fn seat(&mut self, inner: &'a EngineInner<R>, slot: usize, job: Job) {
        // One past the last patch is where the lane table advances to on the
        // final step, so the budget is strictly below the slot's capacity.
        if job.seq_len + job.max_len >= self.capacity {
            job.reply.fail(too_large(&job, self.capacity));
            return;
        }
        let ref_feat = match inner.reference(job.voice.as_deref()) {
            Ok(ref_feat) => ref_feat,
            Err(err) => return job.reply.fail(err),
        };
        let client = inner.client.as_ref();
        // The handoff cache holds this row's prefix and nothing else: its
        // keys are copied into the slot's lane from position 0, so sizing it
        // to the slot's whole capacity would reserve the batch's budget a
        // second time on every admission. A length that does not match the
        // sequence the layout builds is refused by the prefill itself.
        let prefilled = match inner.model.prefill_batch(
            client,
            &[PrefillRow {
                ref_feat,
                text_token_ids: &job.text_token_ids,
            }],
            job.seq_len,
        ) {
            Ok(prefilled) => prefilled,
            Err(err) => return job.reply.fail(Error::from(err)),
        };
        let options = RowOptions {
            max_len: job.max_len,
            seed: self.seed,
        };
        if let Err(err) = self.state.admit(client, slot, &prefilled, options) {
            return job.reply.fail(Error::from(err));
        }
        self.slots.seat(
            slot,
            Row {
                job,
                ref_feat,
                live: true,
                emitted: 0,
            },
        );
        self.refresh_options();
    }

    /// Step the seated slots once, then reply for and release the slots that
    /// finished.
    ///
    /// The step covers the prefix of slots that hold a request, not the
    /// whole allocation, so a lone request in a wide batch costs what one
    /// row costs. The retire walk below still covers every slot: the slot
    /// table and the per-row records are sized by the allocation.
    ///
    /// A released slot's row is DROPPED here, which drops its reply sender:
    /// that is how a streaming caller's drain loop ends, whether the row ran
    /// to its stop token or was retired mid-stream.
    fn step(&mut self, inner: &'a EngineInner<R>) -> Result<()> {
        let client = inner.client.as_ref();
        let chunk = inner.options.stream_chunk_patches.max(1);
        inner
            .model
            .patch_generator()
            .step(client, &mut self.state, &self.options)?;

        let mut released = false;
        for slot in 0..self.slots.width() {
            let retire = {
                let Some(row) = self.slots.get_mut(slot) else {
                    continue;
                };
                inner.pump_row(&mut self.state, slot, row, chunk);
                let done = !row.live || self.state.finished(slot);
                if done {
                    inner.deliver_row(&self.state, slot, row);
                }
                done
            };
            if retire {
                self.slots.take(slot);
                self.state.release(client, slot)?;
                released = true;
            }
        }
        if released {
            self.refresh_options();
        }
        Ok(())
    }

    /// Report `err` on every row still seated and empty the batch.
    fn fail_rows(&mut self, err: &Error) {
        for (_, row) in self.slots.drain_all() {
            if row.live {
                row.job.reply.fail(Error::ModelError {
                    reason: err.to_string(),
                });
            }
        }
    }

    /// Rebuild the per-row caps from the state's own slot record.
    ///
    /// [`GenerateState::slot_options`] is what `admit` and `release` write,
    /// so the loop's caps cannot disagree with the slots they belong to. A
    /// free slot needs an entry all the same — the options carry one per
    /// slot — and counts no patch, so its cap is the smallest legal one.
    fn refresh_options(&mut self) {
        let seed = self.seed;
        self.options.rows = self
            .state
            .slot_options
            .iter()
            .map(|seated| seated.unwrap_or(RowOptions { max_len: 1, seed }))
            .collect();
    }
}

/// Fail every queued and every later request until the queue closes.
///
/// A caller blocks on its reply channel, so a thread that has stopped
/// generating must keep answering rather than stop reading the queue.
fn refuse_queue(jobs: &Receiver<Job>, err: &Error) {
    let reason = format!("VoxCPM2 continuous batch is not running: {err}");
    while let Ok(job) = jobs.recv() {
        job.reply.fail(Error::ModelError {
            reason: reason.clone(),
        });
    }
}

/// A request no slot of a persistent batch can hold.
fn too_large(job: &Job, capacity: usize) -> Error {
    Error::InvalidArgument {
        arg: "text",
        reason: format!(
            "this request needs {} cache positions ({} for its prefill, {} for its patches, \
             1 past the last) but a continuous batch's slot holds {capacity}; shorten the \
             text, or use a shorter reference recording",
            job.seq_len + job.max_len + 1,
            job.seq_len,
            job.max_len,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxcpm::engine::batch::Reply;
    use std::sync::mpsc::channel;

    fn job(seq_len: usize, max_len: usize) -> (Job, std::sync::mpsc::Receiver<Result<Vec<f32>>>) {
        let (tx, rx) = channel();
        (
            Job {
                voice: None,
                text_token_ids: vec![1, 2],
                max_len,
                seq_len,
                reply: Reply::Buffered(tx),
            },
            rx,
        )
    }

    /// The default slot size holds an ordinary clone request, spelled out:
    /// a reference recording of a few hundred patches, a sentence of around
    /// a hundred tokens, and the patches that sentence generates.
    ///
    /// This is the derivation behind the default, run rather than asserted
    /// in prose — a default that no longer holds an ordinary request would
    /// refuse one at the engine's door.
    #[test]
    fn the_default_slot_holds_an_ordinary_request() {
        use crate::voxcpm::options::VoxCpm2SynthOptions;
        let (t_ref, text_len) = (256usize, 96usize);
        let prefill = t_ref + 2 + text_len;
        let patches = text_len * 6 + 10;
        let capacity = VoxCpm2SynthOptions::default().continuous_capacity;
        assert!(
            prefill + patches < capacity,
            "prefill {prefill} + patches {patches} must leave the {capacity}-position slot \
             the one position the lane table advances past the last patch"
        );
    }

    /// A request whose prefill plus patches fills the slot exactly is one
    /// position too long: the lane table advances once past the last patch.
    #[test]
    fn a_request_that_cannot_fit_a_slot_is_refused_with_its_own_numbers() {
        let (job, rx) = job(100, 28);
        job.reply.fail(too_large(&job, 128));
        let err = rx.recv().expect("reply").expect_err("refused");
        let text = err.to_string();
        assert!(text.contains("129"), "{text}");
        assert!(text.contains("100"), "{text}");
        assert!(text.contains("128"), "{text}");
    }

    /// The refusal reaches the caller rather than leaving it queued.
    #[test]
    fn a_refused_queue_answers_every_waiting_caller() {
        let (tx, rx) = channel();
        let (first, first_reply) = job(4, 4);
        let (second, second_reply) = job(4, 4);
        tx.send(first).expect("queue");
        tx.send(second).expect("queue");
        drop(tx);

        refuse_queue(
            &rx,
            &Error::ModelError {
                reason: "boom".into(),
            },
        );
        for reply in [first_reply, second_reply] {
            let err = reply.recv().expect("reply").expect_err("refused");
            assert!(err.to_string().contains("boom"), "{err}");
        }
    }
}
