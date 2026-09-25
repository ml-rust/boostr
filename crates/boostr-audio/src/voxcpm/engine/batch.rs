//! The queue between the callers and the engine's owner thread: what a
//! request hands over, how the thread groups queued requests into one batch,
//! and how a row's output flows back.
//!
//! A caller resolves its own request (voice, tokens, budget), sends the
//! result as a [`Job`] and blocks on its reply channel. The owner thread
//! blocks on the queue, takes the first job, keeps draining until
//! [`VoxCpm2SynthOptions::batch_window`] expires or the batch is full, and
//! runs the whole group through one batched generation (`super::run`).
//!
//! A job carries a voice ID, never a borrowed tensor: the reference tensors
//! live in [`EngineInner`] and the owner thread looks them up when it builds
//! the batch, so a queued job borrows nothing.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use boostr::model::audio::voxcpm::model::{RowOptions, VoxCpm2Model};

use crate::error::{Error, Result};
use crate::voxcpm::engine::types::{EngineInner, VoxCpm2RenderClient};

/// Row counts the generation loop runs at, ascending.
///
/// A batch of `n` real rows runs at the smallest width that holds them, so
/// the DiT Euler loop's per-shape graph cache sees at most these four keys
/// and never evicts. Padding to a width costs a dead row a full solve on
/// every step, which is why the list is short and stops at 8 rather than
/// covering every count.
pub(super) const BATCH_WIDTHS: [usize; 4] = [1, 2, 4, 8];

/// One queued request. Holds no borrow, so it can sit in the queue while the
/// caller blocks elsewhere.
pub(super) struct Job {
    /// Voice ID to look up on the owner thread; `None` is zero-shot.
    pub(super) voice: Option<String>,
    /// Text tokens with `AUDIO_START_ID` already appended.
    pub(super) text_token_ids: Vec<u32>,
    /// This row's patch cap.
    pub(super) max_len: usize,
    /// This row's prefill sequence length, reference prefix included.
    pub(super) seq_len: usize,
    /// Where this row's output goes.
    pub(super) reply: Reply,
}

/// Where one job's output goes.
pub(super) enum Reply {
    /// A whole-utterance caller: one message, the samples or the error.
    Buffered(Sender<Result<Vec<f32>>>),
    /// A streaming caller: one message per decoded chunk, then the sender
    /// drops, which is how the caller sees the end of the row. The caller's
    /// sink is not `Send`, so it stays on the caller's thread and the chunks
    /// cross the channel instead.
    Streaming {
        chunks: Sender<Result<Vec<f32>>>,
        /// Set by the caller when its sink failed. The owner thread stops
        /// decoding for this row and retires it.
        cancelled: Arc<AtomicBool>,
    },
}

impl Reply {
    /// The streaming channel and its cancel flag, or `None` for a buffered
    /// reply.
    pub(super) fn stream(&self) -> Option<(&Sender<Result<Vec<f32>>>, &AtomicBool)> {
        match self {
            Self::Buffered(_) => None,
            Self::Streaming { chunks, cancelled } => Some((chunks, cancelled)),
        }
    }

    /// Hand one decoded chunk to a streaming caller. `false` once the caller
    /// has cancelled or dropped its receiver, after which the row is dead
    /// and gets no further chunks.
    pub(super) fn send_chunk(&self, samples: Vec<f32>) -> bool {
        let Some((chunks, cancelled)) = self.stream() else {
            return false;
        };
        !cancelled.load(Ordering::Relaxed) && chunks.send(Ok(samples)).is_ok()
    }

    /// Hand the whole waveform to a buffered caller.
    pub(super) fn send_samples(&self, samples: Vec<f32>) {
        if let Self::Buffered(done) = self {
            let _ = done.send(Ok(samples));
        }
    }

    /// Report a failure on this row. A caller that already went away is not
    /// an error in turn, so the send result is dropped.
    pub(super) fn fail(&self, err: Error) {
        let sender = match self {
            Self::Buffered(done) => done,
            Self::Streaming { chunks, .. } => chunks,
        };
        let _ = sender.send(Err(err));
    }
}

/// The width `rows` real rows run at: the smallest allowed width that holds
/// them all. A caller never asks for more than the clamped cap, so the
/// largest width is the ceiling.
pub(super) fn batch_width(rows: usize) -> usize {
    let largest = BATCH_WIDTHS[BATCH_WIDTHS.len() - 1];
    BATCH_WIDTHS
        .iter()
        .copied()
        .find(|width| *width >= rows)
        .unwrap_or(largest)
}

/// [`VoxCpm2SynthOptions::max_batch`](crate::voxcpm::VoxCpm2SynthOptions::max_batch)
/// rounded DOWN to an allowed width, at least 1. Rounding down keeps a batch
/// from ever needing a width the cap forbids.
pub(super) fn clamp_max_batch(max_batch: usize) -> usize {
    BATCH_WIDTHS
        .iter()
        .copied()
        .filter(|width| *width <= max_batch)
        .max()
        .unwrap_or(1)
}

/// Per-row caps and seeds for a batch of `width` rows whose first
/// `real_max_lens.len()` rows are real.
///
/// Every real row draws from the SAME engine seed: row `b` of a step draws
/// `randn_seeded(seed + i)` over its own `[1, feat_dim, patch_size]` and the
/// rows are stacked, so a row's noise depends on neither the batch width nor
/// its neighbours and a batched row renders bit for bit like its own render
/// alone. A seed derived from arrival order or slot index would break that.
///
/// A slot past the last request holds none, so its entry is a placeholder:
/// the loop needs one entry per slot, and a free slot counts no patch.
pub(super) fn row_options(real_max_lens: &[usize], width: usize, seed: u64) -> Vec<RowOptions> {
    (0..width)
        .map(|b| RowOptions {
            max_len: real_max_lens.get(b).copied().unwrap_or(1),
            seed,
        })
        .collect()
}

/// Take `first`, then keep taking queued jobs until the batch is full, the
/// window expires, or the queue closes.
///
/// The owner thread has already blocked on an empty queue to get `first`, so
/// this is the only wait a request pays beyond its own generation. It ends
/// early on a full batch and on a closed queue, so a lone request on an
/// engine being dropped is not held for the window.
pub(super) fn collect_batch(
    jobs: &Receiver<Job>,
    first: Job,
    window: Duration,
    max_batch: usize,
) -> Vec<Job> {
    let mut batch = Vec::with_capacity(max_batch.max(1));
    batch.push(first);
    let deadline = Instant::now() + window;
    while batch.len() < max_batch {
        let now = Instant::now();
        if now >= deadline {
            break;
        }
        match jobs.recv_timeout(deadline - now) {
            Ok(job) => batch.push(job),
            Err(RecvTimeoutError::Timeout | RecvTimeoutError::Disconnected) => break,
        }
    }
    batch
}

/// Start the owner thread and return the queue into it.
///
/// The thread holds an [`Arc`] of the shared state and nothing else, so it
/// ends exactly when the returned [`Sender`] drops.
pub(super) fn spawn_worker<R>(inner: Arc<EngineInner<R>>) -> Result<(Sender<Job>, JoinHandle<()>)>
where
    R: Runtime<DType = DType> + 'static,
    R::Client: VoxCpm2RenderClient<R>,
    VoxCpm2Model<R>: Send + Sync,
    Tensor<R>: Send + Sync,
{
    let (sender, receiver) = channel::<Job>();
    let handle = std::thread::Builder::new()
        .name("voxcpm2-render".to_string())
        .spawn(move || worker_loop(&inner, &receiver))?;
    Ok((sender, handle))
}

/// Block for a job, batch it with whatever else is queued, run the batch,
/// repeat. Returns when the queue closes, which is what [`Drop`] does.
fn worker_loop<R>(inner: &Arc<EngineInner<R>>, jobs: &Receiver<Job>)
where
    R: Runtime<DType = DType> + 'static,
    R::Client: VoxCpm2RenderClient<R>,
    VoxCpm2Model<R>: Send + Sync,
    Tensor<R>: Send + Sync,
{
    let window = inner.options.batch_window;
    let max_batch = clamp_max_batch(inner.options.max_batch);
    while let Ok(first) = jobs.recv() {
        let batch = collect_batch(jobs, first, window, max_batch);
        inner.run_batch(batch);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job(reply: Reply) -> Job {
        Job {
            voice: None,
            text_token_ids: vec![1, 2],
            max_len: 22,
            seq_len: 2,
            reply,
        }
    }

    fn buffered() -> (Job, std::sync::mpsc::Receiver<Result<Vec<f32>>>) {
        let (tx, rx) = channel();
        (job(Reply::Buffered(tx)), rx)
    }

    fn streaming() -> (
        Job,
        std::sync::mpsc::Receiver<Result<Vec<f32>>>,
        Arc<AtomicBool>,
    ) {
        let (tx, rx) = channel();
        let cancelled = Arc::new(AtomicBool::new(false));
        (
            job(Reply::Streaming {
                chunks: tx,
                cancelled: Arc::clone(&cancelled),
            }),
            rx,
            cancelled,
        )
    }

    #[test]
    fn batch_width_rounds_up_to_an_allowed_width() {
        assert_eq!(batch_width(1), 1);
        assert_eq!(batch_width(2), 2);
        assert_eq!(batch_width(3), 4);
        assert_eq!(batch_width(4), 4);
        assert_eq!(batch_width(5), 8);
        assert_eq!(batch_width(8), 8);
    }

    #[test]
    fn max_batch_rounds_down_to_an_allowed_width() {
        assert_eq!(clamp_max_batch(0), 1);
        assert_eq!(clamp_max_batch(1), 1);
        assert_eq!(clamp_max_batch(3), 2);
        assert_eq!(clamp_max_batch(7), 4);
        assert_eq!(clamp_max_batch(8), 8);
        assert_eq!(clamp_max_batch(64), 8);
    }

    /// A clamped cap never lets a batch need a width above the largest
    /// allowed one.
    #[test]
    fn a_clamped_cap_always_admits_its_own_width() {
        for cap in 0..64 {
            let clamped = clamp_max_batch(cap);
            assert_eq!(batch_width(clamped), clamped, "cap {cap}");
        }
    }

    #[test]
    fn row_options_pad_rows_cap_at_one_patch_and_share_the_seed() {
        let opts = row_options(&[30, 12], 4, 7);
        assert_eq!(opts.len(), 4);
        assert_eq!(
            opts[0],
            RowOptions {
                max_len: 30,
                seed: 7
            }
        );
        assert_eq!(
            opts[1],
            RowOptions {
                max_len: 12,
                seed: 7
            }
        );
        assert_eq!(
            opts[2],
            RowOptions {
                max_len: 1,
                seed: 7
            }
        );
        assert_eq!(
            opts[3],
            RowOptions {
                max_len: 1,
                seed: 7
            }
        );
    }

    /// Row `b` of the options is job `b` of the batch, in queue order.
    #[test]
    fn row_options_keep_job_order() {
        let opts = row_options(&[5, 6, 7], 4, 0);
        let caps: Vec<usize> = opts.iter().map(|r| r.max_len).collect();
        assert_eq!(caps, [5, 6, 7, 1]);
    }

    /// Everything queued inside the window rides in one batch. An empty
    /// queue does not end the drain early: the window exists so a request
    /// that arrives a moment later still joins, so the drain runs it out.
    #[test]
    fn queued_jobs_batch_together() {
        let (tx, rx) = channel();
        for _ in 0..3 {
            tx.send(buffered().0).expect("queue");
        }
        let first = rx.recv().expect("first");
        let window = Duration::from_millis(50);
        let started = Instant::now();
        let batch = collect_batch(&rx, first, window, 8);
        assert_eq!(batch.len(), 3);
        assert!(started.elapsed() >= window, "ended the window early");
    }

    /// A full batch starts at once, whatever is left in the window.
    #[test]
    fn a_full_batch_starts_without_waiting() {
        let (tx, rx) = channel();
        for _ in 0..4 {
            tx.send(buffered().0).expect("queue");
        }
        let first = rx.recv().expect("first");
        let started = Instant::now();
        let batch = collect_batch(&rx, first, Duration::from_secs(30), 2);
        assert_eq!(batch.len(), 2, "the cap holds");
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "waited the window out"
        );
    }

    /// Nothing else can arrive on a closed queue, so a lone job does not
    /// wait the window out.
    #[test]
    fn a_lone_job_on_a_closed_queue_starts_at_once() {
        let (tx, rx) = channel();
        tx.send(buffered().0).expect("queue");
        drop(tx);
        let first = rx.recv().expect("first");
        let started = Instant::now();
        let batch = collect_batch(&rx, first, Duration::from_secs(30), 8);
        assert_eq!(batch.len(), 1);
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "waited the window out"
        );
    }

    /// A cap of one turns batching off and skips the window with it.
    #[test]
    fn a_cap_of_one_never_waits() {
        let (tx, rx) = channel();
        for _ in 0..2 {
            tx.send(buffered().0).expect("queue");
        }
        let first = rx.recv().expect("first");
        let started = Instant::now();
        let batch = collect_batch(&rx, first, Duration::from_secs(30), 1);
        assert_eq!(batch.len(), 1);
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "waited with batching off"
        );
    }

    /// A lone job on an open queue costs the window and no more.
    #[test]
    fn a_lone_job_costs_the_window_and_no_more() {
        let (tx, rx) = channel();
        tx.send(buffered().0).expect("queue");
        let first = rx.recv().expect("first");
        let window = Duration::from_millis(20);
        let started = Instant::now();
        let batch = collect_batch(&rx, first, window, 8);
        let waited = started.elapsed();
        assert_eq!(batch.len(), 1);
        assert!(waited >= window, "returned before the window: {waited:?}");
        assert!(
            waited < Duration::from_secs(5),
            "waited far past the window"
        );
    }

    #[test]
    fn a_cancelled_streaming_row_receives_no_further_chunks() {
        let (job, rx, cancelled) = streaming();
        assert!(job.reply.send_chunk(vec![0.5, 0.25]));
        cancelled.store(true, Ordering::Relaxed);
        assert!(!job.reply.send_chunk(vec![1.0]));
        let first = rx.recv().expect("first chunk").expect("ok");
        assert_eq!(first, [0.5, 0.25]);
        drop(job);
        assert!(rx.recv().is_err(), "a cancelled row sent a second chunk");
    }

    #[test]
    fn a_disconnected_streaming_row_stops_receiving_chunks() {
        let (job, rx, _cancelled) = streaming();
        drop(rx);
        assert!(!job.reply.send_chunk(vec![1.0]));
    }

    #[test]
    fn a_buffered_reply_takes_samples_not_chunks() {
        let (job, rx) = buffered();
        assert!(!job.reply.send_chunk(vec![1.0]));
        job.reply.send_samples(vec![0.5]);
        let got = rx.recv().expect("reply").expect("ok");
        assert_eq!(got, [0.5]);
    }

    #[test]
    fn a_failure_reaches_either_kind_of_caller() {
        let (job, rx) = buffered();
        job.reply.fail(Error::ModelError {
            reason: "boom".into(),
        });
        let err = rx.recv().expect("reply").expect_err("error");
        assert!(err.to_string().contains("boom"), "{err}");

        let (job, rx, _cancelled) = streaming();
        job.reply.fail(Error::ModelError {
            reason: "bang".into(),
        });
        let err = rx.recv().expect("reply").expect_err("error");
        assert!(err.to_string().contains("bang"), "{err}");
    }
}
