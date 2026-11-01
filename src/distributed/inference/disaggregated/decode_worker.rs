//! Decode worker: resumes a request from its KV handoff and streams tokens.

use anyhow::{Result, anyhow};
use nexar::{NexarClient, Rank};
use numr::dtype::DType;
use numr::runtime::Runtime;
use std::sync::Arc;
use std::time::Duration;

use super::cancel::CancelBook;
use super::config::DecodeQueueLimits;
use super::decode_queue::{DecodeQueue, Queued};
use super::forward::{DecodeOpenFn, DecodeStepFn};
use super::handoff::{DecodeIntake, spawn_handoff_listener};
use super::protocol::{Event, KvHandoff, Stage, TokenEvent};
use super::sampler::RequestSampler;
use super::wire::{
    TaskGuard, run_blocking, send_error, send_event, spawn_cancel_listener, spawn_shutdown_listener,
};
use crate::distributed::inference::transport::shutdown_shared;
use crate::ops::traits::inference::SamplingOps;

/// One request in decode: its step closure and its sampler.
pub(crate) struct ActiveDecode<R: Runtime> {
    step: DecodeStepFn<R>,
    sampler: RequestSampler,
}

impl<R> ActiveDecode<R>
where
    R: Runtime<DType = DType>,
    R::Client: SamplingOps<R>,
{
    /// Opens `handoff` through `open`.
    ///
    /// # Errors
    ///
    /// The handoff has no resolved seed, or `open` fails.
    pub(crate) fn open(open: &DecodeOpenFn<R>, handoff: KvHandoff) -> Result<Self> {
        let sampler = RequestSampler::resume(handoff.params, handoff.context, handoff.generated)?;
        let step = open(&handoff.kv_cache).map_err(|e| e.context("opening the KV cache failed"))?;
        Ok(Self { step, sampler })
    }

    /// Feeds the last token and samples the next one.
    pub(crate) fn advance(&mut self, client: &R::Client) -> Result<TokenEvent> {
        let token = self
            .sampler
            .last_token()
            .ok_or_else(|| anyhow!("the decode context is empty"))?;
        let position = self.sampler.last_position();
        let logits = (self.step)(token, position)
            .map_err(|e| e.context(format!("decode step at position {position} failed")))?;
        self.sampler.sample(client, &logits)
    }
}

/// Decode worker: takes KV handoffs from any prefill worker, then runs one
/// request at a time to its finish.
///
/// Handoffs wait in a queue bounded by [`DecodeQueueLimits`]. A handoff
/// above `max_kv_transfer_bytes` or past a queue bound gets a
/// [`Stage::Decode`] error event when it arrives, and never waits.
///
/// Each step feeds the last sampled token, samples the next with the
/// request's seeded sampler and streams it to the router. The stream ends on
/// an EOS id, a stop id or `max_new_tokens`. A cancel from the router ends
/// the request between steps and frees its state without another event.
/// Every error becomes a [`Stage::Decode`] error event, and the loop goes on.
pub struct DecodeWorker<R: Runtime> {
    client: Arc<NexarClient>,
    rank: Rank,
    router_rank: Rank,
    prefill_workers: Vec<Rank>,
    max_kv_transfer_bytes: usize,
    queue_limits: DecodeQueueLimits,
    sampler_client: R::Client,
    open_fn: Arc<DecodeOpenFn<R>>,
}

impl<R> DecodeWorker<R>
where
    R: Runtime<DType = DType>,
    R::Client: SamplingOps<R> + 'static,
{
    /// A decode worker.
    ///
    /// - `prefill_workers`: every rank that can send this worker a handoff.
    /// - `sampler_client`: the runtime client each token samples on. It must
    ///   run on the device the step closures return logits on.
    /// - `open_fn`: opens one request from its serialized cache. See [`DecodeOpenFn`].
    pub fn new(
        client: Arc<NexarClient>,
        rank: Rank,
        router_rank: Rank,
        prefill_workers: Vec<Rank>,
        max_kv_transfer_bytes: usize,
        sampler_client: R::Client,
        open_fn: DecodeOpenFn<R>,
    ) -> Self {
        Self {
            client,
            rank,
            router_rank,
            prefill_workers,
            max_kv_transfer_bytes,
            queue_limits: DecodeQueueLimits::default(),
            sampler_client,
            open_fn: Arc::new(open_fn),
        }
    }

    /// Sets the bounds on the handoffs that wait. Pass
    /// [`super::DisaggConfig::decode_queue`]. Default [`DecodeQueueLimits::default`].
    ///
    /// A `max_requests` of 0 counts as 1, so the queue takes at least one handoff.
    pub fn with_queue_limits(mut self, limits: DecodeQueueLimits) -> Self {
        self.queue_limits = DecodeQueueLimits {
            max_requests: limits.max_requests.max(1),
            ..limits
        };
        self
    }

    pub fn rank(&self) -> Rank {
        self.rank
    }

    /// Shuts down this worker's nexar client. Call it after the run loop
    /// returns, before the process exits.
    ///
    /// # Errors
    ///
    /// The error of [`shutdown_shared`].
    pub async fn shutdown(self, release_within: Duration) -> Result<()> {
        shutdown_shared(self.client, release_within).await
    }

    /// Serves handoffs until the router's
    /// [`tags::SHUTDOWN`](super::protocol::tags::SHUTDOWN) notice, the
    /// router's disconnect, or every prefill worker's disconnect. Call
    /// [`DecodeWorker::shutdown`] after it returns.
    ///
    /// # Errors
    ///
    /// A failed send to the router.
    pub async fn run_loop(&self) -> Result<()> {
        tracing::info!(rank = self.rank, "decode worker loop starting");
        let book = Arc::new(CancelBook::default());
        let _cancels = spawn_cancel_listener(
            Arc::clone(&self.client),
            self.router_rank,
            Arc::clone(&book),
        );
        // The channel holds only handoffs with a queue place, so the queue bounds it.
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let intake = Arc::new(DecodeIntake {
            client: Arc::clone(&self.client),
            rank: self.rank,
            router: self.router_rank,
            queue: DecodeQueue::new(self.queue_limits, self.max_kv_transfer_bytes),
        });
        let _listeners: Vec<TaskGuard> = self
            .prefill_workers
            .iter()
            .map(|&src| spawn_handoff_listener(Arc::clone(&intake), src, tx.clone()))
            .collect();
        drop(tx);
        let (mut stop, _stop) = spawn_shutdown_listener(Arc::clone(&self.client), self.router_rank);
        loop {
            let frame = tokio::select! {
                biased;
                _ = &mut stop => {
                    tracing::info!(rank = self.rank, "decode worker received the shutdown notice");
                    break;
                }
                frame = rx.recv() => match frame {
                    Some(frame) => frame,
                    None => break,
                },
            };
            self.serve(frame, &book).await?;
        }
        tracing::info!(rank = self.rank, "decode worker loop ended");
        Ok(())
    }

    /// Runs one queued handoff. Its queue place frees as it leaves the queue.
    async fn serve(&self, queued: Queued, book: &CancelBook) -> Result<()> {
        let handoff = queued.take();
        let id = handoff.request_id;
        let result = self.decode(handoff, book).await;
        book.finish(id);
        result
    }

    async fn decode(&self, handoff: KvHandoff, book: &CancelBook) -> Result<()> {
        let id = handoff.request_id;
        if book.is_cancelled(id) {
            return Ok(());
        }
        let open_fn = Arc::clone(&self.open_fn);
        let opened = run_blocking(move || ActiveDecode::open(&open_fn, handoff)).await;
        let mut active = match opened.and_then(|r| r) {
            Ok(a) => a,
            Err(e) => return self.fail(id, e).await,
        };
        loop {
            if book.is_cancelled(id) {
                tracing::debug!(rank = self.rank, request_id = id, "decode cancelled");
                return Ok(());
            }
            let client = self.sampler_client.clone();
            let stepped = run_blocking(move || {
                let r = active.advance(&client);
                (active, r)
            })
            .await;
            let (back, result) = match stepped {
                Ok(pair) => pair,
                Err(e) => return self.fail(id, e).await,
            };
            active = back;
            match result {
                Ok(token) => {
                    send_event(&self.client, self.router_rank, &Event::token(id, token)).await?;
                    if token.finish.is_some() {
                        return Ok(());
                    }
                }
                Err(e) => return self.fail(id, e).await,
            }
        }
    }

    async fn fail(&self, id: u64, e: anyhow::Error) -> Result<()> {
        send_error(
            &self.client,
            self.router_rank,
            self.rank,
            Stage::Decode,
            id,
            e,
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::super::protocol::{FinishReason, GenerationParams};
    use super::super::test_harness::{MockDecode, mock_kv};
    use super::*;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    fn handoff(params: GenerationParams) -> KvHandoff {
        let mut params = params;
        params.rule.seed = Some(1);
        KvHandoff {
            request_id: 4,
            params,
            generated: 1,
            context: vec![3, 4, 5, 6],
            kv_cache: mock_kv(&[3, 4, 5]),
        }
    }

    fn run(active: &mut ActiveDecode<CpuRuntime>) -> (Vec<u32>, Option<FinishReason>) {
        let client = CpuClient::new(CpuDevice::new());
        let mut tokens = Vec::new();
        loop {
            let ev = active.advance(&client).unwrap();
            assert_eq!(ev.index as usize, tokens.len() + 1);
            tokens.push(ev.token);
            if ev.finish.is_some() {
                return (tokens, ev.finish);
            }
        }
    }

    #[test]
    fn decode_stops_at_eos_with_its_reason() {
        let mut p = GenerationParams::greedy(50);
        p.eos_tokens = vec![9];
        let open = MockDecode::default().open_fn();
        let mut a = ActiveDecode::open(&open, handoff(p)).unwrap();
        assert_eq!(run(&mut a), (vec![7, 8, 9], Some(FinishReason::Eos)));
    }

    #[test]
    fn decode_stops_at_a_stop_id_and_at_length() {
        let mut p = GenerationParams::greedy(50);
        p.stop_tokens = vec![8];
        let open = MockDecode::default().open_fn();
        let mut a = ActiveDecode::open(&open, handoff(p)).unwrap();
        assert_eq!(run(&mut a), (vec![7, 8], Some(FinishReason::Stop)));
        let mut a = ActiveDecode::open(&open, handoff(GenerationParams::greedy(4))).unwrap();
        assert_eq!(run(&mut a), (vec![7, 8, 9], Some(FinishReason::Length)));
    }

    #[test]
    fn seeded_sampled_decode_is_deterministic() {
        let mut p = GenerationParams::greedy(12);
        p.rule.filter.temperature = 1.0;
        let open = MockDecode {
            flat: true,
            ..MockDecode::default()
        }
        .open_fn();
        let a = run(&mut ActiveDecode::open(&open, handoff(p.clone())).unwrap());
        let b = run(&mut ActiveDecode::open(&open, handoff(p)).unwrap());
        assert_eq!(a, b);
        assert_eq!(a.0.len(), 11);
    }

    /// Decode holds request A at its first step and has one queue place.
    /// B takes the place. C finds the queue full, and D's cache passes the
    /// transfer limit: both fail at once while A still holds decode.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_full_queue_and_an_oversize_cache_fail_on_arrival() {
        use super::super::config::DEFAULT_HANDOFF_TIMEOUT;
        use super::super::router::TokenStream;
        use super::super::test_harness::{Parts, prefill_worker};
        use std::sync::atomic::Ordering;

        const WAIT: Duration = Duration::from_secs(10);
        async fn next(s: &mut TokenStream) -> Result<TokenEvent> {
            tokio::time::timeout(WAIT, s.next())
                .await
                .expect("an item arrives")
                .expect("the stream has an item")
        }

        let parts = Parts::start(DEFAULT_HANDOFF_TIMEOUT).await;
        let (mock, gate) = MockDecode::gated();
        let steps = Arc::clone(&mock.steps);
        let sampler = CpuClient::new(CpuDevice::new());
        // The mock cache is 4 bytes per prompt token: 8 bytes fit two tokens.
        let d = DecodeWorker::new(parts.decode, 2, 0, vec![1], 8, sampler, mock.open_fn())
            .with_queue_limits(DecodeQueueLimits {
                max_requests: 1,
                max_bytes: 1 << 20,
            });
        let p = prefill_worker(parts.prefill, None, DEFAULT_HANDOFF_TIMEOUT);
        let _tasks = [
            TaskGuard::new(tokio::spawn(async move { d.run_loop().await })),
            TaskGuard::new(tokio::spawn(async move { p.run_loop().await })),
        ];
        let router = &parts.router;
        let route = move |prompt: &'static [u32]| {
            router.route_request_stream(prompt, GenerationParams::greedy(3), None)
        };

        let mut a = route(&[1]).await.unwrap();
        assert_eq!(next(&mut a).await.unwrap().token, 2);
        let deadline = tokio::time::Instant::now() + WAIT;
        while steps.load(Ordering::Relaxed) == 0 {
            assert!(
                tokio::time::Instant::now() < deadline,
                "A never reached decode"
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        let mut b = route(&[1]).await.unwrap();
        assert_eq!(next(&mut b).await.unwrap().token, 2);
        let mut c = route(&[1]).await.unwrap();
        assert_eq!(next(&mut c).await.unwrap().token, 2);
        let err = format!("{:#}", next(&mut c).await.unwrap_err());
        assert!(err.contains("decode queue is full"), "{err}");
        let mut d = route(&[1, 2, 3]).await.unwrap();
        assert_eq!(next(&mut d).await.unwrap().token, 4);
        let err = format!("{:#}", next(&mut d).await.unwrap_err());
        assert!(
            err.contains("12 bytes") && err.contains("8-byte limit"),
            "{err}"
        );

        for _ in 0..4 {
            gate.send(()).unwrap();
        }
        for s in [a, b] {
            let g = tokio::time::timeout(WAIT, s.collect())
                .await
                .unwrap()
                .unwrap();
            assert_eq!(g.tokens, [3, 4], "A and B finish after the refusals");
        }
    }

    #[test]
    fn step_and_open_errors_surface() {
        let open = MockDecode {
            fail_at_step: Some(2),
            ..MockDecode::default()
        }
        .open_fn();
        let mut a = ActiveDecode::open(&open, handoff(GenerationParams::greedy(9))).unwrap();
        let client = CpuClient::new(CpuDevice::new());
        a.advance(&client).unwrap();
        let err = a.advance(&client).unwrap_err();
        assert!(format!("{err:#}").contains("mock step failure"), "{err:#}");
        let mut bad = handoff(GenerationParams::greedy(9));
        bad.kv_cache = vec![1, 2, 3];
        let err = ActiveDecode::open(&open, bad).err().expect("bad cache");
        assert!(
            format!("{err:#}").contains("opening the KV cache"),
            "{err:#}"
        );
    }
}
