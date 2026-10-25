//! The disaggregated router: request admission, worker choice and event listeners.

use anyhow::{Result, anyhow, bail};
use nexar::{NexarClient, Rank};
use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use super::super::config::DisaggConfig;
use super::super::protocol::{Event, GenerationParams, PrefillRequest, peek_request_id, tags};
use super::super::wire::{TaskGuard, send_frame};
use super::dispatch::Shared;
use super::stream::{Generation, TokenStream};

/// Router: assigns each request a prefill and a decode worker and streams its tokens back.
///
/// Routing policy:
/// - Prefill: the worker with the fewest requests in prefill.
/// - Decode: the worker a session key last used, else round-robin.
///
/// The router listens to every worker on one task per rank. It reorders each
/// request's events by seq and ends a request with an error when its worker
/// disconnects or nexar reports the worker dead.
pub struct DisaggRouter {
    shared: Arc<Shared>,
    config: DisaggConfig,
    next_request_id: AtomicU64,
    decode_cursor: AtomicU64,
    /// Session key → decode rank. A routing hint only.
    kv_affinity: Mutex<HashMap<String, Rank>>,
    _tasks: Vec<TaskGuard>,
}

impl DisaggRouter {
    /// A router over `client`, which must be the client of `config.router_rank`.
    ///
    /// Spawns the listener tasks on the current tokio runtime.
    ///
    /// # Errors
    ///
    /// A config that fails [`DisaggConfig::check`], a client of another rank,
    /// or no tokio runtime on this thread.
    pub fn new(client: Arc<NexarClient>, config: DisaggConfig) -> Result<Self> {
        config.check(client.world_size())?;
        if client.rank() != config.router_rank {
            bail!(
                "the router client is rank {}, but the config names rank {} as the router",
                client.rank(),
                config.router_rank
            );
        }
        let runtime = tokio::runtime::Handle::try_current()
            .map_err(|_| anyhow!("DisaggRouter::new must run inside a tokio runtime"))?;
        let shared = Arc::new(Shared::new(Arc::clone(&client), &config.prefill_workers));
        let ranks: BTreeSet<Rank> = config
            .prefill_workers
            .iter()
            .chain(&config.decode_workers)
            .copied()
            .collect();
        let mut tasks: Vec<TaskGuard> = ranks
            .into_iter()
            .map(|rank| spawn_event_listener(&runtime, Arc::clone(&shared), rank))
            .collect();
        tasks.push(spawn_failure_watch(&runtime, Arc::clone(&shared)));
        Ok(Self {
            shared,
            config,
            next_request_id: AtomicU64::new(1),
            decode_cursor: AtomicU64::new(0),
            kv_affinity: Mutex::new(HashMap::new()),
            _tasks: tasks,
        })
    }

    fn choose_prefill_worker(&self) -> Rank {
        self.shared
            .prefill_loads
            .iter()
            .min_by_key(|l| l.in_flight.load(Ordering::Relaxed))
            .map_or(self.config.prefill_workers[0], |l| l.rank)
    }

    fn choose_decode_worker(&self, session_key: Option<&str>) -> Rank {
        if let Some(key) = session_key {
            let affinity = self.kv_affinity.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(&rank) = affinity.get(key) {
                return rank;
            }
        }
        let workers = &self.config.decode_workers;
        let idx = self.decode_cursor.fetch_add(1, Ordering::Relaxed) % workers.len() as u64;
        workers[idx as usize]
    }

    /// Routes later requests of `session_key` to `decode_rank`.
    pub fn record_kv_affinity(&self, session_key: String, decode_rank: Rank) {
        self.kv_affinity
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(session_key, decode_rank);
    }

    /// Forgets the decode worker of `session_key`.
    pub fn evict_kv_affinity(&self, session_key: &str) {
        self.kv_affinity
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(session_key);
    }

    /// Starts one request and returns its token stream.
    ///
    /// The stream yields the first token as soon as prefill samples it, then
    /// each decode token. Dropping the stream cancels the request.
    ///
    /// # Errors
    ///
    /// An empty prompt, params that fail [`GenerationParams::check`], or a
    /// failed send to the prefill worker. Later errors arrive on the stream.
    pub async fn route_request_stream(
        &self,
        prompt: &[u32],
        params: GenerationParams,
        session_key: Option<&str>,
    ) -> Result<TokenStream> {
        if prompt.is_empty() {
            bail!("the prompt has no tokens: pass at least one token id");
        }
        params.check()?;
        let decode_rank = self.choose_decode_worker(session_key);
        let prefill_rank = self.choose_prefill_worker();
        let request_id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        tracing::debug!(
            request_id,
            prefill_rank,
            decode_rank,
            prompt_len = prompt.len(),
            "routing request"
        );

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        self.shared
            .register(request_id, tx, prefill_rank, decode_rank);
        let stream = TokenStream {
            request_id,
            rx,
            shared: Arc::clone(&self.shared),
            prefill_rank,
            decode_rank,
            ended: false,
        };
        let sent = match (PrefillRequest {
            request_id,
            decode_rank,
            params,
            prompt: prompt.to_vec(),
        })
        .to_bytes()
        {
            Ok(frame) => {
                send_frame(
                    &self.shared.client,
                    prefill_rank,
                    tags::PREFILL_REQUEST,
                    &frame,
                )
                .await
            }
            Err(e) => Err(e),
        };
        if let Err(e) = sent {
            // `forget` untracks the request, so dropping the stream sends no cancel.
            self.shared.forget(request_id);
            return Err(e.context(format!(
                "request {request_id} not sent to prefill rank {prefill_rank}"
            )));
        }
        if let Some(key) = session_key {
            self.record_kv_affinity(key.to_string(), decode_rank);
        }
        Ok(stream)
    }

    /// Runs one request to its end and returns every token.
    ///
    /// # Errors
    ///
    /// Any error of [`Self::route_request_stream`] or of the stream.
    pub async fn route_request(
        &self,
        prompt: &[u32],
        params: GenerationParams,
        session_key: Option<&str>,
    ) -> Result<Generation> {
        self.route_request_stream(prompt, params, session_key)
            .await?
            .collect()
            .await
    }
}

impl Drop for DisaggRouter {
    fn drop(&mut self) {
        self.shared.fail_all("the router was dropped");
    }
}

fn spawn_event_listener(
    runtime: &tokio::runtime::Handle,
    shared: Arc<Shared>,
    rank: Rank,
) -> TaskGuard {
    let handle = runtime.spawn(async move {
        loop {
            match shared.client.recv_bytes_tagged(rank, tags::EVENT).await {
                Ok(buf) => match Event::from_bytes(&buf) {
                    Ok(event) => shared.deliver(rank, event),
                    Err(e) => match peek_request_id(&buf) {
                        Some(id) => {
                            shared.fail(id, e.context(format!("malformed event from rank {rank}")))
                        }
                        None => tracing::warn!(rank, "dropping a malformed event frame: {e:#}"),
                    },
                },
                Err(e) => {
                    shared.fail_rank(rank, &format!("event channel closed: {e}"));
                    break;
                }
            }
        }
    });
    TaskGuard::new(handle)
}

fn spawn_failure_watch(runtime: &tokio::runtime::Handle, shared: Arc<Shared>) -> TaskGuard {
    let mut watch = shared.client.failure_watch();
    let handle = runtime.spawn(async move {
        while watch.changed().await.is_ok() {
            let dead = watch.borrow_and_update().clone();
            for rank in dead {
                shared.fail_rank(rank, "nexar reports the peer dead");
            }
        }
    });
    TaskGuard::new(handle)
}

#[cfg(test)]
mod tests {
    use super::super::super::protocol::{FinishReason, Stage};
    use super::super::super::test_harness::{Cluster, MockDecode};
    use super::super::dispatch::WorkerError;
    use super::*;
    use std::sync::atomic::Ordering as AtomicOrdering;
    use std::time::Duration;

    fn eos_params(max: u32) -> GenerationParams {
        let mut p = GenerationParams::greedy(max);
        p.eos_tokens = vec![9];
        p
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn prompt_first_token_decode_tokens_eos_end_to_end() {
        let c = Cluster::start(None, MockDecode::default()).await;
        let mut s = c
            .router
            .route_request_stream(&[3, 4, 5], eos_params(50), Some("s1"))
            .await
            .unwrap();
        let mut got = Vec::new();
        while let Some(item) = s.next().await {
            got.push(item.unwrap());
        }
        let tokens: Vec<(u32, u32)> = got.iter().map(|t| (t.index, t.token)).collect();
        assert_eq!(tokens, vec![(0, 6), (1, 7), (2, 8), (3, 9)]);
        assert_eq!(got.last().unwrap().finish, Some(FinishReason::Eos));
        assert!(got[..3].iter().all(|t| t.finish.is_none()));

        let g = c
            .router
            .route_request(&[1], GenerationParams::greedy(3), None)
            .await
            .unwrap();
        assert_eq!(
            g,
            Generation {
                tokens: vec![2, 3, 4],
                finish: FinishReason::Length
            }
        );
        let g = c
            .router
            .route_request(&[1], GenerationParams::greedy(1), None)
            .await
            .unwrap();
        assert_eq!(
            g,
            Generation {
                tokens: vec![2],
                finish: FinishReason::Length
            }
        );
        let mut stop = GenerationParams::greedy(9);
        stop.stop_tokens = vec![4];
        let g = c.router.route_request(&[1], stop, None).await.unwrap();
        assert_eq!(
            g,
            Generation {
                tokens: vec![2, 3, 4],
                finish: FinishReason::Stop
            }
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn seeded_sampling_is_deterministic_across_the_cluster() {
        let c = Cluster::start(
            None,
            MockDecode {
                flat: true,
                ..MockDecode::default()
            },
        )
        .await;
        let mut p = GenerationParams::greedy(10);
        p.rule.filter.temperature = 1.0;
        p.rule.seed = Some(1234);
        let a = c
            .router
            .route_request(&[1, 2], p.clone(), None)
            .await
            .unwrap();
        let b = c
            .router
            .route_request(&[1, 2], p.clone(), None)
            .await
            .unwrap();
        assert_eq!(a, b);
        assert_eq!(a.tokens.len(), 10);
        p.rule.seed = Some(4321);
        assert_ne!(
            c.router
                .route_request(&[1, 2], p, None)
                .await
                .unwrap()
                .tokens,
            a.tokens
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn prefill_and_decode_errors_arrive_as_error_items() {
        let c = Cluster::start(Some("prefill exploded"), MockDecode::default()).await;
        let mut s = c
            .router
            .route_request_stream(&[1], eos_params(5), None)
            .await
            .unwrap();
        let err = s.next().await.unwrap().unwrap_err();
        let w = err.downcast_ref::<WorkerError>().expect("worker error");
        assert_eq!((w.stage, w.rank), (Stage::Prefill, 1));
        assert!(w.message.contains("prefill exploded"), "{}", w.message);
        assert!(s.next().await.is_none());

        let c = Cluster::start(
            None,
            MockDecode {
                fail_at_step: Some(2),
                ..MockDecode::default()
            },
        )
        .await;
        let mut s = c
            .router
            .route_request_stream(&[1], GenerationParams::greedy(50), None)
            .await
            .unwrap();
        assert_eq!(s.next().await.unwrap().unwrap().token, 2);
        assert_eq!(s.next().await.unwrap().unwrap().token, 3);
        let err = s.next().await.unwrap().unwrap_err();
        let w = err.downcast_ref::<WorkerError>().expect("worker error");
        assert_eq!((w.stage, w.rank), (Stage::Decode, 2));
        assert!(w.message.contains("mock step failure"), "{}", w.message);
        assert!(s.next().await.is_none());
        let g = c.router.route_request(&[1], eos_params(50), None).await;
        assert!(g.is_err(), "the mock fails every request at step 2");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn tokens_stream_before_decode_finishes() {
        let (decode, gate) = MockDecode::gated();
        let c = Cluster::start(None, decode).await;
        let mut s = c
            .router
            .route_request_stream(&[1], eos_params(50), None)
            .await
            .unwrap();
        let first = tokio::time::timeout(Duration::from_secs(10), s.next())
            .await
            .unwrap();
        assert_eq!(
            first.unwrap().unwrap().token,
            2,
            "the first token arrives while decode waits"
        );
        for expect in 3..=9 {
            gate.send(()).unwrap();
            let t = tokio::time::timeout(Duration::from_secs(10), s.next())
                .await
                .unwrap();
            assert_eq!(t.unwrap().unwrap().token, expect);
        }
        assert!(s.next().await.is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn dropping_the_stream_stops_the_decode_worker() {
        let decode = MockDecode {
            delay: Some(Duration::from_millis(5)),
            ..MockDecode::default()
        };
        let steps = Arc::clone(&decode.steps);
        let c = Cluster::start(None, decode).await;
        let mut s = c
            .router
            .route_request_stream(&[1], GenerationParams::greedy(100_000), None)
            .await
            .unwrap();
        for _ in 0..3 {
            s.next().await.unwrap().unwrap();
        }
        drop(s);
        let mut last = steps.load(AtomicOrdering::Relaxed);
        loop {
            tokio::time::sleep(Duration::from_millis(200)).await;
            let now = steps.load(AtomicOrdering::Relaxed);
            if now == last {
                break;
            }
            last = now;
        }
        assert!(
            last < 1_000,
            "decode stopped after the cancel, at step {last}"
        );
        let g = c
            .router
            .route_request(&[1], GenerationParams::greedy(3), None)
            .await
            .unwrap();
        assert_eq!(
            g.tokens,
            vec![2, 3, 4],
            "the worker serves the next request"
        );
        assert_eq!(c.router.shared.tracked(), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn invalid_requests_are_refused_before_routing() {
        let c = Cluster::start(None, MockDecode::default()).await;
        assert!(
            c.router
                .route_request_stream(&[], GenerationParams::greedy(4), None)
                .await
                .is_err()
        );
        assert!(
            c.router
                .route_request_stream(&[1], GenerationParams::greedy(0), None)
                .await
                .is_err()
        );
        assert_eq!(c.router.shared.tracked(), 0);
    }
}
