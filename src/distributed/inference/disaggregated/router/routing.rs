//! The disaggregated router: request admission, worker choice and event listeners.

use anyhow::{Result, anyhow, bail};
use nexar::{NexarClient, Rank};
use std::collections::{BTreeSet, HashMap};
use std::hash::{BuildHasher, RandomState};
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
/// - Prefill: the live worker with the fewest requests in prefill.
/// - Decode: the live worker a session key last used, else round-robin over
///   the live workers. A session pinned to a lost worker moves to a live one.
///
/// The router listens to every worker on one task per rank. It holds a
/// request's decode events until its first prefill event arrives. It ends a
/// request with an error when a worker it waits on disconnects or nexar
/// reports it dead, and when decode does not confirm the KV handoff within
/// [`DisaggConfig::handoff_timeout`]. A worker lost once stays lost: see
/// [`Self::lost_workers`], [`Self::is_degraded`] and [`Self::fault`].
///
/// Request ids start at a random 64-bit origin drawn when the router starts,
/// then count up. A restarted router therefore reuses no id that a
/// long-running worker still remembers from the previous router.
pub struct DisaggRouter {
    pub(super) shared: Arc<Shared>,
    pub(super) config: DisaggConfig,
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
        let shared = Arc::new(Shared::new(
            Arc::clone(&client),
            &config.prefill_workers,
            config.handoff_timeout,
        ));
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
            next_request_id: AtomicU64::new(request_id_origin()),
            decode_cursor: AtomicU64::new(0),
            kv_affinity: Mutex::new(HashMap::new()),
            _tasks: tasks,
        })
    }

    /// The live prefill worker with the fewest requests in prefill.
    ///
    /// # Errors
    ///
    /// Every prefill worker is lost.
    pub(super) fn choose_prefill_worker(&self) -> Result<Rank> {
        let lost = self.shared.lost_ranks();
        self.shared
            .prefill_loads
            .iter()
            .filter(|l| !lost.contains(&l.rank))
            .min_by_key(|l| l.in_flight.load(Ordering::Relaxed))
            .map(|l| l.rank)
            .ok_or_else(|| {
                anyhow!(
                    "no live prefill worker: the router lost every prefill rank {:?}. Restart \
                     the cluster",
                    self.config.prefill_workers
                )
            })
    }

    /// The live decode worker `session_key` last used, else the next live
    /// worker in round-robin order. A pin to a lost worker is dropped.
    ///
    /// # Errors
    ///
    /// Every decode worker is lost.
    pub(super) fn choose_decode_worker(&self, session_key: Option<&str>) -> Result<Rank> {
        let lost = self.shared.lost_ranks();
        if let Some(key) = session_key {
            let mut affinity = self.kv_affinity.lock().unwrap_or_else(|e| e.into_inner());
            match affinity.get(key) {
                Some(rank) if !lost.contains(rank) => return Ok(*rank),
                Some(_) => {
                    affinity.remove(key);
                }
                None => {}
            }
        }
        let live: Vec<Rank> = self
            .config
            .decode_workers
            .iter()
            .copied()
            .filter(|r| !lost.contains(r))
            .collect();
        if live.is_empty() {
            bail!(
                "no live decode worker: the router lost every decode rank {:?}. Restart the cluster",
                self.config.decode_workers
            );
        }
        let idx = self.decode_cursor.fetch_add(1, Ordering::Relaxed) % live.len() as u64;
        Ok(live[idx as usize])
    }

    #[cfg(test)]
    pub(super) fn kv_affinity_of(&self, session_key: &str) -> Option<Rank> {
        self.kv_affinity
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(session_key)
            .copied()
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
    /// An empty prompt, params that fail [`GenerationParams::check`], no live
    /// prefill or no live decode worker, or a failed send to the prefill
    /// worker. Later errors arrive on the stream.
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
        let decode_rank = self.choose_decode_worker(session_key)?;
        let prefill_rank = self.choose_prefill_worker()?;
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
            .register(request_id, tx, prefill_rank, decode_rank)?;
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

/// The first request id of a new router: 64 random bits from the OS-seeded
/// keys of [`RandomState`], mixed with the clock and the process id.
///
/// Ids then count up from it with wrapping. Two routers collide only when
/// their id ranges overlap, which for `n` requests each has a chance near
/// `2n / 2^64`. A seeded numr generator never feeds it, so a fixed
/// `numr::random` seed cannot replay the ids of an earlier router.
fn request_id_origin() -> u64 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    RandomState::new().hash_one((nanos, std::process::id()))
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
    use super::super::pending::WorkerError;
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

    #[test]
    fn each_router_draws_a_fresh_request_id_origin() {
        let origins: BTreeSet<u64> = (0..64).map(|_| request_id_origin()).collect();
        assert_eq!(origins.len(), 64, "no two origins repeat");
    }

    /// A restarted router must not reuse the ids of the router before it: a
    /// long-running worker drops a cancel of an id it already finished.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_restarted_router_reuses_no_request_id() {
        let mut ids = BTreeSet::new();
        for _ in 0..2 {
            let c = Cluster::start(None, MockDecode::default()).await;
            for _ in 0..3 {
                let s = c
                    .router
                    .route_request_stream(&[1], GenerationParams::greedy(2), None)
                    .await
                    .unwrap();
                assert!(ids.insert(s.request_id()), "id {} repeats", s.request_id());
                s.collect().await.unwrap();
            }
        }
        assert!(!ids.contains(&1), "ids no longer start at 1");
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
