//! Router state shared by its listeners and its streams: in-flight requests,
//! event reordering and prefill load.

use anyhow::anyhow;
use nexar::{NexarClient, Rank};
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use tokio::sync::mpsc::UnboundedSender;

use super::super::protocol::{Event, EventBody, Stage, TokenEvent};

/// The item a request stream yields.
pub(crate) type StreamItem = anyhow::Result<TokenEvent>;

/// A worker reported that it could not finish a request.
///
/// It reaches the caller inside the `anyhow::Error` of a stream item. Use
/// `err.downcast_ref::<WorkerError>()` to read it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerError {
    pub request_id: u64,
    pub stage: Stage,
    pub rank: Rank,
    pub message: String,
}

impl fmt::Display for WorkerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let stage = match self.stage {
            Stage::Prefill => "prefill",
            Stage::Decode => "decode",
        };
        write!(
            f,
            "{stage} worker rank {} failed request {}: {}",
            self.rank, self.request_id, self.message
        )
    }
}

impl std::error::Error for WorkerError {}

/// Per-prefill-worker count of requests in prefill.
pub(crate) struct PrefillLoad {
    pub(crate) rank: Rank,
    pub(crate) in_flight: AtomicU64,
}

/// One in-flight request.
pub(crate) struct Pending {
    tx: UnboundedSender<StreamItem>,
    prefill_rank: Rank,
    decode_rank: Rank,
    /// The seq the stream yields next.
    next_seq: u32,
    /// Events that arrived ahead of `next_seq`, with their source rank.
    early: BTreeMap<u32, (Rank, EventBody)>,
    /// The request still counts against its prefill worker's load.
    in_prefill: bool,
}

/// State the router, its listener tasks and its streams share.
pub(crate) struct Shared {
    pub(crate) client: Arc<NexarClient>,
    pub(crate) prefill_loads: Vec<PrefillLoad>,
    pending: Mutex<HashMap<u64, Pending>>,
}

impl Shared {
    pub(crate) fn new(client: Arc<NexarClient>, prefill_workers: &[Rank]) -> Self {
        Self {
            client,
            prefill_loads: prefill_workers
                .iter()
                .map(|&rank| PrefillLoad {
                    rank,
                    in_flight: AtomicU64::new(0),
                })
                .collect(),
            pending: Mutex::new(HashMap::new()),
        }
    }

    fn lock(&self) -> MutexGuard<'_, HashMap<u64, Pending>> {
        // A panic elsewhere leaves the map consistent: every update is one insert or remove.
        self.pending.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn release_prefill(&self, p: &mut Pending) {
        if !std::mem::take(&mut p.in_prefill) {
            return;
        }
        if let Some(load) = self.prefill_loads.iter().find(|l| l.rank == p.prefill_rank) {
            load.in_flight.fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// Tracks request `id`, counting it against `prefill_rank`.
    pub(crate) fn register(
        &self,
        id: u64,
        tx: UnboundedSender<StreamItem>,
        prefill_rank: Rank,
        decode_rank: Rank,
    ) {
        if let Some(load) = self.prefill_loads.iter().find(|l| l.rank == prefill_rank) {
            load.in_flight.fetch_add(1, Ordering::Relaxed);
        }
        let pending = Pending {
            tx,
            prefill_rank,
            decode_rank,
            next_seq: 0,
            early: BTreeMap::new(),
            in_prefill: true,
        };
        self.lock().insert(id, pending);
    }

    /// Stops tracking `id`. Returns whether it was still tracked.
    pub(crate) fn forget(&self, id: u64) -> bool {
        let removed = self.lock().remove(&id);
        match removed {
            Some(mut p) => {
                self.release_prefill(&mut p);
                true
            }
            None => false,
        }
    }

    /// Delivers `event` from `src`, in seq order. An event of an untracked
    /// request, or of a seq already delivered, is dropped.
    pub(crate) fn deliver(&self, src: Rank, event: Event) {
        let mut map = self.lock();
        let Some(p) = map.get_mut(&event.request_id) else {
            return;
        };
        if event.seq < p.next_seq {
            tracing::warn!(
                src,
                request_id = event.request_id,
                seq = event.seq,
                "dropping a duplicate event"
            );
            return;
        }
        p.early.insert(event.seq, (src, event.body));
        let mut done = false;
        while let Some((rank, body)) = p.early.remove(&p.next_seq) {
            if p.next_seq == 0 || matches!(body, EventBody::Error { .. }) {
                self.release_prefill(p);
            }
            let item = match body {
                EventBody::Token(t) => {
                    done = t.finish.is_some();
                    Ok(t)
                }
                EventBody::Error { stage, message } => {
                    done = true;
                    Err(WorkerError {
                        request_id: event.request_id,
                        stage,
                        rank,
                        message,
                    }
                    .into())
                }
            };
            p.next_seq += 1;
            if p.tx.send(item).is_err() {
                done = true;
            }
            if done {
                break;
            }
        }
        if done && let Some(mut p) = map.remove(&event.request_id) {
            self.release_prefill(&mut p);
        }
    }

    /// Ends request `id` with `error`.
    pub(crate) fn fail(&self, id: u64, error: anyhow::Error) {
        let removed = self.lock().remove(&id);
        if let Some(mut p) = removed {
            self.release_prefill(&mut p);
            let _ = p.tx.send(Err(error));
        }
    }

    /// Ends every request that waits on `rank` with an error naming `reason`.
    ///
    /// A request waits on its prefill worker until its first event, and on
    /// its decode worker until it finishes.
    pub(crate) fn fail_rank(&self, rank: Rank, reason: &str) {
        let mut map = self.lock();
        let ids: Vec<u64> = map
            .iter()
            .filter(|(_, p)| p.decode_rank == rank || (p.in_prefill && p.prefill_rank == rank))
            .map(|(&id, _)| id)
            .collect();
        for id in ids {
            if let Some(mut p) = map.remove(&id) {
                self.release_prefill(&mut p);
                let _ = p.tx.send(Err(anyhow!(
                    "request {id} lost worker rank {rank}: {reason}"
                )));
            }
        }
    }

    /// Ends every tracked request with an error naming `reason`.
    pub(crate) fn fail_all(&self, reason: &str) {
        let drained: Vec<(u64, Pending)> = self.lock().drain().collect();
        for (id, mut p) in drained {
            self.release_prefill(&mut p);
            let _ = p.tx.send(Err(anyhow!("request {id} ended: {reason}")));
        }
    }

    #[cfg(test)]
    pub(crate) fn tracked(&self) -> usize {
        self.lock().len()
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::protocol::FinishReason;
    use super::*;

    fn tok(index: u32, token: u32, finish: Option<FinishReason>) -> Event {
        Event::token(
            7,
            TokenEvent {
                index,
                token,
                finish,
            },
        )
    }

    async fn shared() -> Shared {
        let clients = super::super::super::test_harness::local_clients(1).await;
        let client = Arc::new(clients.into_iter().next().unwrap());
        Shared::new(client, &[1])
    }

    #[tokio::test]
    async fn events_are_reordered_and_the_finish_releases_the_request() {
        let s = shared().await;
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        s.register(7, tx, 1, 2);
        assert_eq!(s.prefill_loads[0].in_flight.load(Ordering::Relaxed), 1);
        s.deliver(2, tok(2, 30, Some(FinishReason::Eos)));
        s.deliver(2, tok(1, 20, None));
        assert!(rx.try_recv().is_err(), "nothing passes before seq 0");
        s.deliver(1, tok(0, 10, None));
        assert_eq!(s.prefill_loads[0].in_flight.load(Ordering::Relaxed), 0);
        let got: Vec<u32> = (0..3)
            .map(|_| rx.try_recv().unwrap().unwrap().token)
            .collect();
        assert_eq!(got, vec![10, 20, 30]);
        assert_eq!(s.tracked(), 0);
        s.deliver(2, tok(3, 40, None));
        assert!(
            rx.try_recv().is_err(),
            "events after the finish are dropped"
        );
    }

    #[tokio::test]
    async fn worker_errors_and_lost_ranks_end_the_stream() {
        let s = shared().await;
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        s.register(7, tx, 1, 2);
        s.deliver(1, tok(0, 10, None));
        s.deliver(2, Event::error(7, 1, Stage::Decode, "nan logits".into()));
        assert_eq!(rx.try_recv().unwrap().unwrap().token, 10);
        let err = rx.try_recv().unwrap().unwrap_err();
        let w = err.downcast_ref::<WorkerError>().expect("a worker error");
        assert_eq!(
            (w.stage, w.rank, w.message.as_str()),
            (Stage::Decode, 2, "nan logits")
        );
        assert_eq!(s.tracked(), 0);

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        s.register(8, tx, 1, 2);
        s.fail_rank(3, "unrelated");
        assert_eq!(s.tracked(), 1);
        s.fail_rank(1, "peer closed");
        assert!(
            rx.try_recv()
                .unwrap()
                .unwrap_err()
                .to_string()
                .contains("peer closed")
        );
        assert_eq!(s.prefill_loads[0].in_flight.load(Ordering::Relaxed), 0);
    }
}
