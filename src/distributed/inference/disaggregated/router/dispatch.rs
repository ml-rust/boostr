//! Router state shared by its listeners and its streams: in-flight requests,
//! prefill load, lost workers and the handoff clock.

use anyhow::anyhow;
use nexar::{NexarClient, Rank};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Weak};
use std::time::Duration;
use tokio::sync::mpsc::UnboundedSender;

use super::super::protocol::Event;
use super::super::wire::spawn_cancel;
use super::pending::{Pending, StreamItem};

/// Per-prefill-worker count of requests in prefill.
pub(crate) struct PrefillLoad {
    pub(crate) rank: Rank,
    pub(crate) in_flight: AtomicU64,
}

/// State the router, its listener tasks and its streams share.
pub(crate) struct Shared {
    pub(crate) client: Arc<NexarClient>,
    pub(crate) prefill_loads: Vec<PrefillLoad>,
    handoff_timeout: Duration,
    pending: Mutex<HashMap<u64, Pending>>,
}

impl Shared {
    pub(crate) fn new(
        client: Arc<NexarClient>,
        prefill_workers: &[Rank],
        handoff_timeout: Duration,
    ) -> Self {
        Self {
            client,
            prefill_loads: prefill_workers
                .iter()
                .map(|&rank| PrefillLoad {
                    rank,
                    in_flight: AtomicU64::new(0),
                })
                .collect(),
            handoff_timeout,
            pending: Mutex::new(HashMap::new()),
        }
    }

    fn lock(&self) -> MutexGuard<'_, HashMap<u64, Pending>> {
        // A panic elsewhere leaves the map consistent: every update is one insert or remove.
        self.pending.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn unload(&self, prefill_rank: Rank) {
        if let Some(load) = self.prefill_loads.iter().find(|l| l.rank == prefill_rank) {
            load.in_flight.fetch_sub(1, Ordering::Relaxed);
        }
    }

    fn release_prefill(&self, p: &mut Pending) {
        if std::mem::take(&mut p.in_prefill) {
            self.unload(p.prefill_rank);
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
        self.lock()
            .insert(id, Pending::new(tx, prefill_rank, decode_rank));
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

    /// Delivers `event` from `src` to its stream. See [`super::pending`] for the order.
    ///
    /// A first token that leaves the request waiting for its handoff starts
    /// the handoff clock. An event of an untracked request is dropped.
    pub(crate) fn deliver(self: &Arc<Self>, src: Rank, event: Event) {
        let id = event.request_id;
        let mut map = self.lock();
        let Some(p) = map.get_mut(&id) else {
            return;
        };
        let received = p.receive(id, src, event.body);
        if received.left_prefill {
            self.unload(p.prefill_rank);
        }
        if received.done {
            if let Some(mut p) = map.remove(&id) {
                self.release_prefill(&mut p);
            }
        } else if received.awaits_handoff {
            self.start_handoff_clock(id);
        }
    }

    /// After the handoff timeout, ends request `id` if decode has not
    /// confirmed its handoff by then.
    fn start_handoff_clock(self: &Arc<Self>, id: u64) {
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            tracing::warn!(
                request_id = id,
                "no tokio runtime: the handoff of this request has no timeout"
            );
            return;
        };
        let shared: Weak<Self> = Arc::downgrade(self);
        let timeout = self.handoff_timeout;
        handle.spawn(async move {
            tokio::time::sleep(timeout).await;
            if let Some(shared) = shared.upgrade() {
                shared.expire_handoff(id);
            }
        });
    }

    fn expire_handoff(&self, id: u64) {
        let mut map = self.lock();
        if !map.get(&id).is_some_and(|p| !p.accepted) {
            return;
        }
        let Some(mut p) = map.remove(&id) else {
            return;
        };
        drop(map);
        self.release_prefill(&mut p);
        p.end(anyhow!(
            "request {id}: decode worker rank {} did not confirm the KV handoff from prefill \
             worker rank {} within {:?}. Check both workers, or raise handoff_timeout",
            p.decode_rank,
            p.prefill_rank,
            self.handoff_timeout
        ));
        spawn_cancel(&self.client, vec![p.prefill_rank, p.decode_rank], id);
    }

    /// Ends request `id` with `error`.
    pub(crate) fn fail(&self, id: u64, error: anyhow::Error) {
        let removed = self.lock().remove(&id);
        if let Some(mut p) = removed {
            self.release_prefill(&mut p);
            p.end(error);
        }
    }

    /// Ends every request that waits on `rank` with an error naming `reason`,
    /// and tells the request's other worker to stop it.
    ///
    /// A request waits on its decode worker until it finishes. It waits on
    /// its prefill worker until decode confirms the KV handoff, so a prefill
    /// worker lost between the first token and that confirmation fails it.
    pub(crate) fn fail_rank(&self, rank: Rank, reason: &str) {
        let mut map = self.lock();
        let ids: Vec<u64> = map
            .iter()
            .filter(|(_, p)| p.decode_rank == rank || (p.prefill_rank == rank && !p.accepted))
            .map(|(&id, _)| id)
            .collect();
        let lost: Vec<(u64, Pending)> = ids
            .into_iter()
            .filter_map(|id| map.remove(&id).map(|p| (id, p)))
            .collect();
        drop(map);
        for (id, mut p) in lost {
            let (error, other) = if p.decode_rank == rank {
                let e = anyhow!("request {id} lost decode worker rank {rank}: {reason}");
                (e, p.in_prefill.then_some(p.prefill_rank))
            } else if p.in_prefill {
                let e =
                    anyhow!("request {id} lost prefill worker rank {rank} in prefill: {reason}");
                (e, None)
            } else {
                let e = anyhow!(
                    "request {id} lost prefill worker rank {rank} before decode worker rank {} \
                     confirmed the KV handoff: {reason}",
                    p.decode_rank
                );
                (e, Some(p.decode_rank))
            };
            self.release_prefill(&mut p);
            p.end(error);
            if let Some(other) = other {
                spawn_cancel(&self.client, vec![other], id);
            }
        }
    }

    /// Ends every tracked request with an error naming `reason`.
    pub(crate) fn fail_all(&self, reason: &str) {
        let drained: Vec<(u64, Pending)> = self.lock().drain().collect();
        for (id, mut p) in drained {
            self.release_prefill(&mut p);
            p.end(anyhow!("request {id} ended: {reason}"));
        }
    }

    #[cfg(test)]
    pub(crate) fn tracked(&self) -> usize {
        self.lock().len()
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::protocol::{EventBody, FinishReason, Stage};
    use super::super::pending::WorkerError;
    use super::*;
    use tokio::sync::mpsc::{UnboundedReceiver, unbounded_channel};

    const LONG: Duration = Duration::from_secs(60);

    fn tok(token: u32, finish: Option<FinishReason>) -> Event {
        Event {
            request_id: 7,
            body: EventBody::Token { token, finish },
        }
    }

    fn accepted() -> Event {
        Event {
            request_id: 7,
            body: EventBody::HandoffAccepted,
        }
    }

    async fn shared(handoff_timeout: Duration) -> Arc<Shared> {
        let clients = super::super::super::test_harness::local_clients(1).await;
        let client = Arc::new(clients.into_iter().next().unwrap());
        Arc::new(Shared::new(client, &[1], handoff_timeout))
    }

    fn start(s: &Shared) -> UnboundedReceiver<StreamItem> {
        let (tx, rx) = unbounded_channel();
        s.register(7, tx, 1, 2);
        rx
    }

    fn load(s: &Shared) -> u64 {
        s.prefill_loads[0].in_flight.load(Ordering::Relaxed)
    }

    fn error_of(rx: &mut UnboundedReceiver<StreamItem>) -> String {
        rx.try_recv().unwrap().unwrap_err().to_string()
    }

    #[tokio::test]
    async fn delivery_releases_the_load_and_the_finish_releases_the_request() {
        let s = shared(LONG).await;
        let mut rx = start(&s);
        assert_eq!(load(&s), 1);
        s.deliver(2, tok(20, None));
        assert_eq!(load(&s), 1, "a decode event leaves the request in prefill");
        s.deliver(1, tok(10, None));
        assert_eq!(load(&s), 0);
        s.deliver(2, tok(30, Some(FinishReason::Eos)));
        let got: Vec<u32> = (0..3)
            .map(|_| rx.try_recv().unwrap().unwrap().token)
            .collect();
        assert_eq!(got, [10, 20, 30]);
        assert_eq!(s.tracked(), 0);
        s.deliver(2, tok(40, None));
        assert!(
            rx.try_recv().is_err(),
            "events after the finish are dropped"
        );
    }

    #[tokio::test]
    async fn prefill_lost_between_first_token_and_confirmation_fails_the_request() {
        let s = shared(LONG).await;
        let mut rx = start(&s);
        s.deliver(1, tok(10, None));
        assert_eq!(rx.try_recv().unwrap().unwrap().token, 10);
        s.fail_rank(1, "peer closed");
        let err = error_of(&mut rx);
        assert!(
            err.contains("prefill worker rank 1")
                && err.contains("before decode worker rank 2 confirmed")
                && err.contains("peer closed"),
            "{err}"
        );
        assert_eq!(s.tracked(), 0);
    }

    #[tokio::test]
    async fn prefill_lost_after_confirmation_leaves_the_request_running() {
        let s = shared(LONG).await;
        let mut rx = start(&s);
        s.deliver(1, tok(10, None));
        s.deliver(1, accepted());
        s.fail_rank(1, "peer closed");
        assert_eq!(s.tracked(), 1);
        s.deliver(2, tok(20, Some(FinishReason::Length)));
        let got: Vec<u32> = (0..2)
            .map(|_| rx.try_recv().unwrap().unwrap().token)
            .collect();
        assert_eq!(got, [10, 20]);
        assert_eq!(s.tracked(), 0);

        let mut rx = start(&s);
        s.deliver(2, tok(20, None));
        s.fail_rank(1, "peer closed");
        assert_eq!(s.tracked(), 1, "a decode event also confirms the handoff");
        s.deliver(1, tok(10, None));
        assert_eq!(rx.try_recv().unwrap().unwrap().token, 10);
    }

    #[tokio::test]
    async fn decode_lost_before_confirmation_fails_the_request() {
        let s = shared(LONG).await;
        let mut rx = start(&s);
        s.deliver(1, tok(10, None));
        rx.try_recv().unwrap().unwrap();
        s.fail_rank(2, "peer closed");
        let err = error_of(&mut rx);
        assert!(
            err.contains("lost decode worker rank 2") && err.contains("peer closed"),
            "{err}"
        );
        assert_eq!(s.tracked(), 0);
    }

    #[tokio::test]
    async fn prefill_lost_in_prefill_fails_the_request_and_frees_the_load() {
        let s = shared(LONG).await;
        let mut rx = start(&s);
        s.fail_rank(3, "unrelated");
        assert_eq!(s.tracked(), 1);
        s.fail_rank(1, "peer closed");
        let err = error_of(&mut rx);
        assert!(
            err.contains("in prefill") && err.contains("peer closed"),
            "{err}"
        );
        assert_eq!(load(&s), 0);
    }

    #[tokio::test]
    async fn worker_errors_end_the_stream() {
        let s = shared(LONG).await;
        let mut rx = start(&s);
        s.deliver(1, tok(10, None));
        s.deliver(2, Event::error(7, Stage::Decode, "nan logits".into()));
        assert_eq!(rx.try_recv().unwrap().unwrap().token, 10);
        let err = rx.try_recv().unwrap().unwrap_err();
        let w = err.downcast_ref::<WorkerError>().expect("a worker error");
        assert_eq!(
            (w.stage, w.rank, w.message.as_str()),
            (Stage::Decode, 2, "nan logits")
        );
        assert_eq!(s.tracked(), 0);
    }

    #[tokio::test]
    async fn an_unconfirmed_handoff_times_out() {
        let s = shared(Duration::from_millis(300)).await;
        let mut rx = start(&s);
        s.deliver(1, tok(10, None));
        rx.try_recv().unwrap().unwrap();
        assert_eq!(s.tracked(), 1);
        let item = tokio::time::timeout(Duration::from_secs(10), rx.recv())
            .await
            .expect("the handoff clock fires")
            .unwrap();
        let err = item.unwrap_err().to_string();
        assert!(
            err.contains("did not confirm the KV handoff") && err.contains("handoff_timeout"),
            "{err}"
        );
        assert_eq!(s.tracked(), 0);
    }

    #[tokio::test]
    async fn a_confirmed_handoff_never_times_out() {
        let s = shared(Duration::from_millis(100)).await;
        let mut rx = start(&s);
        s.deliver(1, tok(10, None));
        s.deliver(1, accepted());
        tokio::time::sleep(Duration::from_millis(500)).await;
        assert_eq!(s.tracked(), 1);
        rx.try_recv().unwrap().unwrap();
        assert!(rx.try_recv().is_err());
    }
}
