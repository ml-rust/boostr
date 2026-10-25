//! KV handoff confirmation.
//!
//! 1. Decode receives a handoff and sends its prefill worker a `HandoffAck`
//!    at once, before the handoff waits for its turn.
//! 2. Prefill sends the router a handoff-accepted event. It travels behind
//!    the first token on prefill's event lane, so the router reads it before
//!    it reads prefill's disconnect.
//! 3. A prefill worker that stops waits for the acks of its last handoffs,
//!    then exits. A graceful prefill exit never fails a running request.

use anyhow::Result;
use nexar::{NexarClient, Rank};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;
use tokio::sync::Notify;
use tokio::sync::mpsc::UnboundedSender;

use super::protocol::{Event, EventBody, HandoffAck, KvHandoff, peek_request_id, tags};
use super::wire::{TaskGuard, send_event, send_frame};

/// A received handoff, or its parse error with the request id when the header has one.
pub(crate) type HandoffItem = Result<KvHandoff, (Option<u64>, anyhow::Error)>;

/// Decode side: drains every handoff from `src` into `tx` and acks each one
/// whose request id it can read.
///
/// The task ends when `src` disconnects or `tx` closes.
pub(crate) fn spawn_handoff_listener(
    client: Arc<NexarClient>,
    src: Rank,
    tx: UnboundedSender<HandoffItem>,
) -> TaskGuard {
    let handle = tokio::spawn(async move {
        loop {
            let buf = match client.recv_bytes_tagged(src, tags::KV_HANDOFF).await {
                Ok(buf) => buf,
                Err(e) => {
                    tracing::debug!(src, "KV handoff listener ended: {e}");
                    break;
                }
            };
            let item = KvHandoff::from_bytes(&buf).map_err(|e| (peek_request_id(&buf), e));
            let id = match &item {
                Ok(h) => Some(h.request_id),
                Err((id, _)) => *id,
            };
            if let Some(request_id) = id {
                let ack = HandoffAck { request_id }.to_bytes();
                if let Err(e) = send_frame(&client, src, tags::HANDOFF_ACK, &ack).await {
                    tracing::warn!(src, request_id, "KV handoff ack not sent: {e:#}");
                }
            }
            if tx.send(item).is_err() {
                break;
            }
        }
    });
    TaskGuard::new(handle)
}

/// Prefill side: handoffs sent and not yet acked, by request id.
#[derive(Default)]
pub(crate) struct Unacked {
    ids: Mutex<HashMap<u64, Rank>>,
    changed: Notify,
}

impl Unacked {
    fn lock(&self) -> MutexGuard<'_, HashMap<u64, Rank>> {
        self.ids.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Tracks the handoff of `id` to `decode_rank`. Call it before the send.
    pub(crate) fn insert(&self, id: u64, decode_rank: Rank) {
        self.lock().insert(id, decode_rank);
    }

    fn contains(&self, id: u64) -> bool {
        self.lock().contains_key(&id)
    }

    /// Stops tracking `id`.
    pub(crate) fn remove(&self, id: u64) {
        if self.lock().remove(&id).is_some() {
            self.changed.notify_waiters();
        }
    }

    /// Stops tracking every handoff to `decode_rank`.
    fn remove_rank(&self, decode_rank: Rank) {
        self.lock().retain(|_, &mut r| r != decode_rank);
        self.changed.notify_waiters();
    }

    /// Waits until no handoff is unacked. Returns how many still are after `within`.
    pub(crate) async fn drained(&self, within: Duration) -> usize {
        let deadline = tokio::time::Instant::now() + within;
        loop {
            let changed = self.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            let left = self.lock().len();
            if left == 0 {
                return 0;
            }
            if tokio::time::timeout_at(deadline, changed).await.is_err() {
                return self.lock().len();
            }
        }
    }
}

/// Prefill side: forwards each ack from `decode_rank` to the router as a
/// handoff-accepted event.
///
/// An ack of a handoff this worker does not track is dropped. When
/// `decode_rank` disconnects, its handoffs stop counting as unacked: the
/// router fails those requests on the decode worker's disconnect.
pub(crate) fn spawn_ack_listener(
    client: Arc<NexarClient>,
    decode_rank: Rank,
    router: Rank,
    unacked: Arc<Unacked>,
) -> TaskGuard {
    let handle = tokio::spawn(async move {
        while let Ok(buf) = client
            .recv_bytes_tagged(decode_rank, tags::HANDOFF_ACK)
            .await
        {
            let ack = match HandoffAck::from_bytes(&buf) {
                Ok(ack) => ack,
                Err(e) => {
                    tracing::warn!(decode_rank, "dropping a malformed handoff ack: {e:#}");
                    continue;
                }
            };
            let id = ack.request_id;
            if !unacked.contains(id) {
                continue;
            }
            let accepted = Event {
                request_id: id,
                body: EventBody::HandoffAccepted,
            };
            // Removed only after the send, so a stopping worker exits after it.
            if let Err(e) = send_event(&client, router, &accepted).await {
                tracing::warn!(
                    request_id = id,
                    "handoff confirmation not sent to the router: {e:#}"
                );
            }
            unacked.remove(id);
        }
        unacked.remove_rank(decode_rank);
    });
    TaskGuard::new(handle)
}

/// The ack listeners of one prefill worker, one per decode rank it handed off to.
pub(crate) struct AckListeners {
    client: Arc<NexarClient>,
    router: Rank,
    unacked: Arc<Unacked>,
    tasks: HashMap<Rank, TaskGuard>,
}

impl AckListeners {
    pub(crate) fn new(client: Arc<NexarClient>, router: Rank) -> Self {
        Self {
            client,
            router,
            unacked: Arc::new(Unacked::default()),
            tasks: HashMap::new(),
        }
    }

    pub(crate) fn unacked(&self) -> &Unacked {
        &self.unacked
    }

    /// Tracks the handoff of `id` to `decode_rank`, listening for that rank's acks.
    pub(crate) fn expect(&mut self, id: u64, decode_rank: Rank) {
        self.tasks.entry(decode_rank).or_insert_with(|| {
            spawn_ack_listener(
                Arc::clone(&self.client),
                decode_rank,
                self.router,
                Arc::clone(&self.unacked),
            )
        });
        self.unacked.insert(id, decode_rank);
    }
}

#[cfg(test)]
mod tests {
    use super::super::config::DEFAULT_HANDOFF_TIMEOUT;
    use super::super::protocol::{GenerationParams, PrefillRequest, TokenEvent};
    use super::super::router::TokenStream;
    use super::super::test_harness::{
        MockDecode, Parts, decode_worker, local_clients, prefill_worker,
    };
    use super::*;

    const WAIT: Duration = Duration::from_secs(10);

    fn eos_params() -> GenerationParams {
        let mut p = GenerationParams::greedy(50);
        p.eos_tokens = vec![9];
        p
    }

    /// The stream's items up to its first error, and that error.
    async fn until_error(s: &mut TokenStream) -> (Vec<u32>, String) {
        let mut tokens = Vec::new();
        loop {
            let item = tokio::time::timeout(WAIT, s.next())
                .await
                .expect("the stream ends")
                .expect("an error item before the end");
            match item {
                Ok(t) => tokens.push(t.token),
                Err(e) => return (tokens, e.to_string()),
            }
        }
    }

    /// Shuts `client` down once every other handle dropped. The router and
    /// decode do not shut down, so the done exchange times out: that is fine.
    async fn shut_down(client: Arc<NexarClient>) {
        let mut client = client;
        let deadline = tokio::time::Instant::now() + WAIT;
        let owned = loop {
            match Arc::try_unwrap(client) {
                Ok(owned) => break owned,
                Err(shared) => {
                    assert!(tokio::time::Instant::now() < deadline, "a handle stayed");
                    client = shared;
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            }
        };
        let _ = owned
            .shutdown_with_timeout(Duration::from_millis(500))
            .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn prefill_lost_between_first_token_and_ack_fails_the_request() {
        let parts = Parts::start(DEFAULT_HANDOFF_TIMEOUT).await;
        let (go, gone) = tokio::sync::oneshot::channel::<()>();
        let prefill = Arc::clone(&parts.prefill);
        let fake = tokio::spawn(async move {
            let buf = prefill
                .recv_bytes_tagged(0, tags::PREFILL_REQUEST)
                .await
                .unwrap();
            let id = PrefillRequest::from_bytes(&buf).unwrap().request_id;
            let first = TokenEvent {
                index: 0,
                token: 5,
                finish: None,
            };
            send_event(&prefill, 0, &Event::token(id, first))
                .await
                .unwrap();
            let _ = gone.await;
            prefill.close();
        });
        let mut s = parts
            .router
            .route_request_stream(&[1], eos_params(), None)
            .await
            .unwrap();
        let first = tokio::time::timeout(WAIT, s.next()).await.unwrap();
        assert_eq!(first.unwrap().unwrap().token, 5);
        go.send(()).unwrap();
        let (tokens, err) = until_error(&mut s).await;
        assert!(tokens.is_empty(), "{tokens:?}");
        assert!(
            err.contains("lost prefill worker rank 1 before decode worker rank 2 confirmed"),
            "{err}"
        );
        fake.await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_graceful_prefill_exit_after_the_ack_keeps_the_request() {
        let parts = Parts::start(DEFAULT_HANDOFF_TIMEOUT).await;
        let (mock, gate) = MockDecode::gated();
        let d = decode_worker(Arc::clone(&parts.decode), &mock);
        let decode = tokio::spawn(async move { d.run_loop().await });
        let p = prefill_worker(Arc::clone(&parts.prefill), None, DEFAULT_HANDOFF_TIMEOUT);
        let prefill = tokio::spawn(async move { p.run_loop().await.map(|()| p) });
        let mut s = parts
            .router
            .route_request_stream(&[1], eos_params(), None)
            .await
            .unwrap();
        let first = tokio::time::timeout(WAIT, s.next()).await.unwrap();
        assert_eq!(first.unwrap().unwrap().token, 2);

        // Prefill exits after the ack; decode finishes the request in progress.
        parts.router.stop_workers().await;
        let p = tokio::time::timeout(WAIT, prefill)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        drop(parts.prefill);
        shut_down(p.into_client()).await;
        tokio::time::sleep(Duration::from_millis(300)).await;

        for expect in 3..=9 {
            gate.send(()).unwrap();
            let t = tokio::time::timeout(WAIT, s.next()).await.unwrap();
            assert_eq!(t.unwrap().unwrap().token, expect);
        }
        assert!(s.next().await.is_none());
        tokio::time::timeout(WAIT, decode)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn decode_lost_before_the_ack_fails_the_request() {
        let parts = Parts::start(DEFAULT_HANDOFF_TIMEOUT).await;
        let p = prefill_worker(Arc::clone(&parts.prefill), None, DEFAULT_HANDOFF_TIMEOUT);
        let _prefill = TaskGuard::new(tokio::spawn(async move { p.run_loop().await }));
        let decode = Arc::clone(&parts.decode);
        let fake = tokio::spawn(async move {
            decode.recv_bytes_tagged(1, tags::KV_HANDOFF).await.unwrap();
            decode.close();
        });
        let mut s = parts
            .router
            .route_request_stream(&[1], eos_params(), None)
            .await
            .unwrap();
        let (_, err) = until_error(&mut s).await;
        assert!(err.contains("lost decode worker rank 2"), "{err}");
        fake.await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_handoff_with_no_ack_times_out() {
        let timeout = Duration::from_millis(300);
        let parts = Parts::start(timeout).await;
        let p = prefill_worker(Arc::clone(&parts.prefill), None, timeout);
        let _prefill = TaskGuard::new(tokio::spawn(async move { p.run_loop().await }));
        let decode = Arc::clone(&parts.decode);
        // A live decode rank that takes the handoff and never acks it.
        let _fake = TaskGuard::new(tokio::spawn(async move {
            let held = decode.recv_bytes_tagged(1, tags::KV_HANDOFF).await;
            std::future::pending::<()>().await;
            drop(held);
        }));
        let mut s = parts
            .router
            .route_request_stream(&[1], eos_params(), None)
            .await
            .unwrap();
        let (tokens, err) = until_error(&mut s).await;
        assert_eq!(tokens, [2]);
        assert!(
            err.contains("decode worker rank 2 did not confirm the KV handoff"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn drained_waits_for_the_last_removal_and_reports_what_stays() {
        let u = Arc::new(Unacked::default());
        assert_eq!(u.drained(Duration::from_millis(10)).await, 0);
        u.insert(1, 5);
        u.insert(2, 6);
        let later = Arc::clone(&u);
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            later.remove(1);
            later.remove_rank(6);
        });
        assert_eq!(u.drained(Duration::from_secs(10)).await, 0);
        u.insert(3, 5);
        assert_eq!(u.drained(Duration::from_millis(20)).await, 1);
    }

    /// Rank 0 is the router, rank 1 prefill, rank 2 decode.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_ack_reaches_the_router_as_handoff_accepted() {
        let mut clients = local_clients(3).await.into_iter().map(Arc::new);
        let (router, prefill, decode) = (
            clients.next().unwrap(),
            clients.next().unwrap(),
            clients.next().unwrap(),
        );
        let mut acks = AckListeners::new(Arc::clone(&prefill), 0);
        acks.expect(9, 2);
        acks.expect(10, 2);
        let ack = HandoffAck { request_id: 9 }.to_bytes();
        send_frame(&decode, 1, tags::HANDOFF_ACK, &ack)
            .await
            .unwrap();
        let buf = router.recv_bytes_tagged(1, tags::EVENT).await.unwrap();
        let ev = Event::from_bytes(&buf).unwrap();
        assert_eq!((ev.request_id, ev.body), (9, EventBody::HandoffAccepted));
        assert_eq!(acks.unacked().drained(Duration::from_millis(20)).await, 1);
    }
}
