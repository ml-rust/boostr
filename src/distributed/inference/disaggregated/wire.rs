//! Frame transport helpers the router and both workers share.

use anyhow::{Result, anyhow};
use nexar::{NexarClient, Rank};
use std::sync::Arc;
use tokio::sync::mpsc::UnboundedSender;

use super::cancel::CancelBook;
use super::protocol::{Cancel, Event, Stage, tags};

/// Aborts its task on drop.
pub(crate) struct TaskGuard(tokio::task::AbortHandle);

impl TaskGuard {
    pub(crate) fn new<T>(handle: tokio::task::JoinHandle<T>) -> Self {
        Self(handle.abort_handle())
    }
}

impl Drop for TaskGuard {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// Sends one frame to `dest` on `tag`.
pub(crate) async fn send_frame(
    client: &NexarClient,
    dest: Rank,
    tag: u64,
    frame: &[u8],
) -> Result<()> {
    client
        .send_bytes_tagged(dest, tag, frame)
        .await
        .map_err(|e| {
            anyhow!(
                "sending a {}-byte frame to rank {dest} failed: {e}",
                frame.len()
            )
        })
}

/// Sends one event to the router.
pub(crate) async fn send_event(client: &NexarClient, router: Rank, event: &Event) -> Result<()> {
    send_frame(client, router, tags::EVENT, &event.to_bytes()?).await
}

/// Logs a failed request of `stage` on worker `rank` and sends the router its error event.
pub(crate) async fn send_error(
    client: &NexarClient,
    router: Rank,
    rank: Rank,
    stage: Stage,
    request_id: u64,
    seq: u32,
    e: anyhow::Error,
) -> Result<()> {
    let name = match stage {
        Stage::Prefill => "prefill",
        Stage::Decode => "decode",
    };
    tracing::warn!(rank, request_id, "{name} failed: {e:#}");
    let ev = Event::error(request_id, seq, stage, format!("{e:#}"));
    send_event(client, router, &ev).await
}

/// Drains every frame from `src` on `tag` into `tx`, parsed by `parse`.
///
/// A dedicated task keeps nexar's per-tag channel empty. nexar holds one lock
/// across all tags of a peer while a full channel waits, so a slow consumer
/// stalls every other tag of that peer. The task ends when the peer
/// disconnects or `tx` closes.
pub(crate) fn spawn_listener<T, F>(
    client: Arc<NexarClient>,
    src: Rank,
    tag: u64,
    tx: UnboundedSender<T>,
    parse: F,
) -> TaskGuard
where
    T: Send + 'static,
    F: Fn(&[u8]) -> T + Send + 'static,
{
    let handle = tokio::spawn(async move {
        loop {
            match client.recv_bytes_tagged(src, tag).await {
                Ok(buf) => {
                    if tx.send(parse(&buf)).is_err() {
                        break;
                    }
                }
                Err(e) => {
                    tracing::debug!(src, tag, "disaggregated listener ended: {e}");
                    break;
                }
            }
        }
    });
    TaskGuard::new(handle)
}

/// Records every cancel the router sends into `book`.
pub(crate) fn spawn_cancel_listener(
    client: Arc<NexarClient>,
    router: Rank,
    book: Arc<CancelBook>,
) -> TaskGuard {
    let handle = tokio::spawn(async move {
        while let Ok(buf) = client.recv_bytes_tagged(router, tags::CANCEL).await {
            match Cancel::from_bytes(&buf) {
                Ok(c) => book.cancel(c.request_id),
                Err(e) => tracing::warn!(router, "dropping a malformed cancel frame: {e:#}"),
            }
        }
    });
    TaskGuard::new(handle)
}

/// A receiver that resolves once `router` sends [`tags::SHUTDOWN`] or disconnects.
pub(crate) fn spawn_shutdown_listener(
    client: Arc<NexarClient>,
    router: Rank,
) -> (tokio::sync::oneshot::Receiver<()>, TaskGuard) {
    let (tx, rx) = tokio::sync::oneshot::channel();
    let handle = tokio::spawn(async move {
        if let Err(e) = client.recv_bytes_tagged(router, tags::SHUTDOWN).await {
            tracing::debug!(router, "shutdown listener ended: {e}");
        }
        let _ = tx.send(());
    });
    (rx, TaskGuard::new(handle))
}

/// Runs `f` on the blocking pool. A panic in `f` returns as an error.
pub(crate) async fn run_blocking<T, F>(f: F) -> Result<T>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| anyhow!("forward task did not complete: {e}"))
}
