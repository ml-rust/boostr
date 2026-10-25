//! Task helpers the swarm and disaggregated modes share.

use anyhow::{Result, anyhow};
use nexar::{NexarClient, Rank};
use std::sync::Arc;

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

/// A receiver that resolves once `src` sends a frame on `tag` or disconnects.
pub(crate) fn spawn_notice_listener(
    client: Arc<NexarClient>,
    src: Rank,
    tag: u64,
) -> (tokio::sync::oneshot::Receiver<()>, TaskGuard) {
    let (tx, rx) = tokio::sync::oneshot::channel();
    let handle = tokio::spawn(async move {
        if let Err(e) = client.recv_bytes_tagged(src, tag).await {
            tracing::debug!(src, tag, "notice listener ended: {e}");
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
