//! Tags, one-frame send and receive, and client shutdown of the swarm protocol.
//!
//! Every message is one frame on `send_bytes_tagged`, on its own tag.
//!
//! - A receive for `(src, tag)` takes only frames of that tag.
//! - Frames of one tag from one sender arrive in send order. Several
//!   requests can be in flight on one stage: each stage serves them in order.
//! - Frames of different tags, or from different senders, have no order.
//!   A [`tags::GEN_REQUEST`] frame holds its header and its payload, so no
//!   receiver pairs two messages. A release travels on the same tag as the
//!   requests, so it never overtakes a request of its id.

use anyhow::{Result, anyhow};
use nexar::{CpuAdapter, DeviceAdapter, NexarClient, Rank};
use std::sync::Arc;
use std::time::Duration;

/// Tags of the swarm protocol.
///
/// nexar numbers its own collective tags upward from 1. These tags sit in a
/// fixed high range that never meets them: ASCII `SWARM` in the top five bytes.
pub mod tags {
    const BASE: u64 = 0x5357_4152_4D00_0000;
    /// Leader → worker: a [`LayerAssignment`](super::super::LayerAssignment) frame.
    pub const LAYER_ASSIGNMENT: u64 = BASE | 1;
    /// Worker → leader: a one-byte readiness frame.
    pub const WORKER_READY: u64 = BASE | 2;
    /// Leader → worker: the model path as UTF-8.
    pub const MODEL_PATH: u64 = BASE | 3;
    /// Previous stage → next stage: a
    /// [`GenRequestHeader`](super::super::GenRequestHeader) frame with its
    /// payload. A release is a frame of this tag with no payload.
    pub const GEN_REQUEST: u64 = BASE | 20;
    /// Last stage → leader: a [`LogitsFrame`](super::super::LogitsFrame).
    pub const LOGITS: u64 = BASE | 11;
    /// Any worker → leader: a [`StageError`](super::super::StageError) frame.
    pub const STAGE_ERROR: u64 = BASE | 12;
    /// Leader → every worker: a one-byte frame that stops the compute loop.
    pub const SHUTDOWN: u64 = BASE | 99;
}

/// Sends one frame to `dest` on `tag`.
pub async fn send_frame(client: &NexarClient, dest: Rank, tag: u64, frame: &[u8]) -> Result<()> {
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

/// Receives the next frame from `src` on `tag`.
pub async fn recv_frame(client: &NexarClient, src: Rank, tag: u64) -> Result<Vec<u8>> {
    client
        .recv_bytes_tagged(src, tag)
        .await
        .map(|buf| buf.to_vec())
        .map_err(|e| anyhow!("receiving a frame from rank {src} failed: {e}"))
}

/// How often [`shutdown_shared`] checks whether the other handles dropped.
const RELEASE_POLL: Duration = Duration::from_millis(10);

/// Shuts down a client that other tasks can still hold.
///
/// Waits up to `release_within` for every other `Arc` handle to drop, then
/// runs [`NexarClient::shutdown`]: data this rank sent reaches every peer
/// that is still running. Call it on every rank before the process exits.
///
/// # Errors
///
/// Another handle still held after `release_within`: this handle drops, and
/// the connections close without the shutdown exchange once the last
/// holder drops. Or the error of [`NexarClient::shutdown`].
pub async fn shutdown_shared(client: Arc<NexarClient>, release_within: Duration) -> Result<()> {
    let deadline = tokio::time::Instant::now() + release_within;
    let rank = client.rank();
    let mut client = client;
    loop {
        match Arc::try_unwrap(client) {
            Ok(owned) => {
                return owned
                    .shutdown()
                    .await
                    .map_err(|e| anyhow!("nexar shutdown of rank {rank} failed: {e}"));
            }
            Err(shared) if tokio::time::Instant::now() >= deadline => {
                let others = Arc::strong_count(&shared) - 1;
                return Err(anyhow!(
                    "rank {rank} skipped the nexar shutdown: {others} other handle(s) still held \
                     the client after {release_within:?}. Drop every worker, router and task \
                     that holds the client before the shutdown"
                ));
            }
            Err(shared) => {
                client = shared;
                tokio::time::sleep(RELEASE_POLL).await;
            }
        }
    }
}

/// Create a `CpuAdapter` for nexar (host memory staging).
pub fn cpu_adapter() -> Arc<dyn DeviceAdapter> {
    Arc::new(CpuAdapter::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn pair() -> (Arc<NexarClient>, Arc<NexarClient>) {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let mut clients = NexarClient::bootstrap_local(2, cpu_adapter())
            .await
            .unwrap();
        clients.sort_by_key(|c| c.rank());
        let c1 = Arc::new(clients.pop().unwrap());
        (Arc::new(clients.pop().unwrap()), c1)
    }

    /// A large frame first, so a later small frame can finish its stream first.
    #[tokio::test]
    async fn test_frames_of_one_tag_arrive_in_send_order() {
        let (c0, c1) = pair().await;
        let frames: Vec<Vec<u8>> = (0..8u8)
            .map(|i| vec![i; if i == 0 { 1 << 20 } else { usize::from(i) }])
            .collect();
        for f in &frames {
            send_frame(&c0, 1, tags::GEN_REQUEST, f).await.unwrap();
            send_frame(&c0, 1, tags::LOGITS, &f[..1]).await.unwrap();
        }
        for i in 0..8u8 {
            assert_eq!(recv_frame(&c1, 0, tags::LOGITS).await.unwrap(), [i]);
        }
        for f in &frames {
            assert_eq!(&recv_frame(&c1, 0, tags::GEN_REQUEST).await.unwrap(), f);
        }
    }

    #[tokio::test]
    async fn test_shutdown_shared_waits_for_other_handles() {
        let (c0, c1) = pair().await;
        let held = Arc::clone(&c0);
        let holder = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            drop(held);
        });
        let peer = tokio::spawn(shutdown_shared(c1, Duration::from_secs(5)));
        shutdown_shared(c0, Duration::from_secs(5)).await.unwrap();
        peer.await.unwrap().unwrap();
        holder.await.unwrap();
    }

    #[tokio::test]
    async fn test_shutdown_shared_names_a_handle_that_stays() {
        let (c0, _c1) = pair().await;
        let _held = Arc::clone(&c0);
        let err = shutdown_shared(c0, Duration::from_millis(30))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("1 other handle"), "{err}");
    }

    #[test]
    fn test_tags_are_distinct() {
        let all = [
            tags::LAYER_ASSIGNMENT,
            tags::WORKER_READY,
            tags::MODEL_PATH,
            tags::GEN_REQUEST,
            tags::LOGITS,
            tags::STAGE_ERROR,
            tags::SHUTDOWN,
        ];
        let set: std::collections::BTreeSet<u64> = all.iter().copied().collect();
        assert_eq!(set.len(), all.len());
    }
}
