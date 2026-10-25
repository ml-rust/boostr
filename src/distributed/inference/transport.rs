//! Swarm pipeline protocol over nexar.
//!
//! Every message is one frame on `send_bytes_tagged`, on its own tag.
//!
//! - A receive for `(src, tag)` takes only frames of that tag.
//! - Frames of one tag from one sender arrive in send order. Several
//!   requests can be in flight on one stage: each stage serves them in order.
//! - Frames of different tags, or from different senders, have no order.
//!   A [`tags::GEN_REQUEST`] frame holds its header and its payload, so no
//!   receiver pairs two messages.

use anyhow::{Result, anyhow, bail};
use nexar::{CpuAdapter, DeviceAdapter, NexarClient, Rank};
use std::sync::Arc;
use std::time::Duration;

/// Tags of the swarm protocol.
///
/// nexar numbers its own collective tags upward from 1. These tags sit in a
/// fixed high range that never meets them: ASCII `SWARM` in the top five bytes.
pub mod tags {
    const BASE: u64 = 0x5357_4152_4D00_0000;
    /// Leader → worker: a [`LayerAssignment`](super::LayerAssignment) frame.
    pub const LAYER_ASSIGNMENT: u64 = BASE | 1;
    /// Worker → leader: a one-byte readiness frame.
    pub const WORKER_READY: u64 = BASE | 2;
    /// Leader → worker: the model path as UTF-8.
    pub const MODEL_PATH: u64 = BASE | 3;
    /// Previous stage → next stage: a [`GenRequestHeader`](super::GenRequestHeader)
    /// frame with its payload.
    pub const GEN_REQUEST: u64 = BASE | 20;
    /// Last stage → leader: the logits as `f32` LE.
    pub const LOGITS: u64 = BASE | 11;
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

/// Layer assignment message: serialized as [start_layer: u32, end_layer: u32, has_embedding: u8, has_lm_head: u8].
#[derive(Debug, Clone, Copy)]
pub struct LayerAssignment {
    pub start_layer: u32,
    pub end_layer: u32,
    pub has_embedding: bool,
    pub has_lm_head: bool,
}

impl LayerAssignment {
    pub fn to_bytes(&self) -> [u8; 10] {
        let mut buf = [0u8; 10];
        buf[0..4].copy_from_slice(&self.start_layer.to_le_bytes());
        buf[4..8].copy_from_slice(&self.end_layer.to_le_bytes());
        buf[8] = self.has_embedding as u8;
        buf[9] = self.has_lm_head as u8;
        buf
    }

    pub fn from_bytes(buf: &[u8; 10]) -> Self {
        Self {
            start_layer: u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]),
            end_layer: u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]),
            has_embedding: buf[8] != 0,
            has_lm_head: buf[9] != 0,
        }
    }

    /// The assignment in a [`tags::LAYER_ASSIGNMENT`] frame.
    ///
    /// # Errors
    ///
    /// A frame that is not 10 bytes.
    pub fn from_frame(frame: &[u8]) -> Result<Self> {
        let buf: &[u8; 10] = frame.try_into().map_err(|_| {
            anyhow!(
                "a layer assignment frame is 10 bytes, and this one is {}",
                frame.len()
            )
        })?;
        Ok(Self::from_bytes(buf))
    }
}

/// Generation request header: [seq_len: u32, max_tokens: u32, position: u32].
///
/// `seq_len` counts the payload's elements: `i64` token ids for the first
/// stage, `f32` activations for every later stage.
#[derive(Debug, Clone, Copy)]
pub struct GenRequestHeader {
    pub seq_len: u32,
    pub max_tokens: u32,
    pub position: u32,
}

impl GenRequestHeader {
    pub fn to_bytes(&self) -> [u8; 12] {
        let mut buf = [0u8; 12];
        buf[0..4].copy_from_slice(&self.seq_len.to_le_bytes());
        buf[4..8].copy_from_slice(&self.max_tokens.to_le_bytes());
        buf[8..12].copy_from_slice(&self.position.to_le_bytes());
        buf
    }

    pub fn from_bytes(buf: &[u8; 12]) -> Self {
        Self {
            seq_len: u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]),
            max_tokens: u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]),
            position: u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]),
        }
    }

    /// A [`tags::GEN_REQUEST`] frame: this header, then `payload`.
    pub fn frame(&self, payload: &[u8]) -> Vec<u8> {
        let mut frame = Vec::with_capacity(12 + payload.len());
        frame.extend_from_slice(&self.to_bytes());
        frame.extend_from_slice(payload);
        frame
    }

    /// The header and payload of a [`tags::GEN_REQUEST`] frame.
    ///
    /// # Errors
    ///
    /// A frame shorter than the 12-byte header.
    pub fn split_frame(frame: &[u8]) -> Result<(Self, &[u8])> {
        let Some((header, payload)) = frame.split_first_chunk::<12>() else {
            bail!(
                "a generation request frame of {} bytes has no 12-byte header",
                frame.len()
            );
        };
        Ok((Self::from_bytes(header), payload))
    }
}

/// Create a `CpuAdapter` for nexar (host memory staging).
pub fn cpu_adapter() -> Arc<dyn DeviceAdapter> {
    Arc::new(CpuAdapter::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_assignment_roundtrip() {
        let orig = LayerAssignment {
            start_layer: 16,
            end_layer: 32,
            has_embedding: false,
            has_lm_head: true,
        };
        let bytes = orig.to_bytes();
        let decoded = LayerAssignment::from_bytes(&bytes);
        assert_eq!(decoded.start_layer, 16);
        assert_eq!(decoded.end_layer, 32);
        assert!(!decoded.has_embedding);
        assert!(decoded.has_lm_head);
    }

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
    fn test_gen_request_header_roundtrip() {
        let orig = GenRequestHeader {
            seq_len: 128,
            max_tokens: 256,
            position: 0,
        };
        let bytes = orig.to_bytes();
        let decoded = GenRequestHeader::from_bytes(&bytes);
        assert_eq!(decoded.seq_len, 128);
        assert_eq!(decoded.max_tokens, 256);
        assert_eq!(decoded.position, 0);
    }

    #[test]
    fn test_frames_carry_the_header_and_payload_and_refuse_short_input() {
        let header = GenRequestHeader {
            seq_len: 2,
            max_tokens: 3,
            position: 4,
        };
        let frame = header.frame(&[9, 8, 7]);
        let (back, payload) = GenRequestHeader::split_frame(&frame).unwrap();
        assert_eq!((back.seq_len, back.max_tokens, back.position), (2, 3, 4));
        assert_eq!(payload, [9, 8, 7]);
        assert!(GenRequestHeader::split_frame(&frame[..11]).is_err());

        let a = LayerAssignment {
            start_layer: 1,
            end_layer: 2,
            has_embedding: true,
            has_lm_head: false,
        };
        let back = LayerAssignment::from_frame(&a.to_bytes()).unwrap();
        assert_eq!((back.start_layer, back.end_layer), (1, 2));
        let err = LayerAssignment::from_frame(&[0; 9]).unwrap_err();
        assert!(err.to_string().contains("this one is 9"), "{err}");
    }
}
