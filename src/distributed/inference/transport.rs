//! Activation transfer protocol over nexar.
//!
//! Transfers hidden-state tensors between pipeline stages using nexar's
//! point-to-point send/recv with CPU staging via `CpuAdapter`.

use anyhow::{Result, anyhow};
use nexar::{CpuAdapter, DeviceAdapter, NexarClient, Rank};
use std::sync::Arc;

/// Tags for nexar messages in the swarm protocol.
pub mod tags {
    /// Control plane: layer assignment from leader to worker.
    pub const LAYER_ASSIGNMENT: u32 = 1;
    /// Control plane: worker readiness acknowledgment.
    pub const WORKER_READY: u32 = 2;
    /// Control plane: model path broadcast from leader to worker.
    pub const MODEL_PATH: u32 = 3;
    /// Data plane: hidden state activation tensor transfer.
    pub const ACTIVATION: u32 = 10;
    /// Data plane: logits tensor (final stage → leader).
    pub const LOGITS: u32 = 11;
    /// Control plane: new generation request (leader → first stage).
    pub const GEN_REQUEST: u32 = 20;
    /// Control plane: generation complete signal.
    pub const GEN_COMPLETE: u32 = 21;
    /// Control plane: shutdown signal.
    pub const SHUTDOWN: u32 = 99;
}

/// Sends a byte buffer to a remote rank.
///
/// Uses `CpuAdapter` staging: the data pointer is treated as host memory.
pub async fn send_bytes(client: &NexarClient, data: &[u8], dest: Rank, tag: u32) -> Result<()> {
    let ptr = data.as_ptr() as u64;
    let size = data.len();
    unsafe {
        client
            .send(ptr, size, dest, tag)
            .await
            .map_err(|e| anyhow!("nexar send failed: {}", e))
    }
}

/// Receives bytes from a remote rank into a pre-allocated buffer.
pub async fn recv_bytes(client: &NexarClient, buf: &mut [u8], src: Rank, tag: u32) -> Result<()> {
    let ptr = buf.as_mut_ptr() as u64;
    let size = buf.len();
    unsafe {
        client
            .recv(ptr, size, src, tag)
            .await
            .map_err(|e| anyhow!("nexar recv failed: {}", e))
    }
}

/// Sends an f32 tensor (as raw bytes) to a remote rank.
pub async fn send_tensor_f32(
    client: &NexarClient,
    data: &[f32],
    dest: Rank,
    tag: u32,
) -> Result<()> {
    let bytes = bytemuck::cast_slice::<f32, u8>(data);
    send_bytes(client, bytes, dest, tag).await
}

/// Receives an f32 tensor from a remote rank.
pub async fn recv_tensor_f32(
    client: &NexarClient,
    buf: &mut [f32],
    src: Rank,
    tag: u32,
) -> Result<()> {
    let bytes = bytemuck::cast_slice_mut::<f32, u8>(buf);
    recv_bytes(client, bytes, src, tag).await
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
}

/// Generation request header: [seq_len: u32, max_tokens: u32, position: u32].
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
}
