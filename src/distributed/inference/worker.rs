//! Worker node: receives layer assignments and executes forward passes.
//!
//! The worker connects to the leader's seed node, receives its layer
//! assignment, loads the assigned model layers, and enters a compute
//! loop that receives activations, runs its layers, and forwards
//! results to the next pipeline stage.

use anyhow::Result;
use nexar::{NexarClient, Rank};
use std::sync::Arc;

use super::transport::{self, GenRequestHeader, LayerAssignment, tags};

/// Forward function: `(activation_bytes, size_hint) -> output_bytes`
pub type ForwardFn = Box<dyn Fn(&[u8], usize) -> Vec<u8> + Send + Sync>;

/// Worker state after connecting to the swarm and receiving assignment.
pub struct SwarmWorker {
    /// Nexar client for communication
    client: Arc<NexarClient>,
    /// This worker's rank
    rank: Rank,
    /// Assigned layer range
    assignment: LayerAssignment,
    /// Rank of the previous pipeline stage (sends us activations)
    prev_rank: Option<Rank>,
    /// Rank of the next pipeline stage (we send activations to)
    next_rank: Option<Rank>,
    /// Leader rank (for control messages)
    leader_rank: Rank,
    /// Forward function: takes activation bytes + size hint, returns output bytes.
    ///
    /// Provided by the creator who has access to the model and runtime. The size
    /// hint is `hidden_size` for intermediate stages and `seq_len` for the first
    /// stage (which receives token IDs rather than activations).
    forward_fn: ForwardFn,
}

impl SwarmWorker {
    /// Create a new worker after receiving its assignment.
    ///
    /// `forward_fn` is called on each request with the raw activation bytes and
    /// a size hint. For intermediate stages the hint is `hidden_size`; for the
    /// first stage (no `prev_rank`) the hint is `seq_len` and the bytes are
    /// token IDs encoded as little-endian `i64`. The function must return the
    /// serialised output activation bytes.
    pub fn new(
        client: Arc<NexarClient>,
        rank: Rank,
        assignment: LayerAssignment,
        prev_rank: Option<Rank>,
        next_rank: Option<Rank>,
        leader_rank: Rank,
        forward_fn: ForwardFn,
    ) -> Self {
        Self {
            client,
            rank,
            assignment,
            prev_rank,
            next_rank,
            leader_rank,
            forward_fn,
        }
    }

    /// Wait for layer assignment from the leader.
    pub async fn receive_assignment(
        client: &NexarClient,
        leader_rank: Rank,
    ) -> Result<LayerAssignment> {
        let mut buf = [0u8; 10];
        transport::recv_bytes(client, &mut buf, leader_rank, tags::LAYER_ASSIGNMENT).await?;
        Ok(LayerAssignment::from_bytes(&buf))
    }

    /// Send readiness acknowledgment to leader.
    pub async fn send_ready(client: &NexarClient, leader_rank: Rank) -> Result<()> {
        let ack = [1u8];
        transport::send_bytes(client, &ack, leader_rank, tags::WORKER_READY).await
    }

    /// Run the worker's compute loop.
    ///
    /// This loop:
    /// 1. Receives a generation request header (from leader or prev stage)
    /// 2. Receives activation tensor from previous stage (or creates embedding if first stage)
    /// 3. Runs forward pass through assigned layers
    /// 4. Sends activation to next stage (or logits to leader if last stage)
    /// 5. Repeats for decode steps
    pub async fn run_compute_loop(&self) -> Result<()> {
        tracing::info!(
            rank = self.rank,
            layers = format!(
                "{}..{}",
                self.assignment.start_layer, self.assignment.end_layer
            ),
            "Worker compute loop starting"
        );

        loop {
            // Wait for generation request or shutdown
            let mut header_buf = [0u8; 12];
            let control_src = self.prev_rank.unwrap_or(self.leader_rank);

            // Try to receive — could be a gen request or shutdown
            // First check for shutdown (non-blocking would be ideal, but we use tag dispatch)
            match transport::recv_bytes(
                &self.client,
                &mut header_buf,
                control_src,
                tags::GEN_REQUEST,
            )
            .await
            {
                Ok(()) => {}
                Err(e) => {
                    tracing::warn!("Worker recv error (may be shutdown): {}", e);
                    break;
                }
            }

            let header = GenRequestHeader::from_bytes(&header_buf);
            let hidden_size = header.seq_len as usize; // reused field for tensor size

            tracing::debug!(
                rank = self.rank,
                seq_len = header.seq_len,
                max_tokens = header.max_tokens,
                "Received generation request"
            );

            // Receive activation from previous stage (or token IDs if first stage),
            // run forward pass through assigned layers, then forward to next stage.
            if let Some(prev) = self.prev_rank {
                // Intermediate or last stage: receive f32 activation tensor.
                let tensor_bytes = hidden_size * std::mem::size_of::<f32>();
                let mut activation_buf = vec![0u8; tensor_bytes];
                transport::recv_bytes(&self.client, &mut activation_buf, prev, tags::ACTIVATION)
                    .await?;

                // Run forward pass through assigned layers.
                let output_buf = (self.forward_fn)(&activation_buf, hidden_size);

                if let Some(next) = self.next_rank {
                    transport::send_bytes(&self.client, &output_buf, next, tags::ACTIVATION)
                        .await?;
                } else {
                    // Last stage: send logits to leader.
                    transport::send_bytes(
                        &self.client,
                        &output_buf,
                        self.leader_rank,
                        tags::LOGITS,
                    )
                    .await?;
                }
            } else {
                // First stage: receive token IDs from leader (encoded as i64 LE bytes),
                // compute embeddings + run assigned layers.
                let token_bytes = header.seq_len as usize * std::mem::size_of::<i64>();
                let mut token_buf = vec![0u8; token_bytes];
                transport::recv_bytes(
                    &self.client,
                    &mut token_buf,
                    self.leader_rank,
                    tags::ACTIVATION,
                )
                .await?;

                // The forward_fn receives the raw token ID bytes; the size hint
                // is seq_len so the function can determine the number of tokens.
                let output_buf = (self.forward_fn)(&token_buf, header.seq_len as usize);

                if let Some(next) = self.next_rank {
                    transport::send_bytes(&self.client, &output_buf, next, tags::ACTIVATION)
                        .await?;
                } else {
                    // Single-stage model: this node is both first and last.
                    transport::send_bytes(
                        &self.client,
                        &output_buf,
                        self.leader_rank,
                        tags::LOGITS,
                    )
                    .await?;
                }
            }

            tracing::debug!(rank = self.rank, "Forward pass complete for request");
        }

        tracing::info!(rank = self.rank, "Worker compute loop ended");
        Ok(())
    }

    pub fn rank(&self) -> Rank {
        self.rank
    }

    pub fn assignment(&self) -> &LayerAssignment {
        &self.assignment
    }
}
