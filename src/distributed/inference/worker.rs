//! Worker node: receives layer assignments and executes forward passes.
//!
//! The worker connects to the leader's seed node, receives its layer
//! assignment, loads the assigned model layers, and enters a compute
//! loop that receives activations, runs its layers, and forwards
//! results to the next pipeline stage.

use anyhow::{Result, anyhow, bail};
use nexar::{NexarClient, Rank};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::oneshot;

use super::transport::{self, GenRequestHeader, LayerAssignment, tags};

/// Forward function: `(activation_bytes, size_hint) -> output_bytes`
pub type ForwardFn = Box<dyn Fn(&[u8], usize) -> Vec<u8> + Send + Sync>;

/// Aborts its task on drop.
struct AbortOnDrop(tokio::task::AbortHandle);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// A receiver that resolves once `leader` sends [`tags::SHUTDOWN`] or disconnects.
fn spawn_shutdown_listener(
    client: &Arc<NexarClient>,
    leader: Rank,
) -> (oneshot::Receiver<()>, AbortOnDrop) {
    let (tx, rx) = oneshot::channel();
    let client = Arc::clone(client);
    let task = tokio::spawn(async move {
        if let Err(e) = client.recv_bytes_tagged(leader, tags::SHUTDOWN).await {
            tracing::debug!(leader, "shutdown listener ended: {e}");
        }
        let _ = tx.send(());
    });
    (rx, AbortOnDrop(task.abort_handle()))
}

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
    /// Forward function: takes the payload bytes and their element count,
    /// returns output bytes.
    ///
    /// Provided by the creator who has access to the model and runtime.
    forward_fn: ForwardFn,
}

impl SwarmWorker {
    /// Create a new worker after receiving its assignment.
    ///
    /// `forward_fn` is called on each request with the payload bytes and
    /// their element count: `f32` LE activations on a stage with a
    /// `prev_rank`, `i64` LE token ids on the first stage. The function must
    /// return the serialised output activation bytes, or the logits on the
    /// last stage.
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
        let frame = transport::recv_frame(client, leader_rank, tags::LAYER_ASSIGNMENT).await?;
        LayerAssignment::from_frame(&frame)
    }

    /// Wait for the model path from the leader.
    ///
    /// # Errors
    ///
    /// A failed receive, or a path that is not UTF-8.
    pub async fn receive_model_path(client: &NexarClient, leader_rank: Rank) -> Result<String> {
        let frame = transport::recv_frame(client, leader_rank, tags::MODEL_PATH).await?;
        String::from_utf8(frame).map_err(|e| anyhow!("the model path is not UTF-8: {e}"))
    }

    /// Send readiness acknowledgment to leader.
    pub async fn send_ready(client: &NexarClient, leader_rank: Rank) -> Result<()> {
        transport::send_frame(client, leader_rank, tags::WORKER_READY, &[1]).await
    }

    /// Run the worker's compute loop.
    ///
    /// Each request is one [`tags::GEN_REQUEST`] frame from the previous
    /// stage, or from the leader when this worker has no previous stage. Its
    /// payload is `i64` LE token ids on the first stage and `f32` LE
    /// activations on every later stage. The loop runs the forward closure,
    /// then sends the output in a request frame to the next stage, or as a
    /// [`tags::LOGITS`] frame to the leader on the last stage.
    ///
    /// The loop ends on the leader's [`tags::SHUTDOWN`] notice, after the
    /// request in progress, or when the previous stage disconnects. Call
    /// [`SwarmWorker::shutdown`] after it returns.
    ///
    /// # Errors
    ///
    /// A request whose payload does not match its header, or a failed send.
    pub async fn run_compute_loop(&self) -> Result<()> {
        tracing::info!(
            rank = self.rank,
            layers = format!(
                "{}..{}",
                self.assignment.start_layer, self.assignment.end_layer
            ),
            "Worker compute loop starting"
        );
        let (mut stop, _notice) = spawn_shutdown_listener(&self.client, self.leader_rank);
        let control_src = self.prev_rank.unwrap_or(self.leader_rank);
        // The first stage reads token ids, every later stage reads activations.
        let elem = if self.prev_rank.is_some() {
            std::mem::size_of::<f32>()
        } else {
            std::mem::size_of::<i64>()
        };

        loop {
            let received = tokio::select! {
                biased;
                _ = &mut stop => {
                    tracing::info!(rank = self.rank, "Worker received the shutdown notice");
                    break;
                }
                r = transport::recv_frame(&self.client, control_src, tags::GEN_REQUEST) => r,
            };
            let frame = match received {
                Ok(frame) => frame,
                Err(e) => {
                    tracing::warn!(
                        rank = self.rank,
                        "Worker stops: no request from rank {control_src}: {e}"
                    );
                    break;
                }
            };
            let (header, payload) = GenRequestHeader::split_frame(&frame)?;
            let count = header.seq_len as usize;
            if payload.len() != count * elem {
                bail!(
                    "rank {} got a request of {} elements and a {}-byte payload, not {} bytes",
                    self.rank,
                    count,
                    payload.len(),
                    count * elem
                );
            }
            tracing::debug!(
                rank = self.rank,
                seq_len = header.seq_len,
                max_tokens = header.max_tokens,
                "Received generation request"
            );

            let output = (self.forward_fn)(payload, count);
            match self.next_rank {
                Some(next) => self.send_next(next, header, &output).await?,
                None => {
                    transport::send_frame(&self.client, self.leader_rank, tags::LOGITS, &output)
                        .await?
                }
            }
            tracing::debug!(rank = self.rank, "Forward pass complete for request");
        }

        tracing::info!(rank = self.rank, "Worker compute loop ended");
        Ok(())
    }

    /// Send the next stage one request frame that holds `activation`.
    ///
    /// The forwarded header's `seq_len` is the `f32` count of `activation`.
    async fn send_next(
        &self,
        next: Rank,
        header: GenRequestHeader,
        activation: &[u8],
    ) -> Result<()> {
        let header = GenRequestHeader {
            seq_len: u32::try_from(activation.len() / std::mem::size_of::<f32>())
                .map_err(|_| anyhow!("an activation of {} bytes is too large", activation.len()))?,
            ..header
        };
        transport::send_frame(
            &self.client,
            next,
            tags::GEN_REQUEST,
            &header.frame(activation),
        )
        .await
    }

    /// Shut down this worker's nexar client. Call it after
    /// [`SwarmWorker::run_compute_loop`] returns, before the process exits.
    ///
    /// # Errors
    ///
    /// The error of [`transport::shutdown_shared`].
    pub async fn shutdown(self, release_within: Duration) -> Result<()> {
        transport::shutdown_shared(self.client, release_within).await
    }

    pub fn rank(&self) -> Rank {
        self.rank
    }

    pub fn assignment(&self) -> &LayerAssignment {
        &self.assignment
    }
}
