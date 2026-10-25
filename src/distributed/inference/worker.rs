//! Worker node: receives its layer assignment and serves its pipeline stage.
//!
//! The worker joins the leader's cluster, receives its layer assignment,
//! loads its stage, and runs a compute loop. The loop takes request frames
//! from the previous stage, runs them through its [`Stage`], and sends the
//! result to the next stage, or the logits to the leader on the last stage.

use anyhow::{Result, anyhow};
use nexar::{NexarClient, Rank};
use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::time::Duration;

use super::stage::{Stage, StageOutput};
use super::transport::tasks::{run_blocking, spawn_notice_listener};
use super::transport::{
    self, GenRequestHeader, LayerAssignment, LogitsFrame, RequestFrame, StageError, StageOp,
    StagePayload, tags,
};

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
    /// The stage this worker serves. `None` only while a forward runs.
    stage: Option<Box<dyn Stage>>,
}

impl SwarmWorker {
    /// A worker that serves `stage` for `assignment`.
    ///
    /// Requests come from `prev_rank`, or from the leader when there is no
    /// previous stage. The stage with the embedding takes token ids, every
    /// other stage takes activations. Output goes to `next_rank`, or to the
    /// leader as logits when there is no next stage.
    pub fn new(
        client: Arc<NexarClient>,
        rank: Rank,
        assignment: LayerAssignment,
        prev_rank: Option<Rank>,
        next_rank: Option<Rank>,
        leader_rank: Rank,
        stage: Box<dyn Stage>,
    ) -> Self {
        Self {
            client,
            rank,
            assignment,
            prev_rank,
            next_rank,
            leader_rank,
            stage: Some(stage),
        }
    }

    /// Wait for layer assignment from the leader.
    ///
    /// # Errors
    ///
    /// A failed receive, or a frame [`LayerAssignment::from_frame`] refuses.
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
    /// A failed receive, another protocol version, or a path that is not UTF-8.
    pub async fn receive_model_path(client: &NexarClient, leader_rank: Rank) -> Result<String> {
        let frame = transport::recv_frame(client, leader_rank, tags::MODEL_PATH).await?;
        transport::model_path_from_frame(&frame)
    }

    /// Send readiness acknowledgment to leader.
    pub async fn send_ready(client: &NexarClient, leader_rank: Rank) -> Result<()> {
        transport::send_frame(client, leader_rank, tags::WORKER_READY, &[1]).await
    }

    /// Run the worker's compute loop.
    ///
    /// Each frame is one [`tags::GEN_REQUEST`] frame from the previous stage,
    /// or from the leader when this worker has no previous stage. The loop
    /// serves frames in arrival order:
    /// - prefill or decode: runs the stage, then sends the activation to the
    ///   next stage, or a [`LogitsFrame`] to the leader on the last stage
    /// - release: frees the request, then forwards the release to the next stage
    ///
    /// A request the stage fails gets a [`StageError`] frame to the leader.
    /// The worker frees that request and keeps serving. A panic in the stage
    /// counts as an error of that request.
    ///
    /// The loop ends on the leader's [`tags::SHUTDOWN`] notice, after the
    /// frame in progress, or when the previous stage disconnects. It then
    /// frees every request. Call [`SwarmWorker::shutdown`] after it returns.
    ///
    /// # Errors
    ///
    /// A frame of another protocol version or with a malformed header, or a
    /// failed send to the next stage or the leader.
    pub async fn run_compute_loop(&mut self) -> Result<()> {
        tracing::info!(
            rank = self.rank,
            layers = format!(
                "{}..{}",
                self.assignment.start_layer, self.assignment.end_layer
            ),
            "Worker compute loop starting"
        );
        let (mut stop, _notice) =
            spawn_notice_listener(Arc::clone(&self.client), self.leader_rank, tags::SHUTDOWN);
        let src = self.prev_rank.unwrap_or(self.leader_rank);
        let result = loop {
            let received = tokio::select! {
                biased;
                _ = &mut stop => {
                    tracing::info!(rank = self.rank, "Worker received the shutdown notice");
                    break Ok(());
                }
                r = transport::recv_frame(&self.client, src, tags::GEN_REQUEST) => r,
            };
            let frame = match received {
                Ok(frame) => frame,
                Err(e) => {
                    tracing::warn!(
                        rank = self.rank,
                        "Worker stops: no request from rank {src}: {e}"
                    );
                    break Ok(());
                }
            };
            if let Err(e) = self.serve(&frame).await {
                break Err(e);
            }
        };
        if let Some(stage) = self.stage.as_mut() {
            stage.release_all();
        }
        tracing::info!(rank = self.rank, "Worker compute loop ended");
        result
    }

    /// Serves one request frame. An error ends the loop.
    async fn serve(&mut self, frame: &[u8]) -> Result<()> {
        let RequestFrame {
            header,
            frame_id,
            payload,
        } = GenRequestHeader::split_frame(frame).map_err(|e| {
            e.context(format!(
                "rank {} got a request frame it cannot read",
                self.rank
            ))
        })?;
        if header.op == StageOp::Release {
            self.stage_mut()?.release(header.request_id);
            if let Some(next) = self.next_rank {
                let release = header.frame(frame_id, &[]);
                transport::send_frame(&self.client, next, tags::GEN_REQUEST, &release).await?;
            }
            return Ok(());
        }
        let output =
            match StagePayload::decode(payload, header.seq_len, self.assignment.has_embedding) {
                Ok(payload) => self.forward(header, payload).await?,
                Err(e) => Err(e),
            };
        match (output, self.next_rank) {
            (Ok(StageOutput::Activation(a)), Some(next)) => {
                match StagePayload::Activation(a).encode() {
                    Ok(bytes) => {
                        let out = header.frame(frame_id, &bytes);
                        transport::send_frame(&self.client, next, tags::GEN_REQUEST, &out).await
                    }
                    Err(e) => self.fail(&header, frame_id, e).await,
                }
            }
            (Ok(StageOutput::Logits(logits)), None) => {
                let out = LogitsFrame {
                    request_id: header.request_id,
                    frame_id,
                    logits,
                }
                .to_frame();
                transport::send_frame(&self.client, self.leader_rank, tags::LOGITS, &out).await
            }
            (Ok(StageOutput::Activation(_)), None) => {
                let e = anyhow!("the last stage returned an activation, not logits");
                self.fail(&header, frame_id, e).await
            }
            (Ok(StageOutput::Logits(_)), Some(next)) => {
                let e = anyhow!("the stage returned logits, and the stage on rank {next} follows");
                self.fail(&header, frame_id, e).await
            }
            (Err(e), _) => self.fail(&header, frame_id, e).await,
        }
    }

    /// Runs the stage on the blocking pool. The outer error is fatal: the
    /// stage did not come back. The inner error fails this request only.
    async fn forward(
        &mut self,
        header: GenRequestHeader,
        payload: StagePayload,
    ) -> Result<Result<StageOutput>> {
        let mut stage = self.stage.take().ok_or_else(|| stage_lost(self.rank))?;
        let (stage, result) = run_blocking(move || {
            let run =
                std::panic::catch_unwind(AssertUnwindSafe(|| stage.forward(&header, payload)));
            let result = run.unwrap_or_else(|panic| {
                let what = panic
                    .downcast_ref::<&str>()
                    .map(|s| s.to_string())
                    .or_else(|| panic.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "a non-string panic".into());
                Err(anyhow!("the stage panicked: {what}"))
            });
            (stage, result)
        })
        .await?;
        self.stage = Some(stage);
        Ok(result)
    }

    /// Frees the request of `header` and sends the leader the [`StageError`]
    /// of frame `frame_id`.
    async fn fail(
        &mut self,
        header: &GenRequestHeader,
        frame_id: u64,
        e: anyhow::Error,
    ) -> Result<()> {
        tracing::warn!(
            rank = self.rank,
            request_id = header.request_id,
            "stage failed: {e:#}"
        );
        self.stage_mut()?.release(header.request_id);
        let err = StageError {
            request_id: header.request_id,
            frame_id,
            rank: self.rank,
            message: format!("{e:#}"),
        };
        transport::send_frame(
            &self.client,
            self.leader_rank,
            tags::STAGE_ERROR,
            &err.to_frame(),
        )
        .await
    }

    fn stage_mut(&mut self) -> Result<&mut Box<dyn Stage>> {
        let rank = self.rank;
        self.stage.as_mut().ok_or_else(|| stage_lost(rank))
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

/// The error once the stage did not come back from the blocking pool.
fn stage_lost(rank: Rank) -> anyhow::Error {
    anyhow!("rank {rank} lost its stage during a forward")
}
