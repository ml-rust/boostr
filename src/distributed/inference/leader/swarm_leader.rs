//! [`SwarmLeader`]: sends each request frame into the pipeline.
//!
//! The leader holds rank 0. It runs the leading stages itself, through its
//! own [`Stage`](super::super::stage::Stage), and sends their output to the
//! first worker stage. A dispatcher routes each reply back by request id,
//! so several requests fill the pipeline at once.

use anyhow::{Result, anyhow, bail};
use nexar::{NexarClient, Rank};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use super::super::pipeline::PipelineSchedule;
use super::super::topology::{SwarmNode, SwarmRole};
use super::super::transport::{self, GenRequestHeader, StageOp, StagePayload, tags};
use super::dispatch::Dispatch;
use super::route::{Route, route};

/// Leader node state after cluster formation.
pub struct SwarmLeader {
    /// Nexar client for communication (leader is rank 0)
    pub(super) client: Arc<NexarClient>,
    /// Pipeline schedule
    pub(super) schedule: PipelineSchedule,
    /// Rank → node mapping
    rank_nodes: HashMap<Rank, SwarmNode>,
    /// The worker part of the pipeline and its dispatcher. `None` when
    /// every stage runs on the leader.
    remote: Option<(Route, Dispatch)>,
    next_request_id: AtomicU64,
    /// The frame id of the next request frame. Release frames use 0.
    next_frame_id: AtomicU64,
}

/// A waiter of [`SwarmLeader::pipeline_forward`]. Dropping it, when the call
/// ends or its future drops, removes the waiter of its frame, so the
/// request id takes its next frame. A late reply to the frame drops.
struct WaitGuard<'a> {
    dispatch: &'a Dispatch,
    request_id: u64,
    frame_id: u64,
}

impl Drop for WaitGuard<'_> {
    fn drop(&mut self) {
        self.dispatch.forget(self.request_id, Some(self.frame_id));
    }
}

impl SwarmLeader {
    /// A leader whose schedule runs stage `r` on rank `r`.
    ///
    /// See [`PipelineSchedule::by_rank`]. A worker's previous stage is
    /// `rank - 1` and its next stage is `rank + 1`, so ranks in arrival order
    /// fit the schedule. The leader runs stage 0.
    ///
    /// # Errors
    ///
    /// A client that is not rank 0, more ranks than layers, or an error of
    /// [`SwarmLeader::new`].
    pub fn by_rank(
        client: Arc<NexarClient>,
        total_layers: usize,
        vocab_size: usize,
    ) -> Result<Self> {
        if client.rank() != 0 {
            return Err(anyhow!(
                "the swarm leader must hold rank 0, and this client holds rank {}",
                client.rank()
            ));
        }
        let schedule = PipelineSchedule::by_rank(client.world_size(), total_layers)?;
        let rank_nodes = schedule
            .stages()
            .iter()
            .enumerate()
            .map(|(r, stage)| {
                let node = SwarmNode {
                    node_id: stage.node_id.clone(),
                    address: String::new(),
                    role: if r == 0 {
                        SwarmRole::Leader
                    } else {
                        SwarmRole::Worker
                    },
                    gpu_count: 1,
                    vram_per_gpu: Vec::new(),
                    assigned_layers: Some((stage.start_layer, stage.end_layer)),
                };
                (r as Rank, node)
            })
            .collect();
        Self::new(client, schedule, rank_nodes, vocab_size)
    }

    /// A leader over a formed cluster.
    ///
    /// The leader runs only leading stages, and the last stage runs on a
    /// worker. With a worker stage, the dispatcher starts on the current
    /// tokio runtime and listens for every worker's replies.
    ///
    /// # Errors
    ///
    /// A stage with no rank in `rank_nodes`, a leader stage after a worker
    /// stage, or no tokio runtime on this thread.
    pub fn new(
        client: Arc<NexarClient>,
        schedule: PipelineSchedule,
        rank_nodes: HashMap<Rank, SwarmNode>,
        vocab_size: usize,
    ) -> Result<Self> {
        let rank_of = |id: &str| {
            rank_nodes
                .iter()
                .find(|(_, node)| node.node_id == id)
                .map(|(&rank, _)| rank)
        };
        let remote = match route(&schedule, client.rank(), rank_of)? {
            Some(r) => {
                let dispatch = Dispatch::start(&client, r.last, &r.workers, vocab_size)?;
                Some((r, dispatch))
            }
            None => None,
        };
        Ok(Self {
            client,
            schedule,
            rank_nodes,
            remote,
            next_request_id: AtomicU64::new(1),
            next_frame_id: AtomicU64::new(1),
        })
    }

    /// A request id no earlier call on this leader returned.
    pub fn next_request_id(&self) -> u64 {
        self.next_request_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Stop every worker, stop the dispatcher, then shut down the leader's
    /// nexar client. Call it before the process exits. A request still
    /// waiting gets an error.
    ///
    /// # Errors
    ///
    /// The error of [`transport::shutdown_shared`].
    pub async fn shutdown(mut self, release_within: Duration) -> Result<()> {
        self.shutdown_workers().await?;
        // The dispatcher's listeners hold the client. Stop them first.
        self.remote = None;
        transport::shutdown_shared(self.client, release_within).await
    }

    /// Sends one prefill or decode frame into the worker stages and waits
    /// for its logits: one row of `vocab_size` for the frame's last position.
    ///
    /// When the leader runs the leading stages, `payload` is their output
    /// [`StagePayload::Activation`]. When it runs none, `payload` is the
    /// token ids. Calls for different request ids run concurrently. One
    /// request takes one frame at a time: wait for its reply before its
    /// next frame. Call [`SwarmLeader::release`] when the request ends, also
    /// after an error.
    ///
    /// Dropping the returned future stops the wait: the id takes its next
    /// frame at once, and the reply to the dropped frame drops when it
    /// arrives. The stages still run the dropped frame, so release the id
    /// before a new prefill of it.
    ///
    /// # Errors
    ///
    /// - a release header: use [`SwarmLeader::release`]
    /// - no worker stage, or a payload of the wrong kind for the first worker stage
    /// - token ids that do not match `header.seq_len`
    /// - a request id that already has a frame in the pipeline
    /// - a failed send, or a lost stage
    /// - the stage's error, as a [`StageError`](super::super::transport::StageError)
    ///   inside the error: `err.downcast_ref::<StageError>()`
    /// - logits of another length than `vocab_size`
    pub async fn pipeline_forward(
        &self,
        header: GenRequestHeader,
        payload: &StagePayload,
    ) -> Result<Vec<f32>> {
        let id = header.request_id;
        if header.op == StageOp::Release {
            bail!("request {id}: send a release with SwarmLeader::release");
        }
        let (route, dispatch) = self.remote()?;
        if payload.is_tokens() != route.entry_takes_tokens {
            bail!(
                "request {id}: the first worker stage on rank {} takes {}",
                route.entry,
                if route.entry_takes_tokens {
                    "token ids"
                } else {
                    "the activation of the leader's stage"
                }
            );
        }
        if let StagePayload::Tokens(ids) = payload
            && ids.len() != header.seq_len as usize
        {
            bail!(
                "request {id}: the header covers {} positions and the payload holds {} token ids",
                header.seq_len,
                ids.len()
            );
        }
        let frame_id = self.next_frame_id.fetch_add(1, Ordering::Relaxed);
        let frame = header.frame(frame_id, &payload.encode()?);
        let reply = dispatch.expect(id, frame_id)?;
        let _waiter = WaitGuard {
            dispatch,
            request_id: id,
            frame_id,
        };
        transport::send_frame(&self.client, route.entry, tags::GEN_REQUEST, &frame)
            .await
            .map_err(|e| e.context(format!("request {id}")))?;
        reply
            .await
            .map_err(|_| anyhow!("request {id}: the swarm leader shut down before its reply"))?
    }

    /// Frees `request_id` on every worker stage.
    ///
    /// The release travels the same path as the request frames, so it never
    /// overtakes one. Each stage frees the request and forwards the release.
    /// No reply comes back. Free the leader's own stage separately.
    ///
    /// # Errors
    ///
    /// No worker stage, or a failed send.
    pub async fn release(&self, request_id: u64) -> Result<()> {
        let (route, dispatch) = self.remote()?;
        dispatch.forget(request_id, None);
        let frame = GenRequestHeader::release(request_id).frame(0, &[]);
        transport::send_frame(&self.client, route.entry, tags::GEN_REQUEST, &frame).await
    }

    fn remote(&self) -> Result<(&Route, &Dispatch)> {
        self.remote
            .as_ref()
            .map(|(r, d)| (r, d))
            .ok_or_else(|| anyhow!("every pipeline stage runs on the leader: no worker stage"))
    }

    pub(super) fn rank_for_node(&self, node_id: &str) -> Option<Rank> {
        self.rank_nodes
            .iter()
            .find(|(_, node)| node.node_id == node_id)
            .map(|(&rank, _)| rank)
    }

    pub fn schedule(&self) -> &PipelineSchedule {
        &self.schedule
    }

    pub fn client(&self) -> &Arc<NexarClient> {
        &self.client
    }
}
