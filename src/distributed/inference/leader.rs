//! Leader node: coordinates the swarm, assigns layers, routes activations.
//!
//! The leader runs the seed node for cluster formation, computes layer
//! assignments based on node capabilities, and manages the pipeline
//! execution for inference requests.

use anyhow::{Result, anyhow};
use nexar::{ClusterConfig, NexarClient, Rank, SeedNode};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use super::pipeline::PipelineSchedule;
use super::topology::{SwarmNode, SwarmRole};
use super::transport::{self, LayerAssignment, tags};

/// Leader node state after cluster formation.
pub struct SwarmLeader {
    /// Nexar client for communication (leader is rank 0)
    client: Arc<NexarClient>,
    /// Pipeline schedule
    schedule: PipelineSchedule,
    /// Rank → node mapping
    rank_nodes: HashMap<Rank, SwarmNode>,
    /// Vocabulary size (needed to determine logits output size).
    vocab_size: usize,
    /// Held across one request's send and its logits receive, so concurrent
    /// [`SwarmLeader::pipeline_forward`] calls never take each other's logits.
    forward_lock: tokio::sync::Mutex<()>,
}

impl SwarmLeader {
    /// Form the cluster as rank 0: bind the seed on `seed_addr` and wait
    /// until every rank in `cfg.world_size` joins.
    ///
    /// Each worker process calls [`NexarClient::bootstrap_join`] with the same
    /// `seed_addr`, world size and cluster token. `cfg.bind_addr` and
    /// `cfg.advertise_addr` set the leader's mesh listener. A non-loopback
    /// address needs a cluster token, from `cfg` or `NEXAR_CLUSTER_TOKEN`.
    ///
    /// # Errors
    ///
    /// nexar's typed formation error: `TokenRequired`, `ClusterTokenMismatch`,
    /// `ClusterFormationTimeout`, `BootstrapAborted` and the others.
    pub async fn form_cluster(
        seed_addr: SocketAddr,
        cfg: &ClusterConfig,
    ) -> nexar::Result<NexarClient> {
        let seed = SeedNode::from_config(seed_addr, cfg)?;
        Self::form_cluster_with(seed, cfg).await
    }

    /// Like [`SwarmLeader::form_cluster`], with a seed the caller bound.
    ///
    /// Bind with [`SeedNode::from_config`] and read [`SeedNode::local_addr`]
    /// to learn the port of a `seed_addr` with port 0.
    ///
    /// # Errors
    ///
    /// As [`SwarmLeader::form_cluster`].
    pub async fn form_cluster_with(
        seed: SeedNode,
        cfg: &ClusterConfig,
    ) -> nexar::Result<NexarClient> {
        let cfg = cfg.clone().with_requested_rank(0);
        NexarClient::bootstrap_seed_with(seed, &cfg, transport::cpu_adapter()).await
    }

    /// A leader whose schedule runs stage `r` on rank `r`.
    ///
    /// See [`PipelineSchedule::by_rank`]. A worker's previous stage is
    /// `rank - 1` and its next stage is `rank + 1`, so ranks in arrival order
    /// fit the schedule.
    ///
    /// # Errors
    ///
    /// A client that is not rank 0, or more ranks than layers.
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
        Ok(Self::new(client, schedule, rank_nodes, vocab_size))
    }

    /// Create a leader from a pre-formed cluster.
    pub fn new(
        client: Arc<NexarClient>,
        schedule: PipelineSchedule,
        rank_nodes: HashMap<Rank, SwarmNode>,
        vocab_size: usize,
    ) -> Self {
        Self {
            client,
            schedule,
            rank_nodes,
            vocab_size,
            forward_lock: tokio::sync::Mutex::new(()),
        }
    }

    /// Send layer assignments to all workers.
    pub async fn assign_layers(&self) -> Result<()> {
        for stage in self.schedule.stages() {
            let rank = self
                .rank_nodes
                .iter()
                .find(|(_, node)| node.node_id == stage.node_id)
                .map(|(&rank, _)| rank);

            if let Some(rank) = rank {
                if rank == self.client.rank() {
                    tracing::info!(
                        rank = rank,
                        layers = format!("{}..{}", stage.start_layer, stage.end_layer),
                        "Leader assigned layers (local)"
                    );
                    continue;
                }

                let assignment = LayerAssignment {
                    start_layer: stage.start_layer as u32,
                    end_layer: stage.end_layer as u32,
                    has_embedding: stage.has_embedding,
                    has_lm_head: stage.has_lm_head,
                };
                let bytes = assignment.to_bytes();
                transport::send_frame(&self.client, rank, tags::LAYER_ASSIGNMENT, &bytes).await?;
                tracing::info!(
                    rank = rank,
                    layers = format!("{}..{}", stage.start_layer, stage.end_layer),
                    "Sent layer assignment to worker"
                );
            }
        }
        Ok(())
    }

    /// Send every worker the path of the model to load.
    pub async fn send_model_path(&self, path: &str) -> Result<()> {
        for rank in 1..self.client.world_size() {
            transport::send_frame(&self.client, rank, tags::MODEL_PATH, path.as_bytes()).await?;
            tracing::info!(rank, path, "Sent model path to worker");
        }
        Ok(())
    }

    /// Wait for all workers to report readiness.
    pub async fn wait_for_workers(&self) -> Result<()> {
        let world = self.client.world_size();
        for rank in 1..world {
            transport::recv_frame(&self.client, rank, tags::WORKER_READY).await?;
            tracing::info!(rank = rank, "Worker ready");
        }
        tracing::info!("All {} workers ready", world - 1);
        Ok(())
    }

    /// Send every worker the [`tags::SHUTDOWN`] notice.
    ///
    /// Each worker ends its compute loop after the request in progress. A
    /// failed send to one worker is logged, and the others still get theirs.
    pub async fn shutdown_workers(&self) -> Result<()> {
        let world = self.client.world_size();
        for rank in 1..world {
            if let Err(e) = transport::send_frame(&self.client, rank, tags::SHUTDOWN, &[1]).await {
                tracing::warn!(rank, "shutdown notice not sent: {e}");
            }
        }
        tracing::info!("Shutdown signal sent to all workers");
        Ok(())
    }

    /// Stop every worker, then shut down the leader's nexar client. Call it
    /// before the process exits.
    ///
    /// # Errors
    ///
    /// The error of [`transport::shutdown_shared`].
    pub async fn shutdown(self, release_within: Duration) -> Result<()> {
        self.shutdown_workers().await?;
        transport::shutdown_shared(self.client, release_within).await
    }

    /// Execute a pipeline forward pass across all stages.
    ///
    /// For a single request:
    /// 1. When the first stage runs on another rank, send it `input_data`.
    ///    When it runs on the leader, `input_data` is the output of the
    ///    leader's layers, and it goes to the second stage.
    /// 2. Wait for the logits from the last stage: `seq_len.max(1) * vocab_size`
    ///    values. A pipeline of the leader alone returns `input_data`.
    ///
    /// Concurrent calls run one at a time.
    ///
    /// # Errors
    ///
    /// No stage, a failed send or receive, or logits of the wrong length.
    pub async fn pipeline_forward(
        &self,
        input_data: &[f32],
        seq_len: u32,
        max_tokens: u32,
    ) -> Result<Vec<f32>> {
        let stages = self.schedule.stages();
        let (Some(first_stage), Some(last_stage)) = (stages.first(), stages.last()) else {
            return Err(anyhow!("No pipeline stages configured"));
        };
        let me = self.client.rank();
        let target = match self.rank_for_node(&first_stage.node_id) {
            Some(first) if first != me => Some(first),
            _ => stages
                .get(1)
                .and_then(|second| self.rank_for_node(&second.node_id)),
        };
        let Some(target) = target else {
            return Ok(input_data.to_vec());
        };

        let header = transport::GenRequestHeader {
            seq_len: u32::try_from(input_data.len())
                .map_err(|_| anyhow!("an input of {} values is too large", input_data.len()))?,
            max_tokens,
            position: 0,
        };
        let input_bytes = bytemuck::cast_slice::<f32, u8>(input_data);
        let _one_request = self.forward_lock.lock().await;
        transport::send_frame(
            &self.client,
            target,
            tags::GEN_REQUEST,
            &header.frame(input_bytes),
        )
        .await?;

        let last = self
            .rank_for_node(&last_stage.node_id)
            .ok_or_else(|| anyhow!("no rank runs the last stage {}", last_stage.node_id))?;
        let want = (seq_len as usize).max(1) * self.vocab_size;
        let frame = transport::recv_frame(&self.client, last, tags::LOGITS).await?;
        if frame.len() != want * std::mem::size_of::<f32>() {
            return Err(anyhow!(
                "rank {last} sent {} bytes of logits, and {want} f32 values need {}",
                frame.len(),
                want * std::mem::size_of::<f32>()
            ));
        }
        Ok(frame
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| f32::from_le_bytes(c))
            .collect())
    }

    fn rank_for_node(&self, node_id: &str) -> Option<Rank> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::inference::worker::SwarmWorker;
    use std::future::Future;
    use std::thread::JoinHandle;

    const WAIT: Duration = Duration::from_secs(10);

    fn loopback() -> SocketAddr {
        "127.0.0.1:0".parse().unwrap()
    }

    fn cfg(world: u32) -> ClusterConfig {
        ClusterConfig::new(world)
            .with_bind_addr(loopback())
            .with_cluster_token("swarm-test-token")
            .with_join_timeout(Duration::from_secs(30))
            .with_nexar(nexar::NexarConfig::default())
    }

    fn install_provider() {
        // nexar's QUIC needs a process-wide rustls provider. A second install is a no-op error.
        let _ = rustls::crypto::ring::default_provider().install_default();
    }

    /// Run `f` on its own thread with its own tokio runtime, as its own process would.
    fn on_thread<F, Fut, T>(f: F) -> JoinHandle<T>
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: Future<Output = T>,
        T: Send + 'static,
    {
        std::thread::spawn(move || {
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .unwrap()
                .block_on(f())
        })
    }

    /// Adds 1.0 to every `f32` of the activation.
    fn add_one() -> super::super::worker::ForwardFn {
        Box::new(|bytes: &[u8], _| {
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|&c| (f32::from_le_bytes(c) + 1.0).to_le_bytes())
                .collect()
        })
    }

    #[tokio::test]
    async fn test_form_cluster_single_node() {
        install_provider();
        let client = SwarmLeader::form_cluster(loopback(), &cfg(1))
            .await
            .unwrap();
        assert_eq!((client.rank(), client.world_size()), (0, 1));
        client.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_form_cluster_without_a_token_refuses_a_public_bind() {
        install_provider();
        let open = ClusterConfig::new(1).with_cluster_token("");
        let err = SwarmLeader::form_cluster(loopback(), &open)
            .await
            .err()
            .unwrap();
        assert!(
            matches!(err, nexar::NexarError::TokenRequired { .. }),
            "{err}"
        );
    }

    /// One worker process: join, take the assignment, run the loop, shut down.
    fn start_worker(seed: SocketAddr) -> JoinHandle<Result<(Rank, LayerAssignment)>> {
        on_thread(move || async move {
            let client = NexarClient::bootstrap_join(seed, &cfg(3), transport::cpu_adapter())
                .await
                .map_err(|e| anyhow!("join: {e}"))?;
            let client = Arc::new(client);
            let (rank, world) = (client.rank(), client.world_size());
            let assignment = SwarmWorker::receive_assignment(&client, 0).await?;
            SwarmWorker::send_ready(&client, 0).await?;
            let next = (rank + 1 < world).then_some(rank + 1);
            let worker =
                SwarmWorker::new(client, rank, assignment, Some(rank - 1), next, 0, add_one());
            worker.run_compute_loop().await?;
            worker.shutdown(WAIT).await?;
            Ok((rank, assignment))
        })
    }

    /// Three processes: the schedule follows the ranks, an activation crosses
    /// both worker stages, and the shutdown notice ends every loop.
    #[test]
    fn test_swarm_assigns_by_rank_pipelines_and_shuts_down() {
        install_provider();
        let (tx, rx) = std::sync::mpsc::channel();
        let leader = on_thread(move || async move {
            let seed = SeedNode::from_config(loopback(), &cfg(3)).map_err(|e| anyhow!("{e}"))?;
            tx.send(seed.local_addr()).unwrap();
            let client = SwarmLeader::form_cluster_with(seed, &cfg(3))
                .await
                .map_err(|e| anyhow!("form: {e}"))?;
            let leader = SwarmLeader::by_rank(Arc::new(client), 6, 4)?;
            leader.assign_layers().await?;
            leader.wait_for_workers().await?;
            // Stage 0 runs on the leader: its output goes to rank 1, and
            // rank 2 returns one row of `vocab_size` logits.
            let out = leader.pipeline_forward(&[1.0, 2.0, 3.0, 4.0], 1, 1).await?;
            leader.shutdown(WAIT).await?;
            Ok::<_, anyhow::Error>(out)
        });
        let seed = rx.recv().unwrap();
        let workers = [start_worker(seed), start_worker(seed)];
        assert_eq!(leader.join().unwrap().unwrap(), [3.0, 4.0, 5.0, 6.0]);
        let mut got: Vec<_> = workers
            .into_iter()
            .map(|w| w.join().unwrap().unwrap())
            .map(|(r, a)| (r, a.start_layer, a.end_layer, a.has_lm_head))
            .collect();
        got.sort();
        assert_eq!(got, [(1, 2, 4, false), (2, 4, 6, true)]);
    }
}
