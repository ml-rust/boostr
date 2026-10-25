//! Cluster formation and worker setup of a [`SwarmLeader`]: the seed, layer
//! assignments, the model path, readiness and the shutdown notice.

use anyhow::Result;
use nexar::{ClusterConfig, NexarClient, SeedNode};

use super::super::transport::{self, LayerAssignment, tags};
use super::swarm_leader::SwarmLeader;

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
        seed_addr: std::net::SocketAddr,
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

    /// Send layer assignments to all workers.
    pub async fn assign_layers(&self) -> Result<()> {
        for stage in self.schedule.stages() {
            let Some(rank) = self.rank_for_node(&stage.node_id) else {
                continue;
            };
            let layers = format!("{}..{}", stage.start_layer, stage.end_layer);
            if rank == self.client.rank() {
                tracing::info!(rank, layers, "Leader assigned layers (local)");
                continue;
            }
            let assignment = LayerAssignment {
                start_layer: stage.start_layer as u32,
                end_layer: stage.end_layer as u32,
                has_embedding: stage.has_embedding,
                has_lm_head: stage.has_lm_head,
            };
            let frame = assignment.to_frame();
            transport::send_frame(&self.client, rank, tags::LAYER_ASSIGNMENT, &frame).await?;
            tracing::info!(rank, layers, "Sent layer assignment to worker");
        }
        Ok(())
    }

    /// The leader's own assignment: its leading stage. `None` when the
    /// leader runs no stage.
    ///
    /// Build the leader's [`StageRunner`](super::super::stage::StageRunner)
    /// for it. Its output goes to [`SwarmLeader::pipeline_forward`].
    pub fn local_assignment(&self) -> Option<LayerAssignment> {
        let me = self.client.rank();
        self.schedule
            .stages()
            .iter()
            .find(|s| self.rank_for_node(&s.node_id) == Some(me))
            .map(|s| LayerAssignment {
                start_layer: s.start_layer as u32,
                end_layer: s.end_layer as u32,
                has_embedding: s.has_embedding,
                has_lm_head: s.has_lm_head,
            })
    }

    /// Send every worker the path of the model to load.
    pub async fn send_model_path(&self, path: &str) -> Result<()> {
        let frame = transport::model_path_frame(path);
        for rank in 1..self.client.world_size() {
            transport::send_frame(&self.client, rank, tags::MODEL_PATH, &frame).await?;
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
    /// Each worker ends its compute loop after the frame in progress and
    /// frees every request. A failed send to one worker is logged, and the
    /// others still get theirs.
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
}

#[cfg(test)]
mod tests {
    use super::super::super::stage::{Stage, StageOutput};
    use super::super::super::transport::{Activation, GenRequestHeader, StageOp, StagePayload};
    use super::super::super::worker::SwarmWorker;
    use super::*;
    use anyhow::{anyhow, bail};
    use nexar::Rank;
    use std::future::Future;
    use std::net::SocketAddr;
    use std::sync::Arc;
    use std::thread::JoinHandle;
    use std::time::Duration;

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

    /// Adds 1.0 to every hidden value. The last stage returns them as logits.
    struct AddOne {
        last: bool,
    }

    impl Stage for AddOne {
        fn forward(&mut self, _: &GenRequestHeader, payload: StagePayload) -> Result<StageOutput> {
            let StagePayload::Activation(a) = payload else {
                bail!("AddOne takes an activation");
            };
            let hidden: Vec<f32> = a.hidden.iter().map(|v| v + 1.0).collect();
            Ok(if self.last {
                StageOutput::Logits(hidden)
            } else {
                StageOutput::Activation(Activation { hidden, mlp: None })
            })
        }
        fn release(&mut self, _: u64) {}
        fn release_all(&mut self) {}
    }

    #[tokio::test]
    async fn test_form_cluster_single_node() {
        install_provider();
        let client = SwarmLeader::form_cluster(loopback(), &cfg(1))
            .await
            .unwrap();
        assert_eq!((client.rank(), client.world_size()), (0, 1));
        let leader = SwarmLeader::by_rank(Arc::new(client), 4, 8).unwrap();
        assert_eq!(leader.local_assignment().map(|a| a.end_layer), Some(4));
        let err = leader.release(1).await.unwrap_err();
        assert!(err.to_string().contains("no worker stage"), "{err}");
        leader.shutdown(WAIT).await.unwrap();
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
            let stage = Box::new(AddOne {
                last: next.is_none(),
            });
            let mut worker =
                SwarmWorker::new(client, rank, assignment, Some(rank - 1), next, 0, stage);
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
            assert_eq!(leader.local_assignment().map(|a| a.end_layer), Some(2));
            leader.assign_layers().await?;
            leader.wait_for_workers().await?;
            // Stage 0 runs on the leader: its output goes to rank 1, and
            // rank 2 returns one row of `vocab_size` logits.
            let header = GenRequestHeader {
                request_id: leader.next_request_id(),
                op: StageOp::Prefill,
                position: 0,
                seq_len: 1,
            };
            let payload = StagePayload::Activation(Activation {
                hidden: vec![1.0, 2.0, 3.0, 4.0],
                mlp: None,
            });
            let out = leader.pipeline_forward(header, &payload).await?;
            leader.release(header.request_id).await?;
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
