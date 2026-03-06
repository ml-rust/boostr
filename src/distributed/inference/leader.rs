//! Leader node: coordinates the swarm, assigns layers, routes activations.
//!
//! The leader runs the seed node for cluster formation, computes layer
//! assignments based on node capabilities, and manages the pipeline
//! execution for inference requests.

use anyhow::{Result, anyhow};
use nexar::{NexarClient, Rank};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use super::pipeline::PipelineSchedule;
use super::topology::SwarmNode;
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
}

impl SwarmLeader {
    /// Form a cluster by starting a seed node and waiting for workers.
    ///
    /// Returns (leader_client, all_clients_for_mesh) after cluster formation.
    pub async fn form_cluster(
        _listen_addr: SocketAddr,
        world_size: u32,
    ) -> Result<Vec<NexarClient>> {
        let adapter = super::transport::cpu_adapter();
        let clients = NexarClient::bootstrap_local(world_size, adapter)
            .await
            .map_err(|e| anyhow!("Cluster formation failed: {}", e))?;
        Ok(clients)
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
                transport::send_bytes(&self.client, &bytes, rank, tags::LAYER_ASSIGNMENT).await?;
                tracing::info!(
                    rank = rank,
                    layers = format!("{}..{}", stage.start_layer, stage.end_layer),
                    "Sent layer assignment to worker"
                );
            }
        }
        Ok(())
    }

    /// Wait for all workers to report readiness.
    pub async fn wait_for_workers(&self) -> Result<()> {
        let world = self.client.world_size();
        for rank in 1..world {
            let mut ack = [0u8; 1];
            transport::recv_bytes(&self.client, &mut ack, rank, tags::WORKER_READY).await?;
            tracing::info!(rank = rank, "Worker ready");
        }
        tracing::info!("All {} workers ready", world - 1);
        Ok(())
    }

    /// Send shutdown signal to all workers.
    pub async fn shutdown_workers(&self) -> Result<()> {
        let world = self.client.world_size();
        let shutdown_msg = [0u8; 1];
        for rank in 1..world {
            let _ = transport::send_bytes(&self.client, &shutdown_msg, rank, tags::SHUTDOWN).await;
        }
        tracing::info!("Shutdown signal sent to all workers");
        Ok(())
    }

    /// Execute a pipeline forward pass across all stages.
    ///
    /// For a single request:
    /// 1. Send generation request to first stage
    /// 2. If leader is first stage, run local layers and send activations
    /// 3. Wait for logits from last stage
    pub async fn pipeline_forward(
        &self,
        input_data: &[f32],
        seq_len: u32,
        max_tokens: u32,
    ) -> Result<Vec<f32>> {
        let stages = self.schedule.stages();
        if stages.is_empty() {
            return Err(anyhow!("No pipeline stages configured"));
        }

        let first_stage = &stages[0];
        let last_stage = stages.last().unwrap();

        let first_rank = self.rank_for_node(&first_stage.node_id);
        let last_rank = self.rank_for_node(&last_stage.node_id);

        let header = transport::GenRequestHeader {
            seq_len,
            max_tokens,
            position: 0,
        };
        let header_bytes = header.to_bytes();

        if let Some(first) = first_rank {
            if first != self.client.rank() {
                transport::send_bytes(&self.client, &header_bytes, first, tags::GEN_REQUEST)
                    .await?;
                let input_bytes = bytemuck::cast_slice::<f32, u8>(input_data);
                transport::send_bytes(&self.client, input_bytes, first, tags::ACTIVATION).await?;
            }
        }

        // Wait for logits from last stage
        if let Some(last) = last_rank {
            if last != self.client.rank() {
                // Logits size = batch_size * vocab_size. The input_data length
                // encodes batch_size * hidden_size, so we derive batch_size and
                // compute the correct logits buffer size.
                let batch_size = (seq_len as usize).max(1);
                let logits_size = batch_size * self.vocab_size;
                let mut logits_buf = vec![0f32; logits_size];
                transport::recv_tensor_f32(&self.client, &mut logits_buf, last, tags::LOGITS)
                    .await?;
                return Ok(logits_buf);
            }
        }

        // If leader is both first and last, execute locally
        Ok(input_data.to_vec())
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

    #[tokio::test]
    async fn test_form_cluster_single_node() {
        rustls::crypto::ring::default_provider()
            .install_default()
            .ok(); // ignore if already installed
        let clients = SwarmLeader::form_cluster("127.0.0.1:0".parse().unwrap(), 1)
            .await
            .unwrap();
        assert_eq!(clients.len(), 1);
        assert_eq!(clients[0].rank(), 0);
    }
}
