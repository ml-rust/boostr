//! Swarm mode: multi-node distributed inference via nexar.
//!
//! Enables running large models (e.g., 671B DeepSeek) across multiple machines
//! connected via network. Uses nexar's QUIC transport for inter-node communication
//! and NCCL for intra-node GPU communication.
//!
//! # Architecture
//!
//! ```text
//! Leader Node                    Worker Nodes
//! ┌──────────────────┐          ┌──────────────────┐
//! │ HTTP API Server  │          │                  │
//! │ Request Router   │◄─QUIC──►│ Layer Executor   │
//! │ Pipeline Manager │          │ Expert Pool      │
//! │ KV Cache Coord   │          │ KV Cache Shard   │
//! └──────────────────┘          └──────────────────┘
//! ```
//!
//! - **Leader**: Accepts HTTP requests, coordinates pipeline stages, manages topology
//! - **Worker**: Executes assigned model layers, participates in collective ops
//! - **Discovery**: mDNS for LAN auto-discovery, manual for WAN

use std::collections::HashMap;

/// Role of this node in the swarm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwarmRole {
    /// Coordinates requests, hosts API server
    Leader,
    /// Executes model layers
    Worker,
}

/// Configuration for swarm mode.
#[derive(Debug, Clone)]
pub struct SwarmConfig {
    /// Role of this node
    pub role: SwarmRole,
    /// Authentication token for swarm membership
    pub token: String,
    /// Listen address for nexar transport
    pub listen_addr: String,
    /// Leader address (workers connect to this; leader ignores)
    pub leader_addr: Option<String>,
    /// Enable mDNS auto-discovery on LAN
    pub mdns_discovery: bool,
}

impl Default for SwarmConfig {
    fn default() -> Self {
        Self {
            role: SwarmRole::Leader,
            token: String::new(),
            listen_addr: "0.0.0.0:4001".to_string(),
            leader_addr: None,
            mdns_discovery: true,
        }
    }
}

/// A node in the swarm topology.
#[derive(Debug, Clone)]
pub struct SwarmNode {
    /// Unique node identifier
    pub node_id: String,
    /// Nexar endpoint address
    pub address: String,
    /// Role
    pub role: SwarmRole,
    /// GPU count on this node
    pub gpu_count: usize,
    /// Available VRAM in bytes per GPU
    pub vram_per_gpu: Vec<u64>,
    /// Assigned pipeline stage range [start_layer, end_layer)
    pub assigned_layers: Option<(usize, usize)>,
}

/// Manages the swarm topology and layer assignment.
pub struct SwarmManager {
    config: SwarmConfig,
    nodes: HashMap<String, SwarmNode>,
}

impl SwarmManager {
    pub fn new(config: SwarmConfig) -> Self {
        Self {
            config,
            nodes: HashMap::new(),
        }
    }

    /// Register a new node in the swarm.
    pub fn register_node(&mut self, node: SwarmNode) {
        tracing::info!(
            node_id = %node.node_id,
            address = %node.address,
            gpus = node.gpu_count,
            role = ?node.role,
            "Node registered in swarm"
        );
        self.nodes.insert(node.node_id.clone(), node);
    }

    /// Remove a node from the swarm.
    pub fn remove_node(&mut self, node_id: &str) {
        if self.nodes.remove(node_id).is_some() {
            tracing::info!(node_id = %node_id, "Node removed from swarm");
        }
    }

    /// Compute optimal layer assignment across nodes based on available VRAM.
    ///
    /// Returns a mapping of node_id → (start_layer, end_layer).
    pub fn compute_layer_assignment(&self, total_layers: usize) -> HashMap<String, (usize, usize)> {
        let mut nodes: Vec<(&String, &SwarmNode)> = self.nodes.iter().collect();
        if nodes.is_empty() {
            return HashMap::new();
        }

        // Sort by total VRAM (largest first for balanced assignment)
        nodes.sort_by(|a, b| {
            let vram_a: u64 = a.1.vram_per_gpu.iter().sum();
            let vram_b: u64 = b.1.vram_per_gpu.iter().sum();
            vram_b.cmp(&vram_a)
        });

        // Simple proportional assignment based on VRAM
        let total_vram: u64 = nodes
            .iter()
            .map(|(_, n)| n.vram_per_gpu.iter().sum::<u64>())
            .sum();
        let mut assignments = HashMap::new();
        let mut current_layer = 0;

        for (i, (node_id, node)) in nodes.iter().enumerate() {
            let node_vram: u64 = node.vram_per_gpu.iter().sum();
            let num_layers = if i == nodes.len() - 1 {
                total_layers - current_layer
            } else {
                ((node_vram as f64 / total_vram as f64) * total_layers as f64).round() as usize
            };

            let end_layer = (current_layer + num_layers).min(total_layers);
            assignments.insert((*node_id).clone(), (current_layer, end_layer));
            current_layer = end_layer;
        }

        assignments
    }

    /// Get the swarm configuration.
    pub fn config(&self) -> &SwarmConfig {
        &self.config
    }

    /// Get the number of registered nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get total GPU count across all nodes.
    pub fn total_gpus(&self) -> usize {
        self.nodes.values().map(|n| n.gpu_count).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_assignment_single_node() {
        let mut mgr = SwarmManager::new(SwarmConfig::default());
        mgr.register_node(SwarmNode {
            node_id: "node-0".into(),
            address: "127.0.0.1:4001".into(),
            role: SwarmRole::Leader,
            gpu_count: 1,
            vram_per_gpu: vec![24 * 1024 * 1024 * 1024],
            assigned_layers: None,
        });

        let assignments = mgr.compute_layer_assignment(32);
        assert_eq!(assignments.get("node-0"), Some(&(0, 32)));
    }

    #[test]
    fn test_layer_assignment_two_nodes() {
        let mut mgr = SwarmManager::new(SwarmConfig::default());
        mgr.register_node(SwarmNode {
            node_id: "node-0".into(),
            address: "10.0.0.1:4001".into(),
            role: SwarmRole::Leader,
            gpu_count: 1,
            vram_per_gpu: vec![24_000_000_000],
            assigned_layers: None,
        });
        mgr.register_node(SwarmNode {
            node_id: "node-1".into(),
            address: "10.0.0.2:4001".into(),
            role: SwarmRole::Worker,
            gpu_count: 1,
            vram_per_gpu: vec![24_000_000_000],
            assigned_layers: None,
        });

        let assignments = mgr.compute_layer_assignment(32);
        // Equal VRAM → ~16 layers each
        let total_assigned: usize = assignments.values().map(|(s, e)| e - s).sum();
        assert_eq!(total_assigned, 32);
        assert_eq!(assignments.len(), 2);
    }
}
