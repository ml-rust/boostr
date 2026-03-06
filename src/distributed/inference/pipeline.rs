//! Pipeline parallelism for multi-node inference.
//!
//! Splits model layers across nodes. Each node processes its assigned
//! layer range and forwards activations to the next node via nexar.
//!
//! # Pipeline stages
//!
//! ```text
//! Node 0 (layers 0-15)    Node 1 (layers 16-31)
//! ┌──────────────┐        ┌──────────────┐
//! │ Embedding    │        │              │
//! │ Layers 0-15  │──act──►│ Layers 16-31 │
//! │              │        │ LM Head      │
//! └──────────────┘        └──────────────┘
//! ```

/// A pipeline stage representing a contiguous range of model layers.
#[derive(Debug, Clone)]
pub struct PipelineStage {
    /// Node ID that owns this stage
    pub node_id: String,
    /// Start layer index (inclusive)
    pub start_layer: usize,
    /// End layer index (exclusive)
    pub end_layer: usize,
    /// Whether this stage includes the embedding layer
    pub has_embedding: bool,
    /// Whether this stage includes the LM head
    pub has_lm_head: bool,
}

/// Pipeline schedule for a model split across multiple nodes.
pub struct PipelineSchedule {
    stages: Vec<PipelineStage>,
}

impl PipelineSchedule {
    /// Create a pipeline schedule from layer assignments.
    pub fn new(
        assignments: &std::collections::HashMap<String, (usize, usize)>,
        total_layers: usize,
    ) -> Self {
        let mut stages: Vec<PipelineStage> = assignments
            .iter()
            .map(|(node_id, &(start, end))| PipelineStage {
                node_id: node_id.clone(),
                start_layer: start,
                end_layer: end,
                has_embedding: start == 0,
                has_lm_head: end == total_layers,
            })
            .collect();

        // Sort by start layer for proper ordering
        stages.sort_by_key(|s| s.start_layer);

        Self { stages }
    }

    /// Get pipeline stages in execution order.
    pub fn stages(&self) -> &[PipelineStage] {
        &self.stages
    }

    /// Get the number of pipeline stages.
    pub fn num_stages(&self) -> usize {
        self.stages.len()
    }

    /// Find the stage responsible for a given layer.
    pub fn stage_for_layer(&self, layer_idx: usize) -> Option<&PipelineStage> {
        self.stages
            .iter()
            .find(|s| layer_idx >= s.start_layer && layer_idx < s.end_layer)
    }

    /// Get the node that should receive the final logits.
    pub fn output_node(&self) -> Option<&str> {
        self.stages.last().map(|s| s.node_id.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_pipeline_schedule() {
        let mut assignments = HashMap::new();
        assignments.insert("node-0".to_string(), (0, 16));
        assignments.insert("node-1".to_string(), (16, 32));

        let schedule = PipelineSchedule::new(&assignments, 32);
        assert_eq!(schedule.num_stages(), 2);
        assert!(schedule.stages()[0].has_embedding);
        assert!(!schedule.stages()[0].has_lm_head);
        assert!(!schedule.stages()[1].has_embedding);
        assert!(schedule.stages()[1].has_lm_head);
        assert_eq!(schedule.output_node(), Some("node-1"));
    }
}
