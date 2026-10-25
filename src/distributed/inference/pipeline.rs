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

/// The node id of the stage on `rank` in a [`PipelineSchedule::by_rank`] schedule.
pub fn rank_node_id(rank: u32) -> String {
    format!("rank-{rank}")
}

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

    /// One stage per rank, in rank order: rank `r` runs stage `r`.
    ///
    /// Splits `total_layers` into contiguous ranges, as even as possible.
    /// The lower ranks take the extra layers. Stage `r` names node
    /// [`rank_node_id`]`(r)`. A stage's previous stage is rank `r - 1` and its
    /// next stage is rank `r + 1`, so a worker derives both from its rank.
    ///
    /// # Errors
    ///
    /// `world_size` of 0, or more ranks than layers.
    pub fn by_rank(world_size: u32, total_layers: usize) -> anyhow::Result<Self> {
        let ranks = world_size as usize;
        if ranks == 0 || ranks > total_layers {
            anyhow::bail!(
                "a pipeline of {total_layers} layers cannot run on {world_size} ranks: \
                 use between 1 and {total_layers} ranks"
            );
        }
        let (base, extra) = (total_layers / ranks, total_layers % ranks);
        let mut start = 0;
        let stages = (0..ranks)
            .map(|r| {
                let end = start + base + usize::from(r < extra);
                let stage = PipelineStage {
                    node_id: rank_node_id(r as u32),
                    start_layer: start,
                    end_layer: end,
                    has_embedding: start == 0,
                    has_lm_head: end == total_layers,
                };
                start = end;
                stage
            })
            .collect();
        Ok(Self { stages })
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

    #[test]
    fn test_by_rank_splits_in_rank_order() {
        let schedule = PipelineSchedule::by_rank(3, 10).unwrap();
        let ranges: Vec<_> = schedule
            .stages()
            .iter()
            .map(|s| (s.node_id.as_str(), s.start_layer, s.end_layer))
            .collect();
        assert_eq!(
            ranges,
            [("rank-0", 0, 4), ("rank-1", 4, 7), ("rank-2", 7, 10)]
        );
        assert!(schedule.stages()[0].has_embedding);
        assert!(schedule.stages()[2].has_lm_head);
        assert!(!schedule.stages()[1].has_embedding && !schedule.stages()[1].has_lm_head);
    }

    #[test]
    fn test_by_rank_refuses_more_ranks_than_layers() {
        assert!(PipelineSchedule::by_rank(0, 4).is_err());
        let err = PipelineSchedule::by_rank(5, 4).err().unwrap();
        assert!(err.to_string().contains("between 1 and 4"), "{err}");
        assert_eq!(PipelineSchedule::by_rank(1, 4).unwrap().num_stages(), 1);
    }
}
