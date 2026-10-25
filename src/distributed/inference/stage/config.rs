//! Limits of one stage, and the models and layer ranges a stage accepts.

use anyhow::{Result, bail};
use numr::dtype::DType;
use numr::ops::IndexingOps;
use numr::runtime::Runtime;

use super::super::transport::LayerAssignment;
use crate::model::LoadedModel;

/// Limits and cache dtype of one stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StageConfig {
    /// Requests the stage holds a KV cache for at once. A prefill past it fails.
    pub max_requests: usize,
    /// Positions each new cache allocates up front. It grows to `max_seq_len`.
    pub initial_capacity: usize,
    /// Positions one request can reach. At most the model's `max_seq_len`.
    pub max_seq_len: usize,
    /// Element type of the K and V tensors.
    pub kv_dtype: DType,
}

impl StageConfig {
    /// # Errors
    ///
    /// `max_requests` of 0, `initial_capacity` outside `1..=max_seq_len`, or
    /// `max_seq_len` past `model_max_seq_len`.
    pub fn check(&self, model_max_seq_len: usize) -> Result<()> {
        if self.max_requests == 0 {
            bail!("a stage needs max_requests of 1 or more");
        }
        if self.initial_capacity == 0 || self.initial_capacity > self.max_seq_len {
            bail!(
                "a stage's initial_capacity of {} must be in 1..={}, its max_seq_len",
                self.initial_capacity,
                self.max_seq_len
            );
        }
        if self.max_seq_len > model_max_seq_len {
            bail!(
                "a stage's max_seq_len of {} is past the model's max_seq_len of \
                 {model_max_seq_len}",
                self.max_seq_len
            );
        }
        Ok(())
    }
}

/// Why `model` cannot run as a pipeline stage, or `None` when it can.
///
/// A stage carries one KV cache per request and nothing else. A model with
/// per-layer recurrent state needs that state on the stage that runs the
/// layer, so it is refused here, never run without it.
pub(super) fn refusal<R>(model: &LoadedModel<R>) -> Option<String>
where
    R: Runtime<DType = DType>,
    R::Client: IndexingOps<R>,
{
    match model {
        LoadedModel::Llama(_) => None,
        LoadedModel::Multimodal(m) => refusal(m.llm()),
        LoadedModel::LlamaTp(_) => Some(
            "a tensor-parallel shard cannot run as a pipeline stage. Load the model on each \
             stage without tensor parallelism"
                .into(),
        ),
        LoadedModel::Mamba1(_)
        | LoadedModel::Mamba2(_)
        | LoadedModel::Mamba3(_)
        | LoadedModel::Hybrid(_) => Some(format!(
            "a {} model carries SSM state in its layers, and a pipeline stage carries only a \
             KV cache. Serve it on one node",
            model.model_type()
        )),
        LoadedModel::Qwen35(_) => Some(format!(
            "a {} model carries Gated DeltaNet state in its layers, and a pipeline stage \
             carries only a KV cache. Serve it on one node",
            model.model_type()
        )),
    }
}

/// Checks `assignment` against a model of `num_layers` layers.
///
/// # Errors
///
/// An empty range, a range past `num_layers`, or an embedding or LM head
/// flag that does not match the range.
pub(super) fn check_assignment(assignment: &LayerAssignment, num_layers: usize) -> Result<()> {
    let (start, end) = (
        assignment.start_layer as usize,
        assignment.end_layer as usize,
    );
    if start >= end || end > num_layers {
        bail!(
            "stage layers {start}..{end} are not a non-empty range inside the model's \
             0..{num_layers}"
        );
    }
    if assignment.has_embedding != (start == 0) {
        bail!(
            "stage layers {start}..{end}: the embedding flag is {}, and only the stage that \
             starts at layer 0 holds the embedding",
            assignment.has_embedding
        );
    }
    if assignment.has_lm_head != (end == num_layers) {
        bail!(
            "stage layers {start}..{end}: the LM head flag is {}, and only the stage that \
             ends at layer {num_layers} holds the LM head",
            assignment.has_lm_head
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;

    fn config() -> StageConfig {
        StageConfig {
            max_requests: 2,
            initial_capacity: 4,
            max_seq_len: 16,
            kv_dtype: DType::F32,
        }
    }

    fn range(start: u32, end: u32, has_embedding: bool, has_lm_head: bool) -> LayerAssignment {
        LayerAssignment {
            start_layer: start,
            end_layer: end,
            has_embedding,
            has_lm_head,
        }
    }

    #[test]
    fn test_config_limits_are_checked() {
        config().check(16).unwrap();
        assert!(config().check(8).is_err());
        let zero = StageConfig {
            max_requests: 0,
            ..config()
        };
        assert!(zero.check(16).is_err());
        let wide = StageConfig {
            initial_capacity: 17,
            ..config()
        };
        assert!(wide.check(32).is_err());
    }

    #[test]
    fn test_assignment_matches_its_range() {
        check_assignment(&range(0, 2, true, false), 4).unwrap();
        check_assignment(&range(2, 4, false, true), 4).unwrap();
        check_assignment(&range(0, 4, true, true), 4).unwrap();
        assert!(check_assignment(&range(2, 2, false, false), 4).is_err());
        assert!(check_assignment(&range(2, 5, false, true), 4).is_err());
        assert!(check_assignment(&range(1, 2, true, false), 4).is_err());
        assert!(check_assignment(&range(2, 4, false, false), 4).is_err());
    }

    #[test]
    fn test_an_ssm_model_is_refused() {
        let (_, device) = cpu_setup();
        let mamba = crate::model::mamba::model::tiny_mamba2(&device);
        let model = LoadedModel::Mamba2(Box::new(mamba));
        let reason = refusal(&model).unwrap();
        assert!(reason.contains("SSM state"), "{reason}");
    }
}
