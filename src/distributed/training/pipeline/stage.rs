//! Pipeline stage traits for inference and training.
//!
//! - [`PipelineStage`]: Inference-only (operates on `Tensor<R>`, no autograd overhead).
//! - [`TrainablePipelineStage`]: Training with autograd (operates on `Var<R>`,
//!   saves activations per micro-batch for staggered backward).
//! - [`ZeroBubbleStage`]: Extension of trainable stage that splits backward into
//!   input-gradient (B) and weight-gradient (W) passes.
//! - [`StageContext`]: Concrete helper that manages per-micro-batch activation saving.

use crate::error::Result;
use numr::autograd::Var;
use numr::runtime::Runtime;
use numr::tensor::{Tensor, TensorId};

/// Inference stage — processes micro-batches without autograd tracking.
///
/// Each rank implements this trait for its portion of the model.
/// The pipeline scheduler handles inter-stage communication.
pub trait PipelineStage<R: Runtime>: Send {
    /// Process one micro-batch through this stage.
    ///
    /// `input`: activation tensor from the previous stage (or original input for stage 0).
    /// Returns activation tensor to send to the next stage (or final output for last stage).
    fn forward(&mut self, input: Tensor<R>) -> Result<Tensor<R>>;
}

/// Training stage — autograd-tracked forward with per-micro-batch backward.
///
/// Unlike `PipelineStage`, this trait operates on `Var<R>` to build the autograd
/// graph, and supports staggered backward passes (forward micro-batch `i` while
/// backward on micro-batch `j`).
///
/// Implementations must save activations internally per micro-batch so that
/// `backward()` can be called later with the output gradient from the next stage.
pub trait TrainablePipelineStage<R: Runtime>: Send {
    /// Forward pass for a micro-batch. Returns output activation (tracked for autograd).
    ///
    /// The stage must internally save whatever it needs for the corresponding
    /// `backward()` call (typically the output `Var<R>` and input `Var<R>`).
    fn forward(&mut self, input: Var<R>) -> Result<Var<R>>;

    /// Backward pass for a specific micro-batch.
    ///
    /// `output_grad`: gradient of loss w.r.t. this stage's output.
    /// Returns gradient w.r.t. this stage's input (to send to the previous stage).
    ///
    /// Parameter gradients are accumulated internally.
    fn backward(&mut self, micro_batch_id: usize, output_grad: Tensor<R>) -> Result<Tensor<R>>;

    /// Access accumulated parameter gradients after backward passes.
    ///
    /// Returns `(TensorId, gradient)` pairs for each parameter. The optimizer
    /// uses these to update weights after all micro-batches complete.
    fn param_grads(&self) -> Result<Vec<(TensorId, Tensor<R>)>>;

    /// Number of micro-batches with saved activations (for memory tracking).
    fn num_saved(&self) -> usize;
}

/// Extension of [`TrainablePipelineStage`] that splits backward into two phases:
///
/// - **B pass** (`backward_input`): Compute only the input gradient. This is on
///   the critical path — the previous stage needs it to proceed.
/// - **W pass** (`backward_weights`): Compute only the weight gradients. This can
///   be scheduled into pipeline bubbles since no other stage depends on it.
///
/// This split enables zero-bubble pipeline parallelism by filling bubbles with W passes.
pub trait ZeroBubbleStage<R: Runtime>: TrainablePipelineStage<R> {
    /// Compute only the input gradient (B pass).
    ///
    /// Fast, on the critical path. The result is sent to the previous stage.
    fn backward_input(
        &mut self,
        micro_batch_id: usize,
        output_grad: Tensor<R>,
    ) -> Result<Tensor<R>>;

    /// Compute only the weight gradient (W pass).
    ///
    /// Scheduled into pipeline bubbles. Accumulates into internal parameter gradients.
    fn backward_weights(&mut self, micro_batch_id: usize) -> Result<()>;
}

/// Concrete helper that wraps model layers and manages per-micro-batch activation saving.
///
/// Users provide a forward function (closure or layer set), and `StageContext`
/// handles saving/restoring activations for the backward pass.
pub struct StageContext<R: Runtime> {
    /// Saved output `Var<R>` per micro-batch (needed for backward to traverse the graph).
    saved_outputs: Vec<Option<Var<R>>>,
    /// Saved input `Var<R>` per micro-batch (needed to extract the input gradient).
    saved_inputs: Vec<Option<Var<R>>>,
    /// Accumulated parameter gradients across micro-batches.
    accumulated_grads: Vec<(TensorId, Tensor<R>)>,
}

impl<R: Runtime> StageContext<R> {
    /// Create a new stage context pre-allocated for `num_micro_batches`.
    pub fn new(num_micro_batches: usize) -> Self {
        let mut saved_outputs = Vec::with_capacity(num_micro_batches);
        let mut saved_inputs = Vec::with_capacity(num_micro_batches);
        for _ in 0..num_micro_batches {
            saved_outputs.push(None);
            saved_inputs.push(None);
        }
        Self {
            saved_outputs,
            saved_inputs,
            accumulated_grads: Vec::new(),
        }
    }

    /// Save the input and output activations for a micro-batch.
    pub fn save(&mut self, micro_batch_id: usize, input: Var<R>, output: Var<R>) {
        if micro_batch_id < self.saved_inputs.len() {
            self.saved_inputs[micro_batch_id] = Some(input);
            self.saved_outputs[micro_batch_id] = Some(output);
        }
    }

    /// Take the saved output for backward (removes it from storage).
    pub fn take_output(&mut self, micro_batch_id: usize) -> Option<Var<R>> {
        if micro_batch_id < self.saved_outputs.len() {
            self.saved_outputs[micro_batch_id].take()
        } else {
            None
        }
    }

    /// Take the saved input for extracting the input gradient.
    pub fn take_input(&mut self, micro_batch_id: usize) -> Option<Var<R>> {
        if micro_batch_id < self.saved_inputs.len() {
            self.saved_inputs[micro_batch_id].take()
        } else {
            None
        }
    }

    /// Record a parameter gradient (accumulates across micro-batches).
    pub fn accumulate_grad(&mut self, param_id: TensorId, grad: Tensor<R>) {
        self.accumulated_grads.push((param_id, grad));
    }

    /// Get all accumulated parameter gradients.
    pub fn param_grads(&self) -> &[(TensorId, Tensor<R>)] {
        &self.accumulated_grads
    }

    /// Number of micro-batches that still have saved activations.
    pub fn num_saved(&self) -> usize {
        self.saved_outputs.iter().filter(|o| o.is_some()).count()
    }

    /// Clear all saved state (call after optimizer step).
    pub fn clear(&mut self) {
        for slot in &mut self.saved_outputs {
            *slot = None;
        }
        for slot in &mut self.saved_inputs {
            *slot = None;
        }
        self.accumulated_grads.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_stage_context_save_and_take() {
        let ctx: StageContext<CpuRuntime> = StageContext::new(4);
        assert_eq!(ctx.num_saved(), 0);
    }

    #[test]
    fn test_stage_context_clear() {
        let mut ctx: StageContext<CpuRuntime> = StageContext::new(2);
        ctx.clear();
        assert_eq!(ctx.num_saved(), 0);
        assert!(ctx.param_grads().is_empty());
    }

    #[test]
    fn test_pipeline_stage_trait_send() {
        fn assert_send<T: Send>() {}
        assert_send::<Box<dyn PipelineStage<CpuRuntime>>>();
        assert_send::<Box<dyn TrainablePipelineStage<CpuRuntime>>>();
    }
}
