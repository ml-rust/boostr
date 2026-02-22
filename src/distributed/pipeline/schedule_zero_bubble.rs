//! Zero-bubble pipeline parallelism schedule.
//!
//! Splits the backward pass into two phases:
//! - **B pass** (`backward_input`): compute input gradient (on critical path)
//! - **W pass** (`backward_weights`): compute weight gradient (scheduled into bubbles)
//!
//! This achieves near-zero pipeline bubble when `M >= S`.

use std::sync::Arc;

use super::clock::{PipelineAction, PipelineClock};
use super::comm::{compute_loss_grad, recv_activation, send_activation};
use super::schedule_1f1b::PipelineOutput;
use super::stage::{TrainablePipelineStage, ZeroBubbleStage};
use crate::error::{Error, Result};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::ShapeOps;
use numr::runtime::{Communicator, Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Zero-bubble pipeline schedule.
///
/// Requires stages that implement [`ZeroBubbleStage`], which splits backward
/// into separate input-gradient and weight-gradient passes.
pub struct ScheduleZeroBubble<R: Runtime> {
    stage: Box<dyn ZeroBubbleStage<R>>,
    num_micro_batches: usize,
    pp_comm: Arc<dyn Communicator>,
    device: R::Device,
}

impl<R: Runtime<DType = DType>> ScheduleZeroBubble<R> {
    pub fn new(
        stage: Box<dyn ZeroBubbleStage<R>>,
        num_micro_batches: usize,
        pp_comm: Arc<dyn Communicator>,
        device: R::Device,
    ) -> Result<Self> {
        if num_micro_batches == 0 {
            return Err(Error::DistributedError {
                reason: "num_micro_batches must be > 0".to_string(),
            });
        }
        Ok(Self {
            stage,
            num_micro_batches,
            pp_comm,
            device,
        })
    }

    /// Run the zero-bubble schedule for one training iteration.
    ///
    /// * `micro_batches` — input micro-batches (only on stage 0).
    /// * `loss_fn` — loss function (only on the last stage).
    pub fn run<C>(
        &mut self,
        _client: &C,
        micro_batches: Option<Vec<Tensor<R>>>,
        loss_fn: Option<&super::schedule_1f1b::LossFn<'_, R>>,
    ) -> Result<PipelineOutput<R>>
    where
        C: RuntimeClient<R> + ShapeOps<R>,
    {
        let rank = self.pp_comm.rank();
        let world_size = self.pp_comm.world_size().max(1);
        let is_first = rank == 0;
        let is_last = rank == world_size - 1;

        let clock = PipelineClock::new(world_size, self.num_micro_batches, rank);
        let actions = clock.schedule_zero_bubble();

        let mut forward_outputs: Vec<Option<Var<R>>> = vec![None; self.num_micro_batches];
        let mut losses = Vec::new();
        let mut input_grads: Vec<Option<Tensor<R>>> = vec![None; self.num_micro_batches];

        let mut mb_inputs: Vec<Option<Tensor<R>>> = if is_first {
            let mbs = micro_batches.ok_or_else(|| Error::DistributedError {
                reason: "stage 0 must provide micro_batches".to_string(),
            })?;
            if mbs.len() != self.num_micro_batches {
                return Err(Error::DistributedError {
                    reason: format!(
                        "expected {} micro-batches, got {}",
                        self.num_micro_batches,
                        mbs.len()
                    ),
                });
            }
            mbs.into_iter().map(Some).collect()
        } else {
            vec![None; self.num_micro_batches]
        };

        for action in &actions {
            match *action {
                PipelineAction::Forward(mb_id) => {
                    let input_tensor = if is_first {
                        mb_inputs[mb_id]
                            .take()
                            .ok_or_else(|| Error::DistributedError {
                                reason: format!("micro-batch {mb_id} already consumed"),
                            })?
                    } else {
                        recv_activation::<R>(
                            self.pp_comm.as_ref(),
                            rank - 1,
                            mb_id,
                            false,
                            &self.device,
                        )?
                    };

                    let input_var = Var::new(input_tensor, is_first);
                    let output_var = self.stage.forward(input_var)?;

                    if is_last {
                        forward_outputs[mb_id] = Some(output_var);
                    } else {
                        send_activation::<R>(
                            self.pp_comm.as_ref(),
                            output_var.tensor(),
                            rank + 1,
                            mb_id,
                            false,
                        )?;
                    }
                }
                PipelineAction::BackwardInput(mb_id) => {
                    let output_grad = if is_last {
                        compute_loss_grad(
                            &mut forward_outputs[mb_id],
                            mb_id,
                            loss_fn,
                            &mut losses,
                            &self.device,
                        )?
                    } else {
                        recv_activation::<R>(
                            self.pp_comm.as_ref(),
                            rank + 1,
                            mb_id,
                            true,
                            &self.device,
                        )?
                    };

                    let input_grad = self.stage.backward_input(mb_id, output_grad)?;

                    if is_first {
                        input_grads[mb_id] = Some(input_grad);
                    } else {
                        send_activation::<R>(
                            self.pp_comm.as_ref(),
                            &input_grad,
                            rank - 1,
                            mb_id,
                            true,
                        )?;
                    }
                }
                PipelineAction::BackwardWeights(mb_id) => {
                    self.stage.backward_weights(mb_id)?;
                }
                // Fallback for full backward (shouldn't happen in zero-bubble schedule)
                PipelineAction::Backward(mb_id) => {
                    let output_grad = if is_last {
                        compute_loss_grad(
                            &mut forward_outputs[mb_id],
                            mb_id,
                            loss_fn,
                            &mut losses,
                            &self.device,
                        )?
                    } else {
                        recv_activation::<R>(
                            self.pp_comm.as_ref(),
                            rank + 1,
                            mb_id,
                            true,
                            &self.device,
                        )?
                    };

                    let input_grad =
                        TrainablePipelineStage::backward(&mut *self.stage, mb_id, output_grad)?;

                    if is_first {
                        input_grads[mb_id] = Some(input_grad);
                    } else {
                        send_activation::<R>(
                            self.pp_comm.as_ref(),
                            &input_grad,
                            rank - 1,
                            mb_id,
                            true,
                        )?;
                    }
                }
                PipelineAction::Idle => {}
            }
        }

        Ok(PipelineOutput {
            losses,
            input_grads: input_grads.into_iter().flatten().collect(),
        })
    }

    pub fn num_micro_batches(&self) -> usize {
        self.num_micro_batches
    }

    pub fn param_grads(&self) -> Result<Vec<(numr::tensor::TensorId, Tensor<R>)>> {
        self.stage.param_grads()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::autograd::Var;
    use numr::runtime::NoOpCommunicator;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::TensorId;

    use super::super::stage::TrainablePipelineStage;

    /// Test stage implementing ZeroBubbleStage (identity transform).
    struct IdentityZBStage;

    impl TrainablePipelineStage<CpuRuntime> for IdentityZBStage {
        fn forward(&mut self, input: Var<CpuRuntime>) -> Result<Var<CpuRuntime>> {
            Ok(Var::new(input.tensor().clone(), false))
        }

        fn backward(
            &mut self,
            _mb_id: usize,
            output_grad: Tensor<CpuRuntime>,
        ) -> Result<Tensor<CpuRuntime>> {
            Ok(output_grad)
        }

        fn param_grads(&self) -> Result<Vec<(TensorId, Tensor<CpuRuntime>)>> {
            Ok(Vec::new())
        }

        fn num_saved(&self) -> usize {
            0
        }
    }

    impl ZeroBubbleStage<CpuRuntime> for IdentityZBStage {
        fn backward_input(
            &mut self,
            _mb_id: usize,
            output_grad: Tensor<CpuRuntime>,
        ) -> Result<Tensor<CpuRuntime>> {
            Ok(output_grad)
        }

        fn backward_weights(&mut self, _mb_id: usize) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn test_zero_bubble_single_device() {
        let (client, device) = cpu_setup();
        let comm = Arc::new(NoOpCommunicator);
        let stage = Box::new(IdentityZBStage);
        let mut schedule =
            ScheduleZeroBubble::<CpuRuntime>::new(stage, 2, comm, device.clone()).unwrap();

        let mb0 = Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1], &device);
        let mb1 = Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device);

        let loss_fn = |output: &Var<CpuRuntime>| -> Result<Var<CpuRuntime>> {
            Ok(Var::new(output.tensor().clone(), false))
        };

        let result = schedule
            .run(&client, Some(vec![mb0, mb1]), Some(&loss_fn))
            .unwrap();

        assert_eq!(result.losses.len(), 2);
        assert!((result.losses[0] - 1.0).abs() < 1e-5);
        assert!((result.losses[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_zero_bubble_zero_mb_error() {
        let (_client, device) = cpu_setup();
        let comm = Arc::new(NoOpCommunicator);
        let stage = Box::new(IdentityZBStage);
        let result = ScheduleZeroBubble::<CpuRuntime>::new(stage, 0, comm, device);
        assert!(result.is_err());
    }
}
