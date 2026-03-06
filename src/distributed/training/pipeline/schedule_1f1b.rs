//! 1F1B (one-forward-one-backward) pipeline schedule for training.
//!
//! The standard schedule used by Megatron-LM and DeepSpeed. Minimizes peak
//! memory by limiting the number of in-flight micro-batches per stage.

use std::sync::Arc;

use super::clock::{PipelineAction, PipelineClock};
use super::comm::{compute_loss_grad, recv_activation, send_activation};
use super::stage::TrainablePipelineStage;
use crate::error::{Error, Result};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::ShapeOps;
use numr::runtime::{Communicator, Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Loss function type for pipeline training: takes stage output, returns scalar loss.
pub type LossFn<'a, R> = dyn Fn(&Var<R>) -> Result<Var<R>> + 'a;

/// Output from a pipeline training step.
pub struct PipelineOutput<R: Runtime> {
    /// Per-micro-batch loss values (populated only on the last stage).
    pub losses: Vec<f64>,
    /// Gradient w.r.t. input for each micro-batch (populated only on the first stage).
    pub input_grads: Vec<Tensor<R>>,
}

/// 1F1B pipeline schedule for training.
///
/// Wraps a single [`TrainablePipelineStage`] and drives it through the 1F1B
/// schedule using the pipeline communicator for inter-stage activation transfer.
pub struct Schedule1F1B<R: Runtime> {
    stage: Box<dyn TrainablePipelineStage<R>>,
    num_micro_batches: usize,
    pp_comm: Arc<dyn Communicator>,
    device: R::Device,
}

impl<R: Runtime<DType = DType>> Schedule1F1B<R> {
    pub fn new(
        stage: Box<dyn TrainablePipelineStage<R>>,
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

    /// Run the 1F1B schedule for one training iteration.
    ///
    /// * `client` — runtime client for tensor operations.
    /// * `micro_batches` — input micro-batches (only on stage 0; `None` on other stages).
    /// * `loss_fn` — loss function applied to stage output (only on the last stage).
    ///
    /// Returns [`PipelineOutput`] with losses (last stage) and input grads (first stage).
    pub fn run<C>(
        &mut self,
        _client: &C,
        micro_batches: Option<Vec<Tensor<R>>>,
        loss_fn: Option<&LossFn<'_, R>>,
    ) -> Result<PipelineOutput<R>>
    where
        C: RuntimeClient<R> + ShapeOps<R>,
    {
        let rank = self.pp_comm.rank();
        let world_size = self.pp_comm.world_size();
        let num_stages = world_size.max(1);
        let is_first = rank == 0;
        let is_last = rank == num_stages - 1;

        let clock = PipelineClock::new(num_stages, self.num_micro_batches, rank);
        let actions = clock.schedule_1f1b();

        // Store forward outputs for last-stage loss computation
        let mut forward_outputs: Vec<Option<Var<R>>> = vec![None; self.num_micro_batches];
        let mut losses = Vec::new();
        let mut input_grads: Vec<Option<Tensor<R>>> = vec![None; self.num_micro_batches];

        // Prepare input micro-batches (only stage 0)
        let mb_inputs: Vec<Option<Tensor<R>>> = if is_first {
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
        let mut mb_inputs = mb_inputs;

        for action in &actions {
            match *action {
                PipelineAction::Forward(mb_id) => {
                    // Get input activation
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

                    // Wrap in Var for autograd
                    let input_var = Var::new(input_tensor, is_first);
                    let output_var = self.stage.forward(input_var)?;

                    if is_last {
                        // Save for loss computation during backward
                        forward_outputs[mb_id] = Some(output_var);
                    } else {
                        // Send detached tensor to next stage
                        send_activation::<R>(
                            self.pp_comm.as_ref(),
                            output_var.tensor(),
                            rank + 1,
                            mb_id,
                            false,
                        )?;
                    }
                }
                PipelineAction::Backward(mb_id) => {
                    // Get output gradient
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

                    // Run backward through this stage
                    let input_grad = self.stage.backward(mb_id, output_grad)?;

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
                _ => {
                    // 1F1B only uses Forward and Backward actions
                }
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

    pub fn communicator(&self) -> &dyn Communicator {
        self.pp_comm.as_ref()
    }

    /// Access the stage's accumulated parameter gradients.
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

    /// Test stage that doubles the input (no real autograd, just for schedule testing).
    struct DoubleTrainStage {
        saved_count: usize,
    }

    impl DoubleTrainStage {
        fn new() -> Self {
            Self { saved_count: 0 }
        }
    }

    impl TrainablePipelineStage<CpuRuntime> for DoubleTrainStage {
        fn forward(&mut self, input: Var<CpuRuntime>) -> Result<Var<CpuRuntime>> {
            let data = input.tensor().to_vec::<f32>();
            let doubled: Vec<f32> = data.iter().map(|x| x * 2.0).collect();
            let out = Tensor::from_slice(&doubled, input.tensor().shape(), input.tensor().device());
            self.saved_count += 1;
            Ok(Var::new(out, false))
        }

        fn backward(
            &mut self,
            _micro_batch_id: usize,
            output_grad: Tensor<CpuRuntime>,
        ) -> Result<Tensor<CpuRuntime>> {
            self.saved_count = self.saved_count.saturating_sub(1);
            // Gradient of doubling is 2 * output_grad
            let data = output_grad.to_vec::<f32>();
            let grad: Vec<f32> = data.iter().map(|x| x * 2.0).collect();
            Ok(Tensor::from_slice(
                &grad,
                output_grad.shape(),
                output_grad.device(),
            ))
        }

        fn param_grads(&self) -> Result<Vec<(TensorId, Tensor<CpuRuntime>)>> {
            Ok(Vec::new())
        }

        fn num_saved(&self) -> usize {
            self.saved_count
        }
    }

    #[test]
    fn test_schedule_1f1b_single_device() {
        let (_client, device) = cpu_setup();
        let comm = Arc::new(NoOpCommunicator);
        let stage = Box::new(DoubleTrainStage::new());
        let mut schedule = Schedule1F1B::<CpuRuntime>::new(stage, 2, comm, device.clone()).unwrap();

        let mb0 = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device);
        let mb1 = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 4.0], &[2], &device);

        // Single device: rank 0 is both first and last
        let loss_fn = |output: &Var<CpuRuntime>| -> Result<Var<CpuRuntime>> {
            // Sum as scalar loss
            let data = output.tensor().to_vec::<f32>();
            let sum: f32 = data.iter().sum();
            let loss_t = Tensor::from_slice(&[sum], &[1], output.tensor().device());
            Ok(Var::new(loss_t, false))
        };

        let result = schedule
            .run(&_client, Some(vec![mb0, mb1]), Some(&loss_fn))
            .unwrap();

        // Should have 2 losses (one per micro-batch)
        assert_eq!(result.losses.len(), 2);
        // First micro-batch: [1,2] * 2 = [2,4], sum = 6
        assert!((result.losses[0] - 6.0).abs() < 1e-5);
        // Second micro-batch: [3,4] * 2 = [6,8], sum = 14
        assert!((result.losses[1] - 14.0).abs() < 1e-5);
    }

    #[test]
    fn test_schedule_1f1b_zero_mb_error() {
        let (_client, device) = cpu_setup();
        let comm = Arc::new(NoOpCommunicator);
        let stage = Box::new(DoubleTrainStage::new());
        let result = Schedule1F1B::<CpuRuntime>::new(stage, 0, comm, device);
        assert!(result.is_err());
    }
}
