//! Interleaved 1F1B pipeline schedule for training.
//!
//! Each rank owns multiple virtual stages (non-contiguous layer chunks).
//! Reduces the pipeline bubble ratio from `(S-1)/M` to `(S-1)/(M*V)`
//! where V is the number of virtual stages per rank.

use std::collections::HashMap;
use std::sync::Arc;

use super::clock::{PipelineAction, PipelineClock};
use super::comm::{compute_loss_grad, recv_activation_tagged, send_activation_tagged};
use super::schedule_1f1b::PipelineOutput;
use super::stage::TrainablePipelineStage;
use crate::error::{Error, Result};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::ShapeOps;
use numr::runtime::{Communicator, Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Interleaved 1F1B pipeline schedule.
///
/// Each rank owns `V` virtual stages. Virtual stage `v` on rank `k` corresponds
/// to logical pipeline stage `k + v * num_ranks` in the full pipeline. This
/// interleaving reduces bubbles by keeping more stages busy simultaneously.
pub struct ScheduleInterleaved1F1B<R: Runtime> {
    /// `V` virtual stages owned by this rank.
    stages: Vec<Box<dyn TrainablePipelineStage<R>>>,
    num_micro_batches: usize,
    pp_comm: Arc<dyn Communicator>,
    device: R::Device,
}

impl<R: Runtime<DType = DType>> ScheduleInterleaved1F1B<R> {
    pub fn new(
        stages: Vec<Box<dyn TrainablePipelineStage<R>>>,
        num_micro_batches: usize,
        pp_comm: Arc<dyn Communicator>,
        device: R::Device,
    ) -> Result<Self> {
        if stages.is_empty() {
            return Err(Error::DistributedError {
                reason: "must have at least one virtual stage".to_string(),
            });
        }
        if num_micro_batches == 0 {
            return Err(Error::DistributedError {
                reason: "num_micro_batches must be > 0".to_string(),
            });
        }
        Ok(Self {
            stages,
            num_micro_batches,
            pp_comm,
            device,
        })
    }

    /// Run the interleaved 1F1B schedule.
    ///
    /// * `micro_batches` — input micro-batches (only on stage 0).
    /// * `loss_fn` — loss function (only on the last virtual stage of the last rank).
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
        let num_virtual = self.stages.len();
        let total_stages = world_size * num_virtual;

        let clock = PipelineClock::new(world_size, self.num_micro_batches, rank);
        let schedule = clock.schedule_interleaved(num_virtual);

        // Forward outputs per virtual stage per micro-batch (for last logical stage's loss)
        let mut forward_outputs: Vec<Vec<Option<Var<R>>>> = (0..num_virtual)
            .map(|_| (0..self.num_micro_batches).map(|_| None).collect())
            .collect();

        let mut losses = Vec::new();
        let mut input_grads: Vec<Option<Tensor<R>>> = vec![None; self.num_micro_batches];

        let mut mb_inputs: Vec<Option<Tensor<R>>> = if rank == 0 {
            let mbs = micro_batches.ok_or_else(|| Error::DistributedError {
                reason: "first rank must provide micro_batches".to_string(),
            })?;
            mbs.into_iter().map(Some).collect()
        } else {
            vec![None; self.num_micro_batches]
        };

        // Local buffers for activations passed between virtual stages on the same rank.
        // Key: (logical_stage, mb_id, is_backward)
        let mut local_buffers: HashMap<(usize, usize, bool), Tensor<R>> = HashMap::new();

        for &(v_idx, ref action) in &schedule {
            let logical_stage = rank + v_idx * world_size;
            let is_first_logical = logical_stage == 0;
            let is_last_logical = logical_stage == total_stages - 1;

            // Previous/next logical stage info
            let prev_logical = if logical_stage > 0 {
                Some(logical_stage - 1)
            } else {
                None
            };
            let next_logical = if logical_stage < total_stages - 1 {
                Some(logical_stage + 1)
            } else {
                None
            };

            // Whether prev/next is on same rank (local transfer) or different rank (comm)
            let prev_is_local = prev_logical
                .map(|l| l % world_size == rank)
                .unwrap_or(false);
            let next_is_local = next_logical
                .map(|l| l % world_size == rank)
                .unwrap_or(false);

            let prev_rank_id = prev_logical.map(|l| l % world_size);
            let next_rank_id = next_logical.map(|l| l % world_size);

            // Base tag to avoid collision across virtual stages
            let base_tag = u32::try_from(v_idx * self.num_micro_batches * 4).map_err(|_| {
                Error::DistributedError {
                    reason: "tag overflow".to_string(),
                }
            })?;

            match *action {
                PipelineAction::Forward(mb_id) => {
                    let input_tensor = if is_first_logical {
                        mb_inputs[mb_id]
                            .take()
                            .ok_or_else(|| Error::DistributedError {
                                reason: format!("micro-batch {mb_id} already consumed"),
                            })?
                    } else if prev_is_local {
                        // Get from local buffer (previous virtual stage on same rank)
                        let prev_l = prev_logical
                            .expect("prev_logical guaranteed Some when prev_is_local is true");
                        local_buffers
                            .remove(&(prev_l, mb_id, false))
                            .ok_or_else(|| Error::DistributedError {
                                reason: format!(
                                    "no local activation for logical_stage={prev_l} mb={mb_id}"
                                ),
                            })?
                    } else {
                        let src = prev_rank_id.ok_or_else(|| Error::DistributedError {
                            reason: "no previous stage for recv".to_string(),
                        })?;
                        recv_activation_tagged::<R>(
                            self.pp_comm.as_ref(),
                            src,
                            mb_id,
                            false,
                            base_tag,
                            &self.device,
                        )?
                    };

                    let input_var = Var::new(input_tensor, is_first_logical);
                    let output_var = self.stages[v_idx].forward(input_var)?;

                    if is_last_logical {
                        forward_outputs[v_idx][mb_id] = Some(output_var);
                    } else if next_is_local {
                        // Store in local buffer for the next virtual stage on same rank
                        local_buffers
                            .insert((logical_stage, mb_id, false), output_var.tensor().clone());
                    } else {
                        let dest = next_rank_id.ok_or_else(|| Error::DistributedError {
                            reason: "no next stage for send".to_string(),
                        })?;
                        send_activation_tagged::<R>(
                            self.pp_comm.as_ref(),
                            output_var.tensor(),
                            dest,
                            mb_id,
                            false,
                            base_tag,
                        )?;
                    }
                }
                PipelineAction::Backward(mb_id) => {
                    let output_grad = if is_last_logical {
                        compute_loss_grad(
                            &mut forward_outputs[v_idx][mb_id],
                            mb_id,
                            loss_fn,
                            &mut losses,
                            &self.device,
                        )?
                    } else if next_is_local {
                        let next_l = next_logical
                            .expect("next_logical guaranteed Some when next_is_local is true");
                        local_buffers
                            .remove(&(next_l, mb_id, true))
                            .ok_or_else(|| Error::DistributedError {
                                reason: format!(
                                    "no local grad for logical_stage={next_l} mb={mb_id}"
                                ),
                            })?
                    } else {
                        let src = next_rank_id.ok_or_else(|| Error::DistributedError {
                            reason: "no next stage for backward recv".to_string(),
                        })?;
                        recv_activation_tagged::<R>(
                            self.pp_comm.as_ref(),
                            src,
                            mb_id,
                            true,
                            base_tag,
                            &self.device,
                        )?
                    };

                    let input_grad = self.stages[v_idx].backward(mb_id, output_grad)?;

                    if is_first_logical {
                        input_grads[mb_id] = Some(input_grad);
                    } else if prev_is_local {
                        local_buffers.insert((logical_stage, mb_id, true), input_grad);
                    } else {
                        let dest = prev_rank_id.ok_or_else(|| Error::DistributedError {
                            reason: "no prev stage for backward send".to_string(),
                        })?;
                        send_activation_tagged::<R>(
                            self.pp_comm.as_ref(),
                            &input_grad,
                            dest,
                            mb_id,
                            true,
                            base_tag,
                        )?;
                    }
                }
                _ => {}
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

    pub fn num_virtual_stages(&self) -> usize {
        self.stages.len()
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

    struct IdentityTrainStage;

    impl TrainablePipelineStage<CpuRuntime> for IdentityTrainStage {
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

    #[test]
    fn test_interleaved_single_device() {
        let (client, device) = cpu_setup();
        let comm = Arc::new(NoOpCommunicator);

        let stages: Vec<Box<dyn TrainablePipelineStage<CpuRuntime>>> =
            vec![Box::new(IdentityTrainStage), Box::new(IdentityTrainStage)];

        let mut schedule = ScheduleInterleaved1F1B::new(stages, 2, comm, device.clone()).unwrap();

        let mb0 = Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1], &device);
        let mb1 = Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device);

        let loss_fn = |output: &Var<CpuRuntime>| -> Result<Var<CpuRuntime>> {
            Ok(Var::new(output.tensor().clone(), false))
        };

        let result = schedule
            .run(&client, Some(vec![mb0, mb1]), Some(&loss_fn))
            .unwrap();

        assert_eq!(result.losses.len(), 2);
    }

    #[test]
    fn test_interleaved_empty_stages_error() {
        let (_client, device) = cpu_setup();
        let comm = Arc::new(NoOpCommunicator);
        let stages: Vec<Box<dyn TrainablePipelineStage<CpuRuntime>>> = vec![];
        let result = ScheduleInterleaved1F1B::<CpuRuntime>::new(stages, 2, comm, device);
        assert!(result.is_err());
    }
}
