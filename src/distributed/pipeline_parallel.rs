//! Pipeline parallelism (GPipe-style)
//!
//! Splits a model into stages across devices. Each rank owns one stage.
//! Input is split into micro-batches that are pipelined through stages
//! using point-to-point send/recv between ranks.

use std::sync::Arc;

use crate::distributed::comm_utils::{recv_tensor_with_metadata, send_tensor_with_metadata};
use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::ops::ShapeOps;
use numr::runtime::{Communicator, Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// A single pipeline stage that processes a micro-batch.
///
/// Each rank implements this trait for its portion of the model.
/// The pipeline scheduler handles the inter-stage communication.
pub trait PipelineStage<R: Runtime>: Send {
    /// Process one micro-batch through this stage.
    ///
    /// input: activation tensor from the previous stage (or original input for stage 0)
    /// output: activation tensor to send to the next stage (or final output for last stage)
    fn forward(&mut self, input: Tensor<R>) -> Result<Tensor<R>>;
}

/// GPipe-style pipeline schedule.
///
/// Splits input into micro-batches, pipelines them through stages across ranks.
/// Each rank owns exactly one stage. Inter-stage communication uses
/// point-to-point send/recv via the `Communicator`.
///
/// Schedule: all-forward pass (all micro-batches through all stages),
/// then all-backward pass (not yet implemented — forward-only for inference
/// and the forward half of training).
pub struct PipelineSchedule<R: Runtime> {
    stage: Box<dyn PipelineStage<R>>,
    num_micro_batches: usize,
    comm: Arc<dyn Communicator>,
    device: R::Device,
}

impl<R: Runtime<DType = DType>> PipelineSchedule<R> {
    /// Create a new pipeline schedule.
    ///
    /// * `stage` — this rank's pipeline stage
    /// * `num_micro_batches` — number of micro-batches to split the input into
    /// * `comm` — communicator for inter-stage send/recv
    /// * `device` — device for allocating receive buffers on non-first ranks
    pub fn new(
        stage: Box<dyn PipelineStage<R>>,
        num_micro_batches: usize,
        comm: Arc<dyn Communicator>,
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
            comm,
            device,
        })
    }

    /// Run the forward pipeline.
    ///
    /// * For rank 0: splits `input` into micro-batches along dim 0, processes
    ///   each through the local stage, sends output to rank 1.
    /// * For intermediate ranks: receives from previous rank, processes,
    ///   sends to next rank.
    /// * For the last rank: receives from previous rank, processes, concatenates
    ///   micro-batch outputs.
    ///
    /// Returns the final output on the last rank. Other ranks return an empty vec.
    pub fn run<C>(&mut self, client: &C, input: Option<Tensor<R>>) -> Result<Vec<Tensor<R>>>
    where
        C: RuntimeClient<R> + ShapeOps<R>,
    {
        let rank = self.comm.rank();
        let world_size = self.comm.world_size();
        let num_stages = world_size;
        let is_first = rank == 0;
        let is_last = rank == num_stages - 1;

        // Single device: just run all micro-batches through the stage
        if world_size <= 1 {
            return self.run_single_device(client, input);
        }

        let mut outputs = Vec::new();

        // Split input into micro-batches (only rank 0 has input)
        let micro_batches: Vec<Tensor<R>> = if is_first {
            let inp = input.ok_or_else(|| Error::DistributedError {
                reason: "rank 0 must provide input".to_string(),
            })?;
            client.chunk(&inp, self.num_micro_batches, 0)?
        } else {
            Vec::new()
        };

        let mut mb_iter = micro_batches.into_iter();

        for mb_idx in 0..self.num_micro_batches {
            let tag = u32::try_from(mb_idx * 2).map_err(|_| Error::DistributedError {
                reason: format!("micro-batch index {mb_idx} exceeds u32 tag range"),
            })?;

            // Get micro-batch input
            let mb_input = if is_first {
                mb_iter.next().ok_or_else(|| Error::DistributedError {
                    reason: "fewer micro-batches than expected from chunk".to_string(),
                })?
            } else {
                // Receive from previous rank (with shape metadata)
                recv_tensor_with_metadata::<R>(self.comm.as_ref(), rank - 1, tag, &self.device)?
            };

            // Process through local stage
            let mb_output = self.stage.forward(mb_input)?;

            if is_last {
                outputs.push(mb_output);
            } else {
                // Send to next rank (with shape metadata)
                send_tensor_with_metadata(self.comm.as_ref(), &mb_output, rank + 1, tag)?;
            }
        }

        Ok(outputs)
    }

    fn run_single_device<C>(
        &mut self,
        client: &C,
        input: Option<Tensor<R>>,
    ) -> Result<Vec<Tensor<R>>>
    where
        C: RuntimeClient<R> + ShapeOps<R>,
    {
        let inp = input.ok_or_else(|| Error::DistributedError {
            reason: "input required for single-device pipeline".to_string(),
        })?;

        let micro_batches = client.chunk(&inp, self.num_micro_batches, 0)?;
        let mut outputs = Vec::with_capacity(self.num_micro_batches);

        for mb in micro_batches {
            let out = self.stage.forward(mb)?;
            outputs.push(out);
        }

        Ok(outputs)
    }

    /// Receive into a pre-allocated tensor buffer.
    ///
    /// The caller must provide a tensor with the correct shape and dtype
    /// that matches what the sender will send.
    pub fn recv_into(&self, buffer: &Tensor<R>, src: usize, tag: u32) -> Result<()> {
        crate::distributed::comm_utils::recv_into_tensor(self.comm.as_ref(), buffer, src, tag)
    }

    pub fn num_micro_batches(&self) -> usize {
        self.num_micro_batches
    }

    pub fn communicator(&self) -> &dyn Communicator {
        self.comm.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::NoOpCommunicator;
    use numr::runtime::cpu::CpuRuntime;

    /// Simple test stage that doubles the input
    struct DoubleStage;

    impl PipelineStage<CpuRuntime> for DoubleStage {
        fn forward(&mut self, input: Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
            let data = input.to_vec::<f32>();
            let doubled: Vec<f32> = data.iter().map(|x| x * 2.0).collect();
            Ok(Tensor::from_slice(&doubled, input.shape(), input.device()))
        }
    }

    /// Stage that adds 1 to each element
    struct AddOneStage;

    impl PipelineStage<CpuRuntime> for AddOneStage {
        fn forward(&mut self, input: Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
            let data = input.to_vec::<f32>();
            let result: Vec<f32> = data.iter().map(|x| x + 1.0).collect();
            Ok(Tensor::from_slice(&result, input.shape(), input.device()))
        }
    }

    #[test]
    fn test_pipeline_single_device() {
        let (client, device) = cpu_setup();
        let comm = Arc::new(NoOpCommunicator);

        let stage = Box::new(DoubleStage);
        let mut pipeline = PipelineSchedule::new(stage, 2, comm, device.clone()).unwrap();

        let input = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4, 1], &device);
        let outputs = pipeline.run(&client, Some(input)).unwrap();

        assert_eq!(outputs.len(), 2);
        // First micro-batch: [1, 2] * 2 = [2, 4]
        let out0 = outputs[0].to_vec::<f32>();
        assert_eq!(out0, vec![2.0, 4.0]);
        // Second micro-batch: [3, 4] * 2 = [6, 8]
        let out1 = outputs[1].to_vec::<f32>();
        assert_eq!(out1, vec![6.0, 8.0]);
    }

    #[test]
    fn test_pipeline_single_micro_batch() {
        let (client, device) = cpu_setup();
        let comm = Arc::new(NoOpCommunicator);

        let stage = Box::new(AddOneStage);
        let mut pipeline = PipelineSchedule::new(stage, 1, comm, device.clone()).unwrap();

        let input = Tensor::<CpuRuntime>::from_slice(&[10.0f32, 20.0], &[2, 1], &device);
        let outputs = pipeline.run(&client, Some(input)).unwrap();

        assert_eq!(outputs.len(), 1);
        let data = outputs[0].to_vec::<f32>();
        assert_eq!(data, vec![11.0, 21.0]);
    }

    #[test]
    fn test_pipeline_zero_micro_batches_error() {
        let (_client, device) = cpu_setup();
        let comm = Arc::new(NoOpCommunicator);
        let stage = Box::new(DoubleStage);
        let result = PipelineSchedule::<CpuRuntime>::new(stage, 0, comm, device);
        assert!(result.is_err());
    }

    #[test]
    fn test_pipeline_no_input_error() {
        let (client, device) = cpu_setup();
        let comm = Arc::new(NoOpCommunicator);

        let stage = Box::new(DoubleStage);
        let mut pipeline = PipelineSchedule::new(stage, 1, comm, device.clone()).unwrap();

        let result = pipeline.run(&client, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_pipeline_stage_trait_send() {
        fn assert_send<T: Send>() {}
        assert_send::<Box<dyn PipelineStage<CpuRuntime>>>();
    }

    #[test]
    fn test_recv_into() {
        let (_client, device) = cpu_setup();
        let comm = Arc::new(NoOpCommunicator);

        let stage = Box::new(DoubleStage);
        let pipeline = PipelineSchedule::new(stage, 1, comm, device.clone()).unwrap();

        let buffer = Tensor::<CpuRuntime>::zeros(&[3], DType::F32, &device);
        // NoOpCommunicator recv is a no-op
        pipeline.recv_into(&buffer, 0, 0).unwrap();
    }
}
