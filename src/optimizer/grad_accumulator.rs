//! Gradient accumulation across micro-batches
//!
//! Enables effective large batch training by summing gradients from
//! multiple smaller forward/backward passes before applying an optimizer step.

use crate::error::{Error, Result};
use numr::autograd::GradStore;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ScalarOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::TensorId;

/// Accumulates gradients across multiple micro-batches
///
/// Usage pattern:
/// ```ignore
/// let mut accum = GradAccumulator::new(4)?; // 4 micro-batches
///
/// for micro_batch in data.chunks(micro_batch_size) {
///     let loss = model.forward(micro_batch);
///     let grads = backward(&loss, &client)?;
///
///     if let Some(averaged_grads) = accum.accumulate(&client, grads)? {
///         optimizer.step(&client, &mut params, &averaged_grads)?;
///     }
/// }
/// ```
pub struct GradAccumulator<R: Runtime> {
    accum_steps: usize,
    current_step: usize,
    accumulated: Option<GradStore<R>>,
}

impl<R: Runtime<DType = DType>> GradAccumulator<R> {
    pub fn new(accum_steps: usize) -> Result<Self> {
        if accum_steps == 0 {
            return Err(Error::TrainingError {
                reason: "accum_steps must be > 0".to_string(),
            });
        }
        Ok(Self {
            accum_steps,
            current_step: 0,
            accumulated: None,
        })
    }

    /// Accumulate gradients from one micro-batch.
    ///
    /// Returns `Some(averaged_grads)` when `accum_steps` micro-batches have been
    /// accumulated. Returns `None` when still accumulating.
    pub fn accumulate<C>(&mut self, client: &C, grads: GradStore<R>) -> Result<Option<GradStore<R>>>
    where
        C: RuntimeClient<R> + BinaryOps<R> + ScalarOps<R>,
    {
        let mut acc = match self.accumulated.take() {
            None => grads,
            Some(mut acc) => {
                for id in grads.keys().copied().collect::<Vec<TensorId>>() {
                    let new_grad = grads.get(id).expect("id came from grads.keys()");
                    let summed = if let Some(existing) = acc.get(id) {
                        client.add(existing, new_grad)?
                    } else {
                        new_grad.clone()
                    };
                    acc.insert(id, summed);
                }
                acc
            }
        };

        self.current_step += 1;

        if self.current_step >= self.accum_steps {
            self.current_step = 0;

            // Average by dividing by accum_steps
            let scale = 1.0 / self.accum_steps as f64;
            let ids: Vec<TensorId> = acc.keys().copied().collect();
            for id in ids {
                let grad = acc.get(id).expect("id came from acc.keys()");
                let scaled = client.mul_scalar(grad, scale)?;
                acc.insert(id, scaled);
            }

            Ok(Some(acc))
        } else {
            self.accumulated = Some(acc);
            Ok(None)
        }
    }

    pub fn accum_steps(&self) -> usize {
        self.accum_steps
    }

    pub fn current_step(&self) -> usize {
        self.current_step
    }

    pub fn reset(&mut self) {
        self.current_step = 0;
        self.accumulated = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::autograd::GradStore;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

    #[test]
    fn test_accumulate_returns_none_before_full() {
        let (client, device) = cpu_setup();
        let mut accum = GradAccumulator::<CpuRuntime>::new(3).unwrap();

        let t = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device);
        let id = t.id();
        let mut grads = GradStore::new();
        grads.insert(id, t);

        let result = accum.accumulate(&client, grads).unwrap();
        assert!(result.is_none());
        assert_eq!(accum.current_step(), 1);
    }

    #[test]
    fn test_accumulate_returns_averaged_grads() {
        let (client, device) = cpu_setup();
        let mut accum = GradAccumulator::<CpuRuntime>::new(2).unwrap();

        let id = numr::tensor::TensorId::new();

        // Micro-batch 1: grads = [2.0, 4.0]
        let t1 = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 4.0], &[2], &device);
        let mut g1 = GradStore::new();
        g1.insert(id, t1);

        assert!(accum.accumulate(&client, g1).unwrap().is_none());

        // Micro-batch 2: grads = [6.0, 8.0]
        let t2 = Tensor::<CpuRuntime>::from_slice(&[6.0f32, 8.0], &[2], &device);
        let mut g2 = GradStore::new();
        g2.insert(id, t2);

        let result = accum.accumulate(&client, g2).unwrap();
        assert!(result.is_some());

        let averaged = result.unwrap();
        let grad = averaged.get(id).unwrap();
        let data = grad.to_vec::<f32>();
        // Average of [2,4] and [6,8] = [4, 6]
        assert!((data[0] - 4.0).abs() < 1e-6);
        assert!((data[1] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_accumulate_resets_after_full() {
        let (client, device) = cpu_setup();
        let mut accum = GradAccumulator::<CpuRuntime>::new(2).unwrap();

        let id = numr::tensor::TensorId::new();

        for _ in 0..2 {
            let t = Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1], &device);
            let mut g = GradStore::new();
            g.insert(id, t);
            accum.accumulate(&client, g).unwrap();
        }

        // After returning averaged grads, counter should reset
        assert_eq!(accum.current_step(), 0);

        // Next accumulation starts fresh
        let t = Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1], &device);
        let mut g = GradStore::new();
        g.insert(id, t);
        assert!(accum.accumulate(&client, g).unwrap().is_none());
        assert_eq!(accum.current_step(), 1);
    }

    #[test]
    fn test_accumulate_single_step() {
        let (client, device) = cpu_setup();
        let mut accum = GradAccumulator::<CpuRuntime>::new(1).unwrap();

        let id = numr::tensor::TensorId::new();
        let t = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 5.0], &[2], &device);
        let mut g = GradStore::new();
        g.insert(id, t);

        // With accum_steps=1, should return immediately (scaled by 1/1 = identity)
        let result = accum.accumulate(&client, g).unwrap();
        assert!(result.is_some());
        let data = result.unwrap().get(id).unwrap().to_vec::<f32>();
        assert!((data[0] - 3.0).abs() < 1e-6);
        assert!((data[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_reset() {
        let (client, device) = cpu_setup();
        let mut accum = GradAccumulator::<CpuRuntime>::new(3).unwrap();

        let id = numr::tensor::TensorId::new();
        let t = Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1], &device);
        let mut g = GradStore::new();
        g.insert(id, t);
        accum.accumulate(&client, g).unwrap();

        assert_eq!(accum.current_step(), 1);
        accum.reset();
        assert_eq!(accum.current_step(), 0);
    }

    #[test]
    fn test_new_rejects_zero_steps() {
        let result = GradAccumulator::<CpuRuntime>::new(0);
        assert!(result.is_err());
    }
}
