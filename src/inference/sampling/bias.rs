//! Per-token logit bias as a device tensor.

use std::collections::HashMap;

use crate::error::Result;
use numr::dtype::DType;
use numr::ops::{BinaryOps, TypeConversionOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Additive biases by token id, as an OpenAI `logit_bias` map holds them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogitBias<'a> {
    biases: &'a HashMap<u32, f32>,
}

impl<'a> LogitBias<'a> {
    /// The biases of `biases`.
    pub fn new(biases: &'a HashMap<u32, f32>) -> Self {
        Self { biases }
    }

    /// No bias is set.
    pub fn is_empty(&self) -> bool {
        self.biases.is_empty()
    }

    /// The `[1, 1, vocab]` bias tensor in `dtype` on `device`.
    ///
    /// An id at or past `vocab` gets no bias. Every other entry is 0. A
    /// `[1, n, vocab]` logits tensor takes it by broadcast. The tensor is
    /// built in F32 and cast, so half-precision logits take the bias in
    /// their own dtype.
    ///
    /// # Errors
    ///
    /// An error when the upload or the cast fails.
    pub fn to_tensor<R>(&self, vocab: usize, dtype: DType, device: &R::Device) -> Result<Tensor<R>>
    where
        R: Runtime<DType = DType>,
        R::Client: TypeConversionOps<R>,
    {
        let mut row = vec![0.0f32; vocab];
        for (&token, &bias) in self.biases {
            if (token as usize) < vocab {
                row[token as usize] = bias;
            }
        }
        let tensor = Tensor::<R>::from_slice(&row, &[1, 1, vocab], device)?;
        Ok(R::default_client(device).cast(&tensor, dtype)?)
    }

    /// `logits + bias`, with the bias of [`Self::to_tensor`] over `vocab` in the logits dtype.
    ///
    /// # Errors
    ///
    /// An error when the bias tensor fails or does not broadcast to `logits`.
    pub fn apply<R>(&self, logits: &Tensor<R>, vocab: usize) -> Result<Tensor<R>>
    where
        R: Runtime<DType = DType>,
        R::Client: TypeConversionOps<R> + BinaryOps<R>,
    {
        let bias = self.to_tensor::<R>(vocab, logits.dtype(), logits.device())?;
        Ok(R::default_client(logits.device()).add(logits, &bias)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn the_tensor_holds_each_in_vocab_bias() -> Result<()> {
        let device = CpuDevice::new();
        let map = HashMap::from([(1, 2.5), (3, -1.0), (9, 7.0)]);
        let bias = LogitBias::new(&map).to_tensor::<CpuRuntime>(4, DType::F32, &device)?;
        assert_eq!(bias.shape(), &[1, 1, 4]);
        assert_eq!(bias.to_vec::<f32>(), vec![0.0, 2.5, 0.0, -1.0]);
        Ok(())
    }

    #[test]
    fn apply_adds_the_bias_to_every_position() -> Result<()> {
        let device = CpuDevice::new();
        let map = HashMap::from([(0, 1.0), (2, -2.0)]);
        let logits = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[1, 2, 3],
            &device,
        )?;
        let out = LogitBias::new(&map).apply(&logits, 3)?;
        assert_eq!(out.to_vec::<f32>(), vec![2.0, 2.0, 1.0, 5.0, 5.0, 4.0]);
        assert!(LogitBias::new(&HashMap::new()).is_empty());
        Ok(())
    }
}
