//! Dataset trait and batch type for training data pipelines.

use crate::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// A batch of training data: input tensors and corresponding targets.
///
/// For language modeling: `inputs` and `targets` are both `[batch_size, seq_len]`
/// tensors of token IDs, where targets are inputs shifted by one position.
///
/// For other tasks, shapes depend on the domain.
pub struct Batch<R: Runtime> {
    pub inputs: Tensor<R>,
    pub targets: Tensor<R>,
}

/// Trait for indexable datasets that produce batches.
///
/// Datasets must return CPU-resident tensors. GPU transfer (if needed)
/// is handled by the caller after batching — one bulk transfer per batch
/// is far more efficient than per-sample transfers.
///
/// # Example
///
/// ```ignore
/// struct TokenDataset { tokens: Vec<u32>, seq_len: usize }
///
/// impl Dataset<CpuRuntime> for TokenDataset {
///     fn len(&self) -> usize {
///         self.tokens.len().saturating_sub(1) / self.seq_len
///     }
///     fn get(&self, idx: usize, device: &CpuDevice) -> Result<Batch<CpuRuntime>> {
///         let start = idx * self.seq_len;
///         let inputs = Tensor::from_slice(&self.tokens[start..start + self.seq_len], &[self.seq_len], device);
///         let targets = Tensor::from_slice(&self.tokens[start + 1..start + self.seq_len + 1], &[self.seq_len], device);
///         Ok(Batch { inputs, targets })
///     }
/// }
/// ```
pub trait Dataset<R: Runtime>: Send + Sync {
    /// Number of samples in the dataset.
    fn len(&self) -> usize;

    /// Whether the dataset is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a single sample by index.
    ///
    /// The returned tensors should be unbatched (no batch dimension).
    /// The `DataLoader` handles collation into batches.
    fn get(&self, idx: usize, device: &R::Device) -> Result<Batch<R>>;
}
