//! DataLoader with shuffling, batching, and background prefetching.
//!
//! Iterates over a `Dataset`, producing collated `Batch<CpuRuntime>` tensors
//! with a leading batch dimension. Shuffles indices deterministically
//! per epoch from a seed. Prefetches batches in a background thread
//! to overlap data loading with compute.

use std::sync::mpsc;
use std::thread;

use crate::data::collate::{collate_batch, shuffled_indices};
use crate::data::dataset::{Batch, Dataset};
use crate::error::Result;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};

/// DataLoader that iterates a dataset with shuffling, batching, and prefetching.
///
/// Operates on CPU tensors only. Transfer the resulting batches to GPU
/// with `Tensor::to_device()` in the training loop for efficient bulk transfer.
///
/// # Lifecycle
///
/// ```ignore
/// let loader = DataLoader::new(dataset, batch_size, seed, &device);
/// for epoch in 0..num_epochs {
///     for batch in loader.iter(epoch) {
///         let batch = batch?;
///         // batch.inputs: [batch_size, ...]
///         // batch.targets: [batch_size, ...]
///     }
/// }
/// ```
pub struct DataLoader<D> {
    dataset: D,
    batch_size: usize,
    seed: u64,
    prefetch_count: usize,
    device: CpuDevice,
}

impl<D> DataLoader<D>
where
    D: Dataset<CpuRuntime>,
{
    /// Create a new DataLoader.
    ///
    /// # Arguments
    /// * `dataset` - The dataset to iterate
    /// * `batch_size` - Number of samples per batch
    /// * `seed` - Random seed for deterministic shuffling
    /// * `device` - CPU device for tensor creation
    pub fn new(dataset: D, batch_size: usize, seed: u64, device: CpuDevice) -> Self {
        Self {
            dataset,
            batch_size,
            seed,
            prefetch_count: 2,
            device,
        }
    }

    /// Set the number of batches to prefetch in the background thread.
    ///
    /// Default is 2. Set to 0 to disable prefetching (synchronous iteration).
    pub fn with_prefetch(mut self, count: usize) -> Self {
        self.prefetch_count = count;
        self
    }

    /// Number of batches per epoch.
    pub fn num_batches(&self) -> usize {
        self.dataset.len() / self.batch_size
    }

    /// The underlying dataset.
    pub fn dataset(&self) -> &D {
        &self.dataset
    }

    /// Create a synchronous iterator for one epoch.
    ///
    /// Indices are shuffled deterministically from `seed + epoch`.
    /// The last incomplete batch (if any) is dropped.
    pub fn iter(&self, epoch: u64) -> DataLoaderIter<'_, D> {
        let indices = shuffled_indices(self.dataset.len(), self.seed.wrapping_add(epoch));
        let num_batches = indices.len() / self.batch_size;
        let indices: Vec<usize> = indices[..num_batches * self.batch_size].to_vec();

        DataLoaderIter {
            loader: self,
            indices,
            batch_idx: 0,
            num_batches,
        }
    }
}

/// Synchronous iterator over batches in one epoch.
///
/// Implements `Iterator<Item = Result<Batch<CpuRuntime>>>`.
pub struct DataLoaderIter<'a, D> {
    loader: &'a DataLoader<D>,
    indices: Vec<usize>,
    batch_idx: usize,
    num_batches: usize,
}

impl<'a, D> DataLoaderIter<'a, D>
where
    D: Dataset<CpuRuntime>,
{
    /// Number of batches remaining in this epoch.
    pub fn remaining(&self) -> usize {
        self.num_batches - self.batch_idx
    }

    fn advance(&mut self) -> Result<Option<Batch<CpuRuntime>>> {
        if self.batch_idx >= self.num_batches {
            return Ok(None);
        }

        let start = self.batch_idx * self.loader.batch_size;
        let end = start + self.loader.batch_size;
        let batch_indices = &self.indices[start..end];

        let batch = collate_batch(&self.loader.dataset, batch_indices, &self.loader.device)?;

        self.batch_idx += 1;
        Ok(Some(batch))
    }
}

impl<D> Iterator for DataLoaderIter<'_, D>
where
    D: Dataset<CpuRuntime>,
{
    type Item = Result<Batch<CpuRuntime>>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.advance() {
            Ok(Some(batch)) => Some(Ok(batch)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.remaining();
        (remaining, Some(remaining))
    }
}

impl<D> ExactSizeIterator for DataLoaderIter<'_, D> where D: Dataset<CpuRuntime> {}

/// Prefetching iterator that loads batches in a background thread.
///
/// Created via [`DataLoader::prefetch_iter`]. Uses a bounded channel
/// to pipeline batch loading with compute.
///
/// Implements `Iterator<Item = Result<Batch<CpuRuntime>>>`.
pub struct PrefetchIter {
    receiver: mpsc::Receiver<Result<Batch<CpuRuntime>>>,
    _handle: Option<thread::JoinHandle<()>>,
}

impl<D> DataLoader<D>
where
    D: Dataset<CpuRuntime> + Clone + 'static,
{
    /// Create a prefetching iterator that loads batches in a background thread.
    ///
    /// The background thread loads up to `prefetch_count` batches ahead.
    /// Requires `D: Clone` because the dataset is moved into the background thread.
    pub fn prefetch_iter(&self, epoch: u64) -> PrefetchIter {
        let indices = shuffled_indices(self.dataset.len(), self.seed.wrapping_add(epoch));
        let num_batches = indices.len() / self.batch_size;
        let indices: Vec<usize> = indices[..num_batches * self.batch_size].to_vec();

        let capacity = self.prefetch_count.max(1);
        let (tx, rx) = mpsc::sync_channel::<Result<Batch<CpuRuntime>>>(capacity);

        let dataset = self.dataset.clone();
        let batch_size = self.batch_size;
        let device = self.device.clone();

        let handle = thread::spawn(move || {
            for batch_idx in 0..num_batches {
                let start = batch_idx * batch_size;
                let end = start + batch_size;
                let batch_indices = &indices[start..end];

                let result = collate_batch(&dataset, batch_indices, &device);
                if tx.send(result).is_err() {
                    break; // Receiver dropped, stop prefetching
                }
            }
        });

        PrefetchIter {
            receiver: rx,
            _handle: Some(handle),
        }
    }
}

impl Iterator for PrefetchIter {
    type Item = Result<Batch<CpuRuntime>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.receiver.recv().ok()
    }
}

impl Drop for PrefetchIter {
    fn drop(&mut self) {
        // Drop the receiver first to signal the background thread to stop,
        // then wait for it to finish.
        // The receiver is dropped when self is dropped (it's a field).
        if let Some(handle) = self._handle.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::collate::shuffled_indices;
    use crate::data::dataset::Batch;
    use numr::runtime::cpu::CpuDevice;
    use numr::tensor::Tensor;

    #[derive(Clone)]
    struct SeqDataset {
        size: usize,
    }

    impl Dataset<CpuRuntime> for SeqDataset {
        fn len(&self) -> usize {
            self.size
        }

        fn get(&self, idx: usize, device: &CpuDevice) -> Result<Batch<CpuRuntime>> {
            let val = idx as f32;
            Ok(Batch {
                inputs: Tensor::from_slice(&[val, val + 0.1], &[2], device),
                targets: Tensor::from_slice(&[val + 1.0, val + 1.1], &[2], device),
            })
        }
    }

    #[test]
    fn test_shuffled_indices_deterministic() {
        let a = shuffled_indices(100, 42);
        let b = shuffled_indices(100, 42);
        assert_eq!(a, b);

        let c = shuffled_indices(100, 43);
        assert_ne!(a, c);
    }

    #[test]
    fn test_shuffled_indices_permutation() {
        let indices = shuffled_indices(10, 123);
        let mut sorted = indices.clone();
        sorted.sort();
        assert_eq!(sorted, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn test_dataloader_basic() {
        let device = CpuDevice::new();
        let ds = SeqDataset { size: 10 };
        let loader = DataLoader::new(ds, 3, 0, device);

        assert_eq!(loader.num_batches(), 3);

        let mut count = 0;
        for batch in loader.iter(0) {
            let batch = batch.expect("batch should not error");
            assert_eq!(batch.inputs.shape(), &[3, 2]);
            assert_eq!(batch.targets.shape(), &[3, 2]);
            count += 1;
        }
        assert_eq!(count, 3);
    }

    #[test]
    fn test_dataloader_different_epochs_different_order() {
        let device = CpuDevice::new();
        let ds = SeqDataset { size: 10 };
        let loader = DataLoader::new(ds, 5, 42, device);

        let epoch0_vals: Vec<f32> = loader
            .iter(0)
            .flat_map(|b| b.unwrap().inputs.to_vec::<f32>())
            .collect();

        let epoch1_vals: Vec<f32> = loader
            .iter(1)
            .flat_map(|b| b.unwrap().inputs.to_vec::<f32>())
            .collect();

        assert_eq!(epoch0_vals.len(), epoch1_vals.len());
        assert_ne!(epoch0_vals, epoch1_vals);
    }

    #[test]
    fn test_dataloader_prefetch() {
        let device = CpuDevice::new();
        let ds = SeqDataset { size: 10 };
        let loader = DataLoader::new(ds, 3, 0, device).with_prefetch(2);

        let mut count = 0;
        for batch in loader.prefetch_iter(0) {
            let batch = batch.expect("prefetch batch should not error");
            assert_eq!(batch.inputs.shape(), &[3, 2]);
            count += 1;
        }
        assert_eq!(count, 3);
    }

    #[test]
    fn test_dataloader_empty() {
        let device = CpuDevice::new();
        let ds = SeqDataset { size: 2 };
        let loader = DataLoader::new(ds, 5, 0, device);

        assert_eq!(loader.num_batches(), 0);
        let mut iter = loader.iter(0);
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_dataloader_exact_size() {
        let device = CpuDevice::new();
        let ds = SeqDataset { size: 10 };
        let loader = DataLoader::new(ds, 3, 0, device);

        let iter = loader.iter(0);
        assert_eq!(iter.len(), 3);
    }
}
