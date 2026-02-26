//! Sharded dataset wrapper for distributed training.
//!
//! Each rank reads a disjoint subset of the underlying dataset,
//! ensuring no data duplication across ranks.

use crate::data::dataset::{Batch, Dataset};
use crate::error::{Error, Result};
use numr::runtime::Runtime;

/// Wraps any `Dataset` to expose only a disjoint shard for one rank.
///
/// Indices are assigned round-robin: rank `r` of `W` total ranks
/// sees indices `r, r+W, r+2W, ...` from the underlying dataset.
///
/// # Example
///
/// ```ignore
/// let full_dataset = MmapDataset::open("data.bin", 1024)?;
/// let shard = new_sharded::<CpuRuntime, _>(full_dataset, rank, world_size)?;
/// let loader = DataLoader::new(shard, batch_size, seed);
/// ```
pub struct ShardedDataset<D> {
    inner: D,
    rank: usize,
    world_size: usize,
    shard_len: usize,
}

impl<D> ShardedDataset<D> {
    /// The underlying dataset.
    pub fn inner(&self) -> &D {
        &self.inner
    }

    /// This rank's index.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Total number of ranks.
    pub fn world_size(&self) -> usize {
        self.world_size
    }
}

impl<D> ShardedDataset<D> {
    fn from_parts(inner: D, rank: usize, world_size: usize, total: usize) -> Result<Self> {
        if world_size == 0 {
            return Err(Error::DataError {
                reason: "world_size must be > 0".to_string(),
            });
        }
        if rank >= world_size {
            return Err(Error::DataError {
                reason: format!("rank {rank} >= world_size {world_size}"),
            });
        }
        let shard_len = total.saturating_sub(rank).div_ceil(world_size);
        Ok(Self {
            inner,
            rank,
            world_size,
            shard_len,
        })
    }
}

/// Create a sharded view of `inner`.
///
/// # Arguments
/// * `inner` - The full dataset
/// * `rank` - This rank's index (0-based)
/// * `world_size` - Total number of ranks
pub fn new_sharded<R: Runtime, D: Dataset<R>>(
    inner: D,
    rank: usize,
    world_size: usize,
) -> Result<ShardedDataset<D>> {
    let total = inner.len();
    ShardedDataset::from_parts(inner, rank, world_size, total)
}

impl<R: Runtime, D: Dataset<R>> Dataset<R> for ShardedDataset<D> {
    fn len(&self) -> usize {
        self.shard_len
    }

    fn get(&self, idx: usize, device: &R::Device) -> Result<Batch<R>> {
        if idx >= self.shard_len {
            return Err(Error::DataError {
                reason: format!(
                    "shard index {idx} out of bounds for shard size {}",
                    self.shard_len
                ),
            });
        }
        let global_idx = idx * self.world_size + self.rank;
        self.inner.get(global_idx, device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::dataset::Batch;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;

    struct DummyDataset {
        size: usize,
    }

    impl Dataset<CpuRuntime> for DummyDataset {
        fn len(&self) -> usize {
            self.size
        }

        fn get(
            &self,
            idx: usize,
            device: &<CpuRuntime as Runtime>::Device,
        ) -> Result<Batch<CpuRuntime>> {
            let val = idx as f32;
            Ok(Batch {
                inputs: Tensor::from_slice(&[val], &[1], device),
                targets: Tensor::from_slice(&[val + 0.5], &[1], device),
            })
        }
    }

    #[test]
    fn test_shard_disjoint() {
        let device = CpuDevice::new();
        let ds = DummyDataset { size: 10 };
        let s0 = new_sharded::<CpuRuntime, _>(ds, 0, 3).unwrap();
        let ds = DummyDataset { size: 10 };
        let s1 = new_sharded::<CpuRuntime, _>(ds, 1, 3).unwrap();
        let ds = DummyDataset { size: 10 };
        let s2 = new_sharded::<CpuRuntime, _>(ds, 2, 3).unwrap();

        // Rank 0: indices 0,3,6,9 → 4 samples
        // Rank 1: indices 1,4,7 → 3 samples (10 not valid)
        // Rank 2: indices 2,5,8 → 3 samples
        assert_eq!(s0.len(), 4);
        assert_eq!(s1.len(), 3);
        assert_eq!(s2.len(), 3);

        // Verify rank 0 gets global indices 0, 3, 6, 9
        let b = s0.get(0, &device).unwrap();
        assert_eq!(b.inputs.to_vec::<f32>(), vec![0.0]);
        let b = s0.get(1, &device).unwrap();
        assert_eq!(b.inputs.to_vec::<f32>(), vec![3.0]);
        let b = s0.get(2, &device).unwrap();
        assert_eq!(b.inputs.to_vec::<f32>(), vec![6.0]);
        let b = s0.get(3, &device).unwrap();
        assert_eq!(b.inputs.to_vec::<f32>(), vec![9.0]);
    }

    #[test]
    fn test_shard_invalid_rank() {
        let ds = DummyDataset { size: 10 };
        assert!(new_sharded::<CpuRuntime, _>(ds, 3, 3).is_err());
    }
}
