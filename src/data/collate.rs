//! Batch collation and index shuffling helpers.

use crate::data::dataset::{Batch, Dataset};
use crate::error::{Error, Result};
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

/// Collate individual samples into a batched tensor pair.
///
/// Fetches each sample from the dataset, then stacks inputs and targets
/// along a new leading batch dimension → `[batch_size, ...]`.
///
/// Datasets must return CPU tensors. Use `Tensor::to_device()` on the
/// resulting batch for GPU transfer.
pub(crate) fn collate_batch<D: Dataset<CpuRuntime>>(
    dataset: &D,
    indices: &[usize],
    device: &CpuDevice,
) -> Result<Batch<CpuRuntime>> {
    if indices.is_empty() {
        return Err(Error::DataError {
            reason: "empty batch indices".to_string(),
        });
    }

    let mut all_input_data: Vec<f32> = Vec::new();
    let mut all_target_data: Vec<f32> = Vec::new();
    let mut sample_shape: Option<Vec<usize>> = None;

    for &idx in indices {
        let sample = dataset.get(idx, device)?;

        let input_data: Vec<f32> = sample.inputs.to_vec();
        let target_data: Vec<f32> = sample.targets.to_vec();

        if let Some(ref shape) = sample_shape {
            if sample.inputs.shape() != shape.as_slice() {
                return Err(Error::DataError {
                    reason: format!(
                        "inconsistent sample shapes: expected {:?}, got {:?}",
                        shape,
                        sample.inputs.shape()
                    ),
                });
            }
        } else {
            sample_shape = Some(sample.inputs.shape().to_vec());
        }

        all_input_data.extend_from_slice(&input_data);
        all_target_data.extend_from_slice(&target_data);
    }

    // SAFETY: `indices` is non-empty (checked above) so at least one iteration ran.
    let sample_shape = sample_shape.ok_or_else(|| Error::DataError {
        reason: "no samples produced during collation".to_string(),
    })?;

    let mut batch_shape = Vec::with_capacity(sample_shape.len() + 1);
    batch_shape.push(indices.len());
    batch_shape.extend_from_slice(&sample_shape);

    let inputs = Tensor::<CpuRuntime>::from_slice(&all_input_data, &batch_shape, device);
    let targets = Tensor::<CpuRuntime>::from_slice(&all_target_data, &batch_shape, device);

    Ok(Batch { inputs, targets })
}

/// Generate shuffled indices `[0..len)` deterministically from a seed.
///
/// Uses splitmix64 PRNG + Fisher-Yates shuffle for uniform permutation
/// reproducible across platforms.
pub(crate) fn shuffled_indices(len: usize, seed: u64) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..len).collect();
    if len <= 1 {
        return indices;
    }

    let mut state = seed;
    let mut next_u64 = move || -> u64 {
        state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    };

    // Fisher-Yates shuffle
    for i in (1..len).rev() {
        let j = (next_u64() as usize) % (i + 1);
        indices.swap(i, j);
    }

    indices
}
