//! Memory-mapped dataset for large token files.
//!
//! Reads raw binary files of contiguous `u32` little-endian token IDs.
//! Backed by `mmap` so only accessed pages are loaded into RAM,
//! enabling training on datasets larger than available memory.

use std::path::Path;

use memmap2::Mmap;

use crate::data::dataset::{Batch, Dataset};
use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Memory-mapped dataset over a raw binary token file.
///
/// File format: contiguous little-endian `u32` token IDs, no header.
/// Each sample is a `(inputs, targets)` pair of `[seq_len]` tensors
/// where targets are shifted by one position (causal LM objective).
///
/// # Memory
///
/// Only pages touched by the OS page cache are resident in RAM.
/// A 100GB dataset uses ~0 bytes of RSS until samples are accessed.
pub struct MmapDataset {
    mmap: Mmap,
    seq_len: usize,
    num_tokens: usize,
}

impl MmapDataset {
    /// Open a memory-mapped dataset from a binary token file.
    ///
    /// # Arguments
    /// * `path` - Path to raw `.bin` file (contiguous LE u32 tokens)
    /// * `seq_len` - Sequence length per sample
    pub fn open(path: impl AsRef<Path>, seq_len: usize) -> Result<Self> {
        let path = path.as_ref();
        let file = std::fs::File::open(path).map_err(|e| Error::DataError {
            reason: format!("failed to open {}: {e}", path.display()),
        })?;

        let metadata = file.metadata()?;
        let file_len = metadata.len() as usize;

        if file_len % 4 != 0 {
            return Err(Error::DataError {
                reason: format!(
                    "file size {} is not a multiple of 4 (expected u32 tokens)",
                    file_len
                ),
            });
        }

        let num_tokens = file_len / 4;

        if num_tokens < seq_len + 1 {
            return Err(Error::DataError {
                reason: format!(
                    "file has {num_tokens} tokens but need at least {} for seq_len={seq_len}",
                    seq_len + 1
                ),
            });
        }

        // SAFETY: The file is opened read-only and we only read u32 values from it.
        let mmap = unsafe { Mmap::map(&file)? };

        Ok(Self {
            mmap,
            seq_len,
            num_tokens,
        })
    }

    /// Total number of tokens in the file.
    pub fn num_tokens(&self) -> usize {
        self.num_tokens
    }

    /// Sequence length per sample.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Number of samples (non-overlapping windows of `seq_len + 1` tokens).
    pub fn num_samples(&self) -> usize {
        self.num_tokens / (self.seq_len + 1)
    }

    /// Read a slice of tokens starting at a token offset.
    fn read_tokens(&self, token_offset: usize, count: usize) -> &[u32] {
        let byte_offset = token_offset * 4;
        let byte_end = byte_offset + count * 4;
        let bytes = &self.mmap[byte_offset..byte_end];
        // SAFETY: mmap is aligned to page boundary (4K+), u32 alignment (4) is satisfied.
        // The file size was verified to be a multiple of 4.
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u32, count) }
    }
}

impl<R: Runtime<DType = DType>> Dataset<R> for MmapDataset {
    fn len(&self) -> usize {
        self.num_samples()
    }

    fn get(&self, idx: usize, device: &R::Device) -> Result<Batch<R>> {
        let len = self.num_samples();
        if idx >= len {
            return Err(Error::DataError {
                reason: format!("index {idx} out of bounds for dataset of size {len}"),
            });
        }

        let start = idx * (self.seq_len + 1);
        let tokens = self.read_tokens(start, self.seq_len + 1);

        // Convert u32 tokens to f32 for tensor creation.
        let input_f32: Vec<f32> = tokens[..self.seq_len].iter().map(|&t| t as f32).collect();
        let target_f32: Vec<f32> = tokens[1..=self.seq_len].iter().map(|&t| t as f32).collect();

        let inputs = Tensor::<R>::from_slice(&input_f32, &[self.seq_len], device);
        let targets = Tensor::<R>::from_slice(&target_f32, &[self.seq_len], device);

        Ok(Batch { inputs, targets })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_token_file(tokens: &[u32]) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        for &t in tokens {
            f.write_all(&t.to_le_bytes()).unwrap();
        }
        f.flush().unwrap();
        f
    }

    #[test]
    fn test_mmap_dataset_basic() {
        // 10 tokens, seq_len=3 → each sample needs 4 tokens → 2 samples
        let tokens: Vec<u32> = (0..10).collect();
        let f = write_token_file(&tokens);
        let ds = MmapDataset::open(f.path(), 3).unwrap();

        assert_eq!(ds.num_tokens(), 10);
        assert_eq!(ds.num_samples(), 2); // 10 / (3+1) = 2

        let device = CpuDevice::new();
        let ds: &dyn Dataset<CpuRuntime> = &ds;
        let batch = ds.get(0, &device).unwrap();
        let inputs: Vec<f32> = batch.inputs.to_vec();
        let targets: Vec<f32> = batch.targets.to_vec();
        assert_eq!(inputs, vec![0.0, 1.0, 2.0]);
        assert_eq!(targets, vec![1.0, 2.0, 3.0]);

        let batch1 = ds.get(1, &device).unwrap();
        let inputs1: Vec<f32> = batch1.inputs.to_vec();
        assert_eq!(inputs1, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_mmap_dataset_out_of_bounds() {
        let tokens: Vec<u32> = (0..8).collect();
        let f = write_token_file(&tokens);
        let ds = MmapDataset::open(f.path(), 3).unwrap();
        let device = CpuDevice::new();
        let ds: &dyn Dataset<CpuRuntime> = &ds;
        assert!(ds.get(2, &device).is_err());
    }

    #[test]
    fn test_mmap_dataset_too_small() {
        let tokens: Vec<u32> = vec![0, 1];
        let f = write_token_file(&tokens);
        assert!(MmapDataset::open(f.path(), 3).is_err());
    }

    #[test]
    fn test_mmap_dataset_bad_alignment() {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(&[0u8; 5]).unwrap(); // 5 bytes, not multiple of 4
        f.flush().unwrap();
        assert!(MmapDataset::open(f.path(), 1).is_err());
    }
}
