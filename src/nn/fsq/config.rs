//! Configuration for a Finite Scalar Quantizer.

use crate::error::{Error, Result};

/// Configuration for [`Fsq`](super::quantizer::Fsq).
///
/// Mirrors `FSQ.__init__` from lucidrains/vector-quantize-pytorch
/// (`finite_scalar_quantization.py`), single-codebook case (`num_codebooks = 1`,
/// which is what `ResidualFSQ(num_quantizers = 1)` degenerates to — the setup
/// NeuCodec/WideCodec use).
#[derive(Debug, Clone)]
pub struct FsqConfig {
    /// Per-dimension quantization levels, e.g. `[4; 8]` for NeuCodec/WideCodec
    /// (codebook dim = 8, codebook size = 4^8 = 65_536).
    pub levels: Vec<u32>,
    /// Dimensionality of the tensor fed into `Fsq::quantize` / produced by
    /// `Fsq::indices_to_codes`. When it differs from `levels.len()`, `Fsq`
    /// projects in/out with a `Linear` layer (required, not optional — e.g.
    /// NeuCodec has `input_dim = 2048` against 8 levels).
    pub input_dim: usize,
}

impl FsqConfig {
    /// Construct and validate a configuration.
    pub fn new(levels: Vec<u32>, input_dim: usize) -> Result<Self> {
        let config = Self { levels, input_dim };
        config.validate()?;
        Ok(config)
    }

    /// Number of scalar dimensions actually quantized (`levels.len()`).
    pub fn codebook_dim(&self) -> usize {
        self.levels.len()
    }

    /// Total codebook size: the product of all levels.
    ///
    /// `[4; 8]` gives `4^8 = 65_536`.
    pub fn codebook_size(&self) -> u64 {
        self.levels.iter().map(|&level| level as u64).product()
    }

    /// Whether an input/output `Linear` projection is required because
    /// `input_dim != codebook_dim()`.
    pub fn needs_projection(&self) -> bool {
        self.input_dim != self.codebook_dim()
    }

    /// Validate: non-empty levels, every level >= 2, non-zero `input_dim`, and
    /// `codebook_size()` fits in `i32` (indices are stored as `DType::I32`).
    pub fn validate(&self) -> Result<()> {
        if self.levels.is_empty() {
            return Err(Error::InvalidArgument {
                arg: "levels",
                reason: "must not be empty".to_string(),
            });
        }
        if let Some(&bad_level) = self.levels.iter().find(|&&level| level < 2) {
            return Err(Error::InvalidArgument {
                arg: "levels",
                reason: format!("every level must be >= 2, found {bad_level}"),
            });
        }
        if self.input_dim == 0 {
            return Err(Error::InvalidArgument {
                arg: "input_dim",
                reason: "must not be zero".to_string(),
            });
        }
        let size = self.codebook_size();
        if size > i32::MAX as u64 {
            return Err(Error::InvalidArgument {
                arg: "levels",
                reason: format!(
                    "codebook_size {size} exceeds i32::MAX; indices are stored as DType::I32"
                ),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codebook_size_neucodec() {
        // NeuCodec/WideCodec: levels = [4; 8] -> a single 65_536-entry codebook.
        let config = FsqConfig::new(vec![4; 8], 8).unwrap();
        assert_eq!(config.codebook_size(), 65_536);
        assert_eq!(config.codebook_dim(), 8);
        assert!(!config.needs_projection());
    }

    #[test]
    fn test_codebook_size_toy() {
        let config = FsqConfig::new(vec![4, 4], 2).unwrap();
        assert_eq!(config.codebook_size(), 16);
    }

    #[test]
    fn test_needs_projection_when_dims_differ() {
        // NeuCodec's real setup: dim = 2048 against 8 levels.
        let config = FsqConfig::new(vec![4; 8], 2048).unwrap();
        assert!(config.needs_projection());
    }

    #[test]
    fn test_empty_levels_rejected() {
        let err = FsqConfig::new(vec![], 4).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument { arg: "levels", .. }));
    }

    #[test]
    fn test_level_below_two_rejected() {
        let err = FsqConfig::new(vec![4, 1, 4], 3).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument { arg: "levels", .. }));
    }

    #[test]
    fn test_zero_input_dim_rejected() {
        let err = FsqConfig::new(vec![4, 4], 0).unwrap_err();
        assert!(matches!(
            err,
            Error::InvalidArgument {
                arg: "input_dim",
                ..
            }
        ));
    }

    #[test]
    fn test_codebook_size_overflow_rejected() {
        // 2^32 levels of 2 each would overflow i32::MAX.
        let err = FsqConfig::new(vec![2; 32], 32).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument { arg: "levels", .. }));
    }
}
