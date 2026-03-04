//! Decomposed quantization method descriptors

/// Quantization method for decomposed (non-monolithic) formats.
///
/// Unlike GGUF's block-quantized `QuantTensor`, AWQ and GPTQ store weights
/// as separate component tensors (qweight, scales, qzeros). This enum
/// identifies the specific format so the correct dequantization formula
/// and packing layout are used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecomposedQuantMethod {
    /// AWQ (Activation-aware Weight Quantization)
    ///
    /// Packing: 8 INT4 values per u32, AWQ order shifts [0,16,4,20,8,24,12,28]
    /// Dequant: w = (q - zero) * scale
    Awq { group_size: usize },

    /// GPTQ (Generative Pre-trained Transformer Quantization)
    ///
    /// Packing: 8 INT4 values per u32, sequential 4-bit packing
    /// Dequant: w = q * scale + zero
    /// Additional g_idx tensor for column permutation
    Gptq { group_size: usize },
}

impl DecomposedQuantMethod {
    /// Group size for this method
    pub fn group_size(&self) -> usize {
        match self {
            Self::Awq { group_size } | Self::Gptq { group_size } => *group_size,
        }
    }
}
