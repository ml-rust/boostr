//! Tests for the codec token-space description.

use super::*;

#[test]
fn codec_construction_validates_sizes() {
    assert!(CodecVocab::new(0, 4096).is_err());
    assert!(CodecVocab::new(1, 0).is_err());
    assert!(CodecVocab::with_frame_layout(4096, vec![]).is_err());
    assert!(CodecVocab::with_frame_layout(4096, vec![1, 0, 1]).is_err());
    // A codebook span wider than the u32 id space is rejected.
    assert!(CodecVocab::new(2, usize::MAX / 2).is_err());
    assert!(CodecVocab::new(1, u32::MAX as usize + 1).is_err());
    assert!(CodecVocab::new(1, u32::MAX as usize).is_ok());
}
