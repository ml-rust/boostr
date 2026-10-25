//! The layer range a leader assigns to one worker.

use anyhow::{Result, anyhow};

/// Layer assignment message: serialized as [start_layer: u32, end_layer: u32, has_embedding: u8, has_lm_head: u8].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerAssignment {
    pub start_layer: u32,
    pub end_layer: u32,
    pub has_embedding: bool,
    pub has_lm_head: bool,
}

impl LayerAssignment {
    pub fn to_bytes(&self) -> [u8; 10] {
        let mut buf = [0u8; 10];
        buf[0..4].copy_from_slice(&self.start_layer.to_le_bytes());
        buf[4..8].copy_from_slice(&self.end_layer.to_le_bytes());
        buf[8] = self.has_embedding as u8;
        buf[9] = self.has_lm_head as u8;
        buf
    }

    pub fn from_bytes(buf: &[u8; 10]) -> Self {
        Self {
            start_layer: u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]),
            end_layer: u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]),
            has_embedding: buf[8] != 0,
            has_lm_head: buf[9] != 0,
        }
    }

    /// The assignment in a [`tags::LAYER_ASSIGNMENT`](super::tags::LAYER_ASSIGNMENT) frame.
    ///
    /// # Errors
    ///
    /// A frame that is not 10 bytes.
    pub fn from_frame(frame: &[u8]) -> Result<Self> {
        let buf: &[u8; 10] = frame.try_into().map_err(|_| {
            anyhow!(
                "a layer assignment frame is 10 bytes, and this one is {}",
                frame.len()
            )
        })?;
        Ok(Self::from_bytes(buf))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_assignment_roundtrip() {
        let orig = LayerAssignment {
            start_layer: 16,
            end_layer: 32,
            has_embedding: false,
            has_lm_head: true,
        };
        assert_eq!(LayerAssignment::from_bytes(&orig.to_bytes()), orig);
        assert_eq!(LayerAssignment::from_frame(&orig.to_bytes()).unwrap(), orig);
        let err = LayerAssignment::from_frame(&[0; 9]).unwrap_err();
        assert!(err.to_string().contains("this one is 9"), "{err}");
    }
}
