//! What a leader sends each worker before serving: its layer range and the model path.
//!
//! ```text
//! LAYER_ASSIGNMENT: [version: u8][start_layer: u32 LE][end_layer: u32 LE]
//!                   [has_embedding: u8][has_lm_head: u8]
//! MODEL_PATH:       [version: u8][path: UTF-8]
//! ```

use anyhow::{Result, anyhow};

use super::version::{body, versioned};

/// Byte length of a [`LayerAssignment`] body, after the version byte.
const ASSIGNMENT_LEN: usize = 10;

/// The layer range one stage serves, and whether it holds the embedding and the LM head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerAssignment {
    pub start_layer: u32,
    pub end_layer: u32,
    pub has_embedding: bool,
    pub has_lm_head: bool,
}

impl LayerAssignment {
    /// The 10-byte body, without the version byte.
    pub fn to_bytes(&self) -> [u8; ASSIGNMENT_LEN] {
        let mut buf = [0u8; ASSIGNMENT_LEN];
        buf[0..4].copy_from_slice(&self.start_layer.to_le_bytes());
        buf[4..8].copy_from_slice(&self.end_layer.to_le_bytes());
        buf[8] = self.has_embedding as u8;
        buf[9] = self.has_lm_head as u8;
        buf
    }

    /// The assignment in a 10-byte body, without the version byte.
    pub fn from_bytes(buf: &[u8; ASSIGNMENT_LEN]) -> Self {
        Self {
            start_layer: u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]),
            end_layer: u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]),
            has_embedding: buf[8] != 0,
            has_lm_head: buf[9] != 0,
        }
    }

    /// A [`tags::LAYER_ASSIGNMENT`](super::tags::LAYER_ASSIGNMENT) frame:
    /// the version byte, then the body.
    pub fn to_frame(&self) -> Vec<u8> {
        let mut frame = versioned(ASSIGNMENT_LEN);
        frame.extend_from_slice(&self.to_bytes());
        frame
    }

    /// The assignment in a [`tags::LAYER_ASSIGNMENT`](super::tags::LAYER_ASSIGNMENT) frame.
    ///
    /// # Errors
    ///
    /// Another protocol version, or a body that is not 10 bytes.
    pub fn from_frame(frame: &[u8]) -> Result<Self> {
        let rest = body(frame, "layer assignment frame")?;
        let buf: &[u8; ASSIGNMENT_LEN] = rest.try_into().map_err(|_| {
            anyhow!(
                "a layer assignment body is {ASSIGNMENT_LEN} bytes, and this one is {}",
                rest.len()
            )
        })?;
        Ok(Self::from_bytes(buf))
    }
}

/// A [`tags::MODEL_PATH`](super::tags::MODEL_PATH) frame: the version byte, then `path` as UTF-8.
pub fn model_path_frame(path: &str) -> Vec<u8> {
    let mut frame = versioned(path.len());
    frame.extend_from_slice(path.as_bytes());
    frame
}

/// The path in a [`tags::MODEL_PATH`](super::tags::MODEL_PATH) frame.
///
/// # Errors
///
/// Another protocol version, or a path that is not UTF-8.
pub fn model_path_from_frame(frame: &[u8]) -> Result<String> {
    let rest = body(frame, "model path frame")?;
    String::from_utf8(rest.to_vec()).map_err(|e| anyhow!("the model path is not UTF-8: {e}"))
}

#[cfg(test)]
mod tests {
    use super::super::version::PROTOCOL_VERSION;
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
        let frame = orig.to_frame();
        assert_eq!(frame[0], PROTOCOL_VERSION);
        assert_eq!(LayerAssignment::from_frame(&frame).unwrap(), orig);
        let err = LayerAssignment::from_frame(&frame[..10]).unwrap_err();
        assert!(err.to_string().contains("this one is 9"), "{err}");
    }

    #[test]
    fn test_an_assignment_or_path_of_another_version_is_refused() {
        let mut frame = LayerAssignment {
            start_layer: 0,
            end_layer: 4,
            has_embedding: true,
            has_lm_head: false,
        }
        .to_frame();
        frame[0] = PROTOCOL_VERSION + 1;
        let err = LayerAssignment::from_frame(&frame).unwrap_err().to_string();
        assert!(
            err.contains("layer assignment frame of protocol version"),
            "{err}"
        );

        let frame = model_path_frame("/models/llama");
        assert_eq!(model_path_from_frame(&frame).unwrap(), "/models/llama");
        let mut old = frame.clone();
        old[0] = PROTOCOL_VERSION - 1;
        let err = model_path_from_frame(&old).unwrap_err().to_string();
        assert!(
            err.contains("model path frame of protocol version"),
            "{err}"
        );
        let mut bad = model_path_frame("");
        bad.push(0xFF);
        assert!(model_path_from_frame(&bad).is_err());
        // The unversioned path a pre-version leader sent is refused, not read as a path.
        assert!(model_path_from_frame(b"/models/llama").is_err());
    }
}
