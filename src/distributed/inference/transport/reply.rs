//! Frames a worker sends the leader: one logits row, or a stage error.
//!
//! ```text
//! LOGITS:      [request_id: u64 LE][logit: f32 LE × vocab_size]
//! STAGE_ERROR: [request_id: u64 LE][rank: u32 LE][message: UTF-8]
//! ```

use anyhow::{Result, bail};
use nexar::Rank;

/// The request id at the start of `frame`, if it holds one.
pub(crate) fn peek_request_id(frame: &[u8]) -> Option<u64> {
    frame.first_chunk::<8>().map(|id| u64::from_le_bytes(*id))
}

/// The last stage's logits for the last position of one request frame.
#[derive(Debug, Clone, PartialEq)]
pub struct LogitsFrame {
    pub request_id: u64,
    /// One row of `vocab_size` logits.
    pub logits: Vec<f32>,
}

impl LogitsFrame {
    pub fn to_frame(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(8 + self.logits.len() * 4);
        buf.extend_from_slice(&self.request_id.to_le_bytes());
        for v in &self.logits {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf
    }

    /// # Errors
    ///
    /// A frame shorter than the request id, or a logits part that is not
    /// whole `f32` values.
    pub fn from_frame(frame: &[u8]) -> Result<Self> {
        let Some((id, values)) = frame.split_first_chunk::<8>() else {
            bail!("a logits frame of {} bytes has no request id", frame.len());
        };
        let (chunks, rest) = values.as_chunks::<4>();
        if !rest.is_empty() {
            bail!(
                "the logits frame of request {} holds {} value bytes, not whole f32 values",
                u64::from_le_bytes(*id),
                values.len()
            );
        }
        Ok(Self {
            request_id: u64::from_le_bytes(*id),
            logits: chunks.iter().map(|&c| f32::from_le_bytes(c)).collect(),
        })
    }
}

/// A stage that failed one request. The worker keeps serving the others.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageError {
    pub request_id: u64,
    /// Rank of the stage that failed.
    pub rank: Rank,
    pub message: String,
}

impl StageError {
    pub fn to_frame(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(12 + self.message.len());
        buf.extend_from_slice(&self.request_id.to_le_bytes());
        buf.extend_from_slice(&self.rank.to_le_bytes());
        buf.extend_from_slice(self.message.as_bytes());
        buf
    }

    /// # Errors
    ///
    /// A frame shorter than 12 bytes. A message that is not UTF-8 decodes lossily.
    pub fn from_frame(frame: &[u8]) -> Result<Self> {
        let Some((head, message)) = frame.split_first_chunk::<12>() else {
            bail!(
                "a stage error frame of {} bytes has no 12-byte head",
                frame.len()
            );
        };
        let mut id = [0u8; 8];
        id.copy_from_slice(&head[..8]);
        Ok(Self {
            request_id: u64::from_le_bytes(id),
            rank: Rank::from_le_bytes([head[8], head[9], head[10], head[11]]),
            message: String::from_utf8_lossy(message).into_owned(),
        })
    }
}

impl std::fmt::Display for StageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "the stage on rank {} failed request {}: {}",
            self.rank, self.request_id, self.message
        )
    }
}

impl std::error::Error for StageError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logits_round_trip_and_refuse_a_partial_value() {
        let l = LogitsFrame {
            request_id: 42,
            logits: vec![0.5, -1.0, 3.0],
        };
        let frame = l.to_frame();
        assert_eq!(LogitsFrame::from_frame(&frame).unwrap(), l);
        assert_eq!(peek_request_id(&frame), Some(42));
        let err = LogitsFrame::from_frame(&frame[..frame.len() - 1]).unwrap_err();
        assert!(err.to_string().contains("request 42"), "{err}");
        assert!(LogitsFrame::from_frame(&frame[..7]).is_err());
        assert_eq!(peek_request_id(&frame[..7]), None);
    }

    #[test]
    fn test_a_stage_error_round_trips_and_names_its_stage() {
        let e = StageError {
            request_id: 7,
            rank: 2,
            message: "out of memory".into(),
        };
        let back = StageError::from_frame(&e.to_frame()).unwrap();
        assert_eq!(back, e);
        assert_eq!(
            back.to_string(),
            "the stage on rank 2 failed request 7: out of memory"
        );
        assert!(StageError::from_frame(&[0; 11]).is_err());
    }
}
