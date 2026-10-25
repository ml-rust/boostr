//! Frames a worker sends the leader: one logits row, or a stage error.
//!
//! ```text
//! LOGITS:      [version: u8][request_id: u64 LE][frame_id: u64 LE][logit: f32 LE × vocab_size]
//! STAGE_ERROR: [version: u8][request_id: u64 LE][frame_id: u64 LE][rank: u32 LE][message: UTF-8]
//! ```
//!
//! `frame_id` is the [`RequestFrame::frame_id`](super::RequestFrame::frame_id)
//! of the request frame the reply answers. Every version keeps the request
//! id at byte 1 and the frame id at byte 9, so a reply of another version
//! still fails the frame it answers.

use anyhow::{Result, bail};
use nexar::Rank;

use super::version::{body, versioned};

/// Byte length of the request id and the frame id.
const KEY_LEN: usize = 16;

/// The request id and frame id at bytes 1..17 of `frame`, whatever its
/// version, as far as the frame holds them.
pub(crate) fn peek_reply_key(frame: &[u8]) -> (Option<u64>, Option<u64>) {
    let u64_at = |at: usize| {
        frame
            .get(at..)
            .and_then(|b| b.first_chunk::<8>())
            .map(|b| u64::from_le_bytes(*b))
    };
    (u64_at(1), u64_at(9))
}

/// The request id and frame id at the start of a reply body.
fn split_key(rest: &[u8]) -> Option<(u64, u64, &[u8])> {
    let (key, tail) = rest.split_first_chunk::<KEY_LEN>()?;
    let (id, frame) = key.split_at(8);
    let read = |b: &[u8]| b.try_into().ok().map(u64::from_le_bytes);
    Some((read(id)?, read(frame)?, tail))
}

/// The last stage's logits for the last position of one request frame.
#[derive(Debug, Clone, PartialEq)]
pub struct LogitsFrame {
    pub request_id: u64,
    /// The frame id of the request frame these logits answer.
    pub frame_id: u64,
    /// One row of `vocab_size` logits.
    pub logits: Vec<f32>,
}

impl LogitsFrame {
    pub fn to_frame(&self) -> Vec<u8> {
        let mut buf = versioned(KEY_LEN + self.logits.len() * 4);
        buf.extend_from_slice(&self.request_id.to_le_bytes());
        buf.extend_from_slice(&self.frame_id.to_le_bytes());
        for v in &self.logits {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf
    }

    /// # Errors
    ///
    /// Another protocol version, a frame shorter than the request id and
    /// frame id, or a logits part that is not whole `f32` values.
    pub fn from_frame(frame: &[u8]) -> Result<Self> {
        let rest = body(frame, "logits frame")?;
        let Some((request_id, frame_id, values)) = split_key(rest) else {
            bail!(
                "a logits frame of {} bytes has no request id and frame id",
                frame.len()
            );
        };
        let (chunks, tail) = values.as_chunks::<4>();
        if !tail.is_empty() {
            bail!(
                "the logits frame of request {request_id} holds {} value bytes, not whole f32 \
                 values",
                values.len()
            );
        }
        Ok(Self {
            request_id,
            frame_id,
            logits: chunks.iter().map(|&c| f32::from_le_bytes(c)).collect(),
        })
    }
}

/// A stage that failed one request. The worker keeps serving the others.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageError {
    pub request_id: u64,
    /// The frame id of the request frame that failed.
    pub frame_id: u64,
    /// Rank of the stage that failed.
    pub rank: Rank,
    pub message: String,
}

impl StageError {
    pub fn to_frame(&self) -> Vec<u8> {
        let mut buf = versioned(KEY_LEN + 4 + self.message.len());
        buf.extend_from_slice(&self.request_id.to_le_bytes());
        buf.extend_from_slice(&self.frame_id.to_le_bytes());
        buf.extend_from_slice(&self.rank.to_le_bytes());
        buf.extend_from_slice(self.message.as_bytes());
        buf
    }

    /// # Errors
    ///
    /// Another protocol version, or a body shorter than 20 bytes. A message
    /// that is not UTF-8 decodes lossily.
    pub fn from_frame(frame: &[u8]) -> Result<Self> {
        let rest = body(frame, "stage error frame")?;
        let Some((request_id, frame_id, tail)) = split_key(rest) else {
            bail!(
                "a stage error frame of {} bytes has no request id and frame id",
                frame.len()
            );
        };
        let Some((rank, message)) = tail.split_first_chunk::<4>() else {
            bail!("the stage error frame of request {request_id} has no rank");
        };
        Ok(Self {
            request_id,
            frame_id,
            rank: Rank::from_le_bytes(*rank),
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
    use super::super::version::PROTOCOL_VERSION;
    use super::*;

    #[test]
    fn test_logits_round_trip_and_refuse_a_partial_value() {
        let l = LogitsFrame {
            request_id: 42,
            frame_id: 5,
            logits: vec![0.5, -1.0, 3.0],
        };
        let frame = l.to_frame();
        assert_eq!(frame[0], PROTOCOL_VERSION);
        assert_eq!(LogitsFrame::from_frame(&frame).unwrap(), l);
        assert_eq!(peek_reply_key(&frame), (Some(42), Some(5)));
        let err = LogitsFrame::from_frame(&frame[..frame.len() - 1]).unwrap_err();
        assert!(err.to_string().contains("request 42"), "{err}");
        assert!(LogitsFrame::from_frame(&frame[..16]).is_err());
        assert_eq!(peek_reply_key(&frame[..16]), (Some(42), None));
        assert_eq!(peek_reply_key(&frame[..8]), (None, None));
    }

    #[test]
    fn test_a_stage_error_round_trips_and_names_its_stage() {
        let e = StageError {
            request_id: 7,
            frame_id: 3,
            rank: 2,
            message: "out of memory".into(),
        };
        let back = StageError::from_frame(&e.to_frame()).unwrap();
        assert_eq!(back, e);
        assert_eq!(
            back.to_string(),
            "the stage on rank 2 failed request 7: out of memory"
        );
        let mut short = vec![PROTOCOL_VERSION];
        short.extend_from_slice(&[0; 19]);
        assert!(StageError::from_frame(&short).is_err());
    }

    #[test]
    fn test_a_reply_of_another_version_is_refused_and_still_names_its_frame() {
        let mut logits = LogitsFrame {
            request_id: 9,
            frame_id: 4,
            logits: vec![1.0],
        }
        .to_frame();
        logits[0] = PROTOCOL_VERSION + 1;
        let err = LogitsFrame::from_frame(&logits).unwrap_err().to_string();
        assert!(err.contains("logits frame of protocol version"), "{err}");
        assert_eq!(peek_reply_key(&logits), (Some(9), Some(4)));
        let mut error = StageError {
            request_id: 9,
            frame_id: 4,
            rank: 1,
            message: "x".into(),
        }
        .to_frame();
        error[0] = 0;
        let err = StageError::from_frame(&error).unwrap_err().to_string();
        assert!(
            err.contains("stage error frame of protocol version 0"),
            "{err}"
        );
    }
}
