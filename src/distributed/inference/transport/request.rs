//! The header of a [`tags::GEN_REQUEST`](super::tags::GEN_REQUEST) frame.
//!
//! ```text
//! [version:    u8 = PROTOCOL_VERSION]
//! [op:         u8]                     — StageOp
//! [request_id: u64 LE]
//! [position:   u32 LE]                 — first position of the payload
//! [seq_len:    u32 LE]                 — positions the payload covers
//! [payload]                            — StagePayload, empty for a release
//! ```

use anyhow::{Result, anyhow, bail};

/// Version every request frame writes. A reader refuses any other version.
pub const PROTOCOL_VERSION: u8 = 2;

/// Byte length of [`GenRequestHeader`] on the wire.
pub const HEADER_LEN: usize = 18;

/// What a request frame asks a stage to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum StageOp {
    /// Open the request at position 0 and run its first positions.
    Prefill = 1,
    /// Run the next positions of an open request.
    Decode = 2,
    /// Free the request. The frame has no payload, and each stage forwards it.
    Release = 3,
}

impl StageOp {
    fn from_u8(v: u8) -> Result<Self> {
        Ok(match v {
            1 => Self::Prefill,
            2 => Self::Decode,
            3 => Self::Release,
            other => bail!("unknown swarm stage op {other}"),
        })
    }
}

/// Header of one request frame: the request, the op and the positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenRequestHeader {
    /// Leader-chosen id. Each stage keeps one KV cache per id.
    pub request_id: u64,
    pub op: StageOp,
    /// First position of the payload: the RoPE offset and the cache length
    /// the stage expects.
    pub position: u32,
    /// Positions the payload covers: token ids, or activation rows.
    pub seq_len: u32,
}

impl GenRequestHeader {
    /// The header of a release of `request_id`.
    pub fn release(request_id: u64) -> Self {
        Self {
            request_id,
            op: StageOp::Release,
            position: 0,
            seq_len: 0,
        }
    }

    /// One past the last position of the payload.
    ///
    /// # Errors
    ///
    /// `position + seq_len` past `u32::MAX`.
    pub fn end(&self) -> Result<u32> {
        self.position.checked_add(self.seq_len).ok_or_else(|| {
            anyhow!(
                "request {}: position {} plus {} positions is past u32::MAX",
                self.request_id,
                self.position,
                self.seq_len
            )
        })
    }

    pub fn to_bytes(&self) -> [u8; HEADER_LEN] {
        let mut buf = [0u8; HEADER_LEN];
        buf[0] = PROTOCOL_VERSION;
        buf[1] = self.op as u8;
        buf[2..10].copy_from_slice(&self.request_id.to_le_bytes());
        buf[10..14].copy_from_slice(&self.position.to_le_bytes());
        buf[14..18].copy_from_slice(&self.seq_len.to_le_bytes());
        buf
    }

    /// # Errors
    ///
    /// Another protocol version, or an unknown op.
    pub fn from_bytes(buf: &[u8; HEADER_LEN]) -> Result<Self> {
        if buf[0] != PROTOCOL_VERSION {
            bail!(
                "a swarm request frame of protocol version {} reached a reader of version \
                 {PROTOCOL_VERSION}. Run the same boostr release on every rank",
                buf[0]
            );
        }
        let u32_at = |i: usize| u32::from_le_bytes([buf[i], buf[i + 1], buf[i + 2], buf[i + 3]]);
        let mut id = [0u8; 8];
        id.copy_from_slice(&buf[2..10]);
        Ok(Self {
            request_id: u64::from_le_bytes(id),
            op: StageOp::from_u8(buf[1])?,
            position: u32_at(10),
            seq_len: u32_at(14),
        })
    }

    /// A [`tags::GEN_REQUEST`](super::tags::GEN_REQUEST) frame: this header, then `payload`.
    pub fn frame(&self, payload: &[u8]) -> Vec<u8> {
        let mut frame = Vec::with_capacity(HEADER_LEN + payload.len());
        frame.extend_from_slice(&self.to_bytes());
        frame.extend_from_slice(payload);
        frame
    }

    /// The header and payload of a request frame.
    ///
    /// # Errors
    ///
    /// A frame shorter than the header, a header [`Self::from_bytes`]
    /// refuses, a release with positions or a payload, or a prefill or
    /// decode with no positions.
    pub fn split_frame(frame: &[u8]) -> Result<(Self, &[u8])> {
        let Some((header, payload)) = frame.split_first_chunk::<HEADER_LEN>() else {
            bail!(
                "a swarm request frame of {} bytes has no {HEADER_LEN}-byte header",
                frame.len()
            );
        };
        let header = Self::from_bytes(header)?;
        match header.op {
            StageOp::Release if header.seq_len != 0 || !payload.is_empty() => bail!(
                "the release of request {} carries {} positions and {} payload bytes, not none",
                header.request_id,
                header.seq_len,
                payload.len()
            ),
            StageOp::Prefill | StageOp::Decode if header.seq_len == 0 => bail!(
                "request {}: a {:?} frame covers no positions",
                header.request_id,
                header.op
            ),
            _ => Ok((header, payload)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode(position: u32, seq_len: u32) -> GenRequestHeader {
        GenRequestHeader {
            request_id: 0x0102_0304_0506_0708,
            op: StageOp::Decode,
            position,
            seq_len,
        }
    }

    #[test]
    fn test_header_roundtrips_with_its_payload() {
        let header = decode(4, 2);
        let frame = header.frame(&[9, 8, 7]);
        assert_eq!(frame.len(), HEADER_LEN + 3);
        let (back, payload) = GenRequestHeader::split_frame(&frame).unwrap();
        assert_eq!(back, header);
        assert_eq!(payload, [9, 8, 7]);
        assert!(GenRequestHeader::split_frame(&frame[..HEADER_LEN - 1]).is_err());
    }

    #[test]
    fn test_another_version_and_an_unknown_op_are_refused() {
        let mut bytes = decode(0, 1).to_bytes();
        bytes[0] = 1;
        let err = GenRequestHeader::from_bytes(&bytes).unwrap_err();
        assert!(err.to_string().contains("protocol version 1"), "{err}");
        let mut bytes = decode(0, 1).to_bytes();
        bytes[1] = 9;
        let err = GenRequestHeader::from_bytes(&bytes).unwrap_err();
        assert!(
            err.to_string().contains("unknown swarm stage op 9"),
            "{err}"
        );
    }

    #[test]
    fn test_a_release_is_empty_and_a_forward_is_not() {
        let release = GenRequestHeader::release(5);
        let frame = release.frame(&[]);
        let (back, payload) = GenRequestHeader::split_frame(&frame).unwrap();
        assert_eq!(
            (back.op, back.request_id, payload.len()),
            (StageOp::Release, 5, 0)
        );
        assert!(GenRequestHeader::split_frame(&release.frame(&[1])).is_err());
        assert!(GenRequestHeader::split_frame(&decode(3, 0).frame(&[])).is_err());
    }

    #[test]
    fn test_end_refuses_an_overflow() {
        assert_eq!(decode(4, 2).end().unwrap(), 6);
        assert!(decode(u32::MAX, 1).end().is_err());
    }
}
