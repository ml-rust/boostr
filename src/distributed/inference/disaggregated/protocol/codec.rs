//! Frame header and little-endian field codec of the disaggregated protocol.
//!
//! # Header
//!
//! ```text
//! [magic:      u32 LE = 0xB005_D15A]
//! [version:    u32 LE = PROTOCOL_VERSION]
//! [kind:       u8]                         — FrameKind
//! [request_id: u64 LE]
//! ```
//!
//! Every version keeps `magic` and `request_id` at these offsets. A reader
//! that refuses a version still reads the request id, so the refusal reaches
//! the router as an error frame for that request.
//!
//! The router numbers requests from a random 64-bit origin it draws at
//! start, with wrapping. Ids of one router never repeat, and a restarted
//! router misses the ids a long-running worker remembers from the router
//! before it, except by a chance near `2n / 2^64` for `n` requests each.
//! A worker treats an id as an opaque key: no reader orders or compares
//! ids, so the full `u64` range is valid, 0 included.

use anyhow::{Result, anyhow, bail};

/// Magic of every disaggregated frame.
pub const FRAME_MAGIC: u32 = 0xB005_D15A;

/// Version every frame writes. A reader refuses any other version.
pub const PROTOCOL_VERSION: u32 = 2;

/// Byte length of the header, request id included.
pub const HEADER_LEN: usize = 17;

/// The frame kind byte of the header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FrameKind {
    /// Router → prefill worker: prompt and generation params.
    PrefillRequest = 1,
    /// Prefill worker → decode worker: KV cache, context and params.
    KvHandoff = 2,
    /// Router → worker: stop the request and free its state.
    Cancel = 3,
    /// Worker → router: one token, one error, or a handoff confirmation.
    Event = 4,
    /// Decode worker → prefill worker: decode holds the KV handoff in its
    /// queue, or refused it with an error event to the router.
    HandoffAck = 5,
}

impl FrameKind {
    fn from_u8(v: u8) -> Result<Self> {
        Ok(match v {
            1 => Self::PrefillRequest,
            2 => Self::KvHandoff,
            3 => Self::Cancel,
            4 => Self::Event,
            5 => Self::HandoffAck,
            other => bail!("unknown disaggregated frame kind {other}"),
        })
    }
}

/// The request id of `buf` when its magic matches, whatever its version.
pub fn peek_request_id(buf: &[u8]) -> Option<u64> {
    if buf.len() < HEADER_LEN || buf[0..4] != FRAME_MAGIC.to_le_bytes() {
        return None;
    }
    let mut id = [0u8; 8];
    id.copy_from_slice(&buf[9..17]);
    Some(u64::from_le_bytes(id))
}

/// Appends little-endian fields to a frame buffer.
#[derive(Debug)]
pub struct Writer {
    buf: Vec<u8>,
    /// Largest count or length a `u32` prefix can carry.
    max_len: usize,
    /// The first oversize field. [`Self::finish`] returns it.
    error: Option<anyhow::Error>,
}

impl Writer {
    /// A buffer that starts with the header of `kind` for `request_id`.
    pub fn frame(kind: FrameKind, request_id: u64) -> Self {
        let mut w = Self {
            buf: Vec::new(),
            max_len: u32::MAX as usize,
            error: None,
        };
        w.u32(FRAME_MAGIC);
        w.u32(PROTOCOL_VERSION);
        w.u8(kind as u8);
        w.u64(request_id);
        w
    }

    pub fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    pub fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn f32(&mut self, v: f32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// The header bytes of a frame that has no body.
    pub fn header(kind: FrameKind, request_id: u64) -> Vec<u8> {
        Self::frame(kind, request_id).buf
    }

    /// Writes `len` as a `u32` prefix. A `len` above `max_len` records an
    /// error, writes 0 and leaves the frame unusable.
    fn len_prefix(&mut self, field: &str, len: usize) {
        match u32::try_from(len)
            .ok()
            .filter(|&n| n as usize <= self.max_len)
        {
            Some(n) => self.u32(n),
            None => {
                self.error.get_or_insert_with(|| {
                    anyhow!(
                        "disaggregated frame field `{field}` has {len} entries, above the \
                         {} limit of its u32 length prefix: split the request or shorten the field",
                        self.max_len
                    )
                });
                self.u32(0);
            }
        }
    }

    /// A `u32` count, then each id as `u32` LE.
    pub fn u32s(&mut self, field: &str, v: &[u32]) {
        self.len_prefix(field, v.len());
        if self.error.is_some() {
            return;
        }
        for &x in v {
            self.u32(x);
        }
    }

    /// A `u64` length, then the bytes.
    pub fn bytes(&mut self, v: &[u8]) {
        self.u64(v.len() as u64);
        self.buf.extend_from_slice(v);
    }

    /// A `u32` length, then the UTF-8 bytes.
    pub fn str(&mut self, field: &str, v: &str) {
        self.len_prefix(field, v.len());
        if self.error.is_some() {
            return;
        }
        self.buf.extend_from_slice(v.as_bytes());
    }

    /// The frame bytes.
    ///
    /// # Errors
    ///
    /// A `u32s` or `str` field above the `u32` length limit.
    pub fn finish(self) -> Result<Vec<u8>> {
        match self.error {
            Some(e) => Err(e),
            None => Ok(self.buf),
        }
    }
}

/// Reads little-endian fields from a frame buffer, naming the field on a short read.
#[derive(Debug)]
pub struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    /// Checks the header of `buf` against `kind` and returns the reader and the request id.
    ///
    /// # Errors
    ///
    /// A wrong magic, a version other than [`PROTOCOL_VERSION`], or a kind other than `kind`.
    pub fn frame(buf: &'a [u8], kind: FrameKind) -> Result<(Self, u64)> {
        let mut r = Self { buf, pos: 0 };
        let magic = r.u32("magic")?;
        if magic != FRAME_MAGIC {
            bail!("not a disaggregated frame: magic {magic:#010x}, expected {FRAME_MAGIC:#010x}");
        }
        let version = r.u32("version")?;
        if version != PROTOCOL_VERSION {
            bail!(
                "disaggregated protocol version {version} is not supported: this build reads \
                 version {PROTOCOL_VERSION}. Run the same boostr release on every rank."
            );
        }
        let got = FrameKind::from_u8(r.u8("kind")?)?;
        if got != kind {
            bail!("expected a {kind:?} frame, got {got:?}");
        }
        let request_id = r.u64("request_id")?;
        Ok((r, request_id))
    }

    fn take(&mut self, n: usize, field: &str) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&end| end <= self.buf.len())
            .ok_or_else(|| {
                anyhow!(
                    "disaggregated frame truncated at field `{field}`: needs {n} bytes at \
                     offset {}, frame has {}",
                    self.pos,
                    self.buf.len()
                )
            })?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn array<const N: usize>(&mut self, field: &str) -> Result<[u8; N]> {
        let mut out = [0u8; N];
        out.copy_from_slice(self.take(N, field)?);
        Ok(out)
    }

    pub fn u8(&mut self, field: &str) -> Result<u8> {
        Ok(self.take(1, field)?[0])
    }

    pub fn u32(&mut self, field: &str) -> Result<u32> {
        Ok(u32::from_le_bytes(self.array(field)?))
    }

    pub fn u64(&mut self, field: &str) -> Result<u64> {
        Ok(u64::from_le_bytes(self.array(field)?))
    }

    pub fn f32(&mut self, field: &str) -> Result<f32> {
        Ok(f32::from_le_bytes(self.array(field)?))
    }

    pub fn u32s(&mut self, field: &str) -> Result<Vec<u32>> {
        let n = self.u32(field)? as usize;
        let raw = self.take(n.saturating_mul(4), field)?;
        Ok(raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| u32::from_le_bytes(c))
            .collect())
    }

    pub fn bytes(&mut self, field: &str) -> Result<Vec<u8>> {
        let n = usize::try_from(self.u64(field)?)
            .map_err(|_| anyhow!("field `{field}` length does not fit in usize"))?;
        Ok(self.take(n, field)?.to_vec())
    }

    pub fn str(&mut self, field: &str) -> Result<String> {
        let n = self.u32(field)? as usize;
        let raw = self.take(n, field)?;
        String::from_utf8(raw.to_vec()).map_err(|e| anyhow!("field `{field}` is not UTF-8: {e}"))
    }

    /// Refuses bytes past the last field.
    pub fn end(self) -> Result<()> {
        if self.pos != self.buf.len() {
            bail!(
                "disaggregated frame has {} trailing bytes",
                self.buf.len() - self.pos
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_round_trips_and_peeks() {
        let mut w = Writer::frame(FrameKind::Cancel, 77);
        w.u32s("a", &[1, 2, 3]);
        w.bytes(b"kv");
        w.str("c", "hi");
        w.f32(0.5);
        let buf = w.finish().unwrap();
        assert_eq!(peek_request_id(&buf), Some(77));
        let (mut r, id) = Reader::frame(&buf, FrameKind::Cancel).unwrap();
        assert_eq!(id, 77);
        assert_eq!(r.u32s("a").unwrap(), vec![1, 2, 3]);
        assert_eq!(r.bytes("b").unwrap(), b"kv");
        assert_eq!(r.str("c").unwrap(), "hi");
        assert_eq!(r.f32("d").unwrap(), 0.5);
        r.end().unwrap();
    }

    #[test]
    fn wrong_kind_magic_and_version_are_refused() {
        let buf = Writer::header(FrameKind::Cancel, 5);
        let err = Reader::frame(&buf, FrameKind::Event).unwrap_err();
        assert!(err.to_string().contains("expected a Event frame"), "{err}");

        let mut bad_magic = buf.clone();
        bad_magic[0] ^= 0xFF;
        assert_eq!(peek_request_id(&bad_magic), None);
        assert!(Reader::frame(&bad_magic, FrameKind::Cancel).is_err());

        let mut future = buf.clone();
        future[4..8].copy_from_slice(&(PROTOCOL_VERSION + 1).to_le_bytes());
        let err = Reader::frame(&future, FrameKind::Cancel).unwrap_err();
        assert!(err.to_string().contains("version"), "{err}");
        assert_eq!(
            peek_request_id(&future),
            Some(5),
            "the id survives a version refusal"
        );
    }

    #[test]
    fn truncation_and_trailing_bytes_are_refused() {
        let mut w = Writer::frame(FrameKind::Cancel, 1);
        w.u32s("ids", &[9, 9]);
        let mut buf = w.finish().unwrap();
        buf.pop();
        let (mut r, _) = Reader::frame(&buf, FrameKind::Cancel).unwrap();
        let err = r.u32s("ids").unwrap_err();
        assert!(err.to_string().contains("`ids`"), "{err}");

        let mut buf = Writer::header(FrameKind::Cancel, 1);
        buf.push(0);
        let (r, _) = Reader::frame(&buf, FrameKind::Cancel).unwrap();
        assert!(r.end().is_err());
        assert!(Reader::frame(&buf[..10], FrameKind::Cancel).is_err());
    }

    #[test]
    fn an_oversize_field_fails_finish_and_names_the_field() {
        let mut w = Writer::frame(FrameKind::Cancel, 1);
        w.max_len = 2;
        w.u32s("prompt", &[1, 2]);
        w.str("message", "abc");
        w.u32s("later", &[1, 2, 3, 4]);
        let err = w.finish().unwrap_err().to_string();
        assert!(
            err.contains("`message`") && err.contains("3 entries"),
            "{err}"
        );

        let mut ok = Writer::frame(FrameKind::Cancel, 1);
        ok.max_len = 2;
        ok.u32s("prompt", &[1, 2]);
        assert!(ok.finish().is_ok());
    }
}
