//! The [`TcfWriter::finish`] entry points: the three passes in order, over
//! an in-memory buffer or a `Write + Seek` sink. Section 4.1, Section 5.3.
//!
//! # The three passes
//!
//! The property that makes the order work is that **the directory is computable from headers alone**:
//! `physical_span_bytes` is a function of shape and encoding, and
//! `data_offset` is a running 64-aligned sum. Neither needs tensor data.
//!
//! ```text
//! pass 1   `plan` + `emit_directory`
//!          every physical_span_bytes, data_offset, proof_rel_off
//!          header, all six record arrays, string table
//!
//! pass 2   `emit_payloads`, in directory order, one tensor at a time:
//!          payload_digest, semantic_digest and the 64 proof values
//!          payload written at its precomputed data_offset, zero-filled to
//!          physical_span_bytes (Section 14.4), then the payload dropped
//!
//! pass 3   `finalize`
//!          header_digest, then directory_digest (Section 5.3)
//! ```
//!
//! Pass 2 moves each payload out of the writer, writes it at its final
//! offset, and drops it before the next tensor is touched. The proof vector
//! each tensor produces is written straight into the proof section at its
//! final offset rather than buffered, which leaves the section complete
//! before pass 3 computes the digest covering it.
//!
//! # Memory: the two entry points differ, and only one is streaming
//!
//! Neither [`TcfWriter::finish`] nor [`TcfWriter::finish_into`] holds a
//! second copy of a payload, and neither is one-tensor-in-flight either:
//! [`TcfWriter::add_block_tensor`] and [`TcfWriter::add_raw_tensor`]
//! store their payload in the writer, so every tensor's payload is resident
//! at once by the time pass 2 starts. What pass 2 bounds is the *extra* cost
//! on top of that set, not the set itself.
//!
//! [`TcfWriter::finish_streaming`] is the path that holds one payload at a
//! time. A caller registers each tensor's record with
//! [`TcfWriter::register_block_tensor`] or
//! [`TcfWriter::register_raw_tensor`], supplying no payload, and passes a
//! callback the writer pulls from once per tensor in directory order. That
//! works because the directory is computable from headers alone: only the
//! per-tensor digests need data, and the directory is written last. Peak
//! memory is then the directory plus one tensor.
//!
//! All three write byte-identical files for the same tensors in the same
//! order.
//!
//! Every path buffers `[0, data_off)` — the header, the six record arrays,
//! the string table, and the proof section. That is a function of record
//! **count**, not of payload size. Payload bytes are written to the sink at
//! their precomputed absolute offsets and never enter a buffer that scales
//! with the file. [`TcfWriter::finish`] is [`TcfWriter::finish_into`] over
//! an in-memory sink, and returns the whole file.

use std::io::{Seek, Write};

use crate::tcf::error::TcfError;

use super::layout::{Layout, finalize, seek_write};
use super::records::{LAYOUT_BOUNDS, TcfWriter};

impl TcfWriter {
    /// Emit the complete file as one buffer. Section 4.1,
    ///
    /// The whole file is held in memory, on top of the payloads the writer
    /// already holds. [`TcfWriter::finish_into`] drops the file copy by
    /// writing to a sink; [`TcfWriter::finish_streaming`] drops the payload
    /// set too, and is the one to use for model weights that do not fit.
    /// All three produce identical bytes.
    ///
    /// # Errors
    /// Every error the three passes raise: layout overflow
    /// ([`TcfError::SectionBounds`]), a record the reader would reject (its
    /// own decode error), a shape error from the
    /// proof vector, a payload whose length disagrees with Section 8.0.1
    /// reported as [`TcfError::InvalidQuantShape`], or
    /// [`TcfError::NonzeroReserved`] naming a caller-supplied
    /// `policy_digest`, `contract_digest`, or `relation_digest`. A tensor
    /// registered with [`TcfWriter::register_block_tensor`] or
    /// [`TcfWriter::register_raw_tensor`] and never finished with
    /// [`TcfWriter::finish_streaming`] is [`TcfError::PayloadMismatch`]. The
    /// in-memory sink this method uses raises no I/O error of its own.
    pub fn finish(self) -> Result<Vec<u8>, TcfError> {
        let mut cursor = std::io::Cursor::new(Vec::<u8>::new());
        self.finish_into(&mut cursor)?;
        Ok(cursor.into_inner())
    }

    /// Emit the complete file into `sink`. Section 4.1,
    ///
    /// The streaming form of [`TcfWriter::finish`], and the one a producer of
    /// multi-gigabyte weights uses: only the directory — the header, the six
    /// record arrays, the string table, and the proof section, that is
    /// `[0, data_off)` — is held in memory, and its size tracks record
    /// **count**, never payload bytes. Each tensor's payload goes straight to
    /// `sink` at the absolute `data_offset` pass 1 computed and is dropped
    /// before the next tensor is touched, so no buffer scales with the file.
    ///
    /// Every tensor's payload is still resident when this starts, because
    /// [`TcfWriter::add_block_tensor`] and [`TcfWriter::add_raw_tensor`]
    /// store it. [`TcfWriter::finish_streaming`] is the path that holds one
    /// payload at a time.
    ///
    /// The three produce byte-identical files.
    ///
    /// # Contract
    ///
    /// - **No flush.** This method never calls `flush`. A caller wrapping a
    ///   `BufWriter` owns flushing it, and owns the error that surfaces
    ///   there.
    /// - **Stream position is unspecified on return.** The last write is the
    ///   directory at offset `0`, not the end of the file. A caller that
    ///   needs a particular position seeks itself.
    /// - **Payload before directory.** Pass 2 writes at offsets at or after
    ///   `data_off` while nothing below it has been written yet, so on a real
    ///   file the first writes seek past the end and leave a hole. The single
    ///   final directory write fills it. That ordering is intended: the two
    ///   header digests are only known once every payload digest is.
    /// - **With no tensors** `file_len == data_off`, and that one directory
    ///   write is the whole file.
    ///
    /// # Errors
    /// Every error the three passes raise: layout overflow
    /// ([`TcfError::SectionBounds`]), a record the reader would reject (its
    /// own decode error), a shape error from the
    /// proof vector, a payload whose length disagrees with Section 8.0.1
    /// reported as [`TcfError::InvalidQuantShape`], or
    /// [`TcfError::NonzeroReserved`] naming a caller-supplied
    /// `policy_digest`, `contract_digest`, or `relation_digest`. A tensor
    /// registered with [`TcfWriter::register_block_tensor`] or
    /// [`TcfWriter::register_raw_tensor`] and never finished with
    /// [`TcfWriter::finish_streaming`] is [`TcfError::PayloadMismatch`]. A
    /// failed seek or write on `sink` is [`TcfError::Io`], carrying that
    /// error's `ErrorKind` and message.
    pub fn finish_into<W: Write + Seek>(mut self, sink: &mut W) -> Result<(), TcfError> {
        let (layout, mut directory) = self.prepare()?;
        self.emit_payloads(&mut directory, sink, &layout)?;
        self.write_directory(&mut directory, &layout, sink)
    }

    /// Pass 1 in full: the record digests, the layout, and the directory
    /// buffer with the header, the six record arrays, and the string table
    /// already in it.
    ///
    /// The buffer is `[0, data_off)` only — a function of record count, not
    /// of payload size. Pass 3 hashes it, and it is written to the sink
    /// last. Shared by [`TcfWriter::finish_into`] and
    /// [`TcfWriter::finish_streaming`], which differ only in where pass 2
    /// gets each payload.
    pub(crate) fn prepare(&mut self) -> Result<(Layout, Vec<u8>), TcfError> {
        self.fill_record_digests()?;
        let layout = self.plan()?;
        let size = usize::try_from(layout.data_off).map_err(|_| LAYOUT_BOUNDS)?;
        let mut directory = vec![0u8; size];
        self.emit_directory(&mut directory, &layout)?;
        Ok((layout, directory))
    }

    /// Pass 3, then the single directory write at offset `0`. Section 5.3.
    ///
    /// Runs after every payload digest is in `directory`, which is why the
    /// directory is the last thing to reach the sink on both paths.
    pub(crate) fn write_directory<W: Write + Seek>(
        &self,
        directory: &mut [u8],
        layout: &Layout,
        sink: &mut W,
    ) -> Result<(), TcfError> {
        finalize(directory, layout)?;
        seek_write(sink, 0, directory)
    }
}
