//! The [`TcfWriter::finish`] entry points: the three passes in order, over
//! an in-memory buffer or a `Write + Seek` sink. Section 4.1, Section 5.3.

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
    /// already in it. MIGRATION.md Section 4.5.1.
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
