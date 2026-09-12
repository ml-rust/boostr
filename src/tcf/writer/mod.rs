//! `TcfWriter`: the TCF file writer. `FORMAT.md` Section 4, Section 4.1,
//! Section 5, Section 5.3, Section 8, Section 14.4, Section 15.
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
//!
//! # What the writer owns
//!
//! A caller supplies policy — encoding, `fallback_reason`, residency,
//! telemetry — and never the layout arithmetic. The writer fills in
//! `data_offset`, `logical_payload_bytes`, `physical_span_bytes`,
//! `resident_bytes`, `transfer_bytes`, `semantic_digest`, `payload_digest`,
//! `proof_rel_off`, `proof_count`, and `proof_format` on every tensor. A
//! caller-set non-zero value in any of them is rejected by name rather than
//! silently overwritten: a producer that computed an offset itself and got
//! it wrong must hear about it.
//!
//! The three derived record digests are owned the same way: Section 9
//! requires a producer to compute `policy_digest` (Section 7),
//! `contract_digest` (Section 9), and `relation_digest` (Section 11), so
//! [`TcfWriter::finish`] computes each from the record's own encoded bytes
//! and rejects a caller-supplied non-zero value by name. `contract_digest`
//! is an integrity identity: a carried-over wrong one hides a corrupted
//! record and nothing downstream can tell.
//!
//! `logical_payload_bytes` is determined, never accepted, for a raw tensor
//! as well as a quantized one (Section 8.0.1): it is
//! `product(dims) * width` from the encoding's element size. A caller's
//! byte vector of a different length is rejected rather than stored.
//!
//! Every record the writer emits is decoded back out of the buffer it was
//! just written into. A record that the reader would reject — a bad rank, a
//! non-zero reserved byte, an undefined enum value — fails here, in the
//! producer, instead of in the consumer.
//!
//! # What the writer does not own
//!
//! Payload layout. A block payload is the GGML stream the producer's
//! quantizer emitted, stored as-is; no file in this module holds a bit
//! position, nibble index, or plane order.
//!
//! # Where each pass lives
//!
//! `records` holds the [`TcfWriter`] struct, the record `add_*` and
//! [`TcfWriter::intern`] API, and the derived record digests; `finish`
//! holds the [`TcfWriter::finish`] entry points. The tensor entry points
//! are in `tensors`. Pass 1 and pass 3 are in `layout`; pass 2 is in
//! `payload`; [`TcfWriter::finish_streaming`] is in
//! [`crate::tcf::streaming`].

mod finish;
pub(crate) mod layout;
mod payload;
mod records;
mod tensors;

pub use payload::Payload;
pub use records::TcfWriter;

pub(crate) use records::LAYOUT_BOUNDS;
