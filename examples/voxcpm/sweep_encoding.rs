//! The candidate encodings one sensitivity sweep measures, and the
//! quantize -> dequantize round trip each one performs.
//!
//! `sensitivity.rs` owns the measurement — load once, perturb one tensor,
//! re-score, restore. This module owns the ONE question that measurement asks
//! of an encoding: given this tensor's values and shape, what values does the
//! encoding reconstruct, and how many bytes does it actually store them in.
//!
//! # Bytes come from the block table, never from bits per weight
//!
//! [`SweepEncoding::payload_bytes`] asks
//! [`QuantFormat::storage_bytes`](boostr::quant::QuantFormat::storage_bytes),
//! which charges a fixed byte count per block, scales included. It is never
//! `elements * bpw / 8`: an allocator fed that arithmetic would under-count
//! every option it considers, and by a different amount per format, which is
//! exactly the comparison it is trying to make.
//!
//! # Where the round trips come from
//!
//! Nothing here re-implements a codec. The round trip calls boostr's own
//! writers in `quant/cpu/kernels/quantize/` and its own readers in
//! `quant/cpu/kernels/dequant*`, which are the same kernels
//! [`QuantizeOps`](boostr::quant::QuantizeOps) and
//! [`DequantOps`](boostr::quant::DequantOps) dispatch to on CPU.
//!
//! The kernels are called directly rather than through the two traits for two
//! reasons. `QuantizeOps` has a CPU implementation only, so a trait call would
//! make `--device cuda` unable to measure any format at all. And a host round
//! trip keeps the perturbed values identical on every device, so a sweep's
//! records depend on the device only through the forward pass, never through
//! the codec. The values are already on the host either way: the caller
//! snapshots the tensor with `try_to_vec` and writes the result back with
//! `copy_to_device`.
//!
//! # A format without a writer is skipped, never approximated
//!
//! Every format this module accepts has a boostr writer, and
//! `tests/gguf_writer_conformance_llama_cpp.rs` gates each one on byte equality
//! against llama.cpp's own output.
//!
//! Should that stop being true — a format added to the list ahead of its
//! writer — the pair is reported per tensor as skipped, naming what is missing.
//! It is never stood in for by a nearby format or by a bit-width model of one.
//! An approximated damage number is worse than an absent one, because the
//! allocator cannot tell it apart from a measurement.

use boostr::quant::QuantFormat;
use boostr::quant::cpu::kernels::{dequant, quantize as gguf};

/// One GGUF block format a sweep can measure a tensor at.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct SweepEncoding(pub QuantFormat);

/// GGUF block formats this sweep accepts, spelled as
/// [`QuantFormat::name`] spells them.
///
/// The list is the formats an allocator plans a GGUF mix out of. `Q2_K` and
/// `Q3_K` are the rungs BELOW a `Q4_K` base, so a mix can only demote into
/// them if a sweep ranks tensors at them.
pub const GGUF_ENCODINGS: &[(&str, QuantFormat)] = &[
    ("Q2_K", QuantFormat::Q2K),
    ("Q3_K", QuantFormat::Q3K),
    ("Q4_K", QuantFormat::Q4K),
    ("Q5_K", QuantFormat::Q5K),
    ("Q6_K", QuantFormat::Q6K),
    ("Q4_0", QuantFormat::Q4_0),
    ("Q4_1", QuantFormat::Q4_1),
    ("Q8_0", QuantFormat::Q8_0),
];

/// Every accepted identifier, in one comma-joined string for the usage text
/// and for a parse error, so the two can never list different names.
pub fn accepted_names() -> String {
    let names: Vec<&str> = GGUF_ENCODINGS.iter().map(|(name, _)| *name).collect();
    names.join(", ")
}

/// Quantizes a block's worth of floats into packed GGUF bytes.
type GgufWriter = fn(&[f32], &mut [u8]);

/// Decodes packed GGUF bytes back into floats.
type GgufReader = fn(&[u8], &mut [f32]);

/// The GGUF writer for `format`, or `None` when boostr ships no writer for it.
///
/// A function pointer rather than an inline `match` at the call site so the
/// availability question and the call are one lookup: a format this returns
/// `Some` for is a format the round trip below can complete.
fn gguf_writer(format: QuantFormat) -> Option<GgufWriter> {
    match format {
        QuantFormat::Q4_0 => Some(gguf::quantize_q4_0),
        QuantFormat::Q4_1 => Some(gguf::quantize_q4_1),
        QuantFormat::Q8_0 => Some(gguf::quantize_q8_0),
        QuantFormat::Q2K => Some(gguf::quantize_q2k),
        QuantFormat::Q3K => Some(gguf::quantize_q3k),
        QuantFormat::Q4K => Some(gguf::quantize_q4k),
        QuantFormat::Q5K => Some(gguf::quantize_q5k),
        QuantFormat::Q6K => Some(gguf::quantize_q6k),
        _ => None,
    }
}

/// The GGUF reader for `format`, or `None` when boostr ships no reader for it.
fn gguf_reader(format: QuantFormat) -> Option<GgufReader> {
    match format {
        QuantFormat::Q4_0 => Some(dequant::dequant_q4_0),
        QuantFormat::Q4_1 => Some(dequant::dequant_q4_1),
        QuantFormat::Q5_0 => Some(dequant::dequant_q5_0),
        QuantFormat::Q5_1 => Some(dequant::dequant_q5_1),
        QuantFormat::Q8_0 => Some(dequant::dequant_q8_0),
        QuantFormat::Q8_1 => Some(dequant::dequant_q8_1),
        QuantFormat::Q2K => Some(dequant::dequant_q2k),
        QuantFormat::Q3K => Some(dequant::dequant_q3k),
        QuantFormat::Q4K => Some(dequant::dequant_q4k),
        QuantFormat::Q5K => Some(dequant::dequant_q5k),
        QuantFormat::Q6K => Some(dequant::dequant_q6k),
        QuantFormat::Q8K => Some(dequant::dequant_q8k),
        _ => None,
    }
}

impl SweepEncoding {
    /// Parse one identifier. The spelling is the codec's own, so a name
    /// printed by any of these examples is a name this accepts.
    pub fn parse(value: &str) -> Result<Self, String> {
        if let Some((_, format)) = GGUF_ENCODINGS.iter().find(|(name, _)| *name == value) {
            return Ok(Self(*format));
        }
        Err(format!(
            "--encoding: expected one of {}, got {value:?}",
            accepted_names()
        ))
    }

    /// Parse a comma-separated list, PRESERVING the order given.
    ///
    /// Order is preserved rather than normalized because the sweep measures
    /// encodings in this order and prints its records in it, so two runs of
    /// one command line diff cleanly. A repeated identifier is an error, not a
    /// silent de-duplication: it would otherwise cost a full forward pass per
    /// tensor to produce a record identical to one already emitted, and a
    /// consumer keying on (tensor, encoding) would see a collision.
    pub fn parse_list(value: &str) -> Result<Vec<Self>, String> {
        let mut parsed: Vec<Self> = Vec::new();
        let mut names: Vec<String> = Vec::new();
        for field in value.split(',') {
            let field = field.trim();
            if field.is_empty() {
                return Err(format!(
                    "--encoding: empty entry in {value:?}; the list is comma separated, e.g. \
                     Q4_K,Q6_K"
                ));
            }
            let encoding = Self::parse(field)?;
            let name = encoding.name();
            if names.contains(&name) {
                return Err(format!("--encoding: {name} is listed more than once"));
            }
            names.push(name);
            parsed.push(encoding);
        }
        if parsed.is_empty() {
            return Err("--encoding: needs at least one encoding".to_string());
        }
        Ok(parsed)
    }

    /// The codec's own name for this encoding.
    pub fn name(self) -> String {
        self.0.name().to_string()
    }

    /// Nominal bits per stored weight.
    ///
    /// Reported for context and used for the `bytes_saved` field the existing
    /// consumers read. It is NOT what [`Self::payload_bytes`] returns and must
    /// not be used to derive a byte count — see the module docs.
    pub fn bits_per_weight(self) -> f64 {
        (self.0.block_bytes() as f64) * 8.0 / (self.0.block_size() as f64)
    }

    /// Exact packed bytes a tensor of `shape` occupies under this encoding,
    /// straight from the block table.
    pub fn payload_bytes(self, shape: &[usize]) -> Result<usize, String> {
        self.0
            .storage_bytes(shape.iter().product())
            .map_err(|e| e.to_string())
    }

    /// Why this encoding cannot hold a tensor of `shape`, or `None` when it
    /// can.
    ///
    /// Asked of the codec's own unit arithmetic rather than re-derived here,
    /// so the answer can never disagree with what the round trip would do.
    /// K-quants need a row that is a whole number of 256-element
    /// super-blocks; the simple formats need 32.
    pub fn shape_error(self, shape: &[usize]) -> Option<String> {
        if shape.is_empty() {
            return Some(format!("{}: tensor shape is empty", self.name()));
        }
        let last = shape.last().copied().unwrap_or(0);
        if last.is_multiple_of(self.0.block_size()) {
            None
        } else {
            Some(format!(
                "{} cannot hold this shape: row of {last} is not a whole number of \
                 {}-element blocks",
                self.name(),
                self.0.block_size()
            ))
        }
    }

    /// Why this encoding has no round trip in this build, or `None` when it
    /// has one. Independent of any tensor: it is a property of the codec.
    pub fn round_trip_error(self) -> Option<String> {
        if gguf_writer(self.0).is_none() {
            return Some(format!(
                "{}: boostr has no quantize writer for this format, so it cannot be \
                 round-tripped. Measuring it against a nearby format's writer would \
                 report a number for a codec that was never run.",
                self.name()
            ));
        }
        if gguf_reader(self.0).is_none() {
            return Some(format!(
                "{}: boostr has no dequantize reader for this format",
                self.name()
            ));
        }
        None
    }

    /// Quantize `values` at this encoding and dequantize straight back,
    /// returning the reconstructed values in the same order.
    ///
    /// `values` is the tensor's own contents in memory order and `shape` its
    /// logical shape. The output always has `values.len()` entries; a codec
    /// that returns a different count is an error, never a silent truncation.
    pub fn round_trip(self, values: &[f32], shape: &[usize]) -> Result<Vec<f32>, String> {
        if let Some(reason) = self.round_trip_error() {
            return Err(reason);
        }
        // The byte buffer below is sized from `shape`, so a `shape` that does
        // not describe `values` would hand a kernel a buffer of the wrong
        // length. Checked rather than assumed: the caller derives both from
        // one tensor, and this is what says so out loud.
        let numel: usize = shape.iter().product();
        if numel != values.len() {
            return Err(format!(
                "{}: shape {shape:?} describes {numel} element(s) but {} value(s) were given",
                self.name(),
                values.len()
            ));
        }
        let write =
            gguf_writer(self.0).ok_or_else(|| format!("{}: no quantize writer", self.name()))?;
        let read =
            gguf_reader(self.0).ok_or_else(|| format!("{}: no dequantize reader", self.name()))?;
        let mut blocks = vec![0u8; self.payload_bytes(shape)?];
        write(values, &mut blocks);
        let mut reconstructed = vec![0f32; values.len()];
        read(&blocks, &mut reconstructed);
        if reconstructed.len() != values.len() {
            return Err(format!(
                "{}: round trip produced {} value(s) for {} element(s)",
                self.name(),
                reconstructed.len(),
                values.len()
            ));
        }
        Ok(reconstructed)
    }
}
