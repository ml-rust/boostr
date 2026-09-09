//! `QuantizeOps::quantize` against llama.cpp's OWN quantizer,
//! `ggml_quantize_chunk`.
//!
//! `tests/gguf_conformance_llama_cpp.rs` gates only the READ direction:
//! `DequantOps::dequantize` against llama.cpp's dequantizer. A writer checked
//! against the reader beside it can agree with itself while both disagree with
//! the format, because the reader accepts whatever layout the writer emits. This
//! file gates the WRITE direction against an external quantizer.
//!
//! # Two bars, one per group of formats
//!
//! **Q4_1, Q4_K, Q5_K, Q6_K — byte equality.** boostr's writer reproduces
//! llama.cpp's own fit for these: the plain min/max fit for Q4_1, the iterative
//! per-sub-block scale search for the three K-quants. Matching bytes is the
//! intended property, not a coincidence of rounding. A future change that lands
//! close but not exact is a regression worth seeing, never something to absorb
//! into an epsilon. llama.cpp produced the expectation: fix the writer, never
//! the fixture.
//!
//! **Q4_0 and Q8_0 — deliberately NOT byte equality.** llama.cpp's
//! `quantize_row_q4_0` and `quantize_row_q8_0` are plain absmax fits with no
//! search. boostr sweeps the block scale, refits it by least squares, and
//! derives every code against the binary16 value the reader loads rather than
//! the wider float it came from. Both changes move codes, so the bytes cannot
//! match llama.cpp. The owner's decision is that accuracy wins over bit-identity
//! with llama.cpp. The two tests here gate the properties that decision
//! requires:
//!
//! - the output is a structurally valid block — byte count, field layout, and
//!   for Q8_0 no code on -128, which is outside the format's `[-127, 127]`,
//! - boostr's reader decodes it back to the source within the format's own
//!   per-element step, so the divergence is a better encoding and not a broken
//!   one,
//! - boostr's reconstruction error is no worse than llama.cpp's on the same
//!   input, and the bytes are not llama.cpp's absmax bytes.
//!
//! `src/quant/cpu/kernels/quantize/tests.rs` carries the same comparison one
//! level down, against the `#[cfg(test)]` `quantize_q4_0_absmax` /
//! `quantize_q8_0_absmax` baselines. Those are crate-private and out of reach
//! from an integration test. Here llama.cpp's own bytes stand in for that
//! baseline, because llama.cpp IS the absmax fit.
//!
//! `writer_q4_0_llama.bin` and `writer_q8_0_llama.bin` are no longer
//! expectations. They are the recorded absmax output the search has to beat, and
//! both tests read them. Do not delete them.
//!
//! boostr writes exactly six formats: `QuantFormat::Q4_0`, `Q4_1`, `Q8_0`,
//! `Q4K`, `Q5K`, `Q6K`. Every other `QuantFormat` variant returns
//! `Error::UnsupportedQuantFormat` from `quantize`, so nothing else needs a gate
//! here.
//!
//! boostr has no CUDA or WGPU quantizer. This file is CPU only.
//!
//! `compressr` has no GGUF block-layout code of its own: its `quantize_gguf`
//! calls straight into `boostr::QuantizeOps`. This gate covers compressr's
//! writer too.
//!
//! # Regenerating or extending the fixtures
//!
//! Requires ggml's C library on the system. Call `ggml_quantize_chunk`
//! directly and link against the installed library — no ggml headers needed:
//!
//! ```c
//! size_t ggml_quantize_chunk(int type, const float *src, void *dst,
//!                            int64_t start, int64_t nrows, int64_t n_per_row,
//!                            const float *imatrix);
//! // gcc gen.c -o gen -lggml-base -lggml-cpu
//! ```
//!
//! ggml type ids used here: Q4_0 2, Q4_1 3, Q8_0 8, Q4_K 12, Q5_K 13, Q6_K 14.
//! Call with `nrows = 8`, `n_per_row = 256`, `imatrix = NULL`.
//!
//! Scale each source row by a different factor. A single scale across all
//! rows lets a per-row stride error produce matching bytes by accident.
//!
//! Run with:
//!   cd boostr && cargo test --test gguf_writer_conformance_llama_cpp

use std::path::PathBuf;

use boostr::quant::{DequantOps, QuantFormat, QuantTensor, QuantizeOps};
use half::f16;
use numr::dtype::DType;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

/// The fixture tensor: 8 rows of 256. Q4_0 and Q8_0 pack 32 elements per block,
/// so their output holds 64 blocks. The K-quants use 256-element superblocks and
/// are gated on bytes, so these constants do not apply to them.
const ELEMENTS: usize = 2048;
const BLOCK_SIZE: usize = 32;
const BLOCKS: usize = ELEMENTS / BLOCK_SIZE;

/// Resolves a fixture relative to the crate, never to an absolute path from
/// whichever machine generated it.
fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/gguf_writer")
        .join(name)
}

/// The source floats: little-endian `f32`, one per element.
fn floats(name: &str) -> Vec<f32> {
    let bytes = std::fs::read(fixture(name)).unwrap();
    assert_eq!(bytes.len() % 4, 0, "{name}: source is not whole f32s");
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn cpu_setup() -> (CpuClient, CpuDevice) {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    (client, device)
}

/// Asserts `got` equals llama.cpp's `ggml_quantize_chunk` output EXACTLY. Always
/// prints one flat `GGUF_WRITER_DIAG` line, pass or fail, so a green run still
/// records what each format produced.
///
/// On mismatch it reports up to 16 differing byte offsets. Each entry names the
/// byte offset, the block index (`offset / block_bytes`) and the offset within
/// that block (`offset % block_bytes`). The within-block offset identifies which
/// field is wrong. A differing scale field means the scale search diverged. A
/// differing payload byte means the packing diverged. llama.cpp produced the
/// expectation, so the fix is always the writer, never the fixture.
///
/// Only the byte-matching formats call this: Q4_1, Q4_K, Q5_K, Q6_K. Q4_0 and
/// Q8_0 diverge on purpose and are gated by the two tests below.
fn assert_writer_matches_llama_cpp(format: &str, block_bytes: usize, got: &[u8], llama: &[u8]) {
    assert_eq!(
        got.len(),
        llama.len(),
        "{format}: boostr wrote {} bytes, llama.cpp wrote {} bytes",
        got.len(),
        llama.len()
    );

    let mut differing = 0usize;
    let mut report = String::new();
    for (offset, (a, b)) in got.iter().zip(llama.iter()).enumerate() {
        if a != b {
            differing += 1;
            if differing <= 16 {
                report.push_str(&format!(
                    "\n  offset {offset:6}  block {:4}  in-block {:4}  boostr 0x{a:02x}  llama.cpp 0x{b:02x}",
                    offset / block_bytes,
                    offset % block_bytes
                ));
            }
        }
    }

    println!(
        "GGUF_WRITER_DIAG format={format} bytes={} differing={differing}",
        got.len()
    );

    assert_eq!(
        differing,
        0,
        "{format}: boostr's writer disagrees with llama.cpp's ggml_quantize_chunk on \
         {differing} of {} bytes. First differing offsets:{report}\n\
         llama.cpp produced the expectation. A differing scale field means the scale \
         search diverged; a differing payload byte means the packing diverged. Fix the \
         writer, never the fixture.",
        got.len()
    );
}

#[test]
fn q4_1_writer_matches_llama_cpp() {
    let src = floats("writer_src.bin");
    let llama = std::fs::read(fixture("writer_q4_1_llama.bin")).unwrap();
    let (client, device) = cpu_setup();
    let input = Tensor::<CpuRuntime>::from_slice(&src, &[8, 256], &device).unwrap();
    let got = client
        .quantize(&input, QuantFormat::Q4_1)
        .unwrap()
        .to_bytes()
        .unwrap();
    assert_writer_matches_llama_cpp("Q4_1", 20, &got, &llama);
}

#[test]
fn q4_k_writer_matches_llama_cpp() {
    let src = floats("writer_src.bin");
    let llama = std::fs::read(fixture("writer_q4_k_llama.bin")).unwrap();
    let (client, device) = cpu_setup();
    let input = Tensor::<CpuRuntime>::from_slice(&src, &[8, 256], &device).unwrap();
    let got = client
        .quantize(&input, QuantFormat::Q4K)
        .unwrap()
        .to_bytes()
        .unwrap();
    assert_writer_matches_llama_cpp("Q4K", 144, &got, &llama);
}

#[test]
fn q5_k_writer_matches_llama_cpp() {
    let src = floats("writer_src.bin");
    let llama = std::fs::read(fixture("writer_q5_k_llama.bin")).unwrap();
    let (client, device) = cpu_setup();
    let input = Tensor::<CpuRuntime>::from_slice(&src, &[8, 256], &device).unwrap();
    let got = client
        .quantize(&input, QuantFormat::Q5K)
        .unwrap()
        .to_bytes()
        .unwrap();
    assert_writer_matches_llama_cpp("Q5K", 176, &got, &llama);
}

#[test]
fn q6_k_writer_matches_llama_cpp() {
    let src = floats("writer_src.bin");
    let llama = std::fs::read(fixture("writer_q6_k_llama.bin")).unwrap();
    let (client, device) = cpu_setup();
    let input = Tensor::<CpuRuntime>::from_slice(&src, &[8, 256], &device).unwrap();
    let got = client
        .quantize(&input, QuantFormat::Q6K)
        .unwrap()
        .to_bytes()
        .unwrap();
    assert_writer_matches_llama_cpp("Q6K", 210, &got, &llama);
}

/// Quantizes the fixture source through the public writer, as `compressr` does.
fn write_blocks(format: QuantFormat) -> Vec<u8> {
    let src = floats("writer_src.bin");
    let (client, device) = cpu_setup();
    let input = Tensor::<CpuRuntime>::from_slice(&src, &[8, 256], &device).unwrap();
    client.quantize(&input, format).unwrap().to_bytes().unwrap()
}

/// Decodes a quantized payload through boostr's reader, the way a loader does.
fn decode(format: QuantFormat, bytes: &[u8]) -> Vec<f32> {
    let (client, device) = cpu_setup();
    let qt = QuantTensor::<CpuRuntime>::from_bytes(bytes, format, &[ELEMENTS], &device).unwrap();
    client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>()
}

/// `Σ(x − x̂)²` in f64: 2048 f32 terms of mixed magnitude cancel in f32.
fn squared_error(src: &[f32], decoded: &[f32]) -> f64 {
    assert_eq!(src.len(), decoded.len());
    src.iter()
        .zip(decoded)
        .map(|(&a, &b)| {
            let d = f64::from(a) - f64::from(b);
            d * d
        })
        .sum()
}

/// The stored binary16 block scale, read back from the first two bytes of a
/// block — the same value the reader loads, never the f32 the writer fitted.
fn stored_scale(block: &[u8]) -> f32 {
    f16::from_le_bytes([block[0], block[1]]).to_f32()
}

/// Index of the largest-magnitude element of a block.
fn argmax_abs(xb: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, v) in xb.iter().enumerate() {
        if v.abs() > xb[best].abs() {
            best = i;
        }
    }
    best
}

/// Asserts the block whose largest-magnitude SOURCE element sits at `arg` also
/// carries its largest-magnitude CODE there.
///
/// This is the index-order gate. Q4_0 splits a block across nibbles — element
/// `j` low, element `j + 16` high. Unpacking `out[2i]`/`out[2i + 1]` permutes
/// every weight while leaving the byte count, block count and tensor RMS intact.
/// A permutation moves the peak code off the peak element, which this catches.
/// Clamping can put several codes on the peak magnitude, so the assertion is
/// equality with the maximum, not uniqueness.
fn assert_peak_code_tracks_peak_element(
    format: &str,
    b: usize,
    xb: &[f32],
    codes: &[i32],
    min_peak: i32,
) {
    let arg = argmax_abs(xb);
    let peak = codes.iter().map(|c| c.abs()).max().unwrap();
    assert_eq!(
        codes[arg].abs(),
        peak,
        "{format}: block {b} carries its largest code {peak} somewhere other than \
         element {arg}, which is the block's largest-magnitude source value \
         ({}). The scale search never moves the peak element off the peak code; a \
         permuted unpack does.",
        xb[arg]
    );

    // A block of pure zeros is the one case with nothing to encode: every code
    // is zero and the format represents nothing else.
    if xb[arg] == 0.0 {
        return;
    }
    assert!(
        peak >= min_peak,
        "{format}: block {b} reaches only code {peak}, below {min_peak}. The sweep \
         spans a fraction of one level either side of the absmax fit, so the peak \
         element always lands near the end of the range. A peak this low means the \
         scale came out far too large and most of the code range is unused."
    );
}

/// Asserts every decoded element lands within `steps` block scales of its
/// source.
///
/// Three effects add up to the bound:
///
/// - the writer rounds each code to nearest against the STORED scale, so a
///   half-step is the floor,
/// - the sweep reaches scales smaller than the absmax fit, which clamps the
///   extreme element to the end of the code range and costs a further step,
/// - the scale itself rounds to binary16.
///
/// Two steps covers all three. It still fails by orders of magnitude on a wrong
/// field offset or a permuted block, where the miss is the block's whole dynamic
/// range rather than a rounding step.
fn assert_decodes_within_steps(
    format: &str,
    src: &[f32],
    decoded: &[f32],
    blocks: &[u8],
    block_bytes: usize,
    steps: f32,
) {
    let mut worst = 0.0f64;
    for (b, block) in blocks.chunks_exact(block_bytes).enumerate() {
        let d = stored_scale(block).abs();
        let tol = f64::from(steps * d) + 1e-9;
        let base = b * BLOCK_SIZE;
        let pairs = src[base..][..BLOCK_SIZE]
            .iter()
            .zip(&decoded[base..][..BLOCK_SIZE]);
        for (k, (&want, &got)) in pairs.enumerate() {
            let i = base + k;
            let err = (f64::from(want) - f64::from(got)).abs();
            worst = worst.max(err / tol);
            assert!(
                err <= tol,
                "{format}: element {i} of block {b} decodes to {got} from source \
                 {want}, an error of {err:.6e} against a bound of {tol:.6e} ({steps} \
                 block scales of {d:.6e}). A miss this size is a layout or index error, \
                 not rounding."
            );
        }
    }
    println!("GGUF_WRITER_DIAG format={format} worst_step_ratio={worst:.4}");
}

/// Asserts boostr reconstructs the source at least as well as llama.cpp's absmax
/// output in `llama`, and that the two are not the same bytes.
///
/// The inequality alone passes on a silent revert to absmax, because the revert
/// reproduces llama.cpp exactly and ties. The byte inequality closes that gap.
/// The search picks a different scale on at least one block of this fixture, so
/// identical bytes mean the search stopped running.
fn assert_beats_absmax(format: &str, fmt: QuantFormat, got: &[u8], llama: &[u8]) {
    let src = floats("writer_src.bin");
    assert_eq!(
        got.len(),
        llama.len(),
        "{format}: boostr wrote {} bytes, llama.cpp's absmax wrote {}",
        got.len(),
        llama.len()
    );

    let ours = squared_error(&src, &decode(fmt, got));
    let theirs = squared_error(&src, &decode(fmt, llama));
    println!("GGUF_WRITER_DIAG format={format} sse_boostr={ours:.9e} sse_absmax={theirs:.9e}");

    assert!(
        ours <= theirs,
        "{format}: boostr's squared reconstruction error {ours:.9e} is WORSE than \
         llama.cpp's plain absmax fit {theirs:.9e} on the same input. The scale search \
         exists to be no worse than absmax on every block, so a regression here is in \
         the search, never in the fixture."
    );

    assert!(
        got != llama,
        "{format}: boostr's bytes are identical to llama.cpp's absmax output. The block \
         scale search picks a different scale on at least one block of this fixture, so \
         identical bytes mean the writer reverted to a plain absmax fit."
    );
}

/// Q4_0 diverges from llama.cpp on purpose. Gates structure, decode and error
/// instead of bytes. See the module docs.
#[test]
fn q4_0_writer_is_valid_and_no_worse_than_absmax() {
    const BLOCK_BYTES: usize = 18;
    let src = floats("writer_src.bin");
    let got = write_blocks(QuantFormat::Q4_0);

    assert_eq!(
        got.len(),
        BLOCKS * BLOCK_BYTES,
        "Q4_0: {BLOCKS} blocks must occupy {} bytes",
        BLOCKS * BLOCK_BYTES
    );

    for (b, block) in got.as_chunks::<BLOCK_BYTES>().0.iter().enumerate() {
        // Bytes 0..2 are the f16 scale, 2..18 the nibble pairs: element `j` in
        // the low nibble, element `j + 16` in the high one.
        let mut codes = [0i32; BLOCK_SIZE];
        for (j, &byte) in block[2..].iter().enumerate() {
            codes[j] = i32::from(byte & 0x0F) - 8;
            codes[j + 16] = i32::from(byte >> 4) - 8;
        }
        assert!(
            stored_scale(block).is_finite(),
            "Q4_0: block {b} stores a non-finite scale, which no reader can use"
        );
        assert_peak_code_tracks_peak_element(
            "Q4_0",
            b,
            &src[b * BLOCK_SIZE..][..BLOCK_SIZE],
            &codes,
            6,
        );
    }

    let decoded = decode(QuantFormat::Q4_0, &got);
    assert_decodes_within_steps("Q4_0", &src, &decoded, &got, BLOCK_BYTES, 2.0);

    let llama = std::fs::read(fixture("writer_q4_0_llama.bin")).unwrap();
    assert_beats_absmax("Q4_0", QuantFormat::Q4_0, &got, &llama);
}

/// Q8_0 diverges from llama.cpp on purpose. Gates structure, decode and error
/// instead of bytes. See the module docs.
#[test]
fn q8_0_writer_is_valid_and_no_worse_than_absmax() {
    const BLOCK_BYTES: usize = 34;
    let src = floats("writer_src.bin");
    let got = write_blocks(QuantFormat::Q8_0);

    assert_eq!(
        got.len(),
        BLOCKS * BLOCK_BYTES,
        "Q8_0: {BLOCKS} blocks must occupy {} bytes",
        BLOCKS * BLOCK_BYTES
    );

    for (b, block) in got.as_chunks::<BLOCK_BYTES>().0.iter().enumerate() {
        // Bytes 0..2 are the f16 scale, 2..34 the 32 signed codes, element `i`
        // at byte `2 + i` — no split-half ordering here.
        let mut codes = [0i32; BLOCK_SIZE];
        for (i, &byte) in block[2..].iter().enumerate() {
            let c = i32::from(byte as i8);
            assert_ne!(
                c, -128,
                "Q8_0: block {b} element {i} stores -128. The format's range is \
                 [-127, 127] — the low end is dropped to keep the block symmetric, and \
                 -128 has no positive counterpart."
            );
            codes[i] = c;
        }
        assert!(
            stored_scale(block).is_finite(),
            "Q8_0: block {b} stores a non-finite scale, which no reader can use"
        );
        assert_peak_code_tracks_peak_element(
            "Q8_0",
            b,
            &src[b * BLOCK_SIZE..][..BLOCK_SIZE],
            &codes,
            100,
        );
    }

    let decoded = decode(QuantFormat::Q8_0, &got);
    assert_decodes_within_steps("Q8_0", &src, &decoded, &got, BLOCK_BYTES, 2.0);

    let llama = std::fs::read(fixture("writer_q8_0_llama.bin")).unwrap();
    assert_beats_absmax("Q8_0", QuantFormat::Q8_0, &got, &llama);
}
