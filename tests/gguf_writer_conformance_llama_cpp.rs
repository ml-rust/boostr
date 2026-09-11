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
//! **Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL — byte equality.** boostr's writer
//! reproduces llama.cpp's own iterative per-sub-block scale search for these
//! six. Matching bytes is the intended property, not a coincidence of
//! rounding. A future change that lands close but not exact is a regression
//! worth seeing, never something to absorb into an epsilon. llama.cpp
//! produced the expectation: fix the writer, never the fixture.
//!
//! The SAME bar applies to the importance-weighted (imatrix) writers, which
//! reproduce llama.cpp's `quantize_row_*_K_impl` / `quantize_row_iq4_nl_impl`
//! — the path each quantizer takes whenever an importance matrix is supplied,
//! and the path nearly every GGUF quant the ecosystem ships comes out of. A
//! round-trip or divergence test beside these proves only that boostr's
//! writer and reader agree with each other, never that either agrees with
//! llama.cpp.
//!
//! Every fixture this file reads is committed. A missing one fails the test
//! loudly, naming the file and the recipe below — see `llama_cpp_bytes`.
//!
//! **Q4_0, Q4_1 and Q8_0 — deliberately NOT byte equality.** llama.cpp's
//! `quantize_row_q4_0`, `quantize_row_q4_1` and `quantize_row_q8_0` are plain
//! direct fits with no search. boostr sweeps the stored fields, refits them by
//! least squares, and derives every code against the binary16 values the reader
//! loads rather than the wider floats they came from. Both changes move codes,
//! so the bytes cannot match llama.cpp. The owner's decision is that accuracy
//! wins over bit-identity with llama.cpp. The three tests here gate the
//! properties that decision requires:
//!
//! - the output is a structurally valid block — byte count, field layout,
//!   finite stored fields, for Q8_0 no code on -128 which is outside the
//!   format's `[-127, 127]`, and for every one of them the extreme code still
//!   on the extreme source element,
//! - boostr's reader decodes it back to the source within the format's own
//!   per-element step, so the divergence is a better encoding and not a broken
//!   one,
//! - boostr's reconstruction error is no worse than llama.cpp's on the same
//!   input, and the bytes are not llama.cpp's direct-fit bytes.
//!
//! `src/quant/cpu/kernels/quantize/tests.rs` carries the same comparison one
//! level down, against the `#[cfg(test)]` `quantize_q4_0_absmax` /
//! `quantize_q4_1_minmax` / `quantize_q8_0_absmax` baselines. Those are
//! crate-private and out of reach from an integration test. Here llama.cpp's own
//! bytes stand in for those baselines, because llama.cpp IS the direct fit.
//!
//! `writer_q4_0_llama.bin`, `writer_q4_1_llama.bin` and `writer_q8_0_llama.bin`
//! are no longer expectations. They are the recorded direct-fit output the
//! searches have to beat, and all three tests read them. Do not delete them.
//!
//! boostr writes exactly nine formats: `QuantFormat::Q4_0`, `Q4_1`, `Q8_0`,
//! `Q2K`, `Q3K`, `Q4K`, `Q5K`, `Q6K`, `IQ4NL`. Every other `QuantFormat`
//! variant returns `Error::UnsupportedQuantFormat` from `quantize`, so
//! nothing else needs a gate here.
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
//! ggml type ids used here: Q4_0 2, Q4_1 3, Q8_0 8, Q2_K 10, Q3_K 11, Q4_K 12,
//! Q5_K 13, Q6_K 14, IQ4_NL 20. Call with `nrows = 8`, `n_per_row = 256`,
//! `imatrix = NULL`.
//!
//! Scale each source row by a different factor. A single scale across all
//! rows lets a per-row stride error produce matching bytes by accident.
//!
//! The six imatrix fixtures — `writer_q2_k_imatrix_llama.bin`,
//! `writer_q3_k_imatrix_llama.bin`, `writer_q4_k_imatrix_llama.bin`,
//! `writer_q5_k_imatrix_llama.bin`, `writer_q6_k_imatrix_llama.bin` and
//! `writer_iq4_nl_imatrix_llama.bin` — come from the same call with the LAST
//! argument non-NULL: a pointer to 256 `float` importance values, one per
//! column, identical for every row. Those 256 values are
//! `tests/fixtures/gguf_writer/writer_imatrix.bin`, little-endian `f32`, and
//! both the generator and this file must read that one file — an importance
//! vector regenerated from a formula on either side stops being the same
//! vector. Generate it once with a spread of magnitudes and at least one exact
//! zero, since a zero importance is legal and takes its own branch.
//!
//! ```c
//! // imatrix: 256 floats read from writer_imatrix.bin, NOT rebuilt here
//! ggml_quantize_chunk(type, src, dst, 0, 8, 256, imatrix);
//! ```
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
/// Only the byte-matching formats call this: Q4_K, Q5_K, Q6_K. Q4_0, Q4_1 and
/// Q8_0 diverge on purpose and are gated by the three tests below.
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

/// Reads a byte-equality fixture. Panics naming the missing file and the
/// generation recipe if it is absent.
///
/// Every fixture this file reads is committed. A missing file here means one
/// was deleted or corrupted, not that it was never generated, and that must
/// fail the run: the round-trip tests decode with boostr's own reader, which
/// agrees with whatever the writer emits, so silently passing without this
/// fixture would leave the layout UNVERIFIED behind a green run.
fn llama_cpp_bytes(name: &str, format: &str) -> Vec<u8> {
    std::fs::read(fixture(name)).unwrap_or_else(|err| {
        panic!(
            "{format}: fixture tests/fixtures/gguf_writer/{name} is missing or unreadable \
             ({err}). It is committed and must not be absent. Regenerate it with the recipe \
             under \"Regenerating or extending the fixtures\" in this file's module docs."
        )
    })
}

#[test]
fn q2_k_writer_matches_llama_cpp() {
    let llama = llama_cpp_bytes("writer_q2_k_llama.bin", "Q2K");
    let src = floats("writer_src.bin");
    let (client, device) = cpu_setup();
    let input = Tensor::<CpuRuntime>::from_slice(&src, &[8, 256], &device).unwrap();
    let got = client
        .quantize(&input, QuantFormat::Q2K)
        .unwrap()
        .to_bytes()
        .unwrap();
    assert_writer_matches_llama_cpp("Q2K", 84, &got, &llama);
}

#[test]
fn q3_k_writer_matches_llama_cpp() {
    let llama = llama_cpp_bytes("writer_q3_k_llama.bin", "Q3K");
    let src = floats("writer_src.bin");
    let (client, device) = cpu_setup();
    let input = Tensor::<CpuRuntime>::from_slice(&src, &[8, 256], &device).unwrap();
    let got = client
        .quantize(&input, QuantFormat::Q3K)
        .unwrap()
        .to_bytes()
        .unwrap();
    assert_writer_matches_llama_cpp("Q3K", 110, &got, &llama);
}

/// Structural gate that runs whether or not the byte fixture is present.
///
/// It is deliberately weaker than byte equality and does not replace it: an
/// encoder with a permuted interleave still writes the right byte count and
/// still decodes through boostr's matching reader. What it does catch is the
/// class of bug that survives a round trip — a field written outside its own
/// span, or a super-block whose stored factors are not values a reader can
/// multiply.
#[test]
fn q2_k_and_q3_k_write_structurally_valid_blocks() {
    // Q2_K: scales@0..16, qs@16..80, d@80..82, dmin@82..84.
    let got = write_blocks(QuantFormat::Q2K);
    assert_eq!(
        got.len(),
        8 * 84,
        "Q2K: 8 super-blocks must occupy 672 bytes"
    );
    for (b, block) in got.as_chunks::<84>().0.iter().enumerate() {
        let d = f16::from_le_bytes([block[80], block[81]]).to_f32();
        let dmin = f16::from_le_bytes([block[82], block[83]]).to_f32();
        assert!(
            d.is_finite() && dmin.is_finite(),
            "Q2K: block {b} stores a non-finite factor, which no reader can use"
        );
        assert!(
            d >= 0.0 && dmin >= 0.0,
            "Q2K: block {b} stores a negative factor ({d}, {dmin}). Both are \
             fractions of a non-negative maximum, and the reader multiplies them \
             by unsigned nibbles."
        );
    }
    assert!(decode(QuantFormat::Q2K, &got).iter().all(|v| v.is_finite()));

    // Q3_K: hmask@0..32, qs@32..96, scales@96..108, d@108..110.
    let got = write_blocks(QuantFormat::Q3K);
    assert_eq!(
        got.len(),
        8 * 110,
        "Q3K: 8 super-blocks must occupy 880 bytes"
    );
    for (b, block) in got.as_chunks::<110>().0.iter().enumerate() {
        let d = f16::from_le_bytes([block[108], block[109]]).to_f32();
        assert!(
            d.is_finite(),
            "Q3K: block {b} stores a non-finite scale, which no reader can use"
        );
    }
    assert!(decode(QuantFormat::Q3K, &got).iter().all(|v| v.is_finite()));
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

#[test]
fn iq4_nl_writer_matches_llama_cpp() {
    let src = floats("writer_src.bin");
    let llama = std::fs::read(fixture("writer_iq4_nl_llama.bin")).unwrap();
    let (client, device) = cpu_setup();
    let input = Tensor::<CpuRuntime>::from_slice(&src, &[8, 256], &device).unwrap();
    let got = client
        .quantize(&input, QuantFormat::IQ4NL)
        .unwrap()
        .to_bytes()
        .unwrap();
    assert_writer_matches_llama_cpp("IQ4_NL", 18, &got, &llama);
}

/// IQ4_XS shares IQ4_NL's codebook on super-block geometry: 256 elements, a
/// 6-bit scale per 32 under one f16 super-scale, 136 bytes.
#[test]
fn iq4_xs_writer_matches_llama_cpp() {
    let src = floats("writer_src.bin");
    let llama = std::fs::read(fixture("writer_iq4_xs_llama.bin")).unwrap();
    let (client, device) = cpu_setup();
    let input = Tensor::<CpuRuntime>::from_slice(&src, &[8, 256], &device).unwrap();
    let got = client
        .quantize(&input, QuantFormat::IQ4XS)
        .unwrap()
        .to_bytes()
        .unwrap();
    assert_writer_matches_llama_cpp("IQ4_XS", 136, &got, &llama);
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

/// Asserts boostr reconstructs the source at least as well as llama.cpp's
/// direct-fit output in `llama`, and that the two are not the same bytes.
///
/// The inequality alone passes on a silent revert to the direct fit, because the
/// revert reproduces llama.cpp exactly and ties. The byte inequality closes that
/// gap. The search picks different stored fields on at least one block of this
/// fixture, so identical bytes mean the search stopped running.
fn assert_beats_direct_fit(format: &str, fmt: QuantFormat, got: &[u8], llama: &[u8]) {
    let src = floats("writer_src.bin");
    assert_eq!(
        got.len(),
        llama.len(),
        "{format}: boostr wrote {} bytes, llama.cpp's direct fit wrote {}",
        got.len(),
        llama.len()
    );

    let ours = squared_error(&src, &decode(fmt, got));
    let theirs = squared_error(&src, &decode(fmt, llama));
    println!("GGUF_WRITER_DIAG format={format} sse_boostr={ours:.9e} sse_direct={theirs:.9e}");

    assert!(
        ours <= theirs,
        "{format}: boostr's squared reconstruction error {ours:.9e} is WORSE than \
         llama.cpp's plain direct fit {theirs:.9e} on the same input. The search exists \
         to be no worse than the direct fit on every block, so a regression here is in \
         the search, never in the fixture."
    );

    assert!(
        got != llama,
        "{format}: boostr's bytes are identical to llama.cpp's direct-fit output. The \
         block search picks different stored fields on at least one block of this \
         fixture, so identical bytes mean the writer reverted to a plain direct fit."
    );
}

/// Q4_0 diverges from llama.cpp on purpose. Gates structure, decode and error
/// instead of bytes. See the module docs.
#[test]
fn q4_0_writer_is_valid_and_no_worse_than_direct_fit() {
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
    assert_beats_direct_fit("Q4_0", QuantFormat::Q4_0, &got, &llama);
}

/// Q8_0 diverges from llama.cpp on purpose. Gates structure, decode and error
/// instead of bytes. See the module docs.
#[test]
fn q8_0_writer_is_valid_and_no_worse_than_direct_fit() {
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
    assert_beats_direct_fit("Q8_0", QuantFormat::Q8_0, &got, &llama);
}

/// Asserts the block's largest SOURCE value carries its largest code and the
/// smallest carries its smallest, then that the two codes span at least
/// `min_span` levels.
///
/// This is Q4_1's index-order gate. Q4_1 splits a block across nibbles — element
/// `j` low, element `j + 16` high. Unpacking `out[2i]`/`out[2i + 1]` permutes
/// every weight while leaving the byte count, block count and tensor RMS intact.
/// A permutation moves the extreme codes off the extreme elements, which this
/// catches. Q4_1's codes are unsigned and one-sided, so both ends are checked;
/// the symmetric formats check magnitude at one end instead. Clamping can put
/// several codes on an extreme, so the assertion is equality with the extreme,
/// not uniqueness.
///
/// The span gate catches the other failure: a scale far too large leaves most of
/// the code range unused. The sweep moves the range by a fraction of one level,
/// so a block with any spread still reaches most of `[0, 15]`.
fn assert_extreme_codes_track_extreme_elements(
    format: &str,
    b: usize,
    xb: &[f32],
    codes: &[i32],
    min_span: i32,
) {
    let mut arg_hi = 0usize;
    let mut arg_lo = 0usize;
    for (i, v) in xb.iter().enumerate() {
        if *v > xb[arg_hi] {
            arg_hi = i;
        }
        if *v < xb[arg_lo] {
            arg_lo = i;
        }
    }
    let hi = *codes.iter().max().unwrap();
    let lo = *codes.iter().min().unwrap();

    assert_eq!(
        codes[arg_hi], hi,
        "{format}: block {b} carries its highest code {hi} somewhere other than \
         element {arg_hi}, which is the block's largest source value ({}). The \
         search never moves an extreme element off its extreme code; a permuted \
         unpack does.",
        xb[arg_hi]
    );
    assert_eq!(
        codes[arg_lo], lo,
        "{format}: block {b} carries its lowest code {lo} somewhere other than \
         element {arg_lo}, which is the block's smallest source value ({}).",
        xb[arg_lo]
    );

    // A constant block has nothing to spread: one code represents it exactly.
    if xb[arg_hi] == xb[arg_lo] {
        return;
    }
    assert!(
        hi - lo >= min_span,
        "{format}: block {b} spans only codes {lo}..={hi}, under {min_span} levels. \
         The sweep moves the range by a fraction of one level either side of the \
         min/max fit, so a span this narrow means the scale came out far too large \
         and most of the code range is unused."
    );
}

/// Q4_1 diverges from llama.cpp on purpose. Gates structure, decode and error
/// instead of bytes. See the module docs.
#[test]
fn q4_1_writer_is_valid_and_no_worse_than_direct_fit() {
    const BLOCK_BYTES: usize = 20;
    let src = floats("writer_src.bin");
    let got = write_blocks(QuantFormat::Q4_1);

    assert_eq!(
        got.len(),
        BLOCKS * BLOCK_BYTES,
        "Q4_1: {BLOCKS} blocks must occupy {} bytes",
        BLOCKS * BLOCK_BYTES
    );

    for (b, block) in got.as_chunks::<BLOCK_BYTES>().0.iter().enumerate() {
        // Bytes 0..2 are the f16 scale `d`, 2..4 the f16 offset `m`, 4..20 the
        // nibble pairs: element `j` in the low nibble, element `j + 16` in the
        // high one. The reader computes `d·q + m` with `q` unsigned — no bias.
        let mut codes = [0i32; BLOCK_SIZE];
        for (j, &byte) in block[4..].iter().enumerate() {
            codes[j] = i32::from(byte & 0x0F);
            codes[j + 16] = i32::from(byte >> 4);
        }
        assert!(
            stored_scale(block).is_finite(),
            "Q4_1: block {b} stores a non-finite scale, which no reader can use"
        );
        assert!(
            f16::from_le_bytes([block[2], block[3]])
                .to_f32()
                .is_finite(),
            "Q4_1: block {b} stores a non-finite offset. The least-squares refit \
             can leave binary16's range where the plain min/max fit could not, so \
             an unstorable pair must never reach the file."
        );
        assert_extreme_codes_track_extreme_elements(
            "Q4_1",
            b,
            &src[b * BLOCK_SIZE..][..BLOCK_SIZE],
            &codes,
            12,
        );
    }

    let decoded = decode(QuantFormat::Q4_1, &got);
    assert_decodes_within_steps("Q4_1", &src, &decoded, &got, BLOCK_BYTES, 2.0);

    let llama = std::fs::read(fixture("writer_q4_1_llama.bin")).unwrap();
    assert_beats_direct_fit("Q4_1", QuantFormat::Q4_1, &got, &llama);
}

/// The importance vector: one entry per COLUMN, so 256 for this fixture's
/// `n_per_row`, and the same vector for all 8 rows — which is what
/// `ggml_quantize_chunk` does with the pointer it is handed.
fn importance_fixture() -> Vec<f32> {
    let bytes = llama_cpp_bytes("writer_imatrix.bin", "imatrix source");
    assert_eq!(
        bytes.len(),
        256 * 4,
        "writer_imatrix.bin: expected 256 f32 entries, one per column"
    );
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// Quantizes the fixture source with an importance vector, as a caller holding
/// a loaded importance matrix does.
fn write_blocks_with_importance(format: QuantFormat, imatrix: &[f32]) -> Vec<u8> {
    let src = floats("writer_src.bin");
    let (client, device) = cpu_setup();
    let input = Tensor::<CpuRuntime>::from_slice(&src, &[8, 256], &device).unwrap();
    client
        .quantize_with_importance(&input, format, Some(imatrix))
        .unwrap()
        .to_bytes()
        .unwrap()
}

/// Byte-equality gate for one importance-weighted writer against
/// `ggml_quantize_chunk` called with a non-NULL imatrix.
fn assert_imatrix_writer_matches_llama_cpp(
    label: &str,
    format: QuantFormat,
    fixture_name: &str,
    block_bytes: usize,
) {
    let llama = llama_cpp_bytes(fixture_name, label);
    let imatrix = importance_fixture();
    let got = write_blocks_with_importance(format, &imatrix);
    assert_writer_matches_llama_cpp(label, block_bytes, &got, &llama);
}

#[test]
fn q2_k_imatrix_writer_matches_llama_cpp() {
    assert_imatrix_writer_matches_llama_cpp(
        "Q2K+imatrix",
        QuantFormat::Q2K,
        "writer_q2_k_imatrix_llama.bin",
        84,
    );
}

#[test]
fn q3_k_imatrix_writer_matches_llama_cpp() {
    assert_imatrix_writer_matches_llama_cpp(
        "Q3K+imatrix",
        QuantFormat::Q3K,
        "writer_q3_k_imatrix_llama.bin",
        110,
    );
}

#[test]
fn q4_k_imatrix_writer_matches_llama_cpp() {
    assert_imatrix_writer_matches_llama_cpp(
        "Q4K+imatrix",
        QuantFormat::Q4K,
        "writer_q4_k_imatrix_llama.bin",
        144,
    );
}

#[test]
fn q5_k_imatrix_writer_matches_llama_cpp() {
    assert_imatrix_writer_matches_llama_cpp(
        "Q5K+imatrix",
        QuantFormat::Q5K,
        "writer_q5_k_imatrix_llama.bin",
        176,
    );
}

#[test]
fn q6_k_imatrix_writer_matches_llama_cpp() {
    assert_imatrix_writer_matches_llama_cpp(
        "Q6K+imatrix",
        QuantFormat::Q6K,
        "writer_q6_k_imatrix_llama.bin",
        210,
    );
}

#[test]
fn iq4_nl_imatrix_writer_matches_llama_cpp() {
    assert_imatrix_writer_matches_llama_cpp(
        "IQ4_NL+imatrix",
        QuantFormat::IQ4NL,
        "writer_iq4_nl_imatrix_llama.bin",
        18,
    );
}

#[test]
fn iq4_xs_imatrix_writer_matches_llama_cpp() {
    assert_imatrix_writer_matches_llama_cpp(
        "IQ4_XS+imatrix",
        QuantFormat::IQ4XS,
        "writer_iq4_xs_imatrix_llama.bin",
        136,
    );
}

/// Runs whether or not any fixture is present, and is deliberately WEAKER than
/// the five byte-equality tests above — it does not replace them.
///
/// Decoding through boostr's own reader proves only that the writer and the
/// reader agree with each other; both can agree and still disagree with the
/// format. What this does catch is the failure that would make every quality
/// comparison against GGUF meaningless: an importance vector that reaches the
/// writer and changes nothing. A supplied importance must move at least one
/// stored field on this fixture, so identical bytes mean the weight never
/// reached the search.
#[test]
fn k_quant_imatrix_writers_use_the_importance_and_decode() {
    // A spread of magnitudes with one exact zero, so the legal zero-importance
    // branch is exercised even before the fixture lands. This is NOT the
    // conformance vector — that one is read from writer_imatrix.bin.
    let imatrix: Vec<f32> = (0..256)
        .map(|i| {
            if i % 61 == 0 {
                0.0
            } else {
                0.05 + (i % 17) as f32 * 0.31
            }
        })
        .collect();

    for (label, format, block_bytes) in [
        ("Q2K", QuantFormat::Q2K, 84usize),
        ("Q3K", QuantFormat::Q3K, 110),
        ("Q4K", QuantFormat::Q4K, 144),
        ("Q5K", QuantFormat::Q5K, 176),
        ("Q6K", QuantFormat::Q6K, 210),
    ] {
        let weighted = write_blocks_with_importance(format, &imatrix);
        let plain = write_blocks(format);
        assert_eq!(
            weighted.len(),
            8 * block_bytes,
            "{label}+imatrix: 8 super-blocks must occupy {} bytes",
            8 * block_bytes
        );
        assert_ne!(
            weighted, plain,
            "{label}+imatrix: the importance-weighted writer produced the unweighted \
             writer's bytes. A silently ignored importance vector writes a file no \
             check downstream can tell from an unweighted one."
        );
        let decoded = decode(format, &weighted);
        assert!(
            decoded.iter().all(|v| v.is_finite()),
            "{label}+imatrix: decoded a non-finite value. A zero importance entry is \
             legal and must never divide by zero."
        );
        println!(
            "GGUF_WRITER_DIAG format={label}+imatrix bytes={}",
            weighted.len()
        );
    }
}

/// A zero importance vector is legal and must not produce NaN.
///
/// Every weighted sum in the search collapses to zero, every division is
/// guarded on a positive denominator, and the result is a zero scale — the same
/// all-zero sub-block the writers already handle.
#[test]
fn all_zero_importance_decodes_finite() {
    let imatrix = vec![0.0f32; 256];
    for format in [
        QuantFormat::Q2K,
        QuantFormat::Q3K,
        QuantFormat::Q4K,
        QuantFormat::Q5K,
        QuantFormat::Q6K,
    ] {
        let got = write_blocks_with_importance(format, &imatrix);
        assert!(
            decode(format, &got).iter().all(|v| v.is_finite()),
            "{format:?}: an all-zero importance vector produced a non-finite decode"
        );
    }
}

/// A malformed importance vector is an ERROR, never a fallback to uniform
/// weights.
///
/// The fallback would write a file byte-indistinguishable from an unweighted
/// quantization, so the mistake would survive every check downstream and show
/// up only as a quality result nobody can account for.
#[test]
fn malformed_importance_is_rejected() {
    let src = floats("writer_src.bin");
    let (client, device) = cpu_setup();
    let input = Tensor::<CpuRuntime>::from_slice(&src, &[8, 256], &device).unwrap();

    let short = vec![1.0f32; 255];
    assert!(
        client
            .quantize_with_importance(&input, QuantFormat::Q4K, Some(&short))
            .is_err(),
        "an importance vector shorter than the quantized axis must be rejected"
    );

    let long = vec![1.0f32; 2048];
    assert!(
        client
            .quantize_with_importance(&input, QuantFormat::Q4K, Some(&long))
            .is_err(),
        "an importance vector as long as the whole tensor must be rejected: it is \
         one entry per COLUMN, not one per element"
    );

    let mut nan = vec![1.0f32; 256];
    nan[7] = f32::NAN;
    assert!(
        client
            .quantize_with_importance(&input, QuantFormat::Q4K, Some(&nan))
            .is_err(),
        "a non-finite importance entry must be rejected: it reaches the file as a \
         scale no reader can use"
    );

    let mut negative = vec![1.0f32; 256];
    negative[3] = -1.0;
    assert!(
        client
            .quantize_with_importance(&input, QuantFormat::Q4K, Some(&negative))
            .is_err(),
        "a negative importance entry must be rejected: an importance is a mean \
         square activation"
    );

    // Formats with no `_impl` writer in `ggml-quants.c` refuse the importance
    // rather than dropping it.
    let uniform = vec![1.0f32; 256];
    assert!(
        client
            .quantize_with_importance(&input, QuantFormat::Q8_0, Some(&uniform))
            .is_err(),
        "Q8_0 has no importance-weighted writer and must say so, not ignore the vector"
    );

    // No importance at all is the plain path, unchanged.
    assert_eq!(
        client
            .quantize_with_importance(&input, QuantFormat::Q4K, None)
            .unwrap()
            .to_bytes()
            .unwrap(),
        write_blocks(QuantFormat::Q4K),
        "quantize_with_importance(None) must be quantize() byte for byte"
    );
}
