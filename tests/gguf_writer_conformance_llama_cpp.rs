//! `QuantizeOps::quantize` against llama.cpp's OWN quantizer,
//! `ggml_quantize_chunk`, byte for byte.
//!
//! `tests/gguf_conformance_llama_cpp.rs` gates only the READ direction:
//! `DequantOps::dequantize` against llama.cpp's dequantizer. A writer checked
//! against the reader beside it can agree with itself while both disagree
//! with the format — the reader accepts whatever layout the writer emits. This
//! file gates the WRITE direction against an external quantizer instead.
//!
//! Byte equality, not a value tolerance, is the correct bar. boostr's writer
//! implements llama.cpp's own iterative per-sub-block scale search, so
//! matching bytes is the intended property, not a coincidence of rounding. A
//! future change that lands close but not exact is a regression worth seeing,
//! never something to absorb into an epsilon.
//!
//! boostr writes exactly six formats: `QuantFormat::Q4_0`, `Q4_1`, `Q8_0`,
//! `Q4K`, `Q5K`, `Q6K`. Every other `QuantFormat` variant returns
//! `Error::UnsupportedQuantFormat` from `quantize`, so there is nothing to
//! gate for them here.
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

use boostr::quant::{QuantFormat, QuantizeOps};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

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

/// Asserts `got` equals llama.cpp's `ggml_quantize_chunk` output EXACTLY, and
/// always prints one flat `GGUF_WRITER_DIAG` line — pass or fail — so a green
/// run still records what each format produced.
///
/// On mismatch, reports up to 16 differing byte offsets. Each entry names the
/// byte offset, the block index (`offset / block_bytes`), and the offset
/// within that block (`offset % block_bytes`) — the within-block offset is
/// what identifies which field is wrong. A differing scale field means the
/// scale search diverged; a differing payload byte means the packing
/// diverged. llama.cpp produced the expectation, so the fix is always the
/// writer, never the fixture.
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
fn q4_0_writer_matches_llama_cpp() {
    let src = floats("writer_src.bin");
    let llama = std::fs::read(fixture("writer_q4_0_llama.bin")).unwrap();
    let (client, device) = cpu_setup();
    let input = Tensor::<CpuRuntime>::from_slice(&src, &[8, 256], &device).unwrap();
    let got = client
        .quantize(&input, QuantFormat::Q4_0)
        .unwrap()
        .to_bytes()
        .unwrap();
    assert_writer_matches_llama_cpp("Q4_0", 18, &got, &llama);
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
fn q8_0_writer_matches_llama_cpp() {
    let src = floats("writer_src.bin");
    let llama = std::fs::read(fixture("writer_q8_0_llama.bin")).unwrap();
    let (client, device) = cpu_setup();
    let input = Tensor::<CpuRuntime>::from_slice(&src, &[8, 256], &device).unwrap();
    let got = client
        .quantize(&input, QuantFormat::Q8_0)
        .unwrap()
        .to_bytes()
        .unwrap();
    assert_writer_matches_llama_cpp("Q8_0", 34, &got, &llama);
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
