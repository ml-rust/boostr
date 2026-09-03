//! `DequantOps::dequantize` against llama.cpp's OWN reference implementation.
//!
//! Every expectation comes from llama.cpp, by one of two routes: the `gguf`
//! Python package maintained in the llama.cpp tree, or ggml's C library
//! directly. No boostr code produces any of them. The provenance tables below
//! name the route per fixture.
//!
//! Keep the reference EXTERNAL. A comparison against boostr's own writer,
//! against `QuantTensor` round-tripping, or against a Rust restatement of the
//! block layout passes while the decode is wrong, and is not a substitute.
//!
//! `tests/gguf_dequant_cpu_cuda_parity.rs` is the other half. It proves only
//! that CPU and CUDA agree with each other, so it passes on any error the two
//! backends share. It does cover the quant-matmul kernels, which this file does
//! not. Neither file subsumes the other.
//!
//! # Fixture provenance
//!
//! Files live in `tests/fixtures/gguf_conformance/`. `*_raw.bin` is the
//! verbatim quantized block payload copied out of the named GGUF file;
//! `*_ref.bin` is little-endian `f32`, one value per element, from
//! `gguf.quants.dequantize`.
//!
//! | fixture | source model file | tensor | blocks | elements |
//! |---|---|---|---|---|
//! | `q4_0_*` | `jina-embeddings-v3-Q4_0.gguf` | `blk.0.attn_output.weight` | 8 | 256 |
//! | `q5_0_*` | `nomic-embed-text-v1.5.Q5_0.gguf` | `token_embd.weight` | 8 | 256 |
//! | `q4_k_*` | `bge-micro-v2-Q4_K_M.gguf` | `blk.0.ffn_down.weight` | 8 | 2048 |
//! | `q8_0_*` | `embeddinggemma-300M-Q8_0.gguf` | `token_embd.weight` | 8 | 256 |
//! | `q6_k_*` | `jina-embeddings-v3-Q4_0.gguf` | `token_embd.weight` | 8 | 2048 |
//! | `q2_k_*` | `nomic-embed-text-v1.5.Q2_K.gguf` | `token_embd.weight` | 8 | 2048 |
//! | `q3_k_*` | `nomic-embed-text-v1.5.Q2_K.gguf` | `blk.0.attn_output.weight` | 8 | 2048 |
//! | `q5_k_*` | `nomic-embed-text-v1.5.Q4_K_M.gguf` | `blk.0.attn_qkv.weight` | 8 | 2048 |
//!
//! Four more formats appear in no model file on hand but CAN be produced by the
//! reference implementation itself, via `gguf.quants.quantize`. Their `_raw.bin`
//! is llama.cpp's own quantizer output over a fixed-seed normal sample, and
//! their `_ref.bin` is llama.cpp's own dequantizer applied to those bytes. No
//! boostr code takes part in either half, so the independence that makes this
//! file worth having is intact:
//!
//! | fixture | produced by | blocks | elements |
//! |---|---|---|---|
//! | `q4_1_*` | `gguf.quants.quantize`, `default_rng(0x9E3779B9).standard_normal` | 8 | 256 |
//! | `q5_1_*` | same | 8 | 256 |
//! | `tq1_0_*` | same | 8 | 2048 |
//! | `tq2_0_*` | same | 8 | 2048 |
//!
//! The ternary pair carries only three distinct magnitudes per block, so it is
//! weaker at exposing an in-block permutation than real weights are. Prefer a
//! model file for those two if one ever turns up.
//!
//! `gguf.quants.quantize` produces no IQ format. Those `_raw.bin` come from
//! llama.cpp's `llama-quantize` binary over an F16 or Q8_0 model, and their
//! `_ref.bin` from `gguf.quants.dequantize` on the bytes it emitted:
//!
//! | fixture | quantized from | tensor | blocks | elements |
//! |---|---|---|---|---|
//! | `iq4_nl_*` | `nomic-embed-text-v1.5.f16.gguf` | `blk.0.attn_output.weight` | 8 | 256 |
//! | `iq4_xs_*` | same | `blk.0.attn_output.weight` | 8 | 2048 |
//! | `iq3_s_*` | same | `blk.0.attn_output.weight` | 8 | 2048 |
//! | `iq3_xxs_*` | same | `blk.0.attn_qkv.weight` | 8 | 2048 |
//! | `iq2_s_*` | `mistral-7b-v0.1.Q8_0.gguf` | `blk.0.attn_k.weight` | 8 | 2048 |
//! | `iq2_xs_*` | same | `blk.0.attn_k.weight` | 8 | 2048 |
//! | `iq2_xxs_*` | same | `blk.0.attn_k.weight` | 8 | 2048 |
//! | `iq1_s_*` | same | `blk.0.attn_k.weight` | 8 | 2048 |
//! | `iq1_m_*` | same | `blk.0.attn_k.weight` | 8 | 2048 |
//!
//! The low-bit half of that list needs a second input: `llama-quantize` refuses
//! IQ1_S, IQ1_M, IQ2_S, IQ2_XS and IQ2_XXS outright without an importance
//! matrix. One from `llama-imatrix` over any text corpus will do — it steers
//! which weights get more bits, and changes nothing about the block layout
//! these fixtures pin. It must come from a CAUSAL model: `llama-imatrix` aborts
//! on an encoder-only embedding model, and on one whose tokenizer appends EOS.
//!
//! Q8_1 and Q8K are activation formats: no model file stores them, and the
//! `gguf` Python package implements neither `quantize` nor `dequantize` for
//! either. Both fixtures instead come from ggml's C routines directly
//! (`libggml-base` / `libggml-cpu`), not from a model file or `gguf`:
//!
//! | fixture | produced by | blocks | elements |
//! |---|---|---|---|
//! | `q8_k_raw.bin` | ggml `quantize_row_q8_K` | 8 | 2048 |
//! | `q8_k_ref.bin` | ggml `dequantize_row_q8_K` on `q8_k_raw.bin` | 8 | 2048 |
//! | `q8_1_raw.bin` | ggml `quantize_row_q8_1` | 8 | 256 |
//! | `q8_1_src.bin` | the pre-quantization floats fed to `quantize_row_q8_1` | 8 | 256 |
//!
//! Eight blocks each: one block cannot expose a per-block indexing or stride
//! error, and two cannot separate "off by one block" from "reversed". Real
//! model weights also defeat a permutation the way a uniform synthetic block
//! cannot — no two elements inside a block share a value by construction.
//!
//! # Regenerating or extending the fixtures
//!
//! Requires `pip install gguf numpy`. Point `MODEL` at a real GGUF file, name
//! a tensor stored in the format of interest, and write both halves:
//!
//! ```python
//! import numpy as np
//! from gguf import GGUFReader
//! from gguf.constants import GGML_QUANT_SIZES
//! from gguf.quants import dequantize
//!
//! MODEL  = "/path/to/model.gguf"
//! TENSOR = "blk.0.attn_output.weight"
//! BLOCKS = 8
//! OUT    = "tests/fixtures/gguf_conformance/q4_0"
//!
//! reader = GGUFReader(MODEL)
//! t = next(t for t in reader.tensors if t.name == TENSOR)
//! qtype = t.tensor_type
//! block_size, type_size = GGML_QUANT_SIZES[qtype]
//!
//! raw = t.data.tobytes()[: BLOCKS * type_size]
//! ref = dequantize(np.frombuffer(raw, dtype=np.uint8), qtype).astype(np.float32)
//! assert ref.size == BLOCKS * block_size
//!
//! open(OUT + "_raw.bin", "wb").write(raw)
//! open(OUT + "_ref.bin", "wb").write(ref.tobytes())  # little-endian f32
//! ```
//!
//! For a format `gguf` can quantize, swap the reader for
//! `quantize(src.reshape(BLOCKS, block_size), qtype)` over a fixed-seed sample
//! and keep the rest. `gguf` quantizes only Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, TQ1_0
//! and TQ2_0; every other format raises `NotImplementedError` and needs a real
//! model file.
//!
//! For Q8_1 and Q8K, call ggml's C routines instead. Declare the prototypes
//! and link against the installed library — no ggml headers needed:
//!
//! ```c
//! void quantize_row_q8_1(const float *x, void *y, int64_t k);
//! void quantize_row_q8_K(const float *x, void *y, int64_t k);
//! void dequantize_row_q8_K(const void *x, float *y, int64_t k);
//! // gcc gen.c -o gen -lggml-base -lggml-cpu
//! ```
//!
//! Scale each block by a different factor when generating the source floats.
//! A single scale across all blocks lets a per-block stride error pass.
//!
//! A new format needs the two files plus one `_cpu` and one `_cuda` test,
//! written out explicitly like the existing ones below.
//!
//! # Comparison is EXACT, except Q8_1
//!
//! Every `QuantFormat` variant is gated against llama.cpp. Every format but
//! Q8_1 uses `assert_matches_llama_cpp`: no tolerance, bit-for-bit against the
//! llama.cpp reference, zero mismatching elements, `max_abs = 0.0`. Treat any
//! future approximate result on one of these as an error requiring review.
//! Never absorb it into a widened epsilon. If CUDA ever differs by an ulp or
//! two because nvcc contracts a scale multiply into an FMA, report the exact
//! indices and magnitudes rather than widening the check here.
//!
//! Q8K has a `dequantize_row_q8_K` in ggml, so it fits this bit-exact gate
//! despite being an activation format. Q8_1 has no `dequantize_row_q8_1` —
//! the format is write-only, produced for `vec_dot` — so it uses
//! `assert_round_trip_within` instead: llama.cpp quantizes known floats, and
//! boostr's dequant must land within a fixed tolerance of `0.04` of them, not
//! match bit-for-bit.
//!
//! Run with:
//!   cd boostr && cargo test --test gguf_conformance_llama_cpp
//!   cd boostr && cargo test --features cuda --test gguf_conformance_llama_cpp

use std::path::PathBuf;

use boostr::quant::{DequantOps, QuantFormat, QuantTensor};
use numr::dtype::DType;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

/// Resolves a fixture relative to the crate, never to an absolute path from
/// whichever machine generated it.
fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/gguf_conformance")
        .join(name)
}

/// The verbatim quantized block bytes copied out of the source GGUF file.
fn raw_blocks(name: &str) -> Vec<u8> {
    std::fs::read(fixture(name)).unwrap()
}

/// The llama.cpp reference values: little-endian `f32`, one per element.
fn llama_cpp_reference(name: &str) -> Vec<f32> {
    let bytes = std::fs::read(fixture(name)).unwrap();
    assert_eq!(bytes.len() % 4, 0, "{name}: reference is not whole f32s");
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

/// Asserts `got` equals the llama.cpp reference EXACTLY, and always prints one
/// flat `GGUF_CONFORMANCE_DIAG` line — pass or fail — so a green run still
/// records what each format produced.
///
/// On error it names the first mismatching indices with BOTH values. The
/// pattern is what identifies the defect: reference values reappearing at a
/// different position inside the same block is a permutation (the split-half
/// vs interleaved nibble ordering), while a constant ratio down the two
/// columns is a scale or field-order error.
fn assert_matches_llama_cpp(format: &str, backend: &str, got: &[f32], reference: &[f32]) {
    assert_eq!(
        got.len(),
        reference.len(),
        "{format}/{backend}: element count {} does not match the llama.cpp reference {}",
        got.len(),
        reference.len()
    );

    let mut mismatches = 0usize;
    let mut max_abs = 0.0f32;
    let mut first_mismatch: i64 = -1;
    let mut report = String::new();
    for i in 0..got.len() {
        let (a, b) = (got[i], reference[i]);
        let diff = (a - b).abs();
        if diff > max_abs {
            max_abs = diff;
        }
        if a.to_bits() != b.to_bits() {
            if first_mismatch < 0 {
                first_mismatch = i as i64;
            }
            mismatches += 1;
            if mismatches <= 16 {
                report.push_str(&format!(
                    "\n  idx {i:5}  boostr {a:14.6}  llama.cpp {b:14.6}"
                ));
            }
        }
    }

    println!(
        "GGUF_CONFORMANCE_DIAG format={format} backend={backend} elements={} \
         mismatches={mismatches} max_abs={max_abs:.6e}",
        got.len()
    );

    assert_eq!(
        mismatches,
        0,
        "{format}/{backend}: boostr disagrees with the llama.cpp reference on {mismatches} \
         of {} elements (first at index {first_mismatch}, max_abs {max_abs:.6e}). \
         First mismatches:{report}\n\
         The reference is INDEPENDENT of boostr (gguf.quants.dequantize) — do not \
         relax this comparison, fix the decode. Reference values reappearing at a \
         different position inside the same block is a permutation; a constant ratio \
         between the columns is a scale or field-order error.",
        got.len()
    );
}

/// Asserts `got` lands within `tol` of `source`, the pre-quantization floats.
/// Q8_1 has no external dequantized reference, so this is a round trip, not a
/// bit-exact comparison: `source` fed llama.cpp's quantizer, and `got` is
/// boostr's dequant of the bytes it produced. Always prints one flat
/// `GGUF_CONFORMANCE_DIAG` line — pass or fail — so a green run still records
/// what each format produced.
fn assert_round_trip_within(format: &str, backend: &str, got: &[f32], source: &[f32], tol: f32) {
    assert_eq!(
        got.len(),
        source.len(),
        "{format}/{backend}: element count {} does not match the pre-quantization source {}",
        got.len(),
        source.len()
    );

    let mut mismatches = 0usize;
    let mut max_abs = 0.0f32;
    let mut first_mismatch: i64 = -1;
    let mut report = String::new();
    for i in 0..got.len() {
        let (a, b) = (got[i], source[i]);
        let diff = (a - b).abs();
        if diff > max_abs {
            max_abs = diff;
        }
        if diff > tol {
            if first_mismatch < 0 {
                first_mismatch = i as i64;
            }
            mismatches += 1;
            if mismatches <= 16 {
                report.push_str(&format!(
                    "\n  idx {i:5}  boostr {a:14.6}  pre-quant {b:14.6}"
                ));
            }
        }
    }

    println!(
        "GGUF_CONFORMANCE_DIAG format={format} backend={backend} elements={} \
         mismatches={mismatches} max_abs={max_abs:.6e}",
        got.len()
    );

    assert_eq!(
        mismatches,
        0,
        "{format}/{backend}: boostr's dequant lands outside tolerance {tol} of the \
         pre-quantization floats on {mismatches} of {} elements (first at index \
         {first_mismatch}, max_abs {max_abs:.6e}). First mismatches:{report}\n\
         This is a ROUND TRIP against the source floats llama.cpp quantized, not a \
         bit-exact reference — Q8_1 has no dequantize_row_q8_1 in ggml. A mismatch \
         within a small multiple of tol is a rounding drift; one off by whole \
         magnitudes is a layout error.",
        got.len()
    );
}

#[test]
fn q4_0_matches_llama_cpp_cpu() {
    let bytes = raw_blocks("q4_0_raw.bin");
    let reference = llama_cpp_reference("q4_0_ref.bin");
    let (client, device) = cpu_setup();
    let qt =
        QuantTensor::<CpuRuntime>::from_bytes(&bytes, QuantFormat::Q4_0, &[256], &device).unwrap();
    let got = client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>();
    assert_matches_llama_cpp("Q4_0", "cpu", &got, &reference);
}

#[test]
fn q5_0_matches_llama_cpp_cpu() {
    let bytes = raw_blocks("q5_0_raw.bin");
    let reference = llama_cpp_reference("q5_0_ref.bin");
    let (client, device) = cpu_setup();
    let qt =
        QuantTensor::<CpuRuntime>::from_bytes(&bytes, QuantFormat::Q5_0, &[256], &device).unwrap();
    let got = client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>();
    assert_matches_llama_cpp("Q5_0", "cpu", &got, &reference);
}

#[test]
fn q4_k_matches_llama_cpp_cpu() {
    let bytes = raw_blocks("q4_k_raw.bin");
    let reference = llama_cpp_reference("q4_k_ref.bin");
    let (client, device) = cpu_setup();
    let qt =
        QuantTensor::<CpuRuntime>::from_bytes(&bytes, QuantFormat::Q4K, &[2048], &device).unwrap();
    let got = client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>();
    assert_matches_llama_cpp("Q4K", "cpu", &got, &reference);
}

#[test]
fn q8_0_matches_llama_cpp_cpu() {
    let bytes = raw_blocks("q8_0_raw.bin");
    let reference = llama_cpp_reference("q8_0_ref.bin");
    let (client, device) = cpu_setup();
    let qt =
        QuantTensor::<CpuRuntime>::from_bytes(&bytes, QuantFormat::Q8_0, &[256], &device).unwrap();
    let got = client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>();
    assert_matches_llama_cpp("Q8_0", "cpu", &got, &reference);
}

/// Q6_K needs an external reference: boostr's own reader can agree with a
/// field-order error, so an internal-only check cannot catch it.
#[test]
fn q6_k_matches_llama_cpp_cpu() {
    let bytes = raw_blocks("q6_k_raw.bin");
    let reference = llama_cpp_reference("q6_k_ref.bin");
    let (client, device) = cpu_setup();
    let qt =
        QuantTensor::<CpuRuntime>::from_bytes(&bytes, QuantFormat::Q6K, &[2048], &device).unwrap();
    let got = client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>();
    assert_matches_llama_cpp("Q6K", "cpu", &got, &reference);
}

#[test]
fn q8_k_matches_llama_cpp_cpu() {
    let bytes = raw_blocks("q8_k_raw.bin");
    let reference = llama_cpp_reference("q8_k_ref.bin");
    let (client, device) = cpu_setup();
    let qt =
        QuantTensor::<CpuRuntime>::from_bytes(&bytes, QuantFormat::Q8K, &[2048], &device).unwrap();
    let got = client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>();
    assert_matches_llama_cpp("Q8K", "cpu", &got, &reference);
}

/// Q8_1 is write-only for `vec_dot`: ggml exports no `dequantize_row_q8_1`, so
/// the gate is a round trip against the floats llama.cpp quantized.
///
/// Tolerance derivation: Q8_1 stores 8-bit values with per-block scale
/// `d = max_abs / 127`. The source reaches magnitude ~7.71, so the largest
/// block's `d` is about `7.71 / 127`. Round-to-nearest error is at most `d / 2`,
/// about 0.030. `0.04` leaves margin without admitting a layout error, which
/// is wrong by whole magnitudes rather than a fraction of a step.
#[test]
fn q8_1_matches_llama_cpp_cpu() {
    let bytes = raw_blocks("q8_1_raw.bin");
    let source = llama_cpp_reference("q8_1_src.bin");
    let (client, device) = cpu_setup();
    let qt =
        QuantTensor::<CpuRuntime>::from_bytes(&bytes, QuantFormat::Q8_1, &[256], &device).unwrap();
    let got = client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>();
    assert_round_trip_within("Q8_1", "cpu", &got, &source, 0.04);
}

#[test]
fn q2_k_matches_llama_cpp_cpu() {
    let bytes = raw_blocks("q2_k_raw.bin");
    let reference = llama_cpp_reference("q2_k_ref.bin");
    let (client, device) = cpu_setup();
    let qt =
        QuantTensor::<CpuRuntime>::from_bytes(&bytes, QuantFormat::Q2K, &[2048], &device).unwrap();
    let got = client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>();
    assert_matches_llama_cpp("Q2K", "cpu", &got, &reference);
}

#[test]
fn q3_k_matches_llama_cpp_cpu() {
    let bytes = raw_blocks("q3_k_raw.bin");
    let reference = llama_cpp_reference("q3_k_ref.bin");
    let (client, device) = cpu_setup();
    let qt =
        QuantTensor::<CpuRuntime>::from_bytes(&bytes, QuantFormat::Q3K, &[2048], &device).unwrap();
    let got = client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>();
    assert_matches_llama_cpp("Q3K", "cpu", &got, &reference);
}

#[test]
fn q5_k_matches_llama_cpp_cpu() {
    let bytes = raw_blocks("q5_k_raw.bin");
    let reference = llama_cpp_reference("q5_k_ref.bin");
    let (client, device) = cpu_setup();
    let qt =
        QuantTensor::<CpuRuntime>::from_bytes(&bytes, QuantFormat::Q5K, &[2048], &device).unwrap();
    let got = client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>();
    assert_matches_llama_cpp("Q5K", "cpu", &got, &reference);
}

#[test]
fn q4_1_matches_llama_cpp_cpu() {
    let bytes = raw_blocks("q4_1_raw.bin");
    let reference = llama_cpp_reference("q4_1_ref.bin");
    let (client, device) = cpu_setup();
    let qt =
        QuantTensor::<CpuRuntime>::from_bytes(&bytes, QuantFormat::Q4_1, &[256], &device).unwrap();
    let got = client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>();
    assert_matches_llama_cpp("Q4_1", "cpu", &got, &reference);
}

#[test]
fn q5_1_matches_llama_cpp_cpu() {
    let bytes = raw_blocks("q5_1_raw.bin");
    let reference = llama_cpp_reference("q5_1_ref.bin");
    let (client, device) = cpu_setup();
    let qt =
        QuantTensor::<CpuRuntime>::from_bytes(&bytes, QuantFormat::Q5_1, &[256], &device).unwrap();
    let got = client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>();
    assert_matches_llama_cpp("Q5_1", "cpu", &got, &reference);
}

#[test]
fn tq1_0_matches_llama_cpp_cpu() {
    let bytes = raw_blocks("tq1_0_raw.bin");
    let reference = llama_cpp_reference("tq1_0_ref.bin");
    let (client, device) = cpu_setup();
    let qt = QuantTensor::<CpuRuntime>::from_bytes(&bytes, QuantFormat::TQ1_0, &[2048], &device)
        .unwrap();
    let got = client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>();
    assert_matches_llama_cpp("TQ1_0", "cpu", &got, &reference);
}

#[test]
fn tq2_0_matches_llama_cpp_cpu() {
    let bytes = raw_blocks("tq2_0_raw.bin");
    let reference = llama_cpp_reference("tq2_0_ref.bin");
    let (client, device) = cpu_setup();
    let qt = QuantTensor::<CpuRuntime>::from_bytes(&bytes, QuantFormat::TQ2_0, &[2048], &device)
        .unwrap();
    let got = client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>();
    assert_matches_llama_cpp("TQ2_0", "cpu", &got, &reference);
}

#[test]
fn iq4_nl_matches_llama_cpp_cpu() {
    let bytes = raw_blocks("iq4_nl_raw.bin");
    let reference = llama_cpp_reference("iq4_nl_ref.bin");
    let (client, device) = cpu_setup();
    let qt =
        QuantTensor::<CpuRuntime>::from_bytes(&bytes, QuantFormat::IQ4NL, &[256], &device).unwrap();
    let got = client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>();
    assert_matches_llama_cpp("IQ4NL", "cpu", &got, &reference);
}

#[test]
fn iq4_xs_matches_llama_cpp_cpu() {
    let bytes = raw_blocks("iq4_xs_raw.bin");
    let reference = llama_cpp_reference("iq4_xs_ref.bin");
    let (client, device) = cpu_setup();
    let qt = QuantTensor::<CpuRuntime>::from_bytes(&bytes, QuantFormat::IQ4XS, &[2048], &device)
        .unwrap();
    let got = client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>();
    assert_matches_llama_cpp("IQ4XS", "cpu", &got, &reference);
}

#[test]
fn iq3_s_matches_llama_cpp_cpu() {
    let bytes = raw_blocks("iq3_s_raw.bin");
    let reference = llama_cpp_reference("iq3_s_ref.bin");
    let (client, device) = cpu_setup();
    let qt =
        QuantTensor::<CpuRuntime>::from_bytes(&bytes, QuantFormat::IQ3S, &[2048], &device).unwrap();
    let got = client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>();
    assert_matches_llama_cpp("IQ3S", "cpu", &got, &reference);
}

#[test]
fn iq3_xxs_matches_llama_cpp_cpu() {
    let bytes = raw_blocks("iq3_xxs_raw.bin");
    let reference = llama_cpp_reference("iq3_xxs_ref.bin");
    let (client, device) = cpu_setup();
    let qt = QuantTensor::<CpuRuntime>::from_bytes(&bytes, QuantFormat::IQ3XXS, &[2048], &device)
        .unwrap();
    let got = client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>();
    assert_matches_llama_cpp("IQ3XXS", "cpu", &got, &reference);
}

#[test]
fn iq2_xxs_matches_llama_cpp_cpu() {
    let bytes = raw_blocks("iq2_xxs_raw.bin");
    let reference = llama_cpp_reference("iq2_xxs_ref.bin");
    let (client, device) = cpu_setup();
    let qt = QuantTensor::<CpuRuntime>::from_bytes(&bytes, QuantFormat::IQ2XXS, &[2048], &device)
        .unwrap();
    let got = client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>();
    assert_matches_llama_cpp("IQ2XXS", "cpu", &got, &reference);
}

#[test]
fn iq2_xs_matches_llama_cpp_cpu() {
    let bytes = raw_blocks("iq2_xs_raw.bin");
    let reference = llama_cpp_reference("iq2_xs_ref.bin");
    let (client, device) = cpu_setup();
    let qt = QuantTensor::<CpuRuntime>::from_bytes(&bytes, QuantFormat::IQ2XS, &[2048], &device)
        .unwrap();
    let got = client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>();
    assert_matches_llama_cpp("IQ2XS", "cpu", &got, &reference);
}

#[test]
fn iq1_s_matches_llama_cpp_cpu() {
    let bytes = raw_blocks("iq1_s_raw.bin");
    let reference = llama_cpp_reference("iq1_s_ref.bin");
    let (client, device) = cpu_setup();
    let qt =
        QuantTensor::<CpuRuntime>::from_bytes(&bytes, QuantFormat::IQ1S, &[2048], &device).unwrap();
    let got = client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>();
    assert_matches_llama_cpp("IQ1S", "cpu", &got, &reference);
}

#[test]
fn iq1_m_matches_llama_cpp_cpu() {
    let bytes = raw_blocks("iq1_m_raw.bin");
    let reference = llama_cpp_reference("iq1_m_ref.bin");
    let (client, device) = cpu_setup();
    let qt =
        QuantTensor::<CpuRuntime>::from_bytes(&bytes, QuantFormat::IQ1M, &[2048], &device).unwrap();
    let got = client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>();
    assert_matches_llama_cpp("IQ1M", "cpu", &got, &reference);
}

#[test]
fn iq2_s_matches_llama_cpp_cpu() {
    let bytes = raw_blocks("iq2_s_raw.bin");
    let reference = llama_cpp_reference("iq2_s_ref.bin");
    let (client, device) = cpu_setup();
    let qt =
        QuantTensor::<CpuRuntime>::from_bytes(&bytes, QuantFormat::IQ2S, &[2048], &device).unwrap();
    let got = client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>();
    assert_matches_llama_cpp("IQ2S", "cpu", &got, &reference);
}

// ---------------------------------------------------------------------------
// CUDA
//
// Same fixtures, same exact comparison, against the CUDA dequant kernels. This
// is the half that catches a CUDA kernel decoding a format wrong while
// erroring nothing — shape, block count and RMS all stay plausible, so only
// this comparison against an external reference reveals it.
//
// A machine without CUDA skips loudly, in the style of
// `tests/gguf_dequant_cpu_cuda_parity.rs`: a silent skip lets exactly that
// class of bug through undetected. An allocation error is a FAILURE, never
// a skip.
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
mod cuda {
    use super::{
        assert_matches_llama_cpp, assert_round_trip_within, llama_cpp_reference, raw_blocks,
    };

    use std::sync::{Mutex, OnceLock};

    use boostr::quant::{DequantOps, QuantFormat, QuantTensor};
    use numr::dtype::DType;
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    use numr::runtime::{Runtime, RuntimeClient};

    static CUDA_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    fn cuda_lock() -> std::sync::MutexGuard<'static, ()> {
        CUDA_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|p| p.into_inner())
    }

    /// Prints the skip banner to stdout AND stderr, then returns.
    fn loud_skip(format: &str) {
        let banner = format!(
            "\n\
             !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\
             !! GGUF_CONFORMANCE_SKIPPED  format=\"{format}\" backend=\"cuda\"\n\
             !! REASON: CUDA is not available on this machine\n\
             !! NOTHING WAS VERIFIED. This test reported success WITHOUT comparing\n\
             !! the CUDA dequant kernel against the llama.cpp reference. Treat the\n\
             !! format as UNTESTED on CUDA, not as green.\n\
             !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
        );
        println!("{banner}");
        eprintln!("{banner}");
    }

    fn cuda_setup() -> (CudaClient, CudaDevice) {
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);
        (client, device)
    }

    #[test]
    fn q4_0_matches_llama_cpp_cuda() {
        if !numr::runtime::cuda::is_cuda_available() {
            loud_skip("Q4_0");
            return;
        }
        let _lock = cuda_lock();
        let bytes = raw_blocks("q4_0_raw.bin");
        let reference = llama_cpp_reference("q4_0_ref.bin");
        let (client, device) = cuda_setup();
        client.synchronize();
        let got = {
            let qt =
                QuantTensor::<CudaRuntime>::from_bytes(&bytes, QuantFormat::Q4_0, &[256], &device)
                    .unwrap_or_else(|e| {
                        panic!(
                            "Q4_0/cuda: device allocation failed: {e}. This is a FAILURE, not a \
                             skip — Q4_0 was NOT compared against llama.cpp."
                        )
                    });
            client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>()
        };
        client.synchronize();
        assert_matches_llama_cpp("Q4_0", "cuda", &got, &reference);
    }

    #[test]
    fn q5_0_matches_llama_cpp_cuda() {
        if !numr::runtime::cuda::is_cuda_available() {
            loud_skip("Q5_0");
            return;
        }
        let _lock = cuda_lock();
        let bytes = raw_blocks("q5_0_raw.bin");
        let reference = llama_cpp_reference("q5_0_ref.bin");
        let (client, device) = cuda_setup();
        client.synchronize();
        let got = {
            let qt =
                QuantTensor::<CudaRuntime>::from_bytes(&bytes, QuantFormat::Q5_0, &[256], &device)
                    .unwrap_or_else(|e| {
                        panic!(
                            "Q5_0/cuda: device allocation failed: {e}. This is a FAILURE, not a \
                             skip — Q5_0 was NOT compared against llama.cpp."
                        )
                    });
            client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>()
        };
        client.synchronize();
        assert_matches_llama_cpp("Q5_0", "cuda", &got, &reference);
    }

    #[test]
    fn q4_k_matches_llama_cpp_cuda() {
        if !numr::runtime::cuda::is_cuda_available() {
            loud_skip("Q4K");
            return;
        }
        let _lock = cuda_lock();
        let bytes = raw_blocks("q4_k_raw.bin");
        let reference = llama_cpp_reference("q4_k_ref.bin");
        let (client, device) = cuda_setup();
        client.synchronize();
        let got = {
            let qt =
                QuantTensor::<CudaRuntime>::from_bytes(&bytes, QuantFormat::Q4K, &[2048], &device)
                    .unwrap_or_else(|e| {
                        panic!(
                            "Q4K/cuda: device allocation failed: {e}. This is a FAILURE, not a \
                             skip — Q4K was NOT compared against llama.cpp."
                        )
                    });
            client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>()
        };
        client.synchronize();
        assert_matches_llama_cpp("Q4K", "cuda", &got, &reference);
    }

    #[test]
    fn q8_0_matches_llama_cpp_cuda() {
        if !numr::runtime::cuda::is_cuda_available() {
            loud_skip("Q8_0");
            return;
        }
        let _lock = cuda_lock();
        let bytes = raw_blocks("q8_0_raw.bin");
        let reference = llama_cpp_reference("q8_0_ref.bin");
        let (client, device) = cuda_setup();
        client.synchronize();
        let got = {
            let qt =
                QuantTensor::<CudaRuntime>::from_bytes(&bytes, QuantFormat::Q8_0, &[256], &device)
                    .unwrap_or_else(|e| {
                        panic!(
                            "Q8_0/cuda: device allocation failed: {e}. This is a FAILURE, not a \
                             skip — Q8_0 was NOT compared against llama.cpp."
                        )
                    });
            client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>()
        };
        client.synchronize();
        assert_matches_llama_cpp("Q8_0", "cuda", &got, &reference);
    }

    #[test]
    fn q6_k_matches_llama_cpp_cuda() {
        if !numr::runtime::cuda::is_cuda_available() {
            loud_skip("Q6K");
            return;
        }
        let _lock = cuda_lock();
        let bytes = raw_blocks("q6_k_raw.bin");
        let reference = llama_cpp_reference("q6_k_ref.bin");
        let (client, device) = cuda_setup();
        client.synchronize();
        let got = {
            let qt =
                QuantTensor::<CudaRuntime>::from_bytes(&bytes, QuantFormat::Q6K, &[2048], &device)
                    .unwrap_or_else(|e| {
                        panic!(
                            "Q6K/cuda: device allocation failed: {e}. This is a FAILURE, not a \
                             skip — Q6K was NOT compared against llama.cpp."
                        )
                    });
            client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>()
        };
        client.synchronize();
        assert_matches_llama_cpp("Q6K", "cuda", &got, &reference);
    }

    #[test]
    fn q8_k_matches_llama_cpp_cuda() {
        if !numr::runtime::cuda::is_cuda_available() {
            loud_skip("Q8K");
            return;
        }
        let _lock = cuda_lock();
        let bytes = raw_blocks("q8_k_raw.bin");
        let reference = llama_cpp_reference("q8_k_ref.bin");
        let (client, device) = cuda_setup();
        client.synchronize();
        let got = {
            let qt =
                QuantTensor::<CudaRuntime>::from_bytes(&bytes, QuantFormat::Q8K, &[2048], &device)
                    .unwrap_or_else(|e| {
                        panic!(
                            "Q8K/cuda: device allocation failed: {e}. This is a FAILURE, not a \
                             skip — Q8K was NOT compared against llama.cpp."
                        )
                    });
            client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>()
        };
        client.synchronize();
        assert_matches_llama_cpp("Q8K", "cuda", &got, &reference);
    }

    #[test]
    fn q8_1_matches_llama_cpp_cuda() {
        if !numr::runtime::cuda::is_cuda_available() {
            loud_skip("Q8_1");
            return;
        }
        let _lock = cuda_lock();
        let bytes = raw_blocks("q8_1_raw.bin");
        let source = llama_cpp_reference("q8_1_src.bin");
        let (client, device) = cuda_setup();
        client.synchronize();
        let got = {
            let qt =
                QuantTensor::<CudaRuntime>::from_bytes(&bytes, QuantFormat::Q8_1, &[256], &device)
                    .unwrap_or_else(|e| {
                        panic!(
                            "Q8_1/cuda: device allocation failed: {e}. This is a FAILURE, not a \
                             skip — Q8_1 was NOT compared against llama.cpp."
                        )
                    });
            client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>()
        };
        client.synchronize();
        assert_round_trip_within("Q8_1", "cuda", &got, &source, 0.04);
    }

    #[test]
    fn q2_k_matches_llama_cpp_cuda() {
        if !numr::runtime::cuda::is_cuda_available() {
            loud_skip("Q2K");
            return;
        }
        let _lock = cuda_lock();
        let bytes = raw_blocks("q2_k_raw.bin");
        let reference = llama_cpp_reference("q2_k_ref.bin");
        let (client, device) = cuda_setup();
        client.synchronize();
        let got = {
            let qt =
                QuantTensor::<CudaRuntime>::from_bytes(&bytes, QuantFormat::Q2K, &[2048], &device)
                    .unwrap_or_else(|e| {
                        panic!(
                            "Q2K/cuda: device allocation failed: {e}. This is a FAILURE, not a \
                             skip — Q2K was NOT compared against llama.cpp."
                        )
                    });
            client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>()
        };
        client.synchronize();
        assert_matches_llama_cpp("Q2K", "cuda", &got, &reference);
    }

    #[test]
    fn q3_k_matches_llama_cpp_cuda() {
        if !numr::runtime::cuda::is_cuda_available() {
            loud_skip("Q3K");
            return;
        }
        let _lock = cuda_lock();
        let bytes = raw_blocks("q3_k_raw.bin");
        let reference = llama_cpp_reference("q3_k_ref.bin");
        let (client, device) = cuda_setup();
        client.synchronize();
        let got = {
            let qt =
                QuantTensor::<CudaRuntime>::from_bytes(&bytes, QuantFormat::Q3K, &[2048], &device)
                    .unwrap_or_else(|e| {
                        panic!(
                            "Q3K/cuda: device allocation failed: {e}. This is a FAILURE, not a \
                             skip — Q3K was NOT compared against llama.cpp."
                        )
                    });
            client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>()
        };
        client.synchronize();
        assert_matches_llama_cpp("Q3K", "cuda", &got, &reference);
    }

    #[test]
    fn q5_k_matches_llama_cpp_cuda() {
        if !numr::runtime::cuda::is_cuda_available() {
            loud_skip("Q5K");
            return;
        }
        let _lock = cuda_lock();
        let bytes = raw_blocks("q5_k_raw.bin");
        let reference = llama_cpp_reference("q5_k_ref.bin");
        let (client, device) = cuda_setup();
        client.synchronize();
        let got = {
            let qt =
                QuantTensor::<CudaRuntime>::from_bytes(&bytes, QuantFormat::Q5K, &[2048], &device)
                    .unwrap_or_else(|e| {
                        panic!(
                            "Q5K/cuda: device allocation failed: {e}. This is a FAILURE, not a \
                             skip — Q5K was NOT compared against llama.cpp."
                        )
                    });
            client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>()
        };
        client.synchronize();
        assert_matches_llama_cpp("Q5K", "cuda", &got, &reference);
    }

    #[test]
    fn q4_1_matches_llama_cpp_cuda() {
        if !numr::runtime::cuda::is_cuda_available() {
            loud_skip("Q4_1");
            return;
        }
        let _lock = cuda_lock();
        let bytes = raw_blocks("q4_1_raw.bin");
        let reference = llama_cpp_reference("q4_1_ref.bin");
        let (client, device) = cuda_setup();
        client.synchronize();
        let got = {
            let qt =
                QuantTensor::<CudaRuntime>::from_bytes(&bytes, QuantFormat::Q4_1, &[256], &device)
                    .unwrap_or_else(|e| {
                        panic!(
                            "Q4_1/cuda: device allocation failed: {e}. This is a FAILURE, not a \
                             skip — Q4_1 was NOT compared against llama.cpp."
                        )
                    });
            client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>()
        };
        client.synchronize();
        assert_matches_llama_cpp("Q4_1", "cuda", &got, &reference);
    }

    #[test]
    fn q5_1_matches_llama_cpp_cuda() {
        if !numr::runtime::cuda::is_cuda_available() {
            loud_skip("Q5_1");
            return;
        }
        let _lock = cuda_lock();
        let bytes = raw_blocks("q5_1_raw.bin");
        let reference = llama_cpp_reference("q5_1_ref.bin");
        let (client, device) = cuda_setup();
        client.synchronize();
        let got = {
            let qt =
                QuantTensor::<CudaRuntime>::from_bytes(&bytes, QuantFormat::Q5_1, &[256], &device)
                    .unwrap_or_else(|e| {
                        panic!(
                            "Q5_1/cuda: device allocation failed: {e}. This is a FAILURE, not a \
                             skip — Q5_1 was NOT compared against llama.cpp."
                        )
                    });
            client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>()
        };
        client.synchronize();
        assert_matches_llama_cpp("Q5_1", "cuda", &got, &reference);
    }

    #[test]
    fn tq1_0_matches_llama_cpp_cuda() {
        if !numr::runtime::cuda::is_cuda_available() {
            loud_skip("TQ1_0");
            return;
        }
        let _lock = cuda_lock();
        let bytes = raw_blocks("tq1_0_raw.bin");
        let reference = llama_cpp_reference("tq1_0_ref.bin");
        let (client, device) = cuda_setup();
        client.synchronize();
        let got = {
            let qt = QuantTensor::<CudaRuntime>::from_bytes(
                &bytes,
                QuantFormat::TQ1_0,
                &[2048],
                &device,
            )
            .unwrap_or_else(|e| {
                panic!(
                    "TQ1_0/cuda: device allocation failed: {e}. This is a FAILURE, not a \
                             skip — TQ1_0 was NOT compared against llama.cpp."
                )
            });
            client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>()
        };
        client.synchronize();
        assert_matches_llama_cpp("TQ1_0", "cuda", &got, &reference);
    }

    #[test]
    fn tq2_0_matches_llama_cpp_cuda() {
        if !numr::runtime::cuda::is_cuda_available() {
            loud_skip("TQ2_0");
            return;
        }
        let _lock = cuda_lock();
        let bytes = raw_blocks("tq2_0_raw.bin");
        let reference = llama_cpp_reference("tq2_0_ref.bin");
        let (client, device) = cuda_setup();
        client.synchronize();
        let got = {
            let qt = QuantTensor::<CudaRuntime>::from_bytes(
                &bytes,
                QuantFormat::TQ2_0,
                &[2048],
                &device,
            )
            .unwrap_or_else(|e| {
                panic!(
                    "TQ2_0/cuda: device allocation failed: {e}. This is a FAILURE, not a \
                             skip — TQ2_0 was NOT compared against llama.cpp."
                )
            });
            client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>()
        };
        client.synchronize();
        assert_matches_llama_cpp("TQ2_0", "cuda", &got, &reference);
    }

    #[test]
    fn iq4_nl_matches_llama_cpp_cuda() {
        if !numr::runtime::cuda::is_cuda_available() {
            loud_skip("IQ4NL");
            return;
        }
        let _lock = cuda_lock();
        let bytes = raw_blocks("iq4_nl_raw.bin");
        let reference = llama_cpp_reference("iq4_nl_ref.bin");
        let (client, device) = cuda_setup();
        client.synchronize();
        let got = {
            let qt =
                QuantTensor::<CudaRuntime>::from_bytes(&bytes, QuantFormat::IQ4NL, &[256], &device)
                    .unwrap_or_else(|e| {
                        panic!(
                            "IQ4NL/cuda: device allocation failed: {e}. This is a FAILURE, not a \
                             skip — IQ4NL was NOT compared against llama.cpp."
                        )
                    });
            client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>()
        };
        client.synchronize();
        assert_matches_llama_cpp("IQ4NL", "cuda", &got, &reference);
    }

    #[test]
    fn iq4_xs_matches_llama_cpp_cuda() {
        if !numr::runtime::cuda::is_cuda_available() {
            loud_skip("IQ4XS");
            return;
        }
        let _lock = cuda_lock();
        let bytes = raw_blocks("iq4_xs_raw.bin");
        let reference = llama_cpp_reference("iq4_xs_ref.bin");
        let (client, device) = cuda_setup();
        client.synchronize();
        let got = {
            let qt = QuantTensor::<CudaRuntime>::from_bytes(
                &bytes,
                QuantFormat::IQ4XS,
                &[2048],
                &device,
            )
            .unwrap_or_else(|e| {
                panic!(
                    "IQ4XS/cuda: device allocation failed: {e}. This is a FAILURE, not a \
                             skip — IQ4XS was NOT compared against llama.cpp."
                )
            });
            client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>()
        };
        client.synchronize();
        assert_matches_llama_cpp("IQ4XS", "cuda", &got, &reference);
    }

    #[test]
    fn iq3_s_matches_llama_cpp_cuda() {
        if !numr::runtime::cuda::is_cuda_available() {
            loud_skip("IQ3S");
            return;
        }
        let _lock = cuda_lock();
        let bytes = raw_blocks("iq3_s_raw.bin");
        let reference = llama_cpp_reference("iq3_s_ref.bin");
        let (client, device) = cuda_setup();
        client.synchronize();
        let got = {
            let qt =
                QuantTensor::<CudaRuntime>::from_bytes(&bytes, QuantFormat::IQ3S, &[2048], &device)
                    .unwrap_or_else(|e| {
                        panic!(
                            "IQ3S/cuda: device allocation failed: {e}. This is a FAILURE, not a \
                             skip — IQ3S was NOT compared against llama.cpp."
                        )
                    });
            client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>()
        };
        client.synchronize();
        assert_matches_llama_cpp("IQ3S", "cuda", &got, &reference);
    }

    #[test]
    fn iq3_xxs_matches_llama_cpp_cuda() {
        if !numr::runtime::cuda::is_cuda_available() {
            loud_skip("IQ3XXS");
            return;
        }
        let _lock = cuda_lock();
        let bytes = raw_blocks("iq3_xxs_raw.bin");
        let reference = llama_cpp_reference("iq3_xxs_ref.bin");
        let (client, device) = cuda_setup();
        client.synchronize();
        let got = {
            let qt = QuantTensor::<CudaRuntime>::from_bytes(
                &bytes,
                QuantFormat::IQ3XXS,
                &[2048],
                &device,
            )
            .unwrap_or_else(|e| {
                panic!(
                    "IQ3XXS/cuda: device allocation failed: {e}. This is a FAILURE, not a \
                             skip — IQ3XXS was NOT compared against llama.cpp."
                )
            });
            client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>()
        };
        client.synchronize();
        assert_matches_llama_cpp("IQ3XXS", "cuda", &got, &reference);
    }

    #[test]
    fn iq2_xxs_matches_llama_cpp_cuda() {
        if !numr::runtime::cuda::is_cuda_available() {
            loud_skip("IQ2XXS");
            return;
        }
        let _lock = cuda_lock();
        let bytes = raw_blocks("iq2_xxs_raw.bin");
        let reference = llama_cpp_reference("iq2_xxs_ref.bin");
        let (client, device) = cuda_setup();
        client.synchronize();
        let got = {
            let qt = QuantTensor::<CudaRuntime>::from_bytes(
                &bytes,
                QuantFormat::IQ2XXS,
                &[2048],
                &device,
            )
            .unwrap_or_else(|e| {
                panic!(
                    "IQ2XXS/cuda: device allocation failed: {e}. This is a FAILURE, not a \
                             skip — IQ2XXS was NOT compared against llama.cpp."
                )
            });
            client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>()
        };
        client.synchronize();
        assert_matches_llama_cpp("IQ2XXS", "cuda", &got, &reference);
    }

    #[test]
    fn iq2_xs_matches_llama_cpp_cuda() {
        if !numr::runtime::cuda::is_cuda_available() {
            loud_skip("IQ2XS");
            return;
        }
        let _lock = cuda_lock();
        let bytes = raw_blocks("iq2_xs_raw.bin");
        let reference = llama_cpp_reference("iq2_xs_ref.bin");
        let (client, device) = cuda_setup();
        client.synchronize();
        let got = {
            let qt = QuantTensor::<CudaRuntime>::from_bytes(
                &bytes,
                QuantFormat::IQ2XS,
                &[2048],
                &device,
            )
            .unwrap_or_else(|e| {
                panic!(
                    "IQ2XS/cuda: device allocation failed: {e}. This is a FAILURE, not a \
                             skip — IQ2XS was NOT compared against llama.cpp."
                )
            });
            client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>()
        };
        client.synchronize();
        assert_matches_llama_cpp("IQ2XS", "cuda", &got, &reference);
    }

    #[test]
    fn iq1_s_matches_llama_cpp_cuda() {
        if !numr::runtime::cuda::is_cuda_available() {
            loud_skip("IQ1S");
            return;
        }
        let _lock = cuda_lock();
        let bytes = raw_blocks("iq1_s_raw.bin");
        let reference = llama_cpp_reference("iq1_s_ref.bin");
        let (client, device) = cuda_setup();
        client.synchronize();
        let got = {
            let qt =
                QuantTensor::<CudaRuntime>::from_bytes(&bytes, QuantFormat::IQ1S, &[2048], &device)
                    .unwrap_or_else(|e| {
                        panic!(
                            "IQ1S/cuda: device allocation failed: {e}. This is a FAILURE, not a \
                             skip — IQ1S was NOT compared against llama.cpp."
                        )
                    });
            client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>()
        };
        client.synchronize();
        assert_matches_llama_cpp("IQ1S", "cuda", &got, &reference);
    }

    #[test]
    fn iq1_m_matches_llama_cpp_cuda() {
        if !numr::runtime::cuda::is_cuda_available() {
            loud_skip("IQ1M");
            return;
        }
        let _lock = cuda_lock();
        let bytes = raw_blocks("iq1_m_raw.bin");
        let reference = llama_cpp_reference("iq1_m_ref.bin");
        let (client, device) = cuda_setup();
        client.synchronize();
        let got = {
            let qt =
                QuantTensor::<CudaRuntime>::from_bytes(&bytes, QuantFormat::IQ1M, &[2048], &device)
                    .unwrap_or_else(|e| {
                        panic!(
                            "IQ1M/cuda: device allocation failed: {e}. This is a FAILURE, not a \
                             skip — IQ1M was NOT compared against llama.cpp."
                        )
                    });
            client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>()
        };
        client.synchronize();
        assert_matches_llama_cpp("IQ1M", "cuda", &got, &reference);
    }

    #[test]
    fn iq2_s_matches_llama_cpp_cuda() {
        if !numr::runtime::cuda::is_cuda_available() {
            loud_skip("IQ2S");
            return;
        }
        let _lock = cuda_lock();
        let bytes = raw_blocks("iq2_s_raw.bin");
        let reference = llama_cpp_reference("iq2_s_ref.bin");
        let (client, device) = cuda_setup();
        client.synchronize();
        let got = {
            let qt =
                QuantTensor::<CudaRuntime>::from_bytes(&bytes, QuantFormat::IQ2S, &[2048], &device)
                    .unwrap_or_else(|e| {
                        panic!(
                            "IQ2S/cuda: device allocation failed: {e}. This is a FAILURE, not a \
                             skip — IQ2S was NOT compared against llama.cpp."
                        )
                    });
            client.dequantize(&qt, DType::F32).unwrap().to_vec::<f32>()
        };
        client.synchronize();
        assert_matches_llama_cpp("IQ2S", "cuda", &got, &reference);
    }
}
