//! `DequantOps::dequantize` against llama.cpp's OWN reference implementation,
//! on verbatim block bytes lifted out of real GGUF model files.
//!
//! # Why this file exists
//!
//! `CLAUDE.md` records the defect this guards against: "Q6_K once shipped with
//! the wrong field order and compressr's own reader agreed with it while every
//! real reader produced NaN." A writer checked against the reader beside it
//! proves nothing, and neither does a reader checked against a reference that
//! restates the reader's own decode.
//!
//! Every fixture here therefore carries an EXTERNAL expectation. The `*_ref.bin`
//! files were produced by `gguf.quants.dequantize` from the `gguf` Python
//! package — llama.cpp's own reference implementation, maintained in the
//! llama.cpp tree. No boostr code took part in producing them. That
//! independence is the entire value of this file.
//!
//! Do NOT "simplify" this into a comparison against boostr's own writer,
//! against `QuantTensor` round-tripping, or against a Rust reimplementation of
//! the block layout. Any of those turns the gate back into the circular check
//! CLAUDE.md warns about, and it will pass while the decode is wrong.
//!
//! # Relationship to `tests/gguf_dequant_cpu_cuda_parity.rs`
//!
//! The two files gate different things and neither subsumes the other:
//!
//! - `gguf_dequant_cpu_cuda_parity.rs` proves CPU and CUDA agree with EACH
//!   OTHER, across all 23 `QuantFormat` variants, on synthetic block bytes.
//! - THIS file proves both backends agree with LLAMA.CPP, for the five formats
//!   real model files were available for, on real model bytes.
//!
//! Agreeing with each other while both being wrong is exactly the failure mode
//! CLAUDE.md describes. Agreeing with llama.cpp on five formats says nothing
//! about the other eighteen. Both gates are required.
//!
//! The CUDA half here is the one that matters most in practice: CUDA decoded
//! 10 of 23 formats wrong until commit `202d6aa`, and a wrong decode errors
//! nothing — shape, block count and tensor RMS all stay plausible while the
//! model silently produces wrong numbers.
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
//! A new format needs the two files plus one `_cpu` and one `_cuda` test below,
//! written out explicitly like the existing ten.
//!
//! # Comparison is EXACT
//!
//! There is no tolerance. Measured on CPU, all five formats match the llama.cpp
//! reference bit-for-bit: zero mismatching elements, `max_abs = 0.0`. A future
//! change that makes any of them merely approximate is a change worth seeing,
//! not one worth absorbing into an epsilon. If CUDA ever differs by an ulp or
//! two because nvcc contracts a scale multiply into an FMA, report the exact
//! indices and magnitudes rather than widening the check here.
//!
//! # Formats NOT covered
//!
//! Only five of the 23 `QuantFormat` variants are gated against llama.cpp,
//! because only these five appear in GGUF model files on hand. Uncovered, and
//! therefore resting solely on the CPU/CUDA mutual-agreement gate: Q4_1, Q5_1,
//! Q8_1, Q2K, Q3K, Q5K, Q8K, IQ1S, IQ1M, IQ2XXS, IQ2XS, IQ2S, IQ3XXS, IQ3S,
//! IQ4NL, IQ4XS, TQ1_0, TQ2_0. That is a real gap, not a closed one — closing
//! it means obtaining a model file carrying the format and following the
//! regeneration recipe above.
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
/// On failure it names the first mismatching indices with BOTH values. The
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

/// Q6_K is the format `CLAUDE.md` names by way of warning: it once shipped with
/// the wrong field order and passed every check that used boostr's own reader.
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

// ---------------------------------------------------------------------------
// CUDA
//
// Same fixtures, same exact comparison, against the CUDA dequant kernels. This
// is the half that makes the pre-`202d6aa` class of defect — a CUDA kernel
// decoding a format wrong while erroring nothing — impossible to reintroduce
// silently for these five formats.
//
// A machine without CUDA skips loudly, in the style of
// `tests/gguf_dequant_cpu_cuda_parity.rs`: a silent skip is how the nibble-order
// defect survived. An allocation error is a FAILURE, never a skip.
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
mod cuda {
    use super::{assert_matches_llama_cpp, llama_cpp_reference, raw_blocks};

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
}
