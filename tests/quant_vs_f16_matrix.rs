//! Ground-truth matrix: every quantization of ONE model vs its f16 build.
//!
//! Needs a directory holding several GGUF builds of the same model, one of
//! which is f16 or f32. Without one the test skips loudly and reports nothing.
//!
//! ```bash
//! BOOSTR_QUANT_MATRIX_DIR=/path/to/models BOOSTR_QUANT_MATRIX_STEM=nomic-embed-text-v1.5 \
//!   cargo nextest run -p boostr --test quant_vs_f16_matrix --no-capture
//! ```
//!
//! `BOOSTR_MODELS_DIR` serves as the fallback when the specific variable is
//! unset. `BOOSTR_QUANT_MATRIX_STEM` defaults to `nomic-embed-text-v1.5`.
//!
//! # Why this exists
//!
//! A dequantization kernel that misreads its block layout does NOT fail loudly.
//! Tensor shape, block count and even RMS all stay correct, so the model loads,
//! runs, and merely produces wrong numbers. Retrieval-level metrics catch it
//! only indirectly and can be confounded by genuine quantization loss.
//!
//! Comparing each quantized tensor against the SAME tensor from an f16 build
//! separates the two cleanly, because the two failure modes live in completely
//! different ranges:
//!
//! - correct dequant, real quantization loss → cosine ≳ 0.95
//!   (measured: Q8_0/Q6_K/Q5_K ≈ 0.999, Q4_K ≈ 0.997, Q3_K ≈ 0.988, Q2_K ≈ 0.957)
//! - mis-indexed block layout → cosine ≈ 0.03
//!   (measured, before the fix: Q5_K scored 0.0307 while Q4_K/Q6_K in the SAME
//!   file scored 0.997+ — which is how the bug was localised)
//!
//! The 0.90 floor sits in the wide empty gap between those, so it holds for
//! every format down to 2-bit without being loose enough to pass a layout bug.

mod common;

use boostr::format::gguf::Gguf;
use common::model_fixture;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};

/// Tensors compared per model. Chosen to span the formats a mixed-precision
/// build actually uses: a `_K_M` file assigns different formats to the
/// embedding table, the attention projections and the FFN, so checking only one
/// would leave whole kernels unverified. (That is exactly what hid the Q5_K bug
/// at first: the fused QKV was the only Q5_K tensor in the file.)
const TENSORS: &[&str] = &[
    "token_embd.weight",
    "blk.0.attn_qkv.weight",
    "blk.0.ffn_down.weight",
    "blk.0.attn_output.weight",
];

/// Cosine below this means the layout is wrong, not merely lossy. See module docs.
const LAYOUT_FLOOR: f64 = 0.90;

fn load(path: &std::path::Path, name: &str, device: &CpuDevice) -> Option<Vec<f32>> {
    let mut g = Gguf::open(path).ok()?;
    g.tensor_info(name).ok()?;
    Some(
        g.load_tensor_f32_streaming::<CpuRuntime>(name, device)
            .ok()?
            .to_vec(),
    )
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum();
    let na: f64 = a
        .iter()
        .map(|x| (*x as f64) * (*x as f64))
        .sum::<f64>()
        .sqrt();
    let nb: f64 = b
        .iter()
        .map(|x| (*x as f64) * (*x as f64))
        .sum::<f64>()
        .sqrt();
    if na == 0.0 || nb == 0.0 {
        return f64::NAN;
    }
    dot / (na * nb)
}

/// Skips loudly when no model directory is configured, in the style of the
/// CUDA-gated tests. `#[ignore]` would hide it from every default run instead,
/// which for a layout gate means it reports nothing on the machines that DO
/// have the models.
#[test]
fn every_quantization_matches_the_f16_reference() {
    let Some(dir) = model_fixture("BOOSTR_QUANT_MATRIX_DIR", "") else {
        common::skip_notice("quant matrix models dir", "BOOSTR_QUANT_MATRIX_DIR");
        return;
    };
    let stem = std::env::var("BOOSTR_QUANT_MATRIX_STEM")
        .unwrap_or_else(|_| "nomic-embed-text-v1.5".to_owned());
    let dir = dir.as_path();

    let mut builds: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            let n = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
            n.starts_with(&stem) && n.ends_with(".gguf")
        })
        .collect();
    builds.sort();
    assert!(
        builds.len() >= 2,
        "need an f16 reference plus at least one quantization matching '{stem}*' in {}",
        dir.display()
    );

    let reference = builds
        .iter()
        .find(|p| {
            let n = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
            n.contains(".f16.") || n.contains(".f32.")
        })
        .unwrap_or_else(|| panic!("no f16/f32 reference build found for '{stem}'"))
        .clone();

    let device = CpuDevice::new();
    let mut failures: Vec<String> = Vec::new();

    eprintln!();
    eprintln!(
        "  reference: {}",
        reference.file_name().unwrap().to_string_lossy()
    );
    eprintln!("  {:<38} {:>22} {:>10}", "BUILD", "TENSOR", "COSINE");
    eprintln!("  {}", "-".repeat(74));

    for build in &builds {
        if build == &reference {
            continue;
        }
        let label = build.file_name().unwrap().to_string_lossy().to_string();
        for name in TENSORS {
            let (Some(a), Some(b)) = (load(&reference, name, &device), load(build, name, &device))
            else {
                continue;
            };
            if a.len() != b.len() {
                failures.push(format!("{label}/{name}: element count differs"));
                continue;
            }
            let c = cosine(&a, &b);
            eprintln!("  {label:<38} {name:>22} {c:>10.6}");
            // NaN counts as a failure too — it means one side had zero norm,
            // i.e. a tensor that dequantized to all zeros.
            if c.is_nan() || c <= LAYOUT_FLOOR {
                failures.push(format!(
                    "{label}/{name}: cosine {c:.6} vs the f16 reference is below {LAYOUT_FLOOR} — \
                     that is a mis-indexed block layout, not quantization loss"
                ));
            }
        }
    }
    eprintln!();

    assert!(
        failures.is_empty(),
        "quantized weights do not match the f16 reference:\n  - {}",
        failures.join("\n  - ")
    );
}
