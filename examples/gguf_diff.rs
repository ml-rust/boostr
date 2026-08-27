//! Compare every tensor in a GGUF file against the safetensors checkpoint it
//! was converted from, and report the dequantization error per tensor.
//!
//! ```text
//! cargo run --release --features audio,f16 --example gguf_diff -- \
//!     MODEL.gguf REFERENCE.safetensors [--top 20] [--limit N]
//! ```
//!
//! # Why a whole-file check, when the converter has unit tests
//!
//! A quantizer's unit test proves its packing round-trips through a reader in
//! isolation. It cannot prove that the FILE is right: tensor offsets, the
//! header's reserved byte counts, shape order, and the mapping from name to
//! data all live in the writer, not the kernel. A single tensor registered
//! with a byte count that disagrees with the bytes actually written shifts
//! every tensor after it, and the file still parses.
//!
//! This ran against a real conversion and caught a Q6_K writer that emitted
//! GGML's `block_q6_K` fields in the wrong order: every value dequantized to
//! NaN while the file itself remained structurally valid.
//!
//! # Reading the output
//!
//! Error is reported as relative RMS — RMS of the difference over RMS of the
//! reference — because an absolute number means nothing without knowing the
//! tensor's scale. Expect roughly the theoretical floor for the bit width:
//! uniform quantization over a group with absmax scaling costs about
//! `step / sqrt(12)`, which lands near 0.5% for 8-bit and 8% for 4-bit. A
//! tensor stored unquantized reads 0. NaN, or a figure in the tens of
//! percent for an 8-bit format, means a layout error, not rounding.
use std::collections::BTreeMap;

use boostr::format::gguf::Gguf;
use boostr::format::safetensors_loader::SafeTensorsLoader;
use boostr::{CpuDevice, CpuRuntime, DType};

const USAGE: &str = "usage: gguf_diff MODEL.gguf REFERENCE.safetensors [--top 20] [--limit N]";

/// One tensor's comparison result.
struct Diff {
    name: String,
    rel_rms: f64,
    elems: usize,
}

/// Relative RMS of `got` against `want`, plus whether every value is finite.
///
/// Accumulates in f64: an f32 sum over 150 M elements loses the tail entirely.
fn relative_rms(got: &[f32], want: &[f32]) -> (f64, bool) {
    let mut ref_sq = 0.0f64;
    let mut err_sq = 0.0f64;
    let mut finite = true;
    for (&a, &b) in got.iter().zip(want) {
        finite &= a.is_finite();
        let b = f64::from(b);
        let d = f64::from(a) - b;
        ref_sq += b * b;
        err_sq += d * d;
    }
    if ref_sq == 0.0 {
        // An all-zero reference has no scale to be relative to. Report the
        // absolute error instead of dividing by zero.
        return (err_sq.sqrt(), finite);
    }
    ((err_sq / ref_sq).sqrt(), finite)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut positional = Vec::new();
    let mut top = 20usize;
    let mut limit = usize::MAX;

    let mut i = 0usize;
    while i < argv.len() {
        match argv[i].as_str() {
            "--top" => {
                i += 1;
                top = argv.get(i).ok_or("--top needs a value")?.parse()?;
            }
            "--limit" => {
                i += 1;
                limit = argv.get(i).ok_or("--limit needs a value")?.parse()?;
            }
            "-h" | "--help" => {
                eprintln!("{USAGE}");
                return Ok(());
            }
            other => positional.push(other.to_string()),
        }
        i += 1;
    }
    let [gguf_path, st_path] = positional.as_slice() else {
        eprintln!("{USAGE}");
        std::process::exit(2);
    };

    let device = CpuDevice::default();
    let mut gguf = Gguf::open(gguf_path)?;
    let mut st = SafeTensorsLoader::open(st_path)?;

    let names: Vec<String> = gguf
        .tensor_names()
        .take(limit)
        .map(str::to_string)
        .collect();
    eprintln!("comparing {} tensors ...", names.len());

    let mut diffs = Vec::with_capacity(names.len());
    let mut missing = Vec::new();
    let mut nonfinite = Vec::new();
    let mut shape_mismatch = Vec::new();

    for name in &names {
        let got = gguf.load_tensor_f32::<CpuRuntime>(name, &device)?;
        let want = match st.load_tensor::<CpuRuntime>(name, &device) {
            Ok(t) => t.to_dtype(DType::F32)?,
            Err(_) => {
                missing.push(name.clone());
                continue;
            }
        };
        if got.shape() != want.shape() {
            shape_mismatch.push(format!(
                "{name}: gguf {:?} vs reference {:?}",
                got.shape(),
                want.shape()
            ));
            continue;
        }

        let gv: Vec<f32> = got.contiguous()?.to_vec();
        let wv: Vec<f32> = want.contiguous()?.to_vec();
        let (rel_rms, finite) = relative_rms(&gv, &wv);
        if !finite {
            nonfinite.push(name.clone());
        }
        diffs.push(Diff {
            name: name.clone(),
            rel_rms,
            elems: wv.len(),
        });
    }

    // Group by the exact error figure: every tensor quantized the same way
    // lands on the same floor, so a single outlier stands out immediately.
    let mut buckets: BTreeMap<String, usize> = BTreeMap::new();
    for d in &diffs {
        let key = if d.rel_rms == 0.0 {
            "exact (stored unquantized)".to_string()
        } else if !d.rel_rms.is_finite() {
            "NON-FINITE".to_string()
        } else {
            format!("{:.1}%", 100.0 * d.rel_rms)
        };
        *buckets.entry(key).or_default() += 1;
    }

    println!("\nrelative RMS distribution:");
    for (bucket, count) in &buckets {
        println!("  {bucket:>28}  {count} tensors");
    }

    diffs.sort_by(|a, b| {
        b.rel_rms
            .partial_cmp(&a.rel_rms)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    println!("\nworst {top}:");
    for d in diffs.iter().take(top) {
        println!(
            "  {:<52} {:>8.4}%  ({} elems)",
            d.name,
            100.0 * d.rel_rms,
            d.elems
        );
    }

    let mut ok = true;
    if !missing.is_empty() {
        println!("\n{} tensors absent from the reference:", missing.len());
        for name in missing.iter().take(top) {
            println!("  {name}");
        }
        ok = false;
    }
    if !shape_mismatch.is_empty() {
        println!("\n{} shape mismatches:", shape_mismatch.len());
        for line in shape_mismatch.iter().take(top) {
            println!("  {line}");
        }
        ok = false;
    }
    if !nonfinite.is_empty() {
        println!("\n{} tensors contain NaN or infinity:", nonfinite.len());
        for name in nonfinite.iter().take(top) {
            println!("  {name}");
        }
        ok = false;
    }

    println!("\n{}", if ok { "VERIFIED" } else { "FAILED" });
    std::process::exit(if ok { 0 } else { 1 });
}
