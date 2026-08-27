//! Compare every tensor in a GGUF file against the safetensors checkpoint it
//! was converted from, and report the dequantization error per tensor.
//!
//! ```text
//! cargo run --release --features audio,f16 --example gguf_diff -- \
//!     MODEL.gguf REFERENCE.safetensors [--top 20] [--limit N] [--names auto]
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
//! # Two naming conventions, so iterate the REFERENCE
//!
//! The comparison walks the REFERENCE checkpoint's tensor names, not the
//! GGUF's, and translates each one forward. That reports coverage of what the
//! model actually needs: a tensor the GGUF is missing is a load failure
//! waiting to happen, while an extra tensor in the GGUF is harmless.
//!
//! `--names` selects the translation. `verbatim` expects the GGUF to carry the
//! checkpoint's own HuggingFace names, which is what `compressr convert`
//! writes. `ggml` expects llama.cpp-conventional names (`tslm.blk.0.attn_q.weight`),
//! which is what third-party VoxCPM2 builds carry. `auto`, the default, probes
//! for a sentinel tensor — `general.architecture` cannot decide it, because
//! both conventions declare `voxcpm2`.
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
use boostr::model::audio::voxcpm::loader::cstr::{GGML_SENTINEL, hf_to_ggml_name};
use boostr::{CpuDevice, CpuRuntime, DType};

const USAGE: &str = "usage: gguf_diff MODEL.gguf REFERENCE.safetensors \
[--top 20] [--limit N] [--names auto|verbatim|ggml]";

/// How a GGUF spells the checkpoint's tensor names.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Naming {
    /// The checkpoint's own HuggingFace names, as `compressr convert` writes.
    Verbatim,
    /// llama.cpp-conventional names, as third-party VoxCPM2 builds carry.
    Ggml,
}

impl Naming {
    /// Translate a reference (HuggingFace) name into this convention.
    ///
    /// `None` means the map does not know the name, which is a gap in the map
    /// rather than a fault in the file — reported separately from a tensor the
    /// GGUF genuinely lacks.
    fn lookup(self, hf_name: &str) -> Option<String> {
        match self {
            Self::Verbatim => Some(hf_name.to_string()),
            Self::Ggml => hf_to_ggml_name(hf_name),
        }
    }
}

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

/// True when `stored` is `expected` with its LEADING unit dims dropped.
///
/// Writers differ on whether a `[1, 1, 1, N]` tensor keeps its unit dims, and
/// dropping them changes no element. Deliberately narrow: any other same-numel
/// pair is a genuine layout difference and must still be reported, since a
/// transposed or mis-strided tensor also preserves the element count.
fn squeezed_unit_dims(stored: &[usize], expected: &[usize]) -> bool {
    let trimmed: Vec<usize> = expected.iter().copied().skip_while(|&d| d == 1).collect();
    stored == trimmed.as_slice()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut positional = Vec::new();
    let mut top = 20usize;
    let mut limit = usize::MAX;
    let mut naming: Option<Naming> = None;

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
            "--names" => {
                i += 1;
                naming = match argv.get(i).ok_or("--names needs a value")?.as_str() {
                    "auto" => None,
                    "verbatim" => Some(Naming::Verbatim),
                    "ggml" => Some(Naming::Ggml),
                    other => {
                        return Err(
                            format!("--names: expected auto|verbatim|ggml, got {other:?}").into(),
                        );
                    }
                };
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

    // `general.architecture` cannot decide this: both conventions declare
    // `voxcpm2`. Probe for a name only one of them can have.
    let naming = match naming {
        Some(explicit) => explicit,
        None if gguf.tensor_info(GGML_SENTINEL).is_ok() => Naming::Ggml,
        None => Naming::Verbatim,
    };

    let mut names = st.tensor_names();
    names.truncate(limit);
    // Deterministic order so two runs of the same pair report identically.
    names.sort();
    eprintln!(
        "comparing {} reference tensors, {naming:?} naming ...",
        names.len()
    );

    let mut diffs = Vec::with_capacity(names.len());
    let mut missing = Vec::new();
    let mut unmapped = Vec::new();
    let mut nonfinite = Vec::new();
    let mut shape_mismatch = Vec::new();
    let mut reshaped = Vec::new();

    for name in &names {
        let Some(gguf_name) = naming.lookup(name) else {
            unmapped.push(name.clone());
            continue;
        };
        let want = st
            .load_tensor::<CpuRuntime>(name, &device)?
            .to_dtype(DType::F32)?;
        let got = match gguf.load_tensor_f32::<CpuRuntime>(&gguf_name, &device) {
            Ok(t) => t,
            Err(_) => {
                missing.push(format!("{name} (looked up as {gguf_name})"));
                continue;
            }
        };
        // A GGUF may squeeze the leading unit dims a checkpoint spells out.
        // That is the same tensor, so reshape and keep comparing the VALUES —
        // reporting it as a mismatch and skipping would cry wolf on a file
        // that is actually correct, and would leave its data unchecked.
        let got = if got.shape() != want.shape() && squeezed_unit_dims(got.shape(), want.shape()) {
            reshaped.push(format!(
                "{name}: gguf {:?} -> reference {:?}",
                got.shape(),
                want.shape()
            ));
            got.reshape(want.shape())?
        } else {
            got
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

    if !reshaped.is_empty() {
        println!(
            "\n{} tensors reshaped (leading unit dims the GGUF squeezed), values still compared:",
            reshaped.len()
        );
        for line in reshaped.iter().take(top) {
            println!("  {line}");
        }
    }

    let mut ok = true;
    if !unmapped.is_empty() {
        println!(
            "\n{} reference tensors the name map does not know:",
            unmapped.len()
        );
        for name in unmapped.iter().take(top) {
            println!("  {name}");
        }
        ok = false;
    }
    if !missing.is_empty() {
        println!(
            "\n{} reference tensors absent from the GGUF:",
            missing.len()
        );
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
