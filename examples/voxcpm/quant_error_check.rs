//! Measure how faithfully TCF and GGUF reconstruct the same BF16 checkpoint.
//!
//! ```text
//! cargo run --release --features audio,f16 --example voxcpm_quant_error_check -- \
//!     SOURCE_DIR MODEL.gguf MODEL.tcf [--top 10]
//! ```
//!
//! `SOURCE_DIR` is the original checkpoint directory: BF16 safetensors plus
//! `config.json`. `MODEL.gguf` and `MODEL.tcf` are two quantizations of that
//! checkpoint.
//!
//! # What this answers
//!
//! Which encoding reconstructs the source weights more faithfully. Both files
//! dequantize to f32, both compare against the same BF16 reference, and the
//! reference never moves. No model runs, no kernel runs, no audio is decoded.
//!
//! This measures error only. It does NOT measure speed: TCF has no kernels on
//! any backend today, so a timing number here would compare a reference
//! decoder against a shipped one and mean nothing.
//!
//! # Fair comparison, or no comparison
//!
//! A tensor counts only when BOTH files store it quantized. compressr keeps
//! rank < 2 tensors raw in TCF, and a GGUF keeps its own set at full
//! precision. Comparing a raw tensor in one file against a quantized tensor in
//! the other would score the raw side at zero error and make its format look
//! artificially good. Those tensors are counted and reported as "not
//! comparable", never folded into the aggregate.
//!
//! GGUF quantization is `GgmlType::is_quantized`. TCF quantization is a native
//! encoding, which is exactly the set for which `TcfTensorInfo`'s
//! `bits_per_weight` is `Some` — a raw encoding stores literal values and has
//! no such width.
//!
//! # Three metrics, three questions
//!
//! - Absolute RMS error says how far the values moved.
//! - Relative RMS — RMS error over the reference's own RMS — is the number
//!   that compares across tensors. Absolute RMS is dominated by whichever
//!   tensor has the largest magnitude.
//! - Max absolute error, with its element index, catches a single blown value
//!   that RMS averages away.
//!
//! The aggregate is the element-count-weighted mean relative RMS, so a
//! 100 M-element projection outweighs a 2 k-element bias.
//!
//! # Exit status
//!
//! This example REPORTS a measurement. A large error is a result, not a
//! failure, and there is no established threshold to gate on. Exit is
//! non-zero only for a real fault: a file that will not open, zero comparable
//! tensors, or a name-matching collapse.
use std::collections::HashMap;

use boostr::format::gguf::name_map::gguf_to_hf_name;
use boostr::format::safetensors_name_map::normalize_hf_name;
use boostr::format::tcf::{TcfLoader, encoding_name};
use boostr::format::{Gguf, SafeTensorsLoader};
use boostr::{CpuDevice, CpuRuntime, DType};

const USAGE: &str = "usage: voxcpm_quant_error_check SOURCE_DIR MODEL.gguf MODEL.tcf [--top 10]";

/// One quantized form's error against the BF16 reference.
struct Errors {
    /// RMS of the difference, in the tensor's own units.
    rms: f64,
    /// RMS of the difference over the RMS of the reference.
    rel_rms: f64,
    /// Largest single-element absolute difference.
    max_abs: f64,
    /// Row-major index where `max_abs` occurs.
    max_idx: usize,
    /// False when any dequantized value is NaN or infinite.
    finite: bool,
}

/// One tensor compared in both formats.
struct Row {
    name: String,
    elems: usize,
    gguf: Errors,
    tcf: Errors,
    /// The GGUF type and TCF encoding that produced these numbers.
    encodings: String,
}

/// Error of `got` against reference `want`, accumulated in f64.
///
/// An f32 sum over 100 M elements drops the tail entirely, so every
/// accumulator is f64 regardless of the input dtype.
///
/// A reference whose RMS is zero has no scale to be relative to. `rel_rms`
/// then carries the absolute RMS instead of dividing by zero.
fn errors(got: &[f32], want: &[f32]) -> Errors {
    let mut ref_sq = 0.0f64;
    let mut err_sq = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut max_idx = 0usize;
    let mut finite = true;
    for (i, (&a, &b)) in got.iter().zip(want).enumerate() {
        finite &= a.is_finite();
        let b = f64::from(b);
        let d = f64::from(a) - b;
        ref_sq += b * b;
        err_sq += d * d;
        let abs = d.abs();
        if abs > max_abs {
            max_abs = abs;
            max_idx = i;
        }
    }
    let n = got.len().max(1) as f64;
    let rms = (err_sq / n).sqrt();
    let ref_rms = (ref_sq / n).sqrt();
    let rel_rms = if ref_rms == 0.0 { rms } else { rms / ref_rms };
    Errors {
        rms,
        rel_rms,
        max_abs,
        max_idx,
        finite,
    }
}

/// The checkpoint's `model_type`, which selects the name normalization.
///
/// An absent or unreadable `config.json` yields an empty string. That is the
/// passthrough case in `normalize_hf_name`, which is correct for every
/// Llama-family checkpoint.
fn model_type(source_dir: &std::path::Path) -> String {
    let Ok(text) = std::fs::read_to_string(source_dir.join("config.json")) else {
        return String::new();
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
        return String::new();
    };
    value
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string()
}

/// Index `entries` by canonical name, dropping every name that collides.
///
/// A collision means two stored tensors normalize to one canonical name. The
/// comparison cannot tell which one the reference means, so both leave the
/// run and the count is reported.
fn index_by_canonical<T>(entries: Vec<(String, T)>) -> (HashMap<String, T>, Vec<String>) {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for (canonical, _) in &entries {
        *counts.entry(canonical.clone()).or_default() += 1;
    }
    let mut map = HashMap::new();
    let mut collided = Vec::new();
    for (canonical, value) in entries {
        if counts.get(&canonical).copied().unwrap_or(0) > 1 {
            collided.push(canonical);
            continue;
        }
        map.insert(canonical, value);
    }
    collided.sort();
    collided.dedup();
    (map, collided)
}

/// Print one format's three metrics on a single line.
fn print_errors(label: &str, e: &Errors) {
    println!(
        "      {label:<5} rel RMS {:>8.4}%  RMS {:.4e}  max abs {:.4e} @ {}{}",
        100.0 * e.rel_rms,
        e.rms,
        e.max_abs,
        e.max_idx,
        if e.finite { "" } else { "  NON-FINITE" }
    );
}

/// The row with the largest `pick`, as (name, value).
///
/// `None` only when `rows` is empty, which the caller rules out first.
fn worst_by(rows: &[Row], pick: fn(&Row) -> f64) -> Option<(&str, f64)> {
    rows.iter()
        .map(|r| (r.name.as_str(), pick(r)))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut positional: Vec<String> = Vec::new();
    let mut top = 10usize;
    let mut i = 0usize;
    while i < argv.len() {
        match argv[i].as_str() {
            "--top" => {
                i += 1;
                top = argv.get(i).ok_or("--top needs a value")?.parse()?;
            }
            "-h" | "--help" => {
                eprintln!("{USAGE}");
                return Ok(());
            }
            other => positional.push(other.to_string()),
        }
        i += 1;
    }
    let [source_dir, gguf_path, tcf_path] = positional.as_slice() else {
        eprintln!("{USAGE}");
        std::process::exit(2);
    };
    let source_dir = std::path::PathBuf::from(source_dir);

    let device = CpuDevice::default();
    let model_type = model_type(&source_dir);
    println!("source {}  model_type {model_type:?}", source_dir.display());

    let mut reference = SafeTensorsLoader::open(&source_dir)?;
    let mut gguf = Gguf::open(gguf_path)?;
    let tcf = TcfLoader::open(tcf_path)?;
    println!(
        "opened: reference {} tensors ({} shards), gguf {} tensors, tcf {} tensors",
        reference.tensor_names().len(),
        reference.num_shards(),
        gguf.len(),
        tcf.len()
    );

    // Both quantized files are indexed by the reference's own naming, so the
    // walk below translates each stored name once instead of guessing per
    // lookup. GGUF carries llama.cpp-conventional names, so it needs
    // `gguf_to_hf_name` first; TCF carries the checkpoint's own names.
    let gguf_entries: Vec<(String, (String, boostr::format::GgmlType))> = gguf
        .tensor_names()
        .map(str::to_string)
        .collect::<Vec<_>>()
        .into_iter()
        .filter_map(|stored| {
            let ty = gguf.tensor_info(&stored).ok()?.ggml_type;
            let canonical = normalize_hf_name(&model_type, &gguf_to_hf_name(&stored));
            Some((canonical, (stored, ty)))
        })
        .collect();
    let (gguf_index, gguf_collided) = index_by_canonical(gguf_entries);

    let tcf_entries: Vec<(String, (String, bool, String))> = tcf
        .tensors()
        .iter()
        .map(|info| {
            let canonical = normalize_hf_name(&model_type, &info.name);
            let quantized = info.bits_per_weight().is_some();
            (
                canonical,
                (info.name.clone(), quantized, encoding_name(info.encoding())),
            )
        })
        .collect();
    let (tcf_index, tcf_collided) = index_by_canonical(tcf_entries);

    let mut names = reference.tensor_names();
    names.sort();

    let mut rows: Vec<Row> = Vec::new();
    let mut unmatched: Vec<String> = Vec::new();
    let mut not_comparable: Vec<String> = Vec::new();
    let mut shape_mismatch: Vec<String> = Vec::new();
    let mut reshaped: Vec<String> = Vec::new();

    for name in &names {
        let canonical = normalize_hf_name(&model_type, name);
        let (Some((gguf_name, ggml_type)), Some((tcf_name, tcf_quantized, tcf_encoding))) =
            (gguf_index.get(&canonical), tcf_index.get(&canonical))
        else {
            unmatched.push(canonical);
            continue;
        };
        let gguf_quantized = ggml_type.is_quantized();
        if !gguf_quantized || !*tcf_quantized {
            not_comparable.push(format!(
                "{canonical}: gguf {ggml_type:?}{}, tcf {tcf_encoding}{}",
                if gguf_quantized { "" } else { " (raw)" },
                if *tcf_quantized { "" } else { " (raw)" }
            ));
            continue;
        }

        let want_tensor = reference
            .load_tensor::<CpuRuntime>(name, &device)?
            .to_dtype(DType::F32)?;
        let want: Vec<f32> = want_tensor.contiguous()?.to_vec();

        let got_gguf_tensor = gguf.load_tensor_f32::<CpuRuntime>(gguf_name, &device)?;
        let gguf_shape = got_gguf_tensor.shape().to_vec();
        let got_gguf: Vec<f32> = got_gguf_tensor.contiguous()?.to_vec();
        let got_tcf = tcf.load_tensor_f32(tcf_name)?;

        // Element count decides comparability. Writers disagree on whether a
        // tensor keeps its leading unit dims, and dropping them moves no
        // value, so a differing shape at equal count is reported and the
        // values are still compared.
        if got_gguf.len() != want.len() || got_tcf.len() != want.len() {
            shape_mismatch.push(format!(
                "{canonical}: reference {:?}, gguf {gguf_shape:?}, tcf {} elems",
                want_tensor.shape(),
                got_tcf.len()
            ));
            continue;
        }
        if gguf_shape != want_tensor.shape() {
            reshaped.push(format!(
                "{canonical}: gguf {gguf_shape:?} vs reference {:?}",
                want_tensor.shape()
            ));
        }

        rows.push(Row {
            name: canonical,
            elems: want.len(),
            gguf: errors(&got_gguf, &want),
            tcf: errors(&got_tcf, &want),
            encodings: format!("gguf {ggml_type:?} / tcf {tcf_encoding}"),
        });
    }

    println!("\ncompared {} tensors quantized in both files", rows.len());
    println!(
        "  {} not comparable (raw in one file)",
        not_comparable.len()
    );
    println!("  {} unmatched by name", unmatched.len());
    println!(
        "  {} name collisions (gguf {}, tcf {})",
        gguf_collided.len() + tcf_collided.len(),
        gguf_collided.len(),
        tcf_collided.len()
    );
    println!("  {} shape mismatches", shape_mismatch.len());

    if rows.is_empty() {
        println!("\nFAILED: no tensor is quantized in both files, so nothing is comparable");
        std::process::exit(1);
    }
    // A handful of unmatched names is a gap in the map. A majority unmatched
    // means the wrong pair of files, or a naming convention this map does not
    // cover, and every number below would be drawn from an unrepresentative
    // remnant.
    if unmatched.len() > rows.len() + not_comparable.len() {
        println!(
            "\nFAILED: name matching collapsed, {} unmatched against {} matched",
            unmatched.len(),
            rows.len() + not_comparable.len()
        );
        for name in unmatched.iter().take(top) {
            println!("  {name}");
        }
        std::process::exit(1);
    }

    // Worst by the larger of the two relative errors, so the table shows the
    // tensors where either format struggles.
    rows.sort_by(|a, b| {
        let ka = a.gguf.rel_rms.max(a.tcf.rel_rms);
        let kb = b.gguf.rel_rms.max(b.tcf.rel_rms);
        kb.partial_cmp(&ka).unwrap_or(std::cmp::Ordering::Equal)
    });
    println!("\nworst {top} by relative RMS:");
    for row in rows.iter().take(top) {
        println!("  {} ({} elems, {})", row.name, row.elems, row.encodings);
        print_errors("gguf", &row.gguf);
        print_errors("tcf", &row.tcf);
    }

    let total_elems: f64 = rows.iter().map(|r| r.elems as f64).sum();
    let weighted = |pick: fn(&Row) -> f64| -> f64 {
        rows.iter().map(|r| r.elems as f64 * pick(r)).sum::<f64>() / total_elems
    };
    let gguf_mean = weighted(|r| r.gguf.rel_rms);
    let tcf_mean = weighted(|r| r.tcf.rel_rms);

    let gguf_worst = worst_by(&rows, |r| r.gguf.rel_rms);
    let tcf_worst = worst_by(&rows, |r| r.tcf.rel_rms);

    println!("\naggregate over {} comparable tensors:", rows.len());
    println!("  element-weighted mean relative RMS:");
    println!("    gguf {:>8.4}%", 100.0 * gguf_mean);
    println!("    tcf  {:>8.4}%", 100.0 * tcf_mean);
    if let Some((name, rel)) = gguf_worst {
        println!("  worst gguf tensor: {name} at {:.4}%", 100.0 * rel);
    }
    if let Some((name, rel)) = tcf_worst {
        println!("  worst tcf tensor:  {name} at {:.4}%", 100.0 * rel);
    }

    if !not_comparable.is_empty() {
        println!(
            "\n{} tensors excluded, raw in one file and quantized in the other:",
            not_comparable.len()
        );
        for line in not_comparable.iter().take(top) {
            println!("  {line}");
        }
    }
    if !unmatched.is_empty() {
        println!(
            "\n{} reference tensors matched in neither file:",
            unmatched.len()
        );
        for name in unmatched.iter().take(top) {
            println!("  {name}");
        }
    }
    if !reshaped.is_empty() {
        println!(
            "\n{} tensors differ in shape at equal element count, values still compared:",
            reshaped.len()
        );
        for line in reshaped.iter().take(top) {
            println!("  {line}");
        }
    }
    if !shape_mismatch.is_empty() {
        println!("\n{} shape mismatches, excluded:", shape_mismatch.len());
        for line in shape_mismatch.iter().take(top) {
            println!("  {line}");
        }
    }
    for (label, collided) in [("gguf", &gguf_collided), ("tcf", &tcf_collided)] {
        if collided.is_empty() {
            continue;
        }
        println!(
            "\n{} {label} canonical names claimed by more than one stored tensor, excluded:",
            collided.len()
        );
        for name in collided.iter().take(top) {
            println!("  {name}");
        }
    }

    let (verdict, margin) = if tcf_mean < gguf_mean {
        ("TCF", gguf_mean / tcf_mean.max(f64::MIN_POSITIVE))
    } else {
        ("GGUF", tcf_mean / gguf_mean.max(f64::MIN_POSITIVE))
    };
    println!(
        "\nVERDICT: {verdict} reconstructs the BF16 source more faithfully, \
{margin:.2}x lower element-weighted relative RMS over {} comparable tensors",
        rows.len()
    );
    Ok(())
}
