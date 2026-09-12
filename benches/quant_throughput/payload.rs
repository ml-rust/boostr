//! Packed weight bytes for one case, and the cache that keeps building them out
//! of the measurement.
//!
//! Every case is measured twice — once with zero iterations, once with `N` — and
//! the two runs must do IDENTICAL setup for the subtraction to isolate the
//! measured loop. Quantizing a 12.6M-element weight is not identical work run to
//! run once rayon is involved, so the bytes are built once, cached on disk, and
//! read back thereafter.

use std::fs;
use std::path::PathBuf;

use boostr::quant::{QuantFormat, QuantizeOps};
use boostr::{CpuClient, CpuDevice, CpuRuntime, Tensor};

/// Bumped whenever `source_values` changes, so a stale cache is never read as
/// if it described the current input.
const CACHE_VERSION: u32 = 1;

/// A deterministic input with sign changes, a flat run, and a spike, so a
/// group's scale and its minimum both move between groups.
///
/// A constant or a pure ramp would let a K-quant find a degenerate fit and
/// would not exercise the second scale level at all.
pub fn source_values(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|i| {
            let x = (i + seed) as f32;
            match (i + seed) % 6 {
                0 => 0.75,
                2 => -(x * 0.011).sin() * 2.5,
                4 => (x * 0.037).cos() * 1.5,
                _ => (x * 0.023).sin() * 1.1 - 0.2,
            }
        })
        .collect()
}

/// The activation for a `[m, k]` matmul. A different seed from the weight, so a
/// systematic correlation cannot flatter one format's accumulator.
pub fn activation_values(m: usize, k: usize) -> Vec<f32> {
    source_values(m * k, 17)
}

/// Packed bytes for a `[n, k]` weight under `format`, from cache when possible.
pub fn packed(format: QuantFormat, n: usize, k: usize) -> Result<Vec<u8>, String> {
    let expected = format
        .storage_bytes(n * k)
        .map_err(|e| format!("payload size for {}: {e}", format.name()))?;

    let path = cache_path(format, n, k);
    if let Some(path) = path.as_ref()
        && let Ok(bytes) = fs::read(path)
        && bytes.len() == expected
    {
        return Ok(bytes);
    }

    let values = source_values(n * k, 0);
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    let input = Tensor::<CpuRuntime>::from_slice(&values, &[n, k], &device)
        .map_err(|e| format!("weight tensor {n}x{k}: {e}"))?;
    let bytes = client
        .quantize(&input, format)
        .map_err(|e| format!("quantize {}: {e}", format.name()))?
        .to_bytes()
        .map_err(|e| format!("read back {}: {e}", format.name()))?;

    if bytes.len() != expected {
        return Err(format!(
            "{} produced {} bytes for [{n}, {k}], expected {expected}",
            format.name(),
            bytes.len(),
        ));
    }

    // A cache that cannot be written costs a rebuild, never a wrong answer, so
    // a write error is dropped rather than failing the case.
    if let Some(path) = path.as_ref()
        && let Some(parent) = path.parent()
        && fs::create_dir_all(parent).is_ok()
    {
        let _ = fs::write(path, &bytes);
    }
    Ok(bytes)
}

/// `$XDG_CACHE_HOME/boostr-quant-bench/`, or `None` when neither that nor
/// `$HOME` is set — in which case every run rebuilds.
fn cache_path(format: QuantFormat, n: usize, k: usize) -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))?;
    Some(
        base.join("boostr-quant-bench")
            .join(format!("v{CACHE_VERSION}_{}_{n}x{k}.bin", format.name())),
    )
}
