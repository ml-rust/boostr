//! Shared helpers for the super-resolution parity tests: layout and dtype
//! conversion at the test boundary, and error metrics against an F32
//! golden in NCHW.

use boostr::format::SafeTensors;
use boostr::model::super_resolution::SrExecution;
use numr::dtype::DType;
use numr::ops::{ConvLayout, TypeConversionOps};
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;
use std::path::{Path, PathBuf};

/// `nchw`, an F32 `[N, C, H, W]` golden input, in `exec`'s layout and dtype.
pub fn sr_input<R, C>(client: &C, nchw: &Tensor<R>, exec: &SrExecution) -> Tensor<R>
where
    R: Runtime<DType = DType>,
    C: TypeConversionOps<R>,
{
    let ordered = match exec.layout {
        ConvLayout::Nchw => nchw.clone(),
        ConvLayout::Nhwc => nchw
            .permute(&[0, 2, 3, 1])
            .and_then(|t| t.contiguous())
            .unwrap_or_else(|e| panic!("permute input {:?} to NHWC: {e}", nchw.shape())),
    };
    if exec.dtype == DType::F32 {
        ordered
    } else {
        client
            .cast(&ordered, exec.dtype)
            .unwrap_or_else(|e| panic!("cast input to {:?}: {e}", exec.dtype))
    }
}

/// `out`, a network output in `exec`'s layout and dtype, as F32 values in
/// NCHW order, plus its NCHW shape.
pub fn sr_output_nchw<R, C>(
    client: &C,
    out: &Tensor<R>,
    exec: &SrExecution,
) -> (Vec<usize>, Vec<f32>)
where
    R: Runtime<DType = DType>,
    C: TypeConversionOps<R>,
{
    let wide = if out.dtype() == DType::F32 {
        out.clone()
    } else {
        client
            .cast(out, DType::F32)
            .unwrap_or_else(|e| panic!("cast output {:?} to F32: {e}", out.dtype()))
    };
    let nchw = match exec.layout {
        ConvLayout::Nchw => wide,
        ConvLayout::Nhwc => wide
            .permute(&[0, 3, 1, 2])
            .unwrap_or_else(|e| panic!("permute output {:?} to NCHW: {e}", wide.shape())),
    };
    let nchw = nchw
        .contiguous()
        .unwrap_or_else(|e| panic!("contiguous output: {e}"));
    (nchw.shape().to_vec(), nchw.to_vec::<f32>())
}

/// Largest absolute difference and its flat index.
pub fn max_abs_diff(got: &[f32], want: &[f32]) -> (f32, usize) {
    assert_eq!(got.len(), want.len(), "length mismatch");
    got.iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs())
        .enumerate()
        .fold((0.0f32, 0usize), |(worst, at), (i, d)| {
            if d > worst { (d, i) } else { (worst, at) }
        })
}

/// PSNR in dB with peak 1.0, over unclamped values.
pub fn psnr_db(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len(), "length mismatch");
    let mse = got
        .iter()
        .zip(want)
        .map(|(g, w)| {
            let d = f64::from(*g) - f64::from(*w);
            d * d
        })
        .sum::<f64>()
        / got.len() as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (1.0 / mse).log10()
    }
}

/// `clamp(x, 0, 1) * 255` rounded half to even, in f32 as torch does it.
pub fn quantise_u8(x: f32) -> i32 {
    (x.clamp(0.0, 1.0) * 255.0).round_ties_even() as i32
}

/// Fraction of values whose u8 quantisation matches, and the largest u8
/// difference.
pub fn u8_agreement(got: &[f32], want: &[f32]) -> (f64, i32) {
    assert_eq!(got.len(), want.len(), "length mismatch");
    let (exact, worst) = got
        .iter()
        .zip(want)
        .map(|(g, w)| (quantise_u8(*g) - quantise_u8(*w)).abs())
        .fold((0usize, 0i32), |(exact, worst), d| {
            (exact + usize::from(d == 0), worst.max(d))
        });
    (exact as f64 / got.len() as f64, worst)
}

/// Directory of the checked-in fixture set `name`, under
/// `tests/fixtures/super_resolution/`.
pub fn fixtures_dir(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/super_resolution")
        .join(name)
}

/// Tensor `name` from the `io.safetensors` of fixture set `fixture`, as `R`
/// on `device`.
pub fn golden<R: Runtime<DType = DType>>(
    fixture: &str,
    name: &str,
    device: &R::Device,
) -> Tensor<R> {
    let mut io = SafeTensors::open(fixtures_dir(fixture).join("io.safetensors"))
        .expect("open io.safetensors");
    io.load_tensor::<R>(name, device)
        .unwrap_or_else(|e| panic!("golden tensor '{name}': {e}"))
}

/// Shape and F32 values of tensor `name` from the `io.safetensors` of
/// fixture set `fixture`.
pub fn golden_values(fixture: &str, name: &str) -> (Vec<usize>, Vec<f32>) {
    let t = golden::<CpuRuntime>(fixture, name, &CpuDevice::new());
    (t.shape().to_vec(), t.to_vec::<f32>())
}

/// Parse the `config.json` of fixture set `fixture`.
pub fn load_config<T: serde::de::DeserializeOwned>(fixture: &str) -> T {
    let path = fixtures_dir(fixture).join("config.json");
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&text).expect("parse config.json")
}

/// Shape and F32 values of tensor `name` in the safetensors file at `path`.
pub fn golden_cpu(path: &Path, name: &str) -> (Vec<usize>, Vec<f32>) {
    let mut io = SafeTensors::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let t = io
        .load_tensor::<CpuRuntime>(name, &CpuDevice::new())
        .unwrap_or_else(|e| panic!("golden tensor '{name}': {e}"));
    (t.shape().to_vec(), t.to_vec::<f32>())
}

/// Pass bounds for one precision against the torch CPU F32 golden.
pub struct Bounds {
    pub max_abs: f32,
    pub min_psnr_db: f64,
    pub min_u8_exact: f64,
    pub max_u8_diff: i32,
}

/// Print the error metrics of `got` against `want` under `label`, then hold
/// them to `bounds`.
pub fn assert_within(label: &str, got: &[f32], want: &[f32], bounds: &Bounds) {
    let (diff, at) = max_abs_diff(got, want);
    let psnr = psnr_db(got, want);
    let (exact, worst_u8) = u8_agreement(got, want);
    eprintln!(
        "{label}: max abs {diff:e} at {at}, PSNR {psnr:.2} dB, u8 exact {:.4}%, max u8 diff {worst_u8}",
        100.0 * exact
    );
    assert!(
        diff <= bounds.max_abs,
        "{label}: max abs {diff:e} at {at} > {:e}",
        bounds.max_abs
    );
    assert!(
        psnr >= bounds.min_psnr_db,
        "{label}: PSNR {psnr:.2} dB < {} dB",
        bounds.min_psnr_db
    );
    assert!(
        exact >= bounds.min_u8_exact,
        "{label}: u8 exact {exact:.5} < {}",
        bounds.min_u8_exact
    );
    assert!(
        worst_u8 <= bounds.max_u8_diff,
        "{label}: u8 diff {worst_u8} > {}",
        bounds.max_u8_diff
    );
}
