//! Real-checkpoint parity for [`SrVggCompact`] on `realesr-general-x4v3`
//! against torch CPU F32.
//!
//! Run with:
//!   `cargo test --release --features super-resolution --test super_resolution_compact_reference`
//!   `cargo test --release --features cuda,f16,super-resolution --test super_resolution_compact_reference`
//!
//! Not checked in. This skips unless both are present:
//!
//! * the checkpoint: `REALESR_GENERAL_X4V3`, else
//!   `$BOOSTR_MODELS_DIR/realesr-general-x4v3/realesr-general-x4v3.pth`
//! * the golden: `SR_REF_DIR`, else `$BOOSTR_MODELS_DIR/sr_reference/compact`,
//!   written by `python3 tests/tools/sr_reference_dump/sr_reference_dump.py`
//!
//! The golden holds one synthetic 3×67×45 frame, its `forward_raw` output
//! (×4) and its `forward` output (×2). Every check runs on both outputs.
//! The u8 checks quantise as `round_half_even(clamp(x, 0, 1) · 255)`.

use boostr::format::SafeTensors;
use boostr::model::super_resolution::{
    SrClient, SrExecution, SrVggCompact, SrVggCompactConfig, load_compact,
};
use numr::dtype::DType;
use numr::ops::ConvLayout;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use std::path::{Path, PathBuf};

mod common;
use common::super_resolution::{max_abs_diff, psnr_db, sr_input, sr_output_nchw, u8_agreement};
use common::{model_fixture, skip_notice};

const CHECKPOINT_VAR: &str = "REALESR_GENERAL_X4V3";
const CHECKPOINT_REL: &str = "realesr-general-x4v3/realesr-general-x4v3.pth";
const GOLDEN_VAR: &str = "SR_REF_DIR";
const GOLDEN_REL: &str = "sr_reference/compact";

/// Pass bounds for one precision against the torch CPU F32 golden.
struct Bounds {
    max_abs: f32,
    min_psnr_db: f64,
    min_u8_exact: f64,
    max_u8_diff: i32,
}

/// F32 bound. Two independent F32 builds of this network, torch CPU F64 and
/// torch CUDA F32 without TF32, stay well inside it and keep nearly every
/// u8 value.
const F32: Bounds = Bounds {
    max_abs: 1e-4,
    min_psnr_db: 80.0,
    min_u8_exact: 0.999,
    max_u8_diff: 1,
};

/// F16 bound. 34 convs each round their output to F16. The output lies in
/// [0, 1], where an F16 ulp is at most 2^-11, so each u8 step of 1/255
/// spans about 8 ulps and a few ulps of drift flip the nearest value.
/// Torch's own CUDA F16 run on this frame passes the bound with margin.
#[cfg(all(feature = "cuda", feature = "f16"))]
const F16: Bounds = Bounds {
    max_abs: 1e-2,
    min_psnr_db: 45.0,
    min_u8_exact: 0.90,
    max_u8_diff: 1,
};

/// BF16 bound. A BF16 ulp is 8× an F16 ulp. Torch's own CUDA BF16 run on
/// this frame passes the bound with margin.
#[cfg(all(feature = "cuda", feature = "f16"))]
const BF16: Bounds = Bounds {
    max_abs: 4e-2,
    min_psnr_db: 45.0,
    min_u8_exact: 0.40,
    max_u8_diff: 6,
};

fn fixtures() -> Option<(PathBuf, PathBuf)> {
    let Some(checkpoint) = model_fixture(CHECKPOINT_VAR, CHECKPOINT_REL) else {
        skip_notice("realesr-general-x4v3.pth", CHECKPOINT_VAR);
        return None;
    };
    let Some(golden) = model_fixture(GOLDEN_VAR, GOLDEN_REL) else {
        skip_notice("the sr_reference_dump golden", GOLDEN_VAR);
        return None;
    };
    Some((checkpoint, golden.join("io.safetensors")))
}

fn golden_cpu(path: &Path, name: &str) -> (Vec<usize>, Vec<f32>) {
    let mut io = SafeTensors::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let t = io
        .load_tensor::<CpuRuntime>(name, &CpuDevice::new())
        .unwrap_or_else(|e| panic!("golden tensor '{name}': {e}"));
    (t.shape().to_vec(), t.to_vec::<f32>())
}

/// Run the golden frame on `device` with `exec` and hold both outputs to
/// `bounds`.
fn check<R>(
    checkpoint: &Path,
    golden: &Path,
    device: &R::Device,
    exec: SrExecution,
    bounds: &Bounds,
    label: &str,
) where
    R: Runtime<DType = DType>,
    R::Client: SrClient<R> + numr::ops::ShapeOps<R>,
{
    let client = R::default_client(device);
    let model: SrVggCompact<R> = load_compact(
        checkpoint,
        device,
        &SrVggCompactConfig::realesr_general_x4v3(),
        exec,
    )
    .unwrap_or_else(|e| panic!("load {}: {e}", checkpoint.display()));
    let mut io =
        SafeTensors::open(golden).unwrap_or_else(|e| panic!("open {}: {e}", golden.display()));
    let frame = io
        .load_tensor::<R>("input", device)
        .unwrap_or_else(|e| panic!("golden tensor 'input': {e}"));
    let input = sr_input(&client, &frame, &exec);
    let (_, _, h, w) = exec.image_dims(input.shape()).expect("input dims");

    let raw = model.forward_raw(&client, &input).expect("forward_raw");
    let resized = model
        .forward_resized(&client, &input, (2 * h, 2 * w))
        .expect("forward_resized");

    for (kind, out) in [("raw", &raw), ("resized", &resized)] {
        let (shape, got) = sr_output_nchw(&client, out, &exec);
        let (want_shape, want) = golden_cpu(golden, kind);
        assert_eq!(shape, want_shape, "{label} {kind}: shape");
        let (diff, at) = max_abs_diff(&got, &want);
        let psnr = psnr_db(&got, &want);
        let (exact, worst_u8) = u8_agreement(&got, &want);
        eprintln!(
            "{label} {kind}: max abs {diff:e} at {at}, PSNR {psnr:.2} dB, u8 exact {:.4}%, max u8 diff {worst_u8}",
            100.0 * exact
        );
        assert!(
            diff <= bounds.max_abs,
            "{label} {kind}: max abs {diff:e} at {at} > {:e}",
            bounds.max_abs
        );
        assert!(
            psnr >= bounds.min_psnr_db,
            "{label} {kind}: PSNR {psnr:.2} dB < {} dB",
            bounds.min_psnr_db
        );
        assert!(
            exact >= bounds.min_u8_exact,
            "{label} {kind}: u8 exact {exact:.5} < {}",
            bounds.min_u8_exact
        );
        assert!(
            worst_u8 <= bounds.max_u8_diff,
            "{label} {kind}: u8 diff {worst_u8} > {}",
            bounds.max_u8_diff
        );
    }
}

#[test]
fn cpu_f32_nchw_matches_torch() {
    let Some((checkpoint, golden)) = fixtures() else {
        return;
    };
    let device = CpuDevice::new();
    check::<CpuRuntime>(
        &checkpoint,
        &golden,
        &device,
        SrExecution::new(DType::F32, ConvLayout::Nchw),
        &F32,
        "cpu f32 nchw",
    );
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use numr::runtime::cuda::{CudaDevice, CudaRuntime};

    #[test]
    fn cuda_f32_nhwc_matches_torch() {
        let Some((checkpoint, golden)) = fixtures() else {
            return;
        };
        let device = CudaDevice::new(0);
        check::<CudaRuntime>(
            &checkpoint,
            &golden,
            &device,
            SrExecution::new(DType::F32, ConvLayout::Nhwc),
            &F32,
            "cuda f32 nhwc",
        );
    }

    #[cfg(feature = "f16")]
    #[test]
    fn cuda_f16_nhwc_matches_torch() {
        let Some((checkpoint, golden)) = fixtures() else {
            return;
        };
        let device = CudaDevice::new(0);
        check::<CudaRuntime>(
            &checkpoint,
            &golden,
            &device,
            SrExecution::new(DType::F16, ConvLayout::Nhwc),
            &F16,
            "cuda f16 nhwc",
        );
    }

    #[cfg(feature = "f16")]
    #[test]
    fn cuda_bf16_nhwc_matches_torch() {
        let Some((checkpoint, golden)) = fixtures() else {
            return;
        };
        let device = CudaDevice::new(0);
        check::<CudaRuntime>(
            &checkpoint,
            &golden,
            &device,
            SrExecution::new(DType::BF16, ConvLayout::Nhwc),
            &BF16,
            "cuda bf16 nhwc",
        );
    }
}
