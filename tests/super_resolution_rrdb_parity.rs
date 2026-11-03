//! Numerical parity for [`RrdbNet`] against a torch oracle, on a seeded tiny
//! ×2 network (num_feat 8, num_grow_ch 4, num_block 1).
//!
//! Fixtures are checked in under `tests/fixtures/super_resolution/rrdb/`.
//! Regenerate with
//! `python3 tests/tools/sr_blocks_oracle/sr_blocks_oracle.py rrdb tests/fixtures/super_resolution/rrdb`
//! (requires torch and safetensors).
//!
//! Run with:
//!   `cargo test --features super-resolution --test super_resolution_rrdb_parity`
//!   `cargo test --features cuda,f16,super-resolution --test super_resolution_rrdb_parity`
//!
//! What this pins: the replicate pad and crop of an odd input (13×11), the
//! pixel-unshuffle head, the dense-block channel slices, both residual
//! scales, the trunk residual, the two nearest upsamples, the OIHW → OHWI
//! permute, and the legacy ESRGAN key remap with `metadata.*`. The modern
//! `weights.pth` and the legacy `weights_legacy.pth` hold the same weights.

use boostr::model::super_resolution::{RrdbConfig, RrdbNet, SrClient, SrExecution, load_rrdb};
use numr::dtype::DType;
use numr::ops::ConvLayout;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};

mod common;
use common::super_resolution::{
    fixtures_dir, golden, golden_values, load_config, max_abs_diff, sr_input, sr_output_nchw,
};

/// Fixture set under `tests/fixtures/super_resolution/`.
const FIXTURE: &str = "rrdb";

/// F32 bound: the same arithmetic in another summation order.
const F32_TOL: f32 = 1e-5;
/// Golden inputs, one per fixture case: `a` is 13×11, `b` is 8×8.
const CASES: [&str; 2] = ["a", "b"];
/// The two files that hold the fixture weights.
const WEIGHTS: [&str; 2] = ["weights.pth", "weights_legacy.pth"];

#[derive(serde::Deserialize)]
struct Config {
    num_in_ch: usize,
    num_out_ch: usize,
    num_feat: usize,
    num_block: usize,
    num_grow_ch: usize,
    scale: usize,
}

fn config() -> RrdbConfig {
    let c: Config = load_config(FIXTURE);
    RrdbConfig {
        num_in_ch: c.num_in_ch,
        num_out_ch: c.num_out_ch,
        num_feat: c.num_feat,
        num_block: c.num_block,
        num_grow_ch: c.num_grow_ch,
        scale: c.scale,
    }
}

/// Load `file` from the fixtures with `config`, as `R` with `exec`.
fn load<R>(
    file: &str,
    device: &R::Device,
    config: &RrdbConfig,
    exec: SrExecution,
) -> boostr::error::Result<RrdbNet<R>>
where
    R: Runtime<DType = DType>,
    R::Client: SrClient<R>,
{
    load_rrdb::<R>(fixtures_dir(FIXTURE).join(file), device, config, exec)
}

/// The error text of loading `file` with `config` on CPU F32 NCHW.
fn load_error(file: &str, config: &RrdbConfig) -> String {
    load::<CpuRuntime>(
        file,
        &CpuDevice::new(),
        config,
        SrExecution::new(DType::F32, ConvLayout::Nchw),
    )
    .err()
    .map(|e| e.to_string())
    .unwrap_or_default()
}

/// Run both cases through `forward` and hold each output to `tol` against
/// the F32 golden.
fn check_against_golden<R>(model: &RrdbNet<R>, device: &R::Device, tol: f32, label: &str)
where
    R: Runtime<DType = DType>,
    R::Client: SrClient<R>,
{
    let client = R::default_client(device);
    let exec = *model.execution();
    for case in CASES {
        let input = sr_input(
            &client,
            &golden::<R>(FIXTURE, &format!("input_{case}"), device),
            &exec,
        );
        let out = model.forward(&client, &input).expect("forward");
        let (shape, got) = sr_output_nchw(&client, &out, &exec);
        let (want_shape, want) = golden_values(FIXTURE, &format!("out_{case}"));
        assert_eq!(shape, want_shape, "{label} out_{case}: shape");
        let (diff, at) = max_abs_diff(&got, &want);
        eprintln!("{label} out_{case}: max abs diff {diff:e} at {at}");
        assert!(
            diff <= tol,
            "{label} out_{case}: max abs diff {diff:e} at {at} (got {} want {}), bound {tol:e}",
            got[at],
            want[at]
        );
    }
}

#[test]
fn cpu_f32_both_layouts_both_key_sets_match_torch() {
    let device = CpuDevice::new();
    for file in WEIGHTS {
        for layout in [ConvLayout::Nchw, ConvLayout::Nhwc] {
            let exec = SrExecution::new(DType::F32, layout);
            let model = load::<CpuRuntime>(file, &device, &config(), exec)
                .unwrap_or_else(|e| panic!("load_rrdb {file}: {e}"));
            check_against_golden(
                &model,
                &device,
                F32_TOL,
                &format!("cpu f32 {layout:?} {file}"),
            );
        }
    }
}

#[test]
fn load_rejects_missing_keys() {
    let wrong = RrdbConfig {
        num_block: 2,
        ..config()
    };
    let err = load_error("weights.pth", &wrong);
    assert!(err.contains("weights.pth"), "{err}");
    assert!(
        err.contains("missing [body.1.rdb1.conv1.bias, body.1.rdb1.conv1.weight"),
        "{err}"
    );
    assert!(err.contains("unexpected none"), "{err}");
}

#[test]
fn load_rejects_unexpected_keys() {
    let err = load_error("weights_unexpected.pth", &config());
    assert!(err.contains("weights_unexpected.pth"), "{err}");
    assert!(
        err.contains("missing none, unexpected [body.0.rdb1.conv6.weight]"),
        "{err}"
    );
}

#[test]
fn load_rejects_duplicate_legacy_targets() {
    let err = load_error("weights_legacy_duplicate.pth", &config());
    assert!(err.contains("weights_legacy_duplicate.pth"), "{err}");
    assert!(
        err.contains("both map to 'body.0.rdb1.conv1.weight'"),
        "{err}"
    );
}

#[test]
fn load_rejects_metadata_of_another_scale() {
    let err = load_error("weights_legacy_bad_metadata.pth", &config());
    assert!(err.contains("weights_legacy_bad_metadata.pth"), "{err}");
    assert!(
        err.contains("metadata.scale 8 with metadata.shuffle_factor 2 is not an x2 network"),
        "{err}"
    );
}

#[test]
fn load_rejects_metadata_of_another_depth() {
    let wrong = RrdbConfig {
        num_block: 2,
        ..config()
    };
    let err = load_error("weights_legacy.pth", &wrong);
    assert!(
        err.contains("metadata.num_blocks is 1, but num_block gives 2"),
        "{err}"
    );
}

#[test]
fn load_rejects_shape_mismatch() {
    let wrong = RrdbConfig {
        num_grow_ch: 8,
        ..config()
    };
    let err = load_error("weights.pth", &wrong);
    assert!(
        err.contains("'body.0.rdb1.conv1.weight' has shape [4, 8, 3, 3], expected [8, 8, 3, 3]"),
        "{err}"
    );
}

#[test]
fn load_rejects_unsupported_scale() {
    let wrong = RrdbConfig {
        scale: 4,
        ..config()
    };
    let err = load_error("weights.pth", &wrong);
    assert!(
        err.contains("scale 4") && err.contains("pass scale 2"),
        "{err}"
    );
}

#[test]
fn forward_rejects_wrong_layout_and_dtype() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let exec = SrExecution::new(DType::F32, ConvLayout::Nhwc);
    let model = load::<CpuRuntime>("weights.pth", &device, &config(), exec).expect("load_rrdb");
    // An NCHW [1, 3, 13, 11] tensor read as NHWC has 11 channels.
    let nchw = golden::<CpuRuntime>(FIXTURE, "input_a", &device);
    let err = model
        .forward(&client, &nchw)
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(err.contains("3 input channels"), "{err}");

    let as_f64 =
        numr::ops::TypeConversionOps::cast(&client, &sr_input(&client, &nchw, &exec), DType::F64)
            .expect("cast to F64");
    let err = model
        .forward(&client, &as_f64)
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(err.contains("cast the input to F32"), "{err}");
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use numr::runtime::cuda::{CudaDevice, CudaRuntime};

    fn load_cuda(file: &str, exec: SrExecution) -> (CudaDevice, RrdbNet<CudaRuntime>) {
        let device = CudaDevice::new(0);
        let model = load::<CudaRuntime>(file, &device, &config(), exec)
            .unwrap_or_else(|e| panic!("load_rrdb {file} on CUDA: {e}"));
        (device, model)
    }

    #[test]
    fn cuda_f32_both_layouts_match_torch() {
        for file in WEIGHTS {
            for layout in [ConvLayout::Nchw, ConvLayout::Nhwc] {
                let (device, model) = load_cuda(file, SrExecution::new(DType::F32, layout));
                check_against_golden(
                    &model,
                    &device,
                    F32_TOL,
                    &format!("cuda f32 {layout:?} {file}"),
                );
            }
        }
    }

    /// F16 bound: 8 ulps of the outputs' top binade [1, 2), 8 · 2^-10.
    ///
    /// Every output has magnitude under 2. The tiny network rounds its
    /// input, 21 conv outputs, and the RRDB residual scale and add to F16.
    /// Each rounding is at most half an ulp. Torch's own CUDA F16 run of
    /// this fixture passes the bound.
    #[cfg(feature = "f16")]
    const F16_TOL: f32 = 8.0 / 1024.0;
    /// BF16 bound: 8 ulps of the top binade [1, 2), 8 · 2^-7, on the same
    /// argument as [`F16_TOL`].
    #[cfg(feature = "f16")]
    const BF16_TOL: f32 = 8.0 / 128.0;

    #[cfg(feature = "f16")]
    #[test]
    fn cuda_f16_nhwc_matches_torch() {
        let (device, model) = load_cuda(
            "weights_legacy.pth",
            SrExecution::new(DType::F16, ConvLayout::Nhwc),
        );
        check_against_golden(&model, &device, F16_TOL, "cuda f16 nhwc");
    }

    #[cfg(feature = "f16")]
    #[test]
    fn cuda_bf16_nhwc_matches_torch() {
        let (device, model) = load_cuda(
            "weights.pth",
            SrExecution::new(DType::BF16, ConvLayout::Nhwc),
        );
        check_against_golden(&model, &device, BF16_TOL, "cuda bf16 nhwc");
    }
}
