//! Numerical parity for [`SrVggCompact`] against a torch oracle, on a seeded
//! tiny network (num_feat 8, num_conv 2, upscale 4).
//!
//! Fixtures are checked in under `tests/fixtures/super_resolution/compact/`.
//! Regenerate with
//! `python3 tests/tools/sr_blocks_oracle/sr_blocks_oracle.py tests/fixtures/super_resolution/compact`
//! (requires torch and safetensors).
//!
//! Run with:
//!   `cargo test --features super-resolution --test super_resolution_compact_parity`
//!   `cargo test --features cuda,f16,super-resolution --test super_resolution_compact_parity`
//!
//! What this pins: the `body.N` key schedule, per-channel PReLU slopes, the
//! OIHW → OHWI permute, pixel-shuffle channel order, the nearest-upsampled
//! residual, and the antialiased bicubic resize to 2×. Both the
//! `weights.safetensors` + `from_varbuilder` path and the `.pth` +
//! `load_compact` path load the same weights.

use boostr::format::SafeTensors;
use boostr::model::super_resolution::{
    SrExecution, SrVggCompact, SrVggCompactConfig, load_compact,
};
use boostr::nn::{VarBuilder, VarMap};
use numr::dtype::DType;
use numr::ops::ConvLayout;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;
use std::path::{Path, PathBuf};

mod common;
use common::super_resolution::{max_abs_diff, sr_input, sr_output_nchw};

/// F32 bound: the same arithmetic in another summation order.
const F32_TOL: f32 = 1e-5;
/// Golden inputs, one per fixture case.
const CASES: [&str; 2] = ["a", "b"];

#[derive(serde::Deserialize)]
struct Config {
    num_in_ch: usize,
    num_out_ch: usize,
    num_feat: usize,
    num_conv: usize,
    upscale: usize,
}

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/super_resolution/compact")
}

fn config() -> SrVggCompactConfig {
    let path = fixtures_dir().join("config.json");
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let c: Config = serde_json::from_str(&text).expect("parse config.json");
    SrVggCompactConfig {
        num_in_ch: c.num_in_ch,
        num_out_ch: c.num_out_ch,
        num_feat: c.num_feat,
        num_conv: c.num_conv,
        upscale: c.upscale,
    }
}

fn golden<R: Runtime<DType = DType>>(name: &str, device: &R::Device) -> Tensor<R> {
    let mut io =
        SafeTensors::open(fixtures_dir().join("io.safetensors")).expect("open io.safetensors");
    io.load_tensor::<R>(name, device)
        .unwrap_or_else(|e| panic!("golden tensor '{name}': {e}"))
}

fn golden_values(name: &str) -> (Vec<usize>, Vec<f32>) {
    let t = golden::<CpuRuntime>(name, &CpuDevice::new());
    (t.shape().to_vec(), t.to_vec::<f32>())
}

/// Run both cases through `forward_raw` and `forward_resized` and hold each
/// output to `tol` against the F32 golden.
fn check_against_golden<R>(
    model: &SrVggCompact<R>,
    client: &R::Client,
    device: &R::Device,
    tol: f32,
    label: &str,
) where
    R: Runtime<DType = DType>,
    R::Client: boostr::model::super_resolution::SrClient<R>,
{
    let exec = *model.execution();
    for case in CASES {
        let input = sr_input(
            client,
            &golden::<R>(&format!("input_{case}"), device),
            &exec,
        );
        let (_, _, h, w) = exec.image_dims(input.shape()).expect("input dims");

        let raw = model.forward_raw(client, &input).expect("forward_raw");
        let resized = model
            .forward_resized(client, &input, (2 * h, 2 * w))
            .expect("forward_resized");

        for (kind, out) in [("raw", &raw), ("resized", &resized)] {
            let (shape, got) = sr_output_nchw(client, out, &exec);
            let (want_shape, want) = golden_values(&format!("{kind}_{case}"));
            assert_eq!(shape, want_shape, "{label} {kind}_{case}: shape");
            let (diff, at) = max_abs_diff(&got, &want);
            eprintln!("{label} {kind}_{case}: max abs diff {diff:e} at {at}");
            assert!(
                diff <= tol,
                "{label} {kind}_{case}: max abs diff {diff:e} at {at} (got {} want {}), bound {tol:e}",
                got[at],
                want[at]
            );
        }
    }
}

fn cpu() -> (CpuDevice, CpuClient) {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    (device, client)
}

#[test]
fn cpu_f32_nchw_safetensors_matches_torch() {
    let (device, client) = cpu();
    let exec = SrExecution::new(DType::F32, ConvLayout::Nchw);
    let mut map =
        VarMap::<CpuRuntime>::from_safetensors(fixtures_dir().join("weights.safetensors"), &device)
            .expect("load weights.safetensors");
    let mut vb = VarBuilder::new(&mut map, &device);
    let model = SrVggCompact::from_varbuilder(&client, &mut vb, &config(), exec)
        .expect("build SrVggCompact from safetensors");
    check_against_golden(
        &model,
        &client,
        &device,
        F32_TOL,
        "cpu f32 nchw safetensors",
    );
}

#[test]
fn cpu_f32_both_layouts_pth_match_torch() {
    let (device, client) = cpu();
    for layout in [ConvLayout::Nchw, ConvLayout::Nhwc] {
        let exec = SrExecution::new(DType::F32, layout);
        let model = load_compact::<CpuRuntime>(
            fixtures_dir().join("weights.pth"),
            &device,
            &config(),
            exec,
        )
        .expect("load_compact weights.pth");
        check_against_golden(
            &model,
            &client,
            &device,
            F32_TOL,
            &format!("cpu f32 {layout:?} pth"),
        );
    }
}

#[test]
fn load_rejects_missing_keys() {
    let (device, _) = cpu();
    let wrong = SrVggCompactConfig {
        num_conv: 3,
        ..config()
    };
    let err = load_compact::<CpuRuntime>(
        fixtures_dir().join("weights.pth"),
        &device,
        &wrong,
        SrExecution::new(DType::F32, ConvLayout::Nchw),
    )
    .err()
    .map(|e| e.to_string())
    .unwrap_or_default();
    assert!(err.contains("weights.pth"), "{err}");
    assert!(
        err.contains("missing [body.7.weight, body.8.bias, body.8.weight]"),
        "{err}"
    );
}

#[test]
fn load_rejects_unexpected_keys() {
    let (device, _) = cpu();
    let wrong = SrVggCompactConfig {
        num_conv: 1,
        ..config()
    };
    let err = load_compact::<CpuRuntime>(
        fixtures_dir().join("weights.pth"),
        &device,
        &wrong,
        SrExecution::new(DType::F32, ConvLayout::Nchw),
    )
    .err()
    .map(|e| e.to_string())
    .unwrap_or_default();
    assert!(
        err.contains("unexpected [body.5.weight, body.6.bias, body.6.weight]"),
        "{err}"
    );
}

#[test]
fn load_rejects_shape_mismatch() {
    let (device, _) = cpu();
    let wrong = SrVggCompactConfig {
        num_feat: 16,
        ..config()
    };
    let err = load_compact::<CpuRuntime>(
        fixtures_dir().join("weights.pth"),
        &device,
        &wrong,
        SrExecution::new(DType::F32, ConvLayout::Nchw),
    )
    .err()
    .map(|e| e.to_string())
    .unwrap_or_default();
    assert!(
        err.contains("'body.0.weight' has shape [8, 3, 3, 3], expected [16, 3, 3, 3]"),
        "{err}"
    );
}

#[test]
fn forward_rejects_wrong_layout_and_dtype() {
    let (device, client) = cpu();
    let exec = SrExecution::new(DType::F32, ConvLayout::Nhwc);
    let model =
        load_compact::<CpuRuntime>(fixtures_dir().join("weights.pth"), &device, &config(), exec)
            .expect("load_compact weights.pth");
    // An NCHW [1, 3, 13, 11] tensor read as NHWC has 11 channels.
    let nchw = golden::<CpuRuntime>("input_a", &device);
    let err = model
        .forward_raw(&client, &nchw)
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(err.contains("3 input channels"), "{err}");

    let as_f64 =
        numr::ops::TypeConversionOps::cast(&client, &sr_input(&client, &nchw, &exec), DType::F64)
            .expect("cast to F64");
    let err = model
        .forward_raw(&client, &as_f64)
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(err.contains("cast the input to F32"), "{err}");
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use numr::runtime::cuda::{CudaDevice, CudaRuntime};

    fn load(
        exec: SrExecution,
    ) -> (
        CudaDevice,
        <CudaRuntime as Runtime>::Client,
        SrVggCompact<CudaRuntime>,
    ) {
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);
        let model = load_compact::<CudaRuntime>(
            fixtures_dir().join("weights.pth"),
            &device,
            &config(),
            exec,
        )
        .expect("load_compact weights.pth on CUDA");
        (device, client, model)
    }

    #[test]
    fn cuda_f32_both_layouts_match_torch() {
        for layout in [ConvLayout::Nchw, ConvLayout::Nhwc] {
            let (device, client, model) = load(SrExecution::new(DType::F32, layout));
            check_against_golden(
                &model,
                &client,
                &device,
                F32_TOL,
                &format!("cuda f32 {layout:?}"),
            );
        }
    }

    /// F16 bound: 4 ulps of the outputs' top binade [1, 2), 4 · 2^-10.
    ///
    /// The tiny network rounds its input, each of its 4 conv outputs, the
    /// residual add and the resize store to F16. Each rounding is at most
    /// half an ulp, so the error stays near 3 ulps without growth. Torch's
    /// own CUDA F16 run of this fixture stays under 1 ulp.
    #[cfg(feature = "f16")]
    const F16_TOL: f32 = 4.0 / 1024.0;
    /// BF16 bound: 4 ulps of the top binade [1, 2), 4 · 2^-7, on the same
    /// argument as [`F16_TOL`].
    #[cfg(feature = "f16")]
    const BF16_TOL: f32 = 4.0 / 128.0;

    #[cfg(feature = "f16")]
    #[test]
    fn cuda_f16_nhwc_matches_torch() {
        let (device, client, model) = load(SrExecution::new(DType::F16, ConvLayout::Nhwc));
        check_against_golden(&model, &client, &device, F16_TOL, "cuda f16 nhwc");
    }

    #[cfg(feature = "f16")]
    #[test]
    fn cuda_bf16_nhwc_matches_torch() {
        let (device, client, model) = load(SrExecution::new(DType::BF16, ConvLayout::Nhwc));
        check_against_golden(&model, &client, &device, BF16_TOL, "cuda bf16 nhwc");
    }
}
