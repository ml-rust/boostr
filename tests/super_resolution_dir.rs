//! `load_sr_dir` and `SrModel` over the checked-in tiny fixtures and, when
//! present, the real checkpoints.
//!
//! Fixtures are under `tests/fixtures/super_resolution/`. The unknown key
//! set test uses `tests/fixtures/torch_pt/wrap_root.pth`, whose keys fit
//! neither network.
//!
//! Run with:
//!   `cargo test --features super-resolution --test super_resolution_dir`
//!
//! Real checkpoints skip with a printed notice unless present:
//!
//! * `realesr-general-x4v3`: `REALESR_GENERAL_X4V3`, else
//!   `$BOOSTR_MODELS_DIR/realesr-general-x4v3/realesr-general-x4v3.pth`
//! * `RealESRGAN_x2plus`: `REALESRGAN_X2PLUS_PTH`, else
//!   `$BOOSTR_MODELS_DIR/RealESRGAN_x2plus/RealESRGAN_x2plus.pth`
//! * `2x_PureVision`: `PUREVISION_2X_PTH`, else
//!   `$BOOSTR_MODELS_DIR/2x_PureVision/2x_PureVision.pth`
//!
//! What this pins: the key set picks the network, the one-file rule, the
//! unknown key set error, and `forward_native` equal to the direct network
//! call on the same input, bit for bit.

use boostr::model::super_resolution::{
    RrdbConfig, SrExecution, SrModel, SrVggCompactConfig, load_compact, load_rrdb, load_sr_dir,
};
use numr::dtype::DType;
use numr::ops::ConvLayout;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;
use std::path::{Path, PathBuf};

mod common;
use common::super_resolution::{fixtures_dir, load_config, max_abs_diff, sr_input, sr_output_nchw};
use common::{model_fixture, skip_notice};

const LAYOUTS: [ConvLayout; 2] = [ConvLayout::Nchw, ConvLayout::Nhwc];

#[derive(serde::Deserialize)]
struct CompactFixture {
    num_in_ch: usize,
    num_out_ch: usize,
    num_feat: usize,
    num_conv: usize,
    upscale: usize,
}

#[derive(serde::Deserialize)]
struct RrdbFixture {
    num_in_ch: usize,
    num_out_ch: usize,
    num_feat: usize,
    num_block: usize,
    num_grow_ch: usize,
    scale: usize,
}

fn compact_config() -> SrVggCompactConfig {
    let c: CompactFixture = load_config("compact");
    SrVggCompactConfig {
        num_in_ch: c.num_in_ch,
        num_out_ch: c.num_out_ch,
        num_feat: c.num_feat,
        num_conv: c.num_conv,
        upscale: c.upscale,
    }
}

fn rrdb_config() -> RrdbConfig {
    let c: RrdbFixture = load_config("rrdb");
    RrdbConfig {
        num_in_ch: c.num_in_ch,
        num_out_ch: c.num_out_ch,
        num_feat: c.num_feat,
        num_block: c.num_block,
        num_grow_ch: c.num_grow_ch,
        scale: c.scale,
    }
}

/// Create an empty directory `name` under the test target directory.
fn fresh_dir(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("super_resolution_dir")
        .join(name);
    if dir.exists() {
        std::fs::remove_dir_all(&dir).unwrap_or_else(|e| panic!("clear {}: {e}", dir.display()));
    }
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("create {}: {e}", dir.display()));
    dir
}

/// A directory `name` holding a copy of each `(source, file name)`.
fn stage(name: &str, files: &[(PathBuf, &str)]) -> PathBuf {
    let dir = fresh_dir(name);
    for (source, file) in files {
        std::fs::copy(source, dir.join(file))
            .unwrap_or_else(|e| panic!("copy {}: {e}", source.display()));
    }
    dir
}

/// A directory `name` holding a symlink to the real checkpoint `source`.
fn stage_link(name: &str, source: &Path) -> PathBuf {
    let dir = fresh_dir(name);
    let file = source.file_name().expect("checkpoint file name");
    std::os::unix::fs::symlink(source, dir.join(file))
        .unwrap_or_else(|e| panic!("link {}: {e}", source.display()));
    dir
}

fn fixture(set: &str, file: &str) -> PathBuf {
    fixtures_dir(set).join(file)
}

/// A deterministic F32 NCHW image of `[1, 3, h, w]`.
fn image(device: &CpuDevice, h: usize, w: usize) -> Tensor<CpuRuntime> {
    let data: Vec<f32> = (0..3 * h * w).map(|i| (i % 23) as f32 / 23.0).collect();
    Tensor::<CpuRuntime>::from_slice(&data, &[1, 3, h, w], device).expect("image tensor")
}

fn error_text<T>(result: boostr::error::Result<T>) -> String {
    result.err().map(|e| e.to_string()).unwrap_or_default()
}

/// Load `dir` in both layouts and hold `forward_native` to the direct
/// network call on the same input. The outputs must match exactly.
fn assert_native_equals_direct(dir: &Path, scale: usize, is_compact: bool) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    for layout in LAYOUTS {
        let exec = SrExecution::new(DType::F32, layout);
        let model = load_sr_dir::<CpuRuntime>(dir, &device, exec)
            .unwrap_or_else(|e| panic!("load_sr_dir {}: {e}", dir.display()));
        assert_eq!(matches!(model, SrModel::Compact(_)), is_compact);
        assert_eq!(model.native_scale(), scale);
        let input = sr_input(&client, &image(&device, 13, 11), &exec);
        let native = model
            .forward_native(&client, &input)
            .expect("forward_native");
        let want = direct_forward(&model, &client, &input);
        let (shape, got) = sr_output_nchw(&client, &native, &exec);
        let (want_shape, want) = sr_output_nchw(&client, &want, &exec);
        assert_eq!(shape, want_shape, "{layout:?}: shape");
        assert_eq!(shape, [1, 3, 13 * scale, 11 * scale], "{layout:?}: scale");
        let (diff, at) = max_abs_diff(&got, &want);
        assert_eq!(diff, 0.0, "{layout:?}: differs at {at}");
    }
}

/// The direct call of whichever network `model` holds.
fn direct_forward(
    model: &SrModel<CpuRuntime>,
    client: &<CpuRuntime as Runtime>::Client,
    x: &Tensor<CpuRuntime>,
) -> Tensor<CpuRuntime> {
    match model {
        SrModel::Compact(net) => net.forward_raw(client, x).expect("forward_raw"),
        SrModel::Rrdb(net) => net.forward(client, x).expect("forward"),
    }
}

#[test]
fn compact_dir_loads_compact_and_native_is_forward_raw() {
    let dir = stage(
        "compact",
        &[
            (fixture("compact", "weights.pth"), "weights.pth"),
            (fixture("compact", "config.json"), "config.json"),
            (
                fixture("compact", "weights.safetensors"),
                "weights.safetensors",
            ),
        ],
    );
    assert_native_equals_direct(&dir, 4, true);
}

#[test]
fn compact_dir_matches_load_compact_weights() {
    let dir = stage(
        "compact_direct",
        &[(fixture("compact", "weights.pth"), "m.pth")],
    );
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let exec = SrExecution::new(DType::F32, ConvLayout::Nchw);
    let model = load_sr_dir::<CpuRuntime>(&dir, &device, exec).expect("load_sr_dir");
    let net = load_compact::<CpuRuntime>(
        fixture("compact", "weights.pth"),
        &device,
        &compact_config(),
        exec,
    )
    .expect("load_compact");
    let input = image(&device, 13, 11);
    let got = model.forward_native(&client, &input).expect("native");
    let want = net.forward_raw(&client, &input).expect("forward_raw");
    let (_, got) = sr_output_nchw(&client, &got, &exec);
    let (_, want) = sr_output_nchw(&client, &want, &exec);
    assert_eq!(max_abs_diff(&got, &want).0, 0.0);
}

#[test]
fn rrdb_current_keys_dir_loads_rrdb_and_native_is_forward() {
    let dir = stage("rrdb", &[(fixture("rrdb", "weights.pth"), "weights.pth")]);
    assert_native_equals_direct(&dir, 2, false);
}

#[test]
fn rrdb_legacy_keys_dir_loads_rrdb_and_native_is_forward() {
    let dir = stage(
        "rrdb_legacy",
        &[(fixture("rrdb", "weights_legacy.pth"), "weights_legacy.pth")],
    );
    assert_native_equals_direct(&dir, 2, false);
}

#[test]
fn rrdb_dir_matches_load_rrdb_weights() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let exec = SrExecution::new(DType::F32, ConvLayout::Nchw);
    let input = image(&device, 13, 11);
    for file in ["weights.pth", "weights_legacy.pth"] {
        let dir = stage(
            &format!("rrdb_direct_{file}"),
            &[(fixture("rrdb", file), file)],
        );
        let model = load_sr_dir::<CpuRuntime>(&dir, &device, exec).expect("load_sr_dir");
        let net = load_rrdb::<CpuRuntime>(fixture("rrdb", file), &device, &rrdb_config(), exec)
            .expect("load_rrdb");
        let got = model.forward_native(&client, &input).expect("native");
        let want = net.forward(&client, &input).expect("forward");
        let (_, got) = sr_output_nchw(&client, &got, &exec);
        let (_, want) = sr_output_nchw(&client, &want, &exec);
        assert_eq!(max_abs_diff(&got, &want).0, 0.0, "{file}");
    }
}

#[test]
fn dir_without_pth_is_an_error_naming_the_directory() {
    let dir = stage(
        "none",
        &[(fixture("compact", "config.json"), "config.json")],
    );
    let err = error_text(load_sr_dir::<CpuRuntime>(
        &dir,
        &CpuDevice::new(),
        SrExecution::new(DType::F32, ConvLayout::Nchw),
    ));
    assert!(err.contains(&dir.display().to_string()), "{err}");
    assert!(err.contains("found 0: none"), "{err}");
}

#[test]
fn dir_with_two_pth_files_names_both() {
    let dir = stage(
        "two",
        &[
            (fixture("compact", "weights.pth"), "a.pth"),
            (fixture("rrdb", "weights.pth"), "b.pth"),
        ],
    );
    let err = error_text(load_sr_dir::<CpuRuntime>(
        &dir,
        &CpuDevice::new(),
        SrExecution::new(DType::F32, ConvLayout::Nchw),
    ));
    assert!(err.contains(&dir.display().to_string()), "{err}");
    assert!(err.contains("found 2: a.pth, b.pth"), "{err}");
}

#[test]
fn missing_dir_is_an_error_naming_the_directory() {
    let dir = fresh_dir("missing").join("absent");
    let err = error_text(load_sr_dir::<CpuRuntime>(
        &dir,
        &CpuDevice::new(),
        SrExecution::new(DType::F32, ConvLayout::Nchw),
    ));
    assert!(err.contains(&dir.display().to_string()), "{err}");
}

#[test]
fn unknown_key_set_names_sample_keys_and_directory() {
    let source =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/torch_pt/wrap_root.pth");
    let dir = stage("unknown", &[(source, "unknown.pth")]);
    let err = error_text(load_sr_dir::<CpuRuntime>(
        &dir,
        &CpuDevice::new(),
        SrExecution::new(DType::F32, ConvLayout::Nchw),
    ));
    assert!(err.contains(&dir.display().to_string()), "{err}");
    assert!(err.contains("conv.weight"), "{err}");
    assert!(err.contains("conv.bias"), "{err}");
}

#[test]
fn recognised_key_set_with_extra_keys_fails_in_the_strict_loader() {
    let dir = stage(
        "unexpected",
        &[(
            fixture("rrdb", "weights_unexpected.pth"),
            "weights_unexpected.pth",
        )],
    );
    let err = error_text(load_sr_dir::<CpuRuntime>(
        &dir,
        &CpuDevice::new(),
        SrExecution::new(DType::F32, ConvLayout::Nchw),
    ));
    assert!(err.contains("weights_unexpected.pth"), "{err}");
    assert!(err.contains("unexpected"), "{err}");
}

/// Load a real checkpoint through `load_sr_dir` and check the variant, the
/// scale, and `forward_native` against the direct call.
fn check_real(var: &str, rel: &str, label: &str, compact: bool, scale: usize) {
    let Some(checkpoint) = model_fixture(var, rel) else {
        skip_notice(label, var);
        return;
    };
    let dir = stage_link(&format!("real_{label}"), &checkpoint);
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let exec = SrExecution::new(DType::F32, ConvLayout::Nchw);
    let model = load_sr_dir::<CpuRuntime>(&dir, &device, exec)
        .unwrap_or_else(|e| panic!("load_sr_dir {label}: {e}"));
    assert_eq!(matches!(model, SrModel::Compact(_)), compact, "{label}");
    assert_eq!(model.native_scale(), scale, "{label}");
    let input = image(&device, 15, 13);
    let native = model
        .forward_native(&client, &input)
        .expect("forward_native");
    let want = direct_forward(&model, &client, &input);
    let (shape, got) = sr_output_nchw(&client, &native, &exec);
    let (_, want) = sr_output_nchw(&client, &want, &exec);
    assert_eq!(shape, [1, 3, 15 * scale, 13 * scale], "{label}");
    assert_eq!(max_abs_diff(&got, &want).0, 0.0, "{label}");
}

#[test]
fn real_realesr_general_x4v3() {
    check_real(
        "REALESR_GENERAL_X4V3",
        "realesr-general-x4v3/realesr-general-x4v3.pth",
        "realesr-general-x4v3",
        true,
        4,
    );
}

#[test]
fn real_realesrgan_x2plus() {
    check_real(
        "REALESRGAN_X2PLUS_PTH",
        "RealESRGAN_x2plus/RealESRGAN_x2plus.pth",
        "RealESRGAN_x2plus",
        false,
        2,
    );
}

#[test]
fn real_purevision_2x() {
    check_real(
        "PUREVISION_2X_PTH",
        "2x_PureVision/2x_PureVision.pth",
        "2x_PureVision",
        false,
        2,
    );
}
