//! `TorchPthSource` and the `.pth` reader: root prefix detection, strided
//! views, storage offsets, and F16/BF16 storages.
//!
//! Fixtures are tiny `torch.save` files under `tests/fixtures/torch_pt/`.
//! `tests/fixtures/torch_pt/make_fixtures.py` regenerates them. The last
//! tests open the real super-resolution checkpoints and skip when absent.

mod common;

use boostr::format::torch_pt::{ROOT_PREFIXES, detect_root};
use boostr::format::weight_source::WeightSource;
use boostr::format::{PtScalar, TorchPthSource, TorchStateDict};
use common::{model_fixture, skip_notice};
use numr::dtype::DType;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use std::path::PathBuf;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/torch_pt")
        .join(name)
}

fn load_f32(source: &mut TorchPthSource, name: &str) -> Vec<f32> {
    let device = CpuDevice::new();
    WeightSource::<CpuRuntime>::load_named(source, name, &device)
        .unwrap_or_else(|e| panic!("load {name}: {e}"))
        .to_vec::<f32>()
}

/// Open `file`, check the detected root, and read `conv.weight` and
/// `conv.bias` through it. The generator scales both by `scale`.
fn check_wrapper(file: &str, root: &str, scale: f32) -> TorchPthSource {
    let mut source = TorchPthSource::open(fixture(file)).expect("open fixture");
    assert_eq!(source.root(), root, "{file}");
    assert!(WeightSource::<CpuRuntime>::has_named(
        &source,
        "conv.weight"
    ));
    assert!(!WeightSource::<CpuRuntime>::has_named(
        &source,
        "metadata.in_nc"
    ));
    let weight: Vec<f32> = (0..6).map(|i| i as f32 * scale).collect();
    assert_eq!(load_f32(&mut source, "conv.weight"), weight, "{file}");
    assert_eq!(load_f32(&mut source, "conv.bias"), vec![scale, -scale]);
    let mut names: Vec<&str> = source.names().collect();
    names.sort_unstable();
    assert_eq!(names, ["conv.bias", "conv.weight"], "{file}");
    source
}

#[test]
fn root_prefixes_keep_reference_order() {
    assert_eq!(ROOT_PREFIXES, ["params_ema.", "params.", "state_dict."]);
}

#[test]
fn params_ema_wins_over_params() {
    let source = check_wrapper("wrap_params_ema.pth", "params_ema.", 1.0);
    let state = source.state_dict();
    assert!(state.has("params.conv.weight"));
    assert_eq!(state.scalar("metadata.in_nc"), Some(&PtScalar::Int(3)));
    assert_eq!(
        state.scalar("metadata.name"),
        Some(&PtScalar::Str("tiny".into()))
    );
}

#[test]
fn params_root_is_detected() {
    let source = check_wrapper("wrap_params.pth", "params.", 2.0);
    let in_nc = source.state_dict().scalar("metadata.in_nc");
    assert_eq!(in_nc, Some(&PtScalar::Int(3)));
}

#[test]
fn state_dict_root_is_detected() {
    let source = check_wrapper("wrap_state_dict.pth", "state_dict.", 3.0);
    let in_nc = source.state_dict().scalar("metadata.in_nc");
    assert_eq!(in_nc, Some(&PtScalar::Int(3)));
}

#[test]
fn bare_state_dict_reads_at_the_root() {
    let source = check_wrapper("wrap_root.pth", "", 4.0);
    assert_eq!(detect_root(source.state_dict()), "");
}

#[test]
fn explicit_root_overrides_detection() {
    let mut source =
        TorchPthSource::open_with_root(fixture("wrap_params_ema.pth"), "params.").expect("open");
    assert_eq!(source.root(), "params.");
    let weight: Vec<f32> = (0..6).map(|i| i as f32 * 2.0).collect();
    assert_eq!(load_f32(&mut source, "conv.weight"), weight);
}

#[test]
fn transposed_tensor_is_an_error_naming_key_shape_and_strides() {
    let state = TorchStateDict::open(fixture("strides.pth")).expect("open");
    let err = state
        .load_tensor::<CpuRuntime>("bad", &CpuDevice::new())
        .err()
        .map(|e| e.to_string())
        .expect("a transposed tensor must not load");
    for part in ["\"bad\"", "[3, 2]", "[1, 3]", "[2, 1]", ".contiguous()"] {
        assert!(err.contains(part), "missing {part} in: {err}");
    }
}

#[test]
fn size_one_dim_with_any_stride_loads() {
    let state = TorchStateDict::open(fixture("strides.pth")).expect("open");
    let t = state
        .load_tensor::<CpuRuntime>("ok_size1", &CpuDevice::new())
        .expect("size-1 dims take any stride");
    assert_eq!(t.shape(), &[1, 3]);
    assert_eq!(t.to_vec::<f32>(), vec![0.0, 1.0, 2.0]);
}

#[test]
fn view_reads_from_its_storage_offset() {
    let state = TorchStateDict::open(fixture("offset.pth")).expect("open");
    let t = state
        .load_tensor::<CpuRuntime>("view", &CpuDevice::new())
        .expect("load view");
    assert_eq!(t.shape(), &[4]);
    assert_eq!(t.to_vec::<f32>(), vec![1.5, 2.0, 2.5, 3.0]);
}

#[cfg(feature = "f16")]
const HALF_VALUES: [f32; 6] = [1.0, -2.5, 0.125, 3.0, 0.0, -0.75];

#[cfg(feature = "f16")]
#[test]
fn f16_and_bf16_round_trip() {
    let state = TorchStateDict::open(fixture("half.pth")).expect("open");
    let device = CpuDevice::new();

    let h = state
        .load_tensor::<CpuRuntime>("f16", &device)
        .expect("load f16");
    assert_eq!(h.dtype(), DType::F16);
    assert_eq!(h.shape(), &[2, 3]);
    let h: Vec<f32> = h.to_vec::<half::f16>().iter().map(|v| v.to_f32()).collect();
    assert_eq!(h, HALF_VALUES);

    let b = state
        .load_tensor::<CpuRuntime>("bf16", &device)
        .expect("load bf16");
    assert_eq!(b.dtype(), DType::BF16);
    assert_eq!(b.shape(), &[2, 3]);
    let b: Vec<f32> = b
        .to_vec::<half::bf16>()
        .iter()
        .map(|v| v.to_f32())
        .collect();
    assert_eq!(b, HALF_VALUES);
}

#[cfg(not(feature = "f16"))]
#[test]
fn half_storages_name_the_f16_feature() {
    let state = TorchStateDict::open(fixture("half.pth")).expect("open");
    for (key, dtype) in [("f16", DType::F16), ("bf16", DType::BF16)] {
        assert_eq!(state.shape(key), Some(&[2usize, 3][..]));
        let err = state
            .load_tensor::<CpuRuntime>(key, &CpuDevice::new())
            .err()
            .map(|e| e.to_string())
            .expect("half storage needs the f16 feature");
        assert!(err.contains(&format!("{key:?}")), "{err}");
        assert!(err.contains(&format!("{dtype:?}")), "{err}");
        assert!(err.contains("`f16` feature"), "{err}");
    }
}

/// Open a real checkpoint, check its root, and read one tensor through it.
fn check_real(var: &str, relative: &str, root: &str, tensor: &str, shape: &[usize]) {
    let Some(path) = model_fixture(var, relative) else {
        skip_notice(relative, var);
        return;
    };
    let mut source = TorchPthSource::open(&path).expect("open checkpoint");
    assert_eq!(source.root(), root, "{}", path.display());
    let t = WeightSource::<CpuRuntime>::load_named(&mut source, tensor, &CpuDevice::new())
        .unwrap_or_else(|e| panic!("load {tensor}: {e}"));
    assert_eq!(t.shape(), shape, "{tensor}");
}

#[test]
fn realesr_general_x4v3_reads_under_params() {
    check_real(
        "REALESR_GENERAL_X4V3_PTH",
        "realesr-general-x4v3/realesr-general-x4v3.pth",
        "params.",
        "body.0.weight",
        &[64, 3, 3, 3],
    );
}

#[test]
fn realesrgan_x2plus_reads_under_params_ema() {
    check_real(
        "REALESRGAN_X2PLUS_PTH",
        "RealESRGAN_x2plus/RealESRGAN_x2plus.pth",
        "params_ema.",
        "conv_first.weight",
        &[64, 12, 3, 3],
    );
}

#[test]
fn purevision_reads_under_params_ema_with_metadata() {
    let (var, relative) = ("PUREVISION_2X_PTH", "2x_PureVision/2x_PureVision.pth");
    check_real(
        var,
        relative,
        "params_ema.",
        "model.0.weight",
        &[64, 12, 3, 3],
    );
    let Some(path) = model_fixture(var, relative) else {
        return;
    };
    let source = TorchPthSource::open(&path).expect("open checkpoint");
    let in_nc = source.state_dict().scalar("metadata.in_nc");
    assert_eq!(in_nc, Some(&PtScalar::Int(12)));
}
