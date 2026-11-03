//! Real-checkpoint parity for [`RrdbNet`] on `RealESRGAN_x2plus` (modern
//! keys) and `2x_PureVision` (legacy ESRGAN keys plus `metadata.*`) against
//! torch CPU F32.
//!
//! Run with:
//!   `cargo test --release --features super-resolution --test super_resolution_rrdb_reference`
//!   `cargo test --release --features cuda,f16,super-resolution --test super_resolution_rrdb_reference`
//!
//! Not checked in. Each network skips unless both are present:
//!
//! * the checkpoint: `REALESRGAN_X2PLUS_PTH`, else
//!   `$BOOSTR_MODELS_DIR/RealESRGAN_x2plus/RealESRGAN_x2plus.pth`, and
//!   `PUREVISION_2X_PTH`, else
//!   `$BOOSTR_MODELS_DIR/2x_PureVision/2x_PureVision.pth`
//! * the golden: `SR_REF_REALESRGAN_X2PLUS_DIR`, else
//!   `$BOOSTR_MODELS_DIR/sr_reference/realesrgan_x2plus`, and
//!   `SR_REF_PUREVISION_DIR`, else
//!   `$BOOSTR_MODELS_DIR/sr_reference/purevision`, written by
//!   `python3 tests/tools/sr_reference_dump/sr_reference_dump.py <network>`
//!
//! Each golden holds one synthetic 3×67×45 frame, odd in both dimensions,
//! and its 3×134×90 output. The u8 checks quantise as
//! `round_half_even(clamp(x, 0, 1) · 255)`. Each golden's `index.json`
//! holds torch's own error on the same frame for every precision below.

use boostr::format::SafeTensors;
use boostr::model::super_resolution::{RrdbConfig, RrdbNet, SrClient, SrExecution, load_rrdb};
use numr::dtype::DType;
use numr::ops::ConvLayout;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use std::path::PathBuf;

mod common;
use common::super_resolution::{Bounds, assert_within, golden_cpu, sr_input, sr_output_nchw};
use common::{model_fixture, skip_notice};

/// One real checkpoint and its golden.
struct Network {
    label: &'static str,
    checkpoint_var: &'static str,
    checkpoint_rel: &'static str,
    golden_var: &'static str,
    golden_rel: &'static str,
}

const REALESRGAN_X2PLUS: Network = Network {
    label: "RealESRGAN_x2plus",
    checkpoint_var: "REALESRGAN_X2PLUS_PTH",
    checkpoint_rel: "RealESRGAN_x2plus/RealESRGAN_x2plus.pth",
    golden_var: "SR_REF_REALESRGAN_X2PLUS_DIR",
    golden_rel: "sr_reference/realesrgan_x2plus",
};

const PUREVISION: Network = Network {
    label: "2x_PureVision",
    checkpoint_var: "PUREVISION_2X_PTH",
    checkpoint_rel: "2x_PureVision/2x_PureVision.pth",
    golden_var: "SR_REF_PUREVISION_DIR",
    golden_rel: "sr_reference/purevision",
};

/// F32 bound. Torch CPU F64 and torch CUDA F32 without TF32 stay far
/// inside it on both checkpoints. Their only u8 flips sit on a rounding
/// boundary.
const F32: Bounds = Bounds {
    max_abs: 3e-4,
    min_psnr_db: 100.0,
    min_u8_exact: 0.999,
    max_u8_diff: 1,
};

/// F16 bound. 351 convs each round their output to F16. The output lies in
/// [0, 1], where an F16 ulp is at most 2^-11, so each u8 step of 1/255
/// spans about 8 ulps and a few ulps of drift flip the nearest value.
/// Torch's own CUDA F16 run on this frame passes the bound with margin on
/// both checkpoints.
#[cfg(all(feature = "cuda", feature = "f16"))]
const F16: Bounds = Bounds {
    max_abs: 1e-2,
    min_psnr_db: 60.0,
    min_u8_exact: 0.88,
    max_u8_diff: 2,
};

/// BF16 bound. A BF16 ulp is 8× an F16 ulp. Torch's own CUDA BF16 run on
/// this frame passes the bound with margin on both checkpoints.
#[cfg(all(feature = "cuda", feature = "f16"))]
const BF16: Bounds = Bounds {
    max_abs: 6e-2,
    min_psnr_db: 42.0,
    min_u8_exact: 0.35,
    max_u8_diff: 8,
};

fn fixtures(network: &Network) -> Option<(PathBuf, PathBuf)> {
    let Some(checkpoint) = model_fixture(network.checkpoint_var, network.checkpoint_rel) else {
        skip_notice(network.checkpoint_rel, network.checkpoint_var);
        return None;
    };
    let Some(golden) = model_fixture(network.golden_var, network.golden_rel) else {
        skip_notice(network.golden_rel, network.golden_var);
        return None;
    };
    Some((checkpoint, golden.join("io.safetensors")))
}

/// Run the golden frame of `network` on `device` with `exec` and hold the
/// output to `bounds`. Skip when the checkpoint or the golden is absent.
fn check<R>(network: &Network, device: &R::Device, exec: SrExecution, bounds: &Bounds, label: &str)
where
    R: Runtime<DType = DType>,
    R::Client: SrClient<R>,
{
    let Some((checkpoint, golden)) = fixtures(network) else {
        return;
    };
    let label = format!("{} {label}", network.label);
    let client = R::default_client(device);
    let model: RrdbNet<R> = load_rrdb(&checkpoint, device, &RrdbConfig::realesrgan_x2plus(), exec)
        .unwrap_or_else(|e| panic!("load {}: {e}", checkpoint.display()));
    let mut io =
        SafeTensors::open(&golden).unwrap_or_else(|e| panic!("open {}: {e}", golden.display()));
    let frame = io
        .load_tensor::<R>("input", device)
        .unwrap_or_else(|e| panic!("golden tensor 'input': {e}"));
    let input = sr_input(&client, &frame, &exec);

    let out = model.forward(&client, &input).expect("forward");
    let (shape, got) = sr_output_nchw(&client, &out, &exec);
    let (want_shape, want) = golden_cpu(&golden, "out");
    assert_eq!(shape, want_shape, "{label}: shape");
    assert_within(&label, &got, &want, bounds);
}

fn cpu_f32_nchw(network: &Network) {
    check::<CpuRuntime>(
        network,
        &CpuDevice::new(),
        SrExecution::new(DType::F32, ConvLayout::Nchw),
        &F32,
        "cpu f32 nchw",
    );
}

#[test]
fn realesrgan_x2plus_cpu_f32_nchw_matches_torch() {
    cpu_f32_nchw(&REALESRGAN_X2PLUS);
}

#[test]
fn purevision_cpu_f32_nchw_matches_torch() {
    cpu_f32_nchw(&PUREVISION);
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use numr::runtime::cuda::{CudaDevice, CudaRuntime};

    fn nhwc(network: &Network, dtype: DType, bounds: &Bounds, label: &str) {
        check::<CudaRuntime>(
            network,
            &CudaDevice::new(0),
            SrExecution::new(dtype, ConvLayout::Nhwc),
            bounds,
            label,
        );
    }

    #[test]
    fn realesrgan_x2plus_cuda_f32_nhwc_matches_torch() {
        nhwc(&REALESRGAN_X2PLUS, DType::F32, &F32, "cuda f32 nhwc");
    }

    #[test]
    fn purevision_cuda_f32_nhwc_matches_torch() {
        nhwc(&PUREVISION, DType::F32, &F32, "cuda f32 nhwc");
    }

    #[cfg(feature = "f16")]
    #[test]
    fn realesrgan_x2plus_cuda_f16_nhwc_matches_torch() {
        nhwc(&REALESRGAN_X2PLUS, DType::F16, &F16, "cuda f16 nhwc");
    }

    #[cfg(feature = "f16")]
    #[test]
    fn purevision_cuda_f16_nhwc_matches_torch() {
        nhwc(&PUREVISION, DType::F16, &F16, "cuda f16 nhwc");
    }

    #[cfg(feature = "f16")]
    #[test]
    fn realesrgan_x2plus_cuda_bf16_nhwc_matches_torch() {
        nhwc(&REALESRGAN_X2PLUS, DType::BF16, &BF16, "cuda bf16 nhwc");
    }

    #[cfg(feature = "f16")]
    #[test]
    fn purevision_cuda_bf16_nhwc_matches_torch() {
        nhwc(&PUREVISION, DType::BF16, &BF16, "cuda bf16 nhwc");
    }
}
