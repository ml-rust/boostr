//! Verify the ported VoxCPM2 `feat_encoder` (local encoder) against the
//! Python model's own output.
//!
//! ```text
//! cargo run --release --features audio,f16 --example voxcpm_locenc_check -- CKPT_DIR FIXTURE_DIR
//! ```
//!
//! `CKPT_DIR` is the VoxCPM2 checkpoint (`config.json` + `model.safetensors`).
//! `FIXTURE_DIR` holds `locenc_fixture.safetensors`, written by
//! `audio/pipeline/make_locenc_fixture.py` from the reference implementation.
//!
//! This is the gate for the port, same posture as `voxcpm_vae_check`: running
//! and producing plausible numbers proves nothing, reproducing the reference
//! output does. Exits non-zero on mismatch.
//!
//! Tolerance is 2e-3 absolute. Observed: ~4e-6 and ~1.5e-5 on a ~22 span,
//! i.e. ~2e-7 to 7e-7 relative - f32 accumulation through 4 transformer
//! layers, not a behavioural difference.
use boostr::format::safetensors_loader::SafeTensorsLoader;
use boostr::model::audio::voxcpm::local_encoder::{LocalEncoder, LocalEncoderConfig};
use numr::autograd::Var;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ck = PathBuf::from(std::env::args().nth(1).expect("checkpoint dir"));
    let fx_dir = PathBuf::from(std::env::args().nth(2).expect("fixture dir"));
    let device = CpuDevice::default();
    let client = CpuClient::new(device.clone());

    let cfg = LocalEncoderConfig::from_config_json(ck.join("config.json"))?;
    println!("cfg loaded: short_factor[0..3] check done via config.json");
    // The checkpoint is BF16; the fixture is f32 because RoPE and RMSNorm
    // upcast internally anyway, so f32 is the cleaner ground truth.
    let enc = LocalEncoder::<CpuRuntime>::from_safetensors_with(
        ck.join("model.safetensors"),
        "feat_encoder",
        cfg,
        &device,
        Some(numr::dtype::DType::F32),
    )?;

    let mut fx = SafeTensorsLoader::open(fx_dir.join("locenc_fixture.safetensors"))?;
    let mut ok = true;
    for c in 0..2 {
        let x = fx.load_tensor::<CpuRuntime>(&format!("case{c}_in"), &device)?;
        let want = fx.load_tensor::<CpuRuntime>(&format!("case{c}_out"), &device)?;
        let got = enc.forward(&client, &Var::new(x.clone(), false))?;
        let got = got.tensor();
        println!(
            "case{c}: {:?} -> got {:?} want {:?}",
            x.shape(),
            got.shape(),
            want.shape()
        );
        assert_eq!(got.shape(), want.shape());
        let g: Vec<f32> = got.contiguous()?.to_vec();
        let w: Vec<f32> = want.contiguous()?.to_vec();
        let max = g
            .iter()
            .zip(&w)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let span =
            w.iter().cloned().fold(f32::MIN, f32::max) - w.iter().cloned().fold(f32::MAX, f32::min);
        let pass = max <= 2e-3;
        ok &= pass;
        println!(
            "  max abs err {max:.3e}  span {span:.4}  rel {:.2e}  {}",
            max / span.max(1e-9),
            if pass { "OK" } else { "MISMATCH" }
        );
    }
    println!("\n{}", if ok { "VERIFIED" } else { "FAILED" });
    std::process::exit(if ok { 0 } else { 1 });
}
