//! Verify the ported VoxCPM2 `base_lm` (`MiniCpm4Model`) against the Python
//! model's own output.
//!
//! ```text
//! cargo run --release --features audio,f16 --example voxcpm_baselm_check -- CKPT_DIR FIXTURE_DIR
//! ```
//!
//! `CKPT_DIR` is the VoxCPM2 checkpoint (`config.json` + `model.safetensors`).
//! `FIXTURE_DIR` holds `baselm_fixture.safetensors`, written by
//! `audio/pipeline/make_baselm_fixture.py` from the reference implementation.
//! `--features f16` is required: the checkpoint is BF16 and the cast path is
//! f16-gated even though the model runs in F32.
//!
//! This is the gate for the port, same posture as `voxcpm_locenc_check` and
//! `voxcpm_vae_check`: running and producing plausible numbers proves
//! nothing, reproducing the reference output does. Exits non-zero on
//! mismatch.
//!
//! Tolerance is 2e-3 absolute, same as the local-encoder gate. Observed:
//! 3.6e-5, 5.5e-5 and 3.9e-4 on spans of 15.5, 30.7 and 44.0, i.e. 2e-6 to
//! 9e-6 relative. The error grows with sequence length because a longer
//! sequence accumulates through more attention terms, not because the
//! behaviour differs - all three run the same 28 layers in f32.
use boostr::format::safetensors_loader::SafeTensorsLoader;
use boostr::model::audio::voxcpm::minicpm4::{MiniCpm4Config, MiniCpm4Model};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ck = PathBuf::from(std::env::args().nth(1).expect("checkpoint dir"));
    let fx_dir = PathBuf::from(std::env::args().nth(2).expect("fixture dir"));
    let device = CpuDevice::default();
    let client = CpuClient::new(device.clone());

    let cfg = MiniCpm4Config::from_config_json(ck.join("config.json"))?;
    // The checkpoint is BF16; RoPE and RMSNorm upcast internally anyway, so
    // f32 is the cleaner ground truth.
    let model = MiniCpm4Model::<CpuRuntime>::from_safetensors_with(
        ck.join("model.safetensors"),
        "base_lm",
        cfg,
        &device,
        Some(DType::F32),
    )?;

    let mut fx = SafeTensorsLoader::open(fx_dir.join("baselm_fixture.safetensors"))?;
    let mut ok = true;
    for c in 0..3 {
        let x = fx.load_tensor::<CpuRuntime>(&format!("case{c}_in"), &device)?;
        let want = fx.load_tensor::<CpuRuntime>(&format!("case{c}_out"), &device)?;
        let got = model.forward(&client, &Var::new(x.clone(), false))?;
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
