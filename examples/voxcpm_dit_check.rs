//! Verify the ported VoxCPM2 `feat_decoder` local DiT ("locdit") estimator
//! forward against the Python model's own output.
//!
//! ```text
//! cargo run --release --features audio,f16 --example voxcpm_dit_check -- CKPT_DIR FIXTURE_DIR
//! ```
//!
//! `CKPT_DIR` is the VoxCPM2 checkpoint (`config.json` + `model.safetensors`).
//! `FIXTURE_DIR` holds `dit_fixture.safetensors`, written by
//! `audio/pipeline/make_dit_fixture.py` from the reference implementation.
//!
//! This is the gate for the port, same posture as `voxcpm_baselm_check` and
//! `voxcpm_locenc_check`: running and producing plausible numbers proves
//! nothing, reproducing the reference output does. Exits non-zero on
//! mismatch. This gate covers the estimator ONLY — `LocalDit::forward`
//! evaluated once per `(x, mu, t, cond, dt)`. The CFM sampler that steps the
//! estimator repeatedly (noise, Euler integration, classifier-free guidance)
//! is a separate unit and is gated separately; a pass here is not
//! end-to-end coverage of sampling.
//!
//! Tolerance is 2e-3 absolute, same as the other VoxCPM2 gates. Observed:
//! 1.9e-6, 3.0e-6, 1.4e-6 and 5.3e-6 on spans of 6.6, 5.4, 5.3 and 6.4, i.e.
//! 3e-7 to 8e-7 relative. Flat across t, which is the point of spanning
//! t=0.001 to t=0.999: a timestep-embedding constant that is off (the
//! half_dim-1 divisor, the scale of 1000, or sin/cos order) skews with t
//! rather than shifting every case equally.
use boostr::format::safetensors_loader::SafeTensorsLoader;
use boostr::model::audio::voxcpm::local_dit::{LocalDit, LocalDitConfig};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ck = PathBuf::from(std::env::args().nth(1).expect("checkpoint dir"));
    let fx_dir = PathBuf::from(std::env::args().nth(2).expect("fixture dir"));
    let device = CpuDevice::default();
    let client = CpuClient::new(device.clone());

    let cfg = LocalDitConfig::from_config_json(ck.join("config.json"))?;
    // The checkpoint is BF16; RoPE and RMSNorm upcast internally anyway, so
    // f32 is the cleaner ground truth. `feat_decoder`'s tensors live under
    // the default `feat_decoder` prefix, so the plain entry point applies.
    let model = LocalDit::<CpuRuntime>::from_safetensors(
        ck.join("model.safetensors"),
        cfg,
        &device,
        Some(DType::F32),
    )?;

    let mut fx = SafeTensorsLoader::open(fx_dir.join("dit_fixture.safetensors"))?;
    let mut ok = true;
    for c in 0..4 {
        let x = fx.load_tensor::<CpuRuntime>(&format!("case{c}_x"), &device)?;
        let mu = fx.load_tensor::<CpuRuntime>(&format!("case{c}_mu"), &device)?;
        let cond = fx.load_tensor::<CpuRuntime>(&format!("case{c}_cond"), &device)?;
        let t = fx.load_tensor::<CpuRuntime>(&format!("case{c}_t"), &device)?;
        let dt = fx.load_tensor::<CpuRuntime>(&format!("case{c}_dt"), &device)?;
        let want = fx.load_tensor::<CpuRuntime>(&format!("case{c}_out"), &device)?;

        let t_vals: Vec<f32> = t.contiguous()?.to_vec();

        let got = model.forward(
            &client,
            &Var::new(x.clone(), false),
            &Var::new(mu.clone(), false),
            &Var::new(t.clone(), false),
            &Var::new(cond.clone(), false),
            &Var::new(dt.clone(), false),
        )?;
        let got = got.tensor();
        println!(
            "case{c}: t={t_vals:?} x={:?} -> got {:?} want {:?}",
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
