//! Verify the ported VoxCPM2 `residual_lm` (`MiniCpm4Model` instantiated as
//! the 8-layer NoPE config) against the Python model's own output.
//!
//! ```text
//! cargo run --release --features audio,f16 --example voxcpm_residual_lm_check -- CKPT_DIR FIXTURE_DIR
//! ```
//!
//! `CKPT_DIR` is the VoxCPM2 checkpoint (`config.json` + `model.safetensors`).
//! `FIXTURE_DIR` holds `residual_lm_fixture.safetensors`, written by
//! `audio/pipeline/make_residual_lm_fixture.py` from the reference
//! implementation. `--features f16` is required: the checkpoint is BF16 and
//! the cast path is f16-gated even though the model runs in F32.
//!
//! This is the gate for the port, same posture as `voxcpm_baselm_check`,
//! `voxcpm_locenc_check` and `voxcpm_vae_check`: running and producing
//! plausible numbers proves nothing, reproducing the reference output does.
//! Exits non-zero on mismatch.
//!
//! `residual_lm` is the SAME `MiniCpm4Model` as `base_lm`, instantiated with
//! a different [`MiniCpm4Config`]: layer count and the NoPE switch come from
//! the checkpoint's top-level `residual_lm_num_layers` /
//! `residual_lm_no_rope` keys, never hardcoded here. After loading, the
//! model's `uses_rope()` is asserted false - a `residual_lm` that silently
//! loaded WITH rotation would still be caught by the numeric checks below,
//! but asserting it directly makes that failure mode legible on its own.
//!
//! Tolerance is 2e-3 absolute, same as the other VoxCPM2 gates. Observed on
//! the full-sequence cases: 1.2e-5, 3.1e-5 and 7.8e-5 on spans of 13.6, 22.4
//! and 23.5, i.e. 9e-7 to 3e-6 relative - tighter than `base_lm` because this
//! model is 8 layers deep, not 28.
//!
//! A fourth check exercises the incremental (KV-cached) `decode_step` path,
//! comparing against the reference's own step-wise output rather than its
//! full-sequence output. Same 2e-3 tolerance. The reference's own
//! full-sequence and step-wise paths differ by 1.43e-5 on this sub-model
//! (vs 9.9e-5 for `base_lm` - `residual_lm` has 8 layers vs 28, so the
//! per-layer reduction-order drift accumulates far less), which is the
//! floor for this check. Observed: 1.8e-5 on a span of 21.6, 8e-7 relative.
use boostr::format::safetensors_loader::SafeTensorsLoader;
use boostr::model::audio::voxcpm::minicpm4::{MiniCpm4Config, MiniCpm4Model};
use numr::autograd::{Var, var_cat, var_narrow, var_reshape};
use numr::dtype::DType;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ck = PathBuf::from(std::env::args().nth(1).expect("checkpoint dir"));
    let fx_dir = PathBuf::from(std::env::args().nth(2).expect("fixture dir"));
    let device = CpuDevice::default();
    let client = CpuClient::new(device.clone());

    let cfg = MiniCpm4Config::residual_lm_from_config_json(ck.join("config.json"))?;
    // The checkpoint is BF16; RoPE and RMSNorm upcast internally anyway, so
    // f32 is the cleaner ground truth.
    let model = MiniCpm4Model::<CpuRuntime>::from_safetensors_with(
        ck.join("model.safetensors"),
        "residual_lm",
        cfg,
        &device,
        Some(DType::F32),
    )?;
    assert!(
        !model.uses_rope(),
        "residual_lm must be a NoPE instantiation"
    );
    println!("uses_rope: {} (expected false)", model.uses_rope());

    let mut fx = SafeTensorsLoader::open(fx_dir.join("residual_lm_fixture.safetensors"))?;
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
    // Case 3: incremental (KV-cached) decode, checked against the
    // reference's own STEP-WISE output (`decode_out`, produced by feeding
    // `decode_in` through the reference's KV cache one position at a
    // time) rather than its full-sequence output. The reference's own
    // full-sequence and step-wise paths differ by 1.43e-5 on this model
    // (fewer layers than base_lm, so less reduction-order drift), so this
    // path is inherently noisier than the three full-sequence cases above
    // and exact agreement is not expected; same 2e-3 absolute tolerance.
    {
        let decode_in = fx.load_tensor::<CpuRuntime>("decode_in", &device)?;
        let want = fx.load_tensor::<CpuRuntime>("decode_out", &device)?;
        let shape = decode_in.shape().to_vec();
        let (batch, steps, hidden) = (shape[0], shape[1], shape[2]);
        let decode_in = Var::new(decode_in, false);

        let mut cache = model.new_kv_cache(batch, steps)?;
        let mut outs = Vec::with_capacity(steps);
        for p in 0..steps {
            let step_in = var_narrow(&decode_in, 1, p, 1)?;
            let step_in = var_reshape(&step_in, &[batch, hidden])?;
            let step_out = model.decode_step(&client, &step_in, &mut cache, p)?;
            outs.push(var_reshape(&step_out, &[batch, 1, hidden])?);
        }
        let out_refs: Vec<&Var<CpuRuntime>> = outs.iter().collect();
        let got = var_cat(&out_refs, 1, &client)?;
        let got = got.tensor();
        println!(
            "decode (step-wise, KV cache): {:?} -> got {:?} want {:?}",
            shape,
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
