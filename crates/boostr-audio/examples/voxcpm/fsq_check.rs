//! Verify the ported VoxCPM2 `fsq_layer` finite-scalar-quantization
//! bottleneck and its six auxiliary projections against the Python model's
//! own output.
//!
//! ```text
//! cargo run --release --features audio,f16 --example voxcpm_fsq_check -- CKPT_DIR FIXTURE_DIR
//! ```
//!
//! `CKPT_DIR` is the VoxCPM2 checkpoint (`config.json` + `model.safetensors`).
//! `FIXTURE_DIR` holds `fsq_fixture.safetensors`, written by
//! `audio/pipeline/make_fsq_fixture.py` from the reference implementation.
//!
//! This is the gate for the port, same posture as the other VoxCPM2 checks:
//! running and producing plausible numbers proves nothing, reproducing the
//! reference output does. Exits non-zero on mismatch.
//!
//! Covers, in order: `ScalarQuantization::forward` on three shapes (rank-3
//! `[2,3,2048]`, rank-3 `[1,7,2048]`, rank-2 `[4,2048]` — the per-step decode
//! shape), a dedicated rounding-tie check, the six `AuxProjections` linear
//! layers, and the `stop_head(silu(stop_proj(x)))` composition.
//!
//! ## The tie check
//!
//! A random input essentially never lands exactly on a rounding tie (post-
//! tanh value of the form `(k + 0.5) / scale`), so a bug that rounds ties
//! away from zero instead of to-even would pass every other case here
//! unnoticed. `tie_in`/`tie_out` in the fixture sit exactly on those ties, so
//! this check is the one that actually exercises the rounding rule.
//!
//! `tie_in` is a bare post-tanh vector, not an input to the full
//! `ScalarQuantization::forward` (which would additionally run it through
//! `in_proj`/`tanh`). `layer.rs` does not expose the quantization step
//! (`scale`, `round_ties_even`, `1/scale`) as a separately callable unit, so
//! this check does not call into `src/`'s port at all — it applies the same
//! rounding rule directly via `numr::tensor::Tensor::round_ties_even`
//! (`round_ties_even(h * scale) / scale`) and compares against `tie_out`.
//! This proves the rounding rule the port depends on is the right one; it
//! does NOT exercise `ScalarQuantization::forward`'s own use of that rule.
//! Correctness of the wiring inside `forward` is covered instead by the unit
//! tests in `src/model/audio/voxcpm/fsq/layer.rs`
//! (`quantization_matches_ties_to_even_not_ties_away`), and indirectly by
//! this file's three FSQ shape cases above matching the reference exactly.
//!
//! Tolerance is 2e-3 absolute, same as the other VoxCPM2 gates. Observed: the
//! three FSQ cases land at 7.2e-7, 9.5e-7 and 1.2e-6 (1e-7 to 2e-7 relative),
//! the six projections and the stop chain between 2.4e-7 and 4.1e-5 (all under
//! 5e-7 relative), and the tie case is EXACT - 0.0, bitwise.
//!
//! That bitwise zero is the point. Every other number here is f32 accumulation
//! noise, but a rounding rule either matches or it does not, so the tie case
//! has no tolerance to hide in.
use boostr::format::safetensors_loader::SafeTensorsLoader;
use boostr::model::audio::voxcpm::fsq::{AuxProjections, FsqConfig, ScalarQuantization};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::ScalarOps;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;
use std::path::PathBuf;

/// Load `<name>_in`/`<name>_out`, run `got = f(in)`, and report shapes / max
/// abs error / span / relative error / OK-or-MISMATCH in the sibling gates'
/// line format. Returns whether this case passed.
fn check(
    label: &str,
    got: &Tensor<CpuRuntime>,
    want: &Tensor<CpuRuntime>,
) -> Result<bool, Box<dyn std::error::Error>> {
    println!("{label}: got {:?} want {:?}", got.shape(), want.shape());
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
    println!(
        "  max abs err {max:.3e}  span {span:.4}  rel {:.2e}  {}",
        max / span.max(1e-9),
        if pass { "OK" } else { "MISMATCH" }
    );
    Ok(pass)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ck = PathBuf::from(std::env::args().nth(1).expect("checkpoint dir"));
    let fx_dir = PathBuf::from(std::env::args().nth(2).expect("fixture dir"));
    let device = CpuDevice::default();
    let client = CpuClient::new(device.clone());

    let cfg = FsqConfig::from_config_json(ck.join("config.json"))?;
    // f32 ground truth, same rationale as the DiT/locenc gates: the
    // checkpoint is BF16 and internal ops upcast anyway.
    let fsq = ScalarQuantization::<CpuRuntime>::from_safetensors(
        ck.join("model.safetensors"),
        cfg,
        &device,
        Some(DType::F32),
    )?;
    let aux = AuxProjections::<CpuRuntime>::from_safetensors(
        ck.join("model.safetensors"),
        cfg,
        &device,
        Some(DType::F32),
    )?;

    let mut fx = SafeTensorsLoader::open(fx_dir.join("fsq_fixture.safetensors"))?;
    let mut ok = true;

    // The three ScalarQuantization::forward shape cases.
    for name in ["fsq0", "fsq1", "fsq2"] {
        let input = fx.load_tensor::<CpuRuntime>(&format!("{name}_in"), &device)?;
        let want = fx.load_tensor::<CpuRuntime>(&format!("{name}_out"), &device)?;
        let got = fsq.forward(&client, &Var::new(input, false))?;
        ok &= check(name, got.tensor(), &want)?;
    }

    // Dedicated rounding-tie check — see the doc header for exactly what
    // this does and does not cover.
    {
        let tie_in = fx.load_tensor::<CpuRuntime>("tie_in", &device)?;
        let tie_out = fx.load_tensor::<CpuRuntime>("tie_out", &device)?;
        let scaled = tie_in.mul_scalar(cfg.scale as f64)?;
        let rounded = scaled.round_ties_even()?;
        let levels = client.div_scalar(&rounded, cfg.scale as f64)?;
        ok &= check(
            "tie (round_ties_even rule, not ScalarQuantization::forward)",
            &levels,
            &tie_out,
        )?;
    }

    // The six auxiliary projections.
    for name in [
        "enc_to_lm_proj",
        "lm_to_dit_proj",
        "res_to_dit_proj",
        "fusion_concat_proj",
        "stop_proj",
        "stop_head",
    ] {
        let input = fx.load_tensor::<CpuRuntime>(&format!("{name}_in"), &device)?;
        let want = fx.load_tensor::<CpuRuntime>(&format!("{name}_out"), &device)?;
        let linear = match name {
            "enc_to_lm_proj" => &aux.enc_to_lm_proj,
            "lm_to_dit_proj" => &aux.lm_to_dit_proj,
            "res_to_dit_proj" => &aux.res_to_dit_proj,
            "fusion_concat_proj" => &aux.fusion_concat_proj,
            "stop_proj" => &aux.stop_proj,
            "stop_head" => &aux.stop_head,
            _ => unreachable!(),
        };
        let got = linear.forward(&client, &Var::new(input, false))?;
        ok &= check(name, got.tensor(), &want)?;
    }

    // The full stop_head(silu(stop_proj(x))) composition.
    {
        let input = fx.load_tensor::<CpuRuntime>("stop_chain_in", &device)?;
        let want = fx.load_tensor::<CpuRuntime>("stop_chain_out", &device)?;
        let got = aux.stop(&client, &Var::new(input, false))?;
        ok &= check("stop_chain", got.tensor(), &want)?;
    }

    println!("\n{}", if ok { "VERIFIED" } else { "FAILED" });
    std::process::exit(if ok { 0 } else { 1 });
}
