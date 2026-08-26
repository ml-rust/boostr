//! Verify the ported VoxCPM2 CFM sampler ([`LocalDit::solve_euler`]) against
//! the Python model's own trajectory.
//!
//! ```text
//! cargo run --release --features audio,f16 --example voxcpm_cfm_check -- CKPT_DIR FIXTURE_DIR
//! ```
//!
//! `CKPT_DIR` is the VoxCPM2 checkpoint (`config.json` + `model.safetensors`).
//! `FIXTURE_DIR` holds `cfm_fixture.safetensors`, written by
//! `audio/pipeline/make_cfm_fixture.py` from the reference implementation.
//!
//! Same posture as `voxcpm_dit_check` and the other VoxCPM2 gates: running and
//! producing plausible numbers proves nothing, reproducing the reference
//! output does. Exits non-zero on mismatch.
//!
//! Unlike the estimator gate, this one cannot stop at a final-output
//! comparison. The sampler's failure mode is plausible-but-wrong output: a
//! wrong guidance branch, a wrong schedule, or a missed warmup step all still
//! produce a `[batch, feat_dim, patch_size]` tensor of a plausible magnitude.
//! A final-output-only check cannot tell those apart, and a mismatch there
//! gives no lead on which of the ten steps introduced it. So this gate checks
//! three things in order, each isolating one failure class from the next:
//!
//! 1. **Schedule** ([`cfm_time_span`]) against the fixture's `t_span`,
//!    checked FIRST and independently of the integrator — a schedule bug
//!    must not be misread as a sampler bug.
//! 2. **Per-step trajectory**: `solve_euler` is run with the fixture's OWN
//!    `t_span` (not the recomputed one), so a schedule failure from step 1
//!    cannot contaminate this check. Every step's `x` is compared against the
//!    reference, one line per step, so a divergence names the exact step it
//!    first appears at. `step0_x` (the state after step 1, including the
//!    forced zero-velocity warmup) is additionally asserted to equal `z`
//!    bitwise.
//! 3. **Final output**: the returned value against `out`.
//!
//! Every input — `z`, `mu`, `cond`, `t_span` — is loaded from the fixture.
//! Nothing is drawn or recomputed: noise is injected rather than seeded
//! because torch's RNG and numr's are not reproducible against each other, so
//! `sample`'s own noise draw is never exercised here — only `solve_euler`.
//!
//! Tolerance is 2e-3 absolute, same as the other VoxCPM2 gates. Observed: the
//! schedule matches bitwise (0.0), step0 equals z bitwise, and the trajectory
//! error grows monotonically 7.7e-7 -> 7.9e-6 over the ten Euler steps, ending
//! at 7.9e-6 on the final output.
//!
//! That growth curve is itself the evidence. Accumulating f32 error through ten
//! sequential estimator calls ramps smoothly; the failure modes this gate exists
//! to catch do not. A swapped conditional/unconditional branch, a missed warmup,
//! or a plain-linspace schedule all break at a specific step and stay broken, so
//! they show up as a jump, not a ramp.
use boostr::format::safetensors_loader::SafeTensorsLoader;
use boostr::model::audio::voxcpm::local_dit::{LocalDit, LocalDitConfig, cfm_time_span};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use std::path::PathBuf;

fn max_abs_err(got: &[f32], want: &[f32]) -> f32 {
    got.iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

fn report(label: &str, max: f32) -> bool {
    let pass = max <= 2e-3;
    println!(
        "  {label}: max abs err {max:.3e}  {}",
        if pass { "OK" } else { "MISMATCH" }
    );
    pass
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ck = PathBuf::from(std::env::args().nth(1).expect("checkpoint dir"));
    let fx_dir = PathBuf::from(std::env::args().nth(2).expect("fixture dir"));
    let device = CpuDevice::default();
    let client = CpuClient::new(device.clone());

    let cfg = LocalDitConfig::from_config_json(ck.join("config.json"))?;
    // Same rationale as `voxcpm_dit_check`: f32 is the cleaner ground truth,
    // and `feat_decoder`'s tensors live under the default prefix.
    let model = LocalDit::<CpuRuntime>::from_safetensors(
        ck.join("model.safetensors"),
        cfg,
        &device,
        Some(DType::F32),
    )?;

    let mut fx = SafeTensorsLoader::open(fx_dir.join("cfm_fixture.safetensors"))?;
    let mut ok = true;

    // 1. Schedule, checked independently of the integrator so a schedule bug
    // is never misread as a sampler bug.
    println!("schedule:");
    let want_t_span = fx.load_tensor::<CpuRuntime>("t_span", &device)?;
    let want_t_span: Vec<f32> = want_t_span.contiguous()?.to_vec();
    let got_t_span = cfm_time_span(10, 1.0)?;
    println!("  got  {got_t_span:?}");
    println!("  want {want_t_span:?}");
    assert_eq!(got_t_span.len(), want_t_span.len());
    ok &= report("t_span", max_abs_err(&got_t_span, &want_t_span));

    // 2 & 3. Per-step trajectory and final output, driven off the fixture's
    // OWN t_span so step 1's outcome cannot contaminate this check.
    let z = fx.load_tensor::<CpuRuntime>("z", &device)?;
    let mu = fx.load_tensor::<CpuRuntime>("mu", &device)?;
    let cond = fx.load_tensor::<CpuRuntime>("cond", &device)?;
    let want_out = fx.load_tensor::<CpuRuntime>("out", &device)?;

    let mut trajectory: Vec<Var<CpuRuntime>> = Vec::new();
    let got_out = model.solve_euler(
        &client,
        &Var::new(z.clone(), false),
        &want_t_span,
        &Var::new(mu, false),
        &Var::new(cond, false),
        2.0,
        true,
        Some(&mut trajectory),
    )?;

    println!("trajectory ({} steps):", trajectory.len());
    assert_eq!(trajectory.len(), 10);
    for (k, step) in trajectory.iter().enumerate() {
        let want_step = fx.load_tensor::<CpuRuntime>(&format!("step{k}_x"), &device)?;
        assert_eq!(step.tensor().shape(), want_step.shape());
        let g: Vec<f32> = step.tensor().contiguous()?.to_vec();
        let w: Vec<f32> = want_step.contiguous()?.to_vec();
        ok &= report(&format!("step{k}"), max_abs_err(&g, &w));
        if k == 0 {
            // step0_x is the state after the forced zero-velocity warmup: x
            // must be untouched, so it must equal z bitwise.
            let z0: Vec<f32> = z.contiguous()?.to_vec();
            assert_eq!(g, z0, "step0_x must equal z bitwise (zero-velocity warmup)");
            println!("  step0 == z bitwise: OK");
        }
    }

    println!("final output:");
    assert_eq!(got_out.tensor().shape(), want_out.shape());
    let g: Vec<f32> = got_out.tensor().contiguous()?.to_vec();
    let w: Vec<f32> = want_out.contiguous()?.to_vec();
    ok &= report("out", max_abs_err(&g, &w));

    println!("\n{}", if ok { "VERIFIED" } else { "FAILED" });
    std::process::exit(if ok { 0 } else { 1 });
}
