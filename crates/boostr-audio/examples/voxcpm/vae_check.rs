//! Verify the ported VoxCPM2 AudioVAE against the Python model's own output.
//!
//! ```text
//! cargo run --release --features audio --example voxcpm_vae_check -- DIR
//! ```
//!
//! `DIR` holds `audiovae.safetensors` — the fixture generator's converted
//! copy; the loader reads the published `audiovae.pth` just as well — and the
//! two fixtures written by `make_vae_fixture.py` / `make_vae_enc_fixture.py`.
//!
//! This is the gate for the port. A Rust reimplementation that merely runs and
//! produces plausible audio proves nothing; the only evidence that matters is
//! reproducing the weights' original output. Exits non-zero on mismatch so it
//! can be wired into a check.
//!
//! Tolerance is 1e-3 absolute. Observed: decoder ~5e-7 on a 0.18 span, encoder
//! ~3e-5 on an 8.7 span - both ~3e-6 relative, which is f32 accumulation
//! through the conv stacks, not a difference in behaviour.
use boostr::format::safetensors_loader::SafeTensorsLoader;
use boostr::model::audio::voxcpm::vae::{AudioVaeDecoder, AudioVaeEncoder};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use std::path::PathBuf;

fn cmp(tag: &str, g: &[f32], w: &[f32]) -> bool {
    let max = g
        .iter()
        .zip(w)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let span =
        w.iter().cloned().fold(f32::MIN, f32::max) - w.iter().cloned().fold(f32::MAX, f32::min);
    let ok = max <= 1e-3;
    println!(
        "  {tag}: max abs err {max:.3e}  span {span:.4}  rel {:.2e}  {}",
        max / span.max(1e-9),
        if ok { "OK" } else { "MISMATCH" }
    );
    ok
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = PathBuf::from(std::env::args().nth(1).expect("dir"));
    let device = CpuDevice::default();
    let client = CpuClient::new(device.clone());
    let vae = dir.join("audiovae.safetensors");
    let mut ok = true;

    println!("decoder:");
    let dec = AudioVaeDecoder::<CpuRuntime>::from_checkpoint(&vae, &device)?;
    let mut fd = SafeTensorsLoader::open(dir.join("vae_decoder_fixture.safetensors"))?;
    for c in 0..2 {
        let latent = fd.load_tensor::<CpuRuntime>(&format!("case{c}_latent"), &device)?;
        let want = fd.load_tensor::<CpuRuntime>(&format!("case{c}_waveform"), &device)?;
        let got = dec.forward(&client, &latent)?;
        assert_eq!(got.shape(), want.shape());
        ok &= cmp(
            &format!("case{c}"),
            &got.contiguous()?.to_vec(),
            &want.contiguous()?.to_vec(),
        );
    }

    println!("encoder:");
    let enc = AudioVaeEncoder::<CpuRuntime>::from_checkpoint(&vae, &device)?;
    let mut fe = SafeTensorsLoader::open(dir.join("vae_encoder_fixture.safetensors"))?;
    for c in 0..2 {
        let wave = fe.load_tensor::<CpuRuntime>(&format!("case{c}_wave"), &device)?;
        let want = fe.load_tensor::<CpuRuntime>(&format!("case{c}_mu"), &device)?;
        let got = enc.forward(&client, &wave)?;
        assert_eq!(got.shape(), want.shape());
        ok &= cmp(
            &format!("case{c}"),
            &got.contiguous()?.to_vec(),
            &want.contiguous()?.to_vec(),
        );
    }
    println!("\n{}", if ok { "ALL VERIFIED" } else { "FAILED" });
    std::process::exit(if ok { 0 } else { 1 });
}
