//! Integration smoke test: load the full Kokoro-82M checkpoint.
//!
//! Usage:
//!   cargo run --example load_kokoro --release -- /path/to/Kokoro-82M

use boostr::Runtime;
use boostr::model::audio::kokoro::load_kokoro_v2;
use boostr::runtime::cpu::{CpuDevice, CpuRuntime};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: load_kokoro PATH/TO/Kokoro-82M")?;
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    println!("loading Kokoro from {path} ...");
    let model = load_kokoro_v2::<CpuRuntime, _>(&client, &path, &device)?;
    println!(
        "ok — hidden_dim={}, style_dim={}, max_dur={}, sample_rate={}",
        model.config.hidden_dim,
        model.config.style_dim,
        model.config.max_dur,
        model.config.sample_rate,
    );
    Ok(())
}
