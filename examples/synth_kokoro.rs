//! Live synthesis: phoneme IDs → waveform via the full Kokoro pipeline.
//!
//! Skips G2P (which needs `espeak-ng` on the system) and hand-encodes a
//! short IPA phoneme sequence directly into token IDs via the inlined
//! `vocab` in `config.json`. Purpose is to exercise every neural stage
//! (ALBERT → text encoder → prosody predictor → decoder → iSTFT) on a real
//! checkpoint and surface any runtime shape/dtype mismatches.
//!
//! Usage:
//!   cargo run --example synth_kokoro --release -- /path/to/Kokoro-82M /path/to/voice.pt

use boostr::Runtime;
use boostr::format::load_voice_pt;
use boostr::model::audio::kokoro::{KokoroPhonemeVocab, load_kokoro_v2, select_voice_style};
use boostr::model::audio::wav_encode::{encode_wav_f32, encode_wav_pcm16};
use boostr::runtime::cpu::{CpuDevice, CpuRuntime};
use boostr::tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = std::env::args()
        .nth(1)
        .ok_or("usage: synth_kokoro MODEL_DIR VOICE_PATH")?;
    let voice_path = std::env::args()
        .nth(2)
        .ok_or("usage: synth_kokoro MODEL_DIR VOICE_PATH")?;

    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    println!("loading checkpoint ...");
    let model = load_kokoro_v2::<CpuRuntime, _>(&client, &model_dir, &device)?;

    let vocab_path = std::path::Path::new(&model_dir).join("config.json");
    let vocab = KokoroPhonemeVocab::from_json_file(&vocab_path)?;
    println!("vocab: {} symbols", vocab.len());

    // Hand-encode "hello world" in IPA. Matches the phoneme sequence
    // espeak-ng produces for American English (approximately).
    // The string MUST be decomposed into the exact tokens present in
    // config.json's `vocab` — single IPA chars + space + stress marks.
    // "hello world" in vocab-compatible IPA. Uses ɜ + ɹ instead of the
    // fused r-colored ɝ which isn't in Kokoro's 114-symbol table.
    let phonemes: Vec<String> = ["h", "ə", "l", "o", "ʊ", " ", "w", "ɜ", "ɹ", "l", "d"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let ids = vocab.encode_strict(&phonemes)?;
    println!("phonemes {} → ids {:?}", phonemes.len(), ids);

    let token_ids_f: Vec<i64> = ids.iter().map(|&x| x as i64).collect();
    let token_ids =
        Tensor::<CpuRuntime>::from_slice(&token_ids_f, &[1, token_ids_f.len()], &device);

    println!("loading voice pack from {voice_path} ...");
    let voice_pack = load_voice_pt::<CpuRuntime>(&voice_path, &device)?;
    println!("voice pack shape: {:?}", voice_pack.shape());
    let voice_row = select_voice_style(&voice_pack, ids.len())?;
    println!("voice row shape: {:?}", voice_row.shape());

    println!("synthesizing ...");
    let start = std::time::Instant::now();
    let waveform = model.synthesize_cpu(&client, &token_ids, &voice_row, 1)?;
    let elapsed = start.elapsed();

    let samples: Vec<f32> = waveform.contiguous()?.to_vec();
    println!(
        "ok — {} samples @ {} Hz = {:.2}s audio, synth took {:.2}s ({:.2}x realtime)",
        samples.len(),
        model.config.sample_rate,
        samples.len() as f32 / model.config.sample_rate as f32,
        elapsed.as_secs_f32(),
        (samples.len() as f32 / model.config.sample_rate as f32) / elapsed.as_secs_f32(),
    );

    // Log amplitude stats to catch clipping / level issues vs reference.
    let min = samples.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = samples.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let rms = (samples.iter().map(|x| x * x).sum::<f32>() / samples.len() as f32).sqrt();
    println!("amplitude: range=[{min:.4}, {max:.4}], rms={rms:.4}");

    // Write both PCM16 (listenable in any player) and raw f32 (full range
    // preserved for numerical comparison with the Python reference).
    let out_dir = std::path::Path::new(&model_dir);
    let pcm_path = out_dir.join("synth_output.wav");
    std::fs::write(
        &pcm_path,
        encode_wav_pcm16(&samples, model.config.sample_rate),
    )?;
    println!("wrote {}", pcm_path.display());
    let f32_path = out_dir.join("synth_output_f32.wav");
    std::fs::write(
        &f32_path,
        encode_wav_f32(&samples, model.config.sample_rate),
    )?;
    println!("wrote {}", f32_path.display());
    Ok(())
}
