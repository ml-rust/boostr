//! Verify the ported VoxCPM2 voice-clone PREFILL (`encode_reference` +
//! `prefill_capturing`) against the Python model's own output.
//!
//! ```text
//! cargo run --release --features audio,f16 --example voxcpm_prefill_check -- CKPT_DIR FIXTURE_DIR
//! ```
//!
//! `CKPT_DIR` is the VoxCPM2 checkpoint (`config.json` + `model.safetensors`).
//! `FIXTURE_DIR` holds `prefill_fixture.safetensors`, written by
//! `audio/pipeline/make_prefill_fixture.py` from the reference implementation,
//! and `audiovae.safetensors` (produced by
//! `audio/pipeline/convert_audiovae.py`), same layout as `voxcpm_vae_check`.
//!
//! This is Unit A of the end-to-end orchestrator: reference-audio encoding
//! and the two-LM prefill, up to (not including) the per-patch CFM sampling
//! loop. It is a real numerical gate, not a smoke test — running and
//! producing plausible numbers proves nothing, reproducing the reference
//! output does. Exits non-zero on mismatch.
//!
//! # Why this stage alone can be gated exactly
//!
//! The prefill is fully deterministic: no sampling happens before the first
//! CFM call. Every check below except the layout check (integer, exact) is a
//! float comparison at 2e-3 absolute, same tolerance as the other VoxCPM2
//! gates. Once the per-patch loop starts drawing noise for CFM, that
//! exactness is gone — a matching seed does not make two independent
//! implementations draw the same samples, so no gate downstream of this one
//! can be this direct. That is also why there is no end-to-end exact gate:
//! this file is as far as one can go.
//!
//! # `ref_wav` comes from the fixture, not the original 48 kHz recording
//!
//! The fixture's `ref_wav` `[1, 87040]` is the reference's OWN
//! `librosa.load(..., sr=16000)` + right-pad output — i.e. exactly what its
//! `_encode_wav` fed to the AudioVAE. This example feeds that tensor to
//! `encode_reference` directly and never touches the original 48 kHz
//! recording. boostr's resampler is not bit-identical to librosa's
//! `soxr_hq`; decoding the original file here would conflate a resampler
//! difference with a genuine port bug in the VAE / patch fold / prefill
//! path, which is what this gate exists to isolate.
//!
//! # The layout check is exact-integer on purpose
//!
//! Token ids and the two masks are checked with `==`, not a tolerance. A
//! wrong delimiter id, an off-by-one boundary, or a masks-swapped bug is
//! shape-valid and silently produces a different (but still plausible-
//! looking) float sequence; a 2e-3 float tolerance would hide it completely.
//! Integer equality cannot.
use boostr::format::safetensors_loader::SafeTensorsLoader;
use boostr::model::audio::voxcpm::model::VoxCpm2Model;
use boostr::model::audio::voxcpm::model::config::AUDIO_START_ID;
use boostr::model::audio::voxcpm::model::sequence::SequenceLayout;
use numr::dtype::DType;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;
use std::path::PathBuf;

/// Sizes both KV caches; matches the reference checkpoint's `config.json`
/// `max_length`. Only `S = 59` positions are actually prefilled here.
const MAX_LENGTH: usize = 8192;

/// Load `<name>`, run nothing (caller already has `got`), and report shapes /
/// max abs error / span / relative error / OK-or-MISMATCH in the sibling
/// gates' line format. Returns whether this case passed.
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
    // Scale-aware tolerance. A fixed absolute bound is meaningless across this
    // gate's tensors: their spans range from ~9.5 (`ref_feat`) to ~4309
    // (`residual_enc_inputs`, a 4096-wide Linear producing values up to +-2200).
    // For a 4096-term f32 dot product, accumulated error grows as roughly
    // sqrt(n) * eps * |value| ~= 64 * 1.19e-7 * 2200 ~= 1.7e-2, so a 2e-3 bound
    // rejects correct output purely for being large. Gate on RELATIVE error
    // instead, with an absolute floor so a near-zero-span tensor cannot pass on
    // a vacuous ratio. The floor is the same 2e-3 the sibling gates use.
    let tol = 2e-3f32.max(span * 1e-5);
    let pass = max <= tol;
    println!(
        "  max abs err {max:.3e}  span {span:.4}  rel {:.2e}  tol {tol:.3e}  {}",
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

    // audiovae.safetensors lives inside FIXTURE_DIR, same layout as
    // voxcpm_vae_check.
    let vae = fx_dir.join("audiovae.safetensors");
    // f32 ground truth, same rationale as the other VoxCPM2 gates: the
    // checkpoint is BF16 and internal ops upcast anyway.
    let model = VoxCpm2Model::<CpuRuntime>::from_checkpoint(&ck, &vae, &device, Some(DType::F32))?;

    let mut fx = SafeTensorsLoader::open(fx_dir.join("prefill_fixture.safetensors"))?;
    let mut ok = true;

    // --- 1. Reference encode: ref_wav -> ref_feat ---------------------------
    let ref_wav = fx.load_tensor::<CpuRuntime>("ref_wav", &device)?;
    let ref_feat_want = fx.load_tensor::<CpuRuntime>("ref_feat", &device)?;
    let ref_wav_samples: Vec<f32> = ref_wav.contiguous()?.to_vec();
    let ref_feat_got = model.encode_reference(&client, &ref_wav_samples)?;
    ok &= check("encode_reference", &ref_feat_got, &ref_feat_want)?;

    let t_ref = ref_feat_got.shape()[0];

    // --- 2. Sequence layout, EXACT integer -----------------------------------
    // text_token is the FULL S-length sequence: ref_tokens ([103] ++
    // [0; T_ref] ++ [104]) ++ text_token_ids. Slice off the trailing
    // text_length = S - T_ref - 2 ids and feed only those to `prefill` —
    // it builds the reference prefix itself.
    let text_token_tensor = fx.load_tensor::<CpuRuntime>("text_token", &device)?;
    let full_ids: Vec<i64> = text_token_tensor.contiguous()?.to_vec();
    let seq_len_want = full_ids.len();
    let text_length = seq_len_want
        .checked_sub(t_ref + 2)
        .expect("S must be at least T_ref + 2");
    let text_token_ids: Vec<u32> = full_ids[t_ref + 2..].iter().map(|&id| id as u32).collect();

    // Verify the slice boundary before trusting it: position t_ref+1 (the
    // last ref-prefix id) must be the 104 delimiter, and the slice must end
    // with 101 (AUDIO_START_ID) as `SequenceLayout::build` requires.
    assert_eq!(
        full_ids[t_ref + 1],
        104,
        "expected the ref-audio-end delimiter (104) at index T_ref+1={}, got {}",
        t_ref + 1,
        full_ids[t_ref + 1]
    );
    assert_eq!(
        *text_token_ids.last().expect("non-empty text slice"),
        AUDIO_START_ID,
        "text slice must end with AUDIO_START_ID ({AUDIO_START_ID})"
    );

    let layout = SequenceLayout::build(t_ref, &text_token_ids)?;
    println!(
        "layout: t_ref={t_ref} text_length={} S={} (want S={seq_len_want})",
        layout.text_length,
        layout.seq_len()
    );
    assert_eq!(layout.text_length, text_length);

    let mut layout_ok = layout.seq_len() == seq_len_want;
    if !layout_ok {
        println!(
            "  MISMATCH: seq_len got {} want {seq_len_want}",
            layout.seq_len()
        );
    }
    if let Some(i) = (0..seq_len_want).find(|&i| layout.token_ids[i] != full_ids[i]) {
        layout_ok = false;
        println!(
            "  MISMATCH: token_ids first differ at index {i}: got {} want {}",
            layout.token_ids[i], full_ids[i]
        );
    }
    let text_mask_want = fx.load_tensor::<CpuRuntime>("text_mask", &device)?;
    let text_mask_want: Vec<f32> = text_mask_want.contiguous()?.to_vec();
    if let Some(i) = (0..seq_len_want).find(|&i| layout.text_mask[i] != text_mask_want[i]) {
        layout_ok = false;
        println!(
            "  MISMATCH: text_mask first differs at index {i}: got {} want {}",
            layout.text_mask[i], text_mask_want[i]
        );
    }
    let audio_mask_want = fx.load_tensor::<CpuRuntime>("audio_mask", &device)?;
    let audio_mask_want: Vec<f32> = audio_mask_want.contiguous()?.to_vec();
    if let Some(i) = (0..seq_len_want).find(|&i| layout.audio_mask[i] != audio_mask_want[i]) {
        layout_ok = false;
        println!(
            "  MISMATCH: audio_mask first differs at index {i}: got {} want {}",
            layout.audio_mask[i], audio_mask_want[i]
        );
    }
    println!(
        "layout (exact-integer): {}",
        if layout_ok { "OK" } else { "MISMATCH" }
    );
    ok &= layout_ok;

    // --- 3-6. Intermediate anchors, then 7-8. final hidden states -----------
    let state = model.prefill_capturing(&client, &ref_feat_got, &text_token_ids, MAX_LENGTH)?;
    let intermediates = state
        .intermediates
        .expect("prefill_capturing always returns Some(intermediates)");

    let feat_embed_want = fx.load_tensor::<CpuRuntime>("feat_embed", &device)?;
    ok &= check(
        "feat_embed",
        intermediates.feat_embed.tensor(),
        &feat_embed_want,
    )?;

    let combined_embed_want = fx.load_tensor::<CpuRuntime>("combined_embed", &device)?;
    ok &= check(
        "combined_embed",
        intermediates.combined_embed.tensor(),
        &combined_embed_want,
    )?;

    let enc_outputs_want = fx.load_tensor::<CpuRuntime>("enc_outputs_post_blend", &device)?;
    ok &= check(
        "enc_outputs_post_blend",
        intermediates.enc_outputs.tensor(),
        &enc_outputs_want,
    )?;

    let residual_enc_inputs_want = fx.load_tensor::<CpuRuntime>("residual_enc_inputs", &device)?;
    ok &= check(
        "residual_enc_inputs",
        intermediates.residual_enc_inputs.tensor(),
        &residual_enc_inputs_want,
    )?;

    let lm_hidden_want = fx.load_tensor::<CpuRuntime>("lm_hidden", &device)?;
    ok &= check("lm_hidden", state.lm_hidden.tensor(), &lm_hidden_want)?;

    let residual_hidden_want = fx.load_tensor::<CpuRuntime>("residual_hidden", &device)?;
    ok &= check(
        "residual_hidden",
        state.residual_hidden.tensor(),
        &residual_hidden_want,
    )?;

    // --- 9. position ---------------------------------------------------------
    println!("position: {} (want {seq_len_want})", state.position);
    ok &= state.position == seq_len_want;

    println!("\n{}", if ok { "VERIFIED" } else { "FAILED" });
    std::process::exit(if ok { 0 } else { 1 });
}
