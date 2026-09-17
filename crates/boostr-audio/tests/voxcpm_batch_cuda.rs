//! Batched VoxCPM2 generation on a real model: two requests of different
//! length, one with a reference voice and one zero-shot, generated together
//! through `prefill_batch` and the batched loop, against each request
//! rendered alone.
//!
//! Gated on a real model and CUDA: set `VOXCPM2_GGUF` to a compressr GGUF
//! that embeds the VAE and the tokenizer, and `VOXCPM2_REF_WAV` to a
//! reference recording (the test skips when either is unset or absent).
//! `VOXCPM2_BATCH_OUT`, optional, is a directory that receives
//! `batch2_row0.wav` / `batch2_row1.wav` (the batched renders) and
//! `batch1_row0.wav` / `batch1_row1.wav` (the single renders).
//! `VOXCPM2_EXPECTED_ROW0_WAV`, optional, is a wav the single render of row 0
//! must equal byte for byte — the clone recipe's known-good output.
//!
//! Quantized matmul tiles are chosen by `M`, so a row's values under `B = 2`
//! are not bit-identical to its `B = 1` render; the test requires the same
//! patch count per row and reports the per-row max abs latent difference and
//! the audio SNR.

#![cfg(all(feature = "voxcpm", feature = "cuda"))]

use std::path::{Path, PathBuf};

use boostr::model::audio::voxcpm::VoxCpm2Weights;
use boostr::model::audio::voxcpm::model::config::AUDIO_START_ID;
use boostr::model::audio::voxcpm::model::{
    GenerateOptions, GenerateOutcome, GenerateState, PrefillRow, RowOptions, VoxCpm2Model,
    unfold_patches, unfold_patches_row,
};
use boostr::model::audio::voxcpm::vae::decoder::SAMPLE_RATE;
use boostr_audio::voxcpm::{load_tokenizer, normalize_whitespace, tokenize};
use boostr_audio::{decode_audio_file_mono_at, encode_wav_pcm16};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
use numr::tensor::Tensor;

/// The clone recipe's prompt, rendered with the reference voice and seed 0.
const ROW0_TEXT: &str =
    "Okay, so lepas ni kita test the new model, tengok sama ada suara dia sound macam saya ke tak.";
/// A shorter zero-shot prompt with its own seed.
const ROW1_TEXT: &str = "Selamat pagi, today we test batching.";
const SEEDS: [u64; 2] = [0, 1];
/// The clone pipeline's patch budget: six per text token plus ten.
const MAX_LEN_CAP: usize = 4096;
/// The AudioVAE encoder's input rate.
const REF_RATE: u32 = 16_000;

fn env_path(name: &str) -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var(name).ok()?);
    path.exists().then_some(path)
}

fn budget(text_len: usize) -> usize {
    (text_len * 6 + 10).min(MAX_LEN_CAP)
}

fn seq_len(t_ref: Option<usize>, text_len: usize) -> usize {
    match t_ref {
        Some(t_ref) => t_ref + 2 + text_len,
        None => text_len,
    }
}

fn to_host(t: &Tensor<CudaRuntime>) -> Vec<f32> {
    t.contiguous().expect("contiguous").to_vec::<f32>()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn snr_db(reference: &[f32], test: &[f32]) -> f64 {
    let signal: f64 = reference.iter().map(|v| f64::from(*v).powi(2)).sum();
    let noise: f64 = reference
        .iter()
        .zip(test)
        .map(|(r, t)| (f64::from(*r) - f64::from(*t)).powi(2))
        .sum();
    if noise == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (signal / noise).log10()
    }
}

fn write_wav(dir: Option<&Path>, name: &str, samples: &[f32]) {
    let Some(dir) = dir else {
        return;
    };
    let bytes = encode_wav_pcm16(samples, SAMPLE_RATE as u32).expect("encode wav");
    std::fs::write(dir.join(name), bytes).expect("write wav");
}

struct SingleRender {
    patches: usize,
    latent: Vec<f32>,
    samples: Vec<f32>,
}

fn render_single(
    client: &CudaClient,
    model: &VoxCpm2Model<CudaRuntime>,
    row: &PrefillRow<'_, CudaRuntime>,
    max_len: usize,
    seed: u64,
) -> SingleRender {
    let t_ref = row.ref_feat.map(|f| f.shape()[0]);
    let max_length = seq_len(t_ref, row.text_token_ids.len()) + max_len;
    let prefill = model
        .prefill(client, row.ref_feat, row.text_token_ids, max_length)
        .expect("prefill");
    let mut state = GenerateState::start(prefill, model.config).expect("start");
    let options = GenerateOptions::new(max_len, seed);
    let outcome = model
        .patch_generator()
        .generate(client, &mut state, &options)
        .expect("generate");
    assert_eq!(
        outcome,
        GenerateOutcome::StopToken,
        "single render truncated"
    );
    let latent = unfold_patches(
        &state.patches,
        model.config.patch_size,
        model.config.feat_dim,
    )
    .expect("unfold");
    let decoded = model
        .decode_patches(client, &state.patches)
        .expect("decode");
    SingleRender {
        patches: state.patches.len(),
        latent: to_host(&latent),
        samples: to_host(&decoded),
    }
}

#[test]
fn batch_of_two_matches_single_renders() {
    let (Some(gguf), Some(ref_wav)) = (env_path("VOXCPM2_GGUF"), env_path("VOXCPM2_REF_WAV"))
    else {
        eprintln!("VOXCPM2_GGUF or VOXCPM2_REF_WAV unset or absent; skipping");
        return;
    };
    let out_dir = env_path("VOXCPM2_BATCH_OUT");
    let expected_row0 = env_path("VOXCPM2_EXPECTED_ROW0_WAV");

    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let model = VoxCpm2Model::<CudaRuntime>::from_gguf(
        &gguf,
        None,
        None,
        &device,
        Some(DType::F32),
        Some(DType::F16),
    )
    .expect("load model");

    let weights = VoxCpm2Weights::Gguf {
        path: gguf.clone(),
        config: None,
    };
    let tokenizer =
        load_tokenizer(&weights.tokenizer_source().expect("tokenizer source")).expect("tokenizer");
    let ids: Vec<Vec<u32>> = [ROW0_TEXT, ROW1_TEXT]
        .iter()
        .map(|text| {
            let mut ids = tokenize(&tokenizer, &normalize_whitespace(text));
            ids.push(AUDIO_START_ID);
            ids
        })
        .collect();

    let ref_samples = decode_audio_file_mono_at(&ref_wav, REF_RATE).expect("reference wav");
    let ref_feat = model
        .encode_reference(&client, &ref_samples)
        .expect("encode reference");
    let rows = [
        PrefillRow {
            ref_feat: Some(&ref_feat),
            text_token_ids: &ids[0],
        },
        PrefillRow {
            ref_feat: None,
            text_token_ids: &ids[1],
        },
    ];
    let max_lens: Vec<usize> = ids.iter().map(|ids| budget(ids.len() - 1)).collect();

    // Each row alone: the clone pipeline's exact path.
    let singles: Vec<SingleRender> = rows
        .iter()
        .zip(&max_lens)
        .zip(SEEDS)
        .map(|((row, &max_len), seed)| render_single(&client, &model, row, max_len, seed))
        .collect();
    if let Some(expected) = expected_row0 {
        let got = encode_wav_pcm16(&singles[0].samples, SAMPLE_RATE as u32).expect("encode");
        let want = std::fs::read(&expected).expect("read expected wav");
        assert!(
            got == want,
            "single render of row 0 differs from {}",
            expected.display()
        );
    }

    // Both rows together, left-padded.
    let s_max = rows
        .iter()
        .map(|row| seq_len(row.ref_feat.map(|f| f.shape()[0]), row.text_token_ids.len()))
        .max()
        .expect("two rows");
    let max_len = *max_lens.iter().max().expect("two rows");
    let prefill = model
        .prefill_batch(&client, &rows, s_max + max_len)
        .expect("prefill_batch");
    assert_eq!(prefill.batch, 2);
    assert!(prefill.kv_start.is_some(), "rows differ in length");
    let mut state = GenerateState::start(prefill, model.config).expect("start");
    let mut options = GenerateOptions::new(max_len, SEEDS[0]);
    options.rows = max_lens
        .iter()
        .zip(SEEDS)
        .map(|(&max_len, seed)| RowOptions { max_len, seed })
        .collect();
    let outcome = model
        .patch_generator()
        .generate(&client, &mut state, &options)
        .expect("generate batch");
    assert_eq!(
        outcome,
        GenerateOutcome::StopToken,
        "batched render truncated"
    );

    for (b, single) in singles.iter().enumerate() {
        assert_eq!(
            state.outcomes[b],
            Some(GenerateOutcome::StopToken),
            "row {b} outcome"
        );
        assert_eq!(state.patch_len[b], single.patches, "row {b} patch count");
        let latent = unfold_patches_row(
            state.row_patches(b).expect("row"),
            b,
            model.config.patch_size,
            model.config.feat_dim,
        )
        .expect("unfold row");
        let latent = to_host(&latent);
        let decoded = model.decode_row(&client, &state, b).expect("decode row");
        let samples = to_host(&decoded);
        assert_eq!(samples.len(), single.samples.len(), "row {b} sample count");
        let latent_diff = max_abs_diff(&latent, &single.latent);
        let snr = snr_db(&single.samples, &samples);
        let latent_scale = single.latent.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        eprintln!(
            "row {b}: {} patches, max abs latent diff {latent_diff:.3e} (latent max abs \
             {latent_scale:.3e}), audio SNR {snr:.1} dB",
            single.patches
        );
        // Per-patch divergence, to tell a gradual autoregressive drift from
        // a wrong first patch: the latent is [1, feat_dim, patches * patch_size].
        let frames_per_patch = model.config.patch_size;
        let frames = single.patches * frames_per_patch;
        let per_patch: Vec<String> = (0..single.patches)
            .map(|k| {
                let mut worst = 0.0f32;
                for c in 0..model.config.feat_dim {
                    let start = c * frames + k * frames_per_patch;
                    let end = start + frames_per_patch;
                    worst = worst.max(max_abs_diff(
                        &latent[start..end],
                        &single.latent[start..end],
                    ));
                }
                format!("{worst:.2e}")
            })
            .collect();
        eprintln!("row {b} per-patch max abs diff: {}", per_patch.join(" "));
        assert!(
            samples.iter().all(|v| v.is_finite()),
            "row {b} has non-finite samples"
        );
        write_wav(out_dir.as_deref(), &format!("batch2_row{b}.wav"), &samples);
        write_wav(
            out_dir.as_deref(),
            &format!("batch1_row{b}.wav"),
            &single.samples,
        );
    }
}

/// The same request twice in one batch, unpadded: the control that puts the
/// padded rows' drift in context. Identical rows do NOT come out bit-identical
/// on CUDA — the stream-k MMQ tiling splits a tile's K walk by its position in
/// the launch, so two rows of one batch sum in different orders, and the
/// Q8 activation quantization plus the sampler amplify that last-bit
/// difference to a visible one. This test reports the divergence between the
/// two rows and against the single render, and checks nothing bitwise.
#[test]
fn duplicate_rows_report_kernel_drift() {
    let (Some(gguf), Some(ref_wav)) = (env_path("VOXCPM2_GGUF"), env_path("VOXCPM2_REF_WAV"))
    else {
        eprintln!("VOXCPM2_GGUF or VOXCPM2_REF_WAV unset or absent; skipping");
        return;
    };
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let model = VoxCpm2Model::<CudaRuntime>::from_gguf(
        &gguf,
        None,
        None,
        &device,
        Some(DType::F32),
        Some(DType::F16),
    )
    .expect("load model");
    let weights = VoxCpm2Weights::Gguf {
        path: gguf.clone(),
        config: None,
    };
    let tokenizer =
        load_tokenizer(&weights.tokenizer_source().expect("tokenizer source")).expect("tokenizer");
    let mut ids = tokenize(&tokenizer, &normalize_whitespace(ROW0_TEXT));
    ids.push(AUDIO_START_ID);
    let ref_samples = decode_audio_file_mono_at(&ref_wav, REF_RATE).expect("reference wav");
    let ref_feat = model
        .encode_reference(&client, &ref_samples)
        .expect("encode reference");
    let row = PrefillRow {
        ref_feat: Some(&ref_feat),
        text_token_ids: &ids,
    };
    let max_len = budget(ids.len() - 1);
    let single = render_single(&client, &model, &row, max_len, SEEDS[0]);

    let s = seq_len(Some(ref_feat.shape()[0]), ids.len());
    let prefill = model
        .prefill_batch(&client, &[row, row], s + max_len)
        .expect("prefill_batch");
    assert!(prefill.kv_start.is_none(), "equal rows carry no pad");
    let lm = to_host(prefill.lm_hidden.tensor());
    let half = lm.len() / 2;
    eprintln!(
        "duplicate rows after prefill: lm_hidden row diff {:.3e}",
        max_abs_diff(&lm[..half], &lm[half..])
    );
    let mut state = GenerateState::start(prefill, model.config).expect("start");
    let options = GenerateOptions::new(max_len, SEEDS[0]);
    let outcome = model
        .patch_generator()
        .generate(&client, &mut state, &options)
        .expect("generate batch");
    assert_eq!(
        outcome,
        GenerateOutcome::StopToken,
        "batched render truncated"
    );

    let (patch_size, feat_dim) = (model.config.patch_size, model.config.feat_dim);
    let rows: Vec<Vec<f32>> = (0..2)
        .map(|b| {
            to_host(
                &unfold_patches_row(state.row_patches(b).expect("row"), b, patch_size, feat_dim)
                    .expect("unfold"),
            )
        })
        .collect();
    // Patch 0 of a `[feat_dim, patches * patch_size]` latent, flattened.
    let patch0 = |latent: &[f32], patches: usize| -> Vec<f32> {
        let frames = patches * patch_size;
        (0..feat_dim)
            .flat_map(|c| latent[c * frames..c * frames + patch_size].to_vec())
            .collect()
    };
    let (p0, p1) = (
        patch0(&rows[0], state.patch_len[0]),
        patch0(&rows[1], state.patch_len[1]),
    );
    eprintln!(
        "duplicate rows: patch counts {:?} (single {}), patch 0 row-vs-row diff {:.3e}, \
         patch 0 row 0 vs single diff {:.3e}",
        state.patch_len,
        single.patches,
        max_abs_diff(&p0, &p1),
        max_abs_diff(&p0, &patch0(&single.latent, single.patches)),
    );
    for (b, latent) in rows.iter().enumerate() {
        assert!(
            latent.iter().all(|v| v.is_finite()),
            "row {b} has non-finite latent values"
        );
    }
}
