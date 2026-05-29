//! Tier-3: load the full `KokoroModelV2` from a checkpoint directory.

use super::primitives::{
    AdainResBlk1dLoadOpts, AdainResBlock1LoadOpts, load_ada_layer_norm, load_adain_resblk1d,
    load_adain_resblock1, load_albert_embeddings, load_albert_layer, missing_bias,
};
use crate::error::{Error, Result};
use crate::model::audio::kokoro::{
    AlbertConfig, AlbertModel, BertEncoder, ConvBlock, Decoder, DurationEncoder, IStftNetGenerator,
    KokoroConfig, KokoroModelV2, MagPhaseHead, ProsodyBranch, ProsodyPredictor, SineGen,
    SourceModuleHnNSF, TextEncoder, UpsampleBlock,
};
use crate::nn::Embedding;
use numr::dtype::DType;
use numr::ops::{BinaryOps, PaddingMode, ReduceOps, TensorOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use std::path::Path;

use super::super::loader::{
    load_bilstm, load_linear_tensors, load_plain_conv1d, load_weight_norm_pair,
    load_weight_normed_conv1d,
};

/// Load a full Kokoro checkpoint into [`KokoroModelV2`]. Reads
/// `config.json` from `model_dir`, then the safetensors in the same directory.
pub fn load_kokoro_v2<R, C>(
    client: &C,
    model_dir: impl AsRef<Path>,
    device: &R::Device,
) -> Result<KokoroModelV2<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + ReduceOps<R> + UnaryOps<R> + BinaryOps<R> + TensorOps<R>,
{
    let model_dir = model_dir.as_ref();
    let config = KokoroConfig::from_json_file(model_dir.join("config.json"))?;
    config.validate()?;
    let mut st = super::super::weight_source::KokoroWeightSource::open(model_dir)?;

    // -------- BertEncoder (bert.* + bert_encoder.*) --------
    let albert_cfg = AlbertConfig {
        hidden_size: config.bert_hidden_size,
        embedding_size: config.bert_embedding_size,
        num_hidden_layers: config.bert_num_layers,
        num_attention_heads: config.bert_num_heads,
        intermediate_size: config.bert_hidden_size * 4 / 3, // placeholder; real value 2048
        max_position_embeddings: 512,
        vocab_size: config.n_symbols,
        type_vocab_size: 2,
        layer_norm_eps: 1e-12,
    };
    let albert_embeddings =
        load_albert_embeddings::<R>(&mut st, "bert.embeddings", &albert_cfg, device)?;
    let (emb_proj_w, emb_proj_b) =
        load_linear_tensors::<R>(&mut st, "bert.encoder.embedding_hidden_mapping_in", device)?;
    let shared_layer = load_albert_layer::<R>(
        &mut st,
        "bert.encoder.albert_layer_groups.0.albert_layers.0",
        device,
    )?;
    let albert = AlbertModel {
        embeddings: albert_embeddings,
        embedding_projection_weight: emb_proj_w,
        embedding_projection_bias: emb_proj_b
            .ok_or_else(missing_bias("bert.encoder.embedding_hidden_mapping_in"))?,
        shared_layer,
        config: albert_cfg,
    };
    let (bert_proj_w, bert_proj_b) = load_linear_tensors::<R>(&mut st, "bert_encoder", device)?;
    let bert = BertEncoder {
        albert,
        projection_weight: bert_proj_w,
        projection_bias: bert_proj_b.ok_or_else(missing_bias("bert_encoder"))?,
        out_dim: config.hidden_dim,
    };

    // -------- TextEncoder (text_encoder.*) --------
    let te_embedding = Embedding::new(
        st.load_tensor::<R>("text_encoder.embedding.weight", device)?,
        false,
    );
    let mut te_blocks = Vec::with_capacity(config.text_conv_depth);
    for i in 0..config.text_conv_depth {
        let conv = load_weight_normed_conv1d::<R, C>(
            client,
            &mut st,
            &format!("text_encoder.cnn.{i}.0"),
            1,
            PaddingMode::Same,
            1,
            1,
            device,
        )?;
        // Custom LayerNorm: gamma/beta keys.
        let gamma = st.load_tensor::<R>(&format!("text_encoder.cnn.{i}.1.gamma"), device)?;
        let beta = st.load_tensor::<R>(&format!("text_encoder.cnn.{i}.1.beta"), device)?;
        te_blocks.push(ConvBlock::new(conv, gamma, beta, 1e-5, config.leaky_slope));
    }
    let te_lstm = load_bilstm::<R>(&mut st, "text_encoder.lstm", device)?;
    let text_encoder = TextEncoder::new(te_embedding, te_blocks, te_lstm, config.hidden_dim)?;

    // -------- ProsodyPredictor (predictor.*) --------
    // DurationEncoder: alternating LSTM / AdaLayerNorm.
    let mut pe_lstms = Vec::new();
    let mut pe_adalns = Vec::new();
    // The upstream nlayers isn't in config.json — we discover it by probing
    // for `predictor.text_encoder.lstms.{2i}.weight_ih_l0` until we run out.
    let mut nlayers = 0usize;
    while st.has_tensor(&format!(
        "predictor.text_encoder.lstms.{}.weight_ih_l0",
        2 * nlayers
    )) {
        pe_lstms.push(load_bilstm::<R>(
            &mut st,
            &format!("predictor.text_encoder.lstms.{}", 2 * nlayers),
            device,
        )?);
        pe_adalns.push(load_ada_layer_norm::<R>(
            &mut st,
            &format!("predictor.text_encoder.lstms.{}", 2 * nlayers + 1),
            1e-5,
            device,
        )?);
        nlayers += 1;
    }
    if nlayers == 0 {
        return Err(Error::ModelError {
            reason: "predictor.text_encoder.lstms is empty — no LSTM layers found".into(),
        });
    }
    let pred_text_encoder =
        DurationEncoder::new(pe_lstms, pe_adalns, config.hidden_dim, config.style_dim)?;

    let pred_lstm = load_bilstm::<R>(&mut st, "predictor.lstm", device)?;
    let (dur_w, dur_b) =
        load_linear_tensors::<R>(&mut st, "predictor.duration_proj.linear_layer", device)?;
    let pred_shared = load_bilstm::<R>(&mut st, "predictor.shared", device)?;

    // F0 and N branches: 3 AdainResBlk1d blocks each, middle block upsamples.
    // Layout from the upstream ProsodyPredictor: F0.0 keeps shape,
    // F0.1 halves channels with stride-2 pool + conv1x1 shortcut,
    // F0.2 keeps the reduced shape.
    let plain = AdainResBlk1dLoadOpts::default();
    let upsample = AdainResBlk1dLoadOpts {
        learned_sc: true,
        upsample_stride: Some(2),
        ..plain
    };
    let f0_blocks: [_; 3] = [
        load_adain_resblk1d::<R, C>(client, &mut st, "predictor.F0.0", device, plain)?,
        load_adain_resblk1d::<R, C>(client, &mut st, "predictor.F0.1", device, upsample)?,
        load_adain_resblk1d::<R, C>(client, &mut st, "predictor.F0.2", device, plain)?,
    ];
    let n_blocks: [_; 3] = [
        load_adain_resblk1d::<R, C>(client, &mut st, "predictor.N.0", device, plain)?,
        load_adain_resblk1d::<R, C>(client, &mut st, "predictor.N.1", device, upsample)?,
        load_adain_resblk1d::<R, C>(client, &mut st, "predictor.N.2", device, plain)?,
    ];
    let f0_proj = load_plain_conv1d::<R>(
        &mut st,
        "predictor.F0_proj",
        1,
        PaddingMode::Valid,
        1,
        1,
        device,
    )?;
    let n_proj = load_plain_conv1d::<R>(
        &mut st,
        "predictor.N_proj",
        1,
        PaddingMode::Valid,
        1,
        1,
        device,
    )?;

    let predictor = ProsodyPredictor {
        text_encoder: pred_text_encoder,
        lstm: pred_lstm,
        duration_proj_weight: dur_w,
        duration_proj_bias: dur_b.ok_or_else(missing_bias("duration_proj"))?,
        shared: pred_shared,
        f0: ProsodyBranch::new(f0_blocks, f0_proj),
        n: ProsodyBranch::new(n_blocks, n_proj),
        d_model: config.hidden_dim,
        style_dim: config.style_dim,
        max_dur: config.max_dur,
    };

    // -------- Decoder (decoder.*) --------
    // asr_res is weight-normed in upstream Kokoro (checkpoint ships
    // `decoder.asr_res.0.weight_g`/`weight_v`, not `.weight`).
    let asr_res = load_weight_normed_conv1d::<R, C>(
        client,
        &mut st,
        "decoder.asr_res.0",
        1,
        PaddingMode::Valid,
        1,
        1,
        device,
    )?;
    let f0_conv = load_weight_normed_conv1d::<R, C>(
        client,
        &mut st,
        "decoder.F0_conv",
        2,
        PaddingMode::Same,
        1,
        1,
        device,
    )?;
    let n_conv = load_weight_normed_conv1d::<R, C>(
        client,
        &mut st,
        "decoder.N_conv",
        2,
        PaddingMode::Same,
        1,
        1,
        device,
    )?;
    let decoder_plain_with_sc = AdainResBlk1dLoadOpts {
        learned_sc: true,
        upsample_stride: None,
        norm_eps: 1e-5,
        leaky_slope: 0.2,
    };
    let decoder_upsample_with_sc = AdainResBlk1dLoadOpts {
        learned_sc: true,
        upsample_stride: Some(2),
        norm_eps: 1e-5,
        leaky_slope: 0.2,
    };
    let dec_encode = load_adain_resblk1d::<R, C>(
        client,
        &mut st,
        "decoder.encode",
        device,
        decoder_plain_with_sc,
    )?;
    let mut dec_blocks = Vec::with_capacity(4);
    for i in 0..4 {
        let is_last = i == 3;
        dec_blocks.push(load_adain_resblk1d::<R, C>(
            client,
            &mut st,
            &format!("decoder.decode.{i}"),
            device,
            if is_last {
                decoder_upsample_with_sc
            } else {
                decoder_plain_with_sc
            },
        )?);
    }

    // Generator under decoder.generator.*
    let source_w = st.load_tensor::<R>("decoder.generator.m_source.l_linear.weight", device)?;
    let source_b = st.load_tensor::<R>("decoder.generator.m_source.l_linear.bias", device)?;
    let m_source = SourceModuleHnNSF::new(
        SineGen::new(config.sample_rate as f32, config.harmonic_num),
        source_w,
        source_b,
    )?;

    let num_upsamples = config.upsample_ratios.len();
    let mut ups = Vec::with_capacity(num_upsamples);
    for (i, (&stride, &k)) in config
        .upsample_ratios
        .iter()
        .zip(config.upsample_kernel_sizes.iter())
        .enumerate()
    {
        let (g, v) =
            load_weight_norm_pair::<R>(&mut st, &format!("decoder.generator.ups.{i}"), device)?;
        let weight = crate::nn::fuse_weight_norm(client, &v, &g, 0)?;
        let bias = st
            .load_tensor::<R>(&format!("decoder.generator.ups.{i}.bias"), device)
            .ok();
        let padding = (k - stride) / 2;
        ups.push(UpsampleBlock::new(
            weight,
            bias,
            stride,
            PaddingMode::Custom(padding, padding, 0, 0),
            0,
            1,
            1,
            0.1,
        ));
    }

    let num_kernels = config.resblock_kernel_sizes.len();
    let mut resblocks = Vec::with_capacity(num_upsamples * num_kernels);
    for stage in 0..num_upsamples {
        for k_idx in 0..num_kernels {
            let kernel = config.resblock_kernel_sizes[k_idx];
            // Dilations are indexed by k_idx (same layout per stage).
            let dilations_vec = config
                .resblock_dilation_sizes
                .get(k_idx)
                .cloned()
                .unwrap_or_else(|| vec![1, 3, 5]);
            let dilations: [usize; 3] = [
                *dilations_vec.first().unwrap_or(&1),
                *dilations_vec.get(1).unwrap_or(&3),
                *dilations_vec.get(2).unwrap_or(&5),
            ];
            let flat_idx = stage * num_kernels + k_idx;
            let opts = AdainResBlock1LoadOpts {
                dilations,
                kernel,
                norm_eps: 1e-5,
                snake_eps: 1e-9,
            };
            resblocks.push(load_adain_resblock1::<R, C>(
                client,
                &mut st,
                &format!("decoder.generator.resblocks.{flat_idx}"),
                device,
                opts,
            )?);
        }
    }

    let conv_post = load_weight_normed_conv1d::<R, C>(
        client,
        &mut st,
        "decoder.generator.conv_post",
        1,
        PaddingMode::Same,
        1,
        1,
        device,
    )?;
    let conv_post_head = MagPhaseHead::new(conv_post, config.n_fft)?;

    // Noise-path conditioning: plain Conv1d + AdaINResBlock1 per upsample
    // stage. Present in the upstream Kokoro checkpoint under
    // `decoder.generator.noise_convs.{i}` / `noise_res.{i}`. If these keys
    // aren't present (e.g. a stripped-down variant), we pass empty vecs and
    // `IStftNetGenerator::new` validates that both-or-neither is populated.
    let mut noise_convs = Vec::with_capacity(num_upsamples);
    let mut noise_res = Vec::with_capacity(num_upsamples);
    let has_noise = st.has_tensor("decoder.generator.noise_convs.0.weight");
    if has_noise {
        // Upstream `noise_convs[i]` convolves the harmonic spectrogram at
        // audio-spec rate down to the trunk rate at stage `i`:
        //
        //   stride_f0 = prod(upsample_rates[i+1..])   # stage 0: 6, stage 1: 1
        //   Conv1d(n_fft+2, c_i, kernel=stride_f0*2 if i+1<N else 1,
        //          stride=stride_f0 if i+1<N else 1,
        //          padding=(stride_f0+1)//2 if i+1<N else 0)
        //
        // We infer kernel+stride from weights but compute padding from the
        // upstream formula (weight shape alone doesn't tell us the padding).
        for stage in 0..num_upsamples {
            let conv_prefix = format!("decoder.generator.noise_convs.{stage}");
            let (weight, bias) = load_linear_tensors::<R>(&mut st, &conv_prefix, device)?;
            let weight_shape = weight.shape().to_vec();
            if weight_shape.len() != 3 {
                return Err(Error::ModelError {
                    reason: format!(
                        "{conv_prefix}.weight must be rank-3 Conv1d weight, got {weight_shape:?}"
                    ),
                });
            }
            let kernel = weight_shape[2];
            let stride_f0: usize = if stage + 1 < num_upsamples {
                config.upsample_ratios[stage + 1..].iter().product()
            } else {
                1
            };
            let stride = if stage + 1 < num_upsamples {
                stride_f0
            } else {
                1
            };
            let pad = if stage + 1 < num_upsamples {
                stride_f0.div_ceil(2)
            } else {
                0
            };
            noise_convs.push(crate::nn::Conv1d::new(
                weight,
                bias,
                stride,
                PaddingMode::Custom(pad, pad, 0, 0),
                1,
                1,
                false,
            ));
            let _ = kernel;

            // noise_res: stage 0 uses kernel=7, stage 1 uses kernel=11
            // (upstream `resblock_kernel_sizes` indices 1 and 2 respectively).
            let res_kernel = if stage == 0 { 7 } else { 11 };
            let res_opts = AdainResBlock1LoadOpts {
                dilations: [1, 3, 5],
                kernel: res_kernel,
                norm_eps: 1e-5,
                snake_eps: 1e-9,
            };
            noise_res.push(load_adain_resblock1::<R, C>(
                client,
                &mut st,
                &format!("decoder.generator.noise_res.{stage}"),
                device,
                res_opts,
            )?);
        }
    }

    let generator = IStftNetGenerator::new(
        m_source,
        ups,
        resblocks,
        noise_convs,
        noise_res,
        conv_post_head,
        crate::model::audio::kokoro::IStftNetGeneratorOpts {
            num_kernels,
            leaky_slope: 0.1,
            stft: crate::model::audio::kokoro::GeneratorStftParams {
                n_fft: config.n_fft,
                hop_length: config.hop_length,
            },
            // Upstream Kokoro uses `ReflectionPad1d((conv_post_kernel-1)/2)`.
            // `conv_post_kernel = 7` → pad = 3.
            last_stage_reflect_pad: 3,
            // The noise-path waveform is at audio-sample rate, which equals
            // `total_upsample * hop_length` samples per frame-rate input. For
            // Kokoro-82M: 60 * 5 = 300.
            f0_upsample_factor: config.total_upsample() * config.hop_length,
        },
    )?;
    let decoder = Decoder::new(asr_res, f0_conv, n_conv, dec_encode, dec_blocks, generator)?;

    Ok(KokoroModelV2 {
        bert,
        text_encoder,
        predictor,
        decoder,
        config,
    })
}

/// Name alias — callers importing from `kokoro::load_kokoro_v2` get the
/// public entry point; this mirror keeps the existing tier-3 stub path
/// (`load_kokoro` in `loader.rs`) discoverable too.
pub fn load_kokoro_full<R, C>(
    client: &C,
    model_dir: impl AsRef<Path>,
    device: &R::Device,
) -> Result<KokoroModelV2<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + ReduceOps<R> + UnaryOps<R> + BinaryOps<R> + TensorOps<R>,
{
    load_kokoro_v2(client, model_dir, device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn load_kokoro_reports_missing_config() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let tmp = std::env::temp_dir().join("boostr_kokoro_v2_missing_config");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        assert!(load_kokoro_v2::<CpuRuntime, _>(&client, &tmp, &device).is_err());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn load_kokoro_reports_missing_safetensors() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let tmp = std::env::temp_dir().join("boostr_kokoro_v2_missing_st");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        // Valid config, no .safetensors file.
        std::fs::write(tmp.join("config.json"), b"{}").unwrap();
        assert!(load_kokoro_v2::<CpuRuntime, _>(&client, &tmp, &device).is_err());
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
