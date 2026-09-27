//! Synthetic-checkpoint builders shared by `super::core`'s and
//! `super::synthesize`'s tests: the same tiny hand-built weight set both
//! exercise `VitsModel::from_varbuilder` against, instead of two divergent
//! ones.

use crate::model::config::VitsConfig;
use crate::nn::varmap::VarMap;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

/// A single-speaker config small enough to hand-build every tensor for,
/// using the deterministic duration predictor (far fewer tensors than
/// the stochastic one — the full stochastic pipeline is exercised by
/// `tests/vits_synthesis_parity.rs` against the real checkpoint).
pub(super) fn tiny_config(num_speakers: usize, speaker_embedding_size: usize) -> VitsConfig {
    VitsConfig {
        vocab_size: 6,
        hidden_size: 4,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        window_size: 0,
        use_bias: true,
        ffn_dim: 8,
        ffn_kernel_size: 3,
        flow_size: 4,
        spectrogram_bins: 4,
        hidden_act: "relu".to_string(),
        layer_norm_eps: 1e-5,
        use_stochastic_duration_prediction: false,
        duration_predictor_kernel_size: 3,
        num_speakers,
        speaker_embedding_size,
        upsample_initial_channel: 8,
        upsample_rates: vec![2, 2],
        upsample_kernel_sizes: vec![4, 4],
        resblock_kernel_sizes: vec![3],
        resblock_dilation_sizes: vec![vec![1]],
        leaky_relu_slope: 0.1,
        wavenet_kernel_size: 3,
        wavenet_dilation_rate: 1,
        prior_encoder_num_wavenet_layers: 1,
        prior_encoder_num_flows: 1,
        ..Default::default()
    }
}

pub fn put(map: &mut VarMap<CpuRuntime>, name: &str, data: Vec<f32>, shape: &[usize]) {
    let device = CpuDevice::new();
    map.insert(
        name.to_string(),
        Tensor::<CpuRuntime>::from_slice(&data, shape, &device).unwrap(),
    );
}

/// Every tensor a `tiny_config` checkpoint needs, named exactly as the
/// real MMS-TTS `model.safetensors` names them.
pub fn build_map(cfg: &VitsConfig) -> VarMap<CpuRuntime> {
    let mut map = VarMap::<CpuRuntime>::new();
    let h = cfg.hidden_size;
    let ffn = cfg.ffn_dim;
    let k = cfg.ffn_kernel_size;
    let flow = cfg.flow_size;
    let half = flow / 2;

    // text_encoder
    put(
        &mut map,
        "text_encoder.embed_tokens.weight",
        vec![0.01; cfg.vocab_size * h],
        &[cfg.vocab_size, h],
    );
    let p = "text_encoder.encoder.layers.0";
    for name in ["q_proj", "k_proj", "v_proj", "out_proj"] {
        put(
            &mut map,
            &format!("{p}.attention.{name}.weight"),
            vec![0.1; h * h],
            &[h, h],
        );
        put(
            &mut map,
            &format!("{p}.attention.{name}.bias"),
            vec![0.0; h],
            &[h],
        );
    }
    put(
        &mut map,
        &format!("{p}.layer_norm.weight"),
        vec![1.0; h],
        &[h],
    );
    put(
        &mut map,
        &format!("{p}.layer_norm.bias"),
        vec![0.0; h],
        &[h],
    );
    put(
        &mut map,
        &format!("{p}.feed_forward.conv_1.weight"),
        vec![0.1; ffn * h * k],
        &[ffn, h, k],
    );
    put(
        &mut map,
        &format!("{p}.feed_forward.conv_1.bias"),
        vec![0.0; ffn],
        &[ffn],
    );
    put(
        &mut map,
        &format!("{p}.feed_forward.conv_2.weight"),
        vec![0.1; h * ffn * k],
        &[h, ffn, k],
    );
    put(
        &mut map,
        &format!("{p}.feed_forward.conv_2.bias"),
        vec![0.0; h],
        &[h],
    );
    put(
        &mut map,
        &format!("{p}.final_layer_norm.weight"),
        vec![1.0; h],
        &[h],
    );
    put(
        &mut map,
        &format!("{p}.final_layer_norm.bias"),
        vec![0.0; h],
        &[h],
    );
    put(
        &mut map,
        "text_encoder.project.weight",
        vec![0.1; flow * 2 * h],
        &[flow * 2, h, 1],
    );
    put(
        &mut map,
        "text_encoder.project.bias",
        vec![0.0; flow * 2],
        &[flow * 2],
    );

    // duration_predictor (deterministic)
    put(
        &mut map,
        "duration_predictor.conv_1.weight",
        vec![0.1; h * h * cfg.duration_predictor_kernel_size],
        &[h, h, cfg.duration_predictor_kernel_size],
    );
    put(
        &mut map,
        "duration_predictor.conv_1.bias",
        vec![0.0; h],
        &[h],
    );
    put(
        &mut map,
        "duration_predictor.norm_1.weight",
        vec![1.0; h],
        &[h],
    );
    put(
        &mut map,
        "duration_predictor.norm_1.bias",
        vec![0.0; h],
        &[h],
    );
    put(
        &mut map,
        "duration_predictor.conv_2.weight",
        vec![0.1; h * h * cfg.duration_predictor_kernel_size],
        &[h, h, cfg.duration_predictor_kernel_size],
    );
    put(
        &mut map,
        "duration_predictor.conv_2.bias",
        vec![0.0; h],
        &[h],
    );
    put(
        &mut map,
        "duration_predictor.norm_2.weight",
        vec![1.0; h],
        &[h],
    );
    put(
        &mut map,
        "duration_predictor.norm_2.bias",
        vec![0.0; h],
        &[h],
    );
    put(
        &mut map,
        "duration_predictor.proj.weight",
        vec![0.05; h],
        &[1, h, 1],
    );
    put(&mut map, "duration_predictor.proj.bias", vec![0.0; 1], &[1]);
    if cfg.speaker_embedding_size != 0 {
        put(
            &mut map,
            "duration_predictor.cond.weight",
            vec![0.02; h * cfg.speaker_embedding_size],
            &[h, cfg.speaker_embedding_size, 1],
        );
        put(&mut map, "duration_predictor.cond.bias", vec![0.0; h], &[h]);
    }

    // flow (one coupling layer)
    put(
        &mut map,
        "flow.flows.0.conv_pre.weight",
        vec![0.05; h * half],
        &[h, half, 1],
    );
    put(&mut map, "flow.flows.0.conv_pre.bias", vec![0.0; h], &[h]);
    put(
        &mut map,
        "flow.flows.0.wavenet.in_layers.0.weight",
        vec![0.1; 2 * h * h * cfg.wavenet_kernel_size],
        &[2 * h, h, cfg.wavenet_kernel_size],
    );
    put(
        &mut map,
        "flow.flows.0.wavenet.in_layers.0.bias",
        vec![0.0; 2 * h],
        &[2 * h],
    );
    put(
        &mut map,
        "flow.flows.0.wavenet.res_skip_layers.0.weight",
        vec![0.1; h * h],
        &[h, h, 1],
    );
    put(
        &mut map,
        "flow.flows.0.wavenet.res_skip_layers.0.bias",
        vec![0.0; h],
        &[h],
    );
    if cfg.speaker_embedding_size != 0 {
        put(
            &mut map,
            "flow.flows.0.wavenet.cond_layer.weight",
            vec![0.02; 2 * h * cfg.speaker_embedding_size],
            &[2 * h, cfg.speaker_embedding_size, 1],
        );
    }
    put(
        &mut map,
        "flow.flows.0.conv_post.weight",
        vec![0.05; half * h],
        &[half, h, 1],
    );
    put(
        &mut map,
        "flow.flows.0.conv_post.bias",
        vec![0.0; half],
        &[half],
    );

    // decoder
    put(
        &mut map,
        "decoder.conv_pre.weight",
        vec![0.05; cfg.upsample_initial_channel * flow * 7],
        &[cfg.upsample_initial_channel, flow, 7],
    );
    put(
        &mut map,
        "decoder.conv_pre.bias",
        vec![0.0; cfg.upsample_initial_channel],
        &[cfg.upsample_initial_channel],
    );
    let mut channels = cfg.upsample_initial_channel;
    let mut resblock_idx = 0;
    for (i, (_rate, &ksize)) in cfg
        .upsample_rates
        .iter()
        .zip(cfg.upsample_kernel_sizes.iter())
        .enumerate()
    {
        let out_c = channels / 2;
        put(
            &mut map,
            &format!("decoder.upsampler.{i}.weight"),
            vec![0.05; channels * out_c * ksize],
            &[channels, out_c, ksize],
        );
        put(
            &mut map,
            &format!("decoder.upsampler.{i}.bias"),
            vec![0.0; out_c],
            &[out_c],
        );
        for (kernel_size, dilations) in cfg
            .resblock_kernel_sizes
            .iter()
            .zip(cfg.resblock_dilation_sizes.iter())
        {
            for j in 0..dilations.len() {
                put(
                    &mut map,
                    &format!("decoder.resblocks.{resblock_idx}.convs1.{j}.weight"),
                    vec![0.05; out_c * out_c * kernel_size],
                    &[out_c, out_c, *kernel_size],
                );
                put(
                    &mut map,
                    &format!("decoder.resblocks.{resblock_idx}.convs1.{j}.bias"),
                    vec![0.0; out_c],
                    &[out_c],
                );
                put(
                    &mut map,
                    &format!("decoder.resblocks.{resblock_idx}.convs2.{j}.weight"),
                    vec![0.05; out_c * out_c * kernel_size],
                    &[out_c, out_c, *kernel_size],
                );
                put(
                    &mut map,
                    &format!("decoder.resblocks.{resblock_idx}.convs2.{j}.bias"),
                    vec![0.0; out_c],
                    &[out_c],
                );
            }
            resblock_idx += 1;
        }
        channels = out_c;
    }
    put(
        &mut map,
        "decoder.conv_post.weight",
        vec![0.05; channels * 7],
        &[1, channels, 7],
    );
    if cfg.speaker_embedding_size != 0 {
        put(
            &mut map,
            "decoder.cond.weight",
            vec![0.02; cfg.upsample_initial_channel * cfg.speaker_embedding_size],
            &[cfg.upsample_initial_channel, cfg.speaker_embedding_size, 1],
        );
        put(
            &mut map,
            "decoder.cond.bias",
            vec![0.0; cfg.upsample_initial_channel],
            &[cfg.upsample_initial_channel],
        );
    }

    if cfg.num_speakers > 1 {
        put(
            &mut map,
            "embed_speaker.weight",
            vec![0.02; cfg.num_speakers * cfg.speaker_embedding_size],
            &[cfg.num_speakers, cfg.speaker_embedding_size],
        );
    }

    map
}
