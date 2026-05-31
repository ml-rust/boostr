use super::*;
use crate::model::encoder::config::HiddenAct;
use crate::test_utils::cpu_setup;
use numr::runtime::cpu::CpuRuntime;

fn make_test_encoder() -> (
    Encoder<CpuRuntime>,
    numr::runtime::cpu::CpuClient,
    numr::runtime::cpu::CpuDevice,
) {
    let (client, device) = cpu_setup();

    let config = EncoderConfig {
        vocab_size: 10,
        hidden_size: 8,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        intermediate_size: 16,
        max_position_embeddings: 32,
        layer_norm_eps: 1e-12,
        hidden_act: HiddenAct::Gelu,
        type_vocab_size: 0,
        arch_family: ArchFamily::Bert,
        padding_token_id: 0,
        compute_dtype: numr::dtype::DType::F32,
        rope_freq_base: 10000.0,
        causal: false,
        ffn_variant: crate::model::encoder::config::FfnVariant::Standard,
        token_type_embed_size: 0,
        num_kv_heads: 0,
        head_dim_explicit: None,
        rms_eps: 1e-6,
        sliding_window: None,
        embed_scale: false,
        max_tokens_per_forward: None,
    };

    let encoder = Encoder::from_weights(config, Pooling::Mean, |name| match name {
        "embeddings.word_embeddings.weight" => {
            Ok(Tensor::from_slice(&vec![0.1f32; 10 * 8], &[10, 8], &device))
        }
        "embeddings.position_embeddings.weight" => Ok(Tensor::from_slice(
            &vec![0.01f32; 32 * 8],
            &[32, 8],
            &device,
        )),
        "embeddings.layer_norm.weight" => Ok(Tensor::from_slice(&[1.0f32; 8], &[8], &device)),
        "embeddings.layer_norm.bias" => Ok(Tensor::from_slice(&[0.0f32; 8], &[8], &device)),
        n if n.ends_with("query.weight")
            || n.ends_with("key.weight")
            || n.ends_with("value.weight") =>
        {
            Ok(Tensor::from_slice(&vec![0.02f32; 8 * 8], &[8, 8], &device))
        }
        n if n.ends_with("query.bias") || n.ends_with("key.bias") || n.ends_with("value.bias") => {
            Ok(Tensor::from_slice(&[0.0f32; 8], &[8], &device))
        }
        n if n.ends_with("attention.output.dense.weight") => {
            Ok(Tensor::from_slice(&vec![0.02f32; 8 * 8], &[8, 8], &device))
        }
        n if n.ends_with("attention.output.dense.bias") => {
            Ok(Tensor::from_slice(&[0.0f32; 8], &[8], &device))
        }
        n if n.ends_with("output.dense.weight") => Ok(Tensor::from_slice(
            &vec![0.02f32; 8 * 16],
            &[8, 16],
            &device,
        )),
        n if n.ends_with("output.dense.bias") => {
            Ok(Tensor::from_slice(&[0.0f32; 8], &[8], &device))
        }
        n if n.ends_with("LayerNorm.weight") => Ok(Tensor::from_slice(&[1.0f32; 8], &[8], &device)),
        n if n.ends_with("LayerNorm.bias") => Ok(Tensor::from_slice(&[0.0f32; 8], &[8], &device)),
        n if n.ends_with("intermediate.dense.weight") => Ok(Tensor::from_slice(
            &vec![0.02f32; 16 * 8],
            &[16, 8],
            &device,
        )),
        n if n.ends_with("intermediate.dense.bias") => {
            Ok(Tensor::from_slice(&[0.0f32; 16], &[16], &device))
        }
        _ => Err(Error::ModelError {
            reason: format!("unknown weight: {name}"),
        }),
    })
    .unwrap();

    (encoder, client, device)
}

fn make_test_encoder_cls() -> (
    Encoder<CpuRuntime>,
    numr::runtime::cpu::CpuClient,
    numr::runtime::cpu::CpuDevice,
) {
    let (client, device) = cpu_setup();

    let config = EncoderConfig {
        vocab_size: 10,
        hidden_size: 8,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        intermediate_size: 16,
        max_position_embeddings: 32,
        layer_norm_eps: 1e-12,
        hidden_act: HiddenAct::Gelu,
        type_vocab_size: 0,
        arch_family: ArchFamily::Bert,
        padding_token_id: 0,
        compute_dtype: numr::dtype::DType::F32,
        rope_freq_base: 10000.0,
        causal: false,
        ffn_variant: crate::model::encoder::config::FfnVariant::Standard,
        token_type_embed_size: 0,
        num_kv_heads: 0,
        head_dim_explicit: None,
        rms_eps: 1e-6,
        sliding_window: None,
        embed_scale: false,
        max_tokens_per_forward: None,
    };

    let device_ref = &device;
    let encoder = Encoder::from_weights(config, Pooling::Cls, |name| match name {
        "embeddings.word_embeddings.weight" => Ok(Tensor::from_slice(
            &vec![0.1f32; 10 * 8],
            &[10, 8],
            device_ref,
        )),
        "embeddings.position_embeddings.weight" => Ok(Tensor::from_slice(
            &vec![0.01f32; 32 * 8],
            &[32, 8],
            device_ref,
        )),
        "embeddings.layer_norm.weight" => Ok(Tensor::from_slice(&[1.0f32; 8], &[8], device_ref)),
        "embeddings.layer_norm.bias" => Ok(Tensor::from_slice(&[0.0f32; 8], &[8], device_ref)),
        n if n.ends_with("query.weight")
            || n.ends_with("key.weight")
            || n.ends_with("value.weight")
            || n.ends_with("attention.output.dense.weight") =>
        {
            Ok(Tensor::from_slice(
                &vec![0.02f32; 8 * 8],
                &[8, 8],
                device_ref,
            ))
        }
        n if n.ends_with("query.bias")
            || n.ends_with("key.bias")
            || n.ends_with("value.bias")
            || n.ends_with("attention.output.dense.bias")
            || n.ends_with("output.dense.bias") =>
        {
            Ok(Tensor::from_slice(&[0.0f32; 8], &[8], device_ref))
        }
        n if n.ends_with("LayerNorm.weight") => {
            Ok(Tensor::from_slice(&[1.0f32; 8], &[8], device_ref))
        }
        n if n.ends_with("LayerNorm.bias") => {
            Ok(Tensor::from_slice(&[0.0f32; 8], &[8], device_ref))
        }
        n if n.ends_with("intermediate.dense.weight") => Ok(Tensor::from_slice(
            &vec![0.02f32; 16 * 8],
            &[16, 8],
            device_ref,
        )),
        n if n.ends_with("intermediate.dense.bias") => {
            Ok(Tensor::from_slice(&[0.0f32; 16], &[16], device_ref))
        }
        n if n.ends_with("output.dense.weight") => Ok(Tensor::from_slice(
            &vec![0.02f32; 8 * 16],
            &[8, 16],
            device_ref,
        )),
        _ => Err(Error::ModelError {
            reason: format!("unknown weight: {name}"),
        }),
    })
    .unwrap();

    (encoder, client, device)
}

#[test]
fn test_encode_output_shape() {
    let (encoder, client, device) = make_test_encoder();
    let input_ids = Tensor::<CpuRuntime>::from_slice(&[1i64, 2, 3], &[1, 3], &device);
    let hidden = encoder.encode(&client, &input_ids, None).unwrap();
    assert_eq!(hidden.shape(), &[1, 3, 8]);
}

#[test]
fn test_embed_mean_pool() {
    let (encoder, client, device) = make_test_encoder();
    let input_ids = Tensor::<CpuRuntime>::from_slice(&[1i64, 2, 3, 4], &[1, 4], &device);
    let emb = encoder.embed(&client, &input_ids, None).unwrap();
    assert_eq!(emb.shape(), &[1, 8]);
}

#[test]
fn test_embed_batched() {
    let (encoder, client, device) = make_test_encoder();
    let input_ids = Tensor::<CpuRuntime>::from_slice(&[1i64, 2, 3, 4, 5, 6], &[2, 3], &device);
    let emb = encoder.embed(&client, &input_ids, None).unwrap();
    assert_eq!(emb.shape(), &[2, 8]);
}

#[test]
fn test_encode_with_none_mask_matches_no_mask() {
    let (encoder, client, device) = make_test_encoder();
    let input_ids = Tensor::<CpuRuntime>::from_slice(&[1i64, 2, 3], &[1, 3], &device);
    let h1 = encoder.encode(&client, &input_ids, None).unwrap();
    let h2 = encoder.encode(&client, &input_ids, None).unwrap();
    let v1: Vec<f32> = h1.tensor().to_vec();
    let v2: Vec<f32> = h2.tensor().to_vec();
    assert_eq!(v1, v2);
}

#[test]
fn test_mask_wrong_shape_returns_error() {
    let (encoder, client, device) = make_test_encoder();
    let input_ids = Tensor::<CpuRuntime>::from_slice(&[1i64, 2, 3], &[1, 3], &device);
    let bad_mask = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 4], &[1, 4], &device);
    let result = encoder.encode(&client, &input_ids, Some(&bad_mask));
    assert!(result.is_err());
}

#[test]
fn test_cls_pooling_batched_produces_correct_shape() {
    let (encoder, client, device) = make_test_encoder_cls();
    let input_ids = Tensor::<CpuRuntime>::from_slice(&[1i64, 2, 3, 4, 5, 6], &[2, 3], &device);
    let emb = encoder.embed(&client, &input_ids, None).unwrap();
    assert_eq!(emb.shape(), &[2, 8]);
}

#[test]
fn test_xlm_roberta_position_ids() {
    let (client, device) = cpu_setup();

    let config = EncoderConfig {
        vocab_size: 10,
        hidden_size: 8,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        intermediate_size: 16,
        max_position_embeddings: 32,
        layer_norm_eps: 1e-12,
        hidden_act: HiddenAct::Gelu,
        type_vocab_size: 0,
        arch_family: ArchFamily::XlmRoberta,
        padding_token_id: 1,
        compute_dtype: numr::dtype::DType::F32,
        rope_freq_base: 10000.0,
        causal: false,
        ffn_variant: crate::model::encoder::config::FfnVariant::Standard,
        token_type_embed_size: 0,
        num_kv_heads: 0,
        head_dim_explicit: None,
        rms_eps: 1e-6,
        sliding_window: None,
        embed_scale: false,
        max_tokens_per_forward: None,
    };

    let device_ref = &device;
    let encoder = Encoder::<CpuRuntime>::from_weights(config, Pooling::Mean, |name| match name {
        "embeddings.word_embeddings.weight" => Ok(Tensor::from_slice(
            &vec![0.1f32; 10 * 8],
            &[10, 8],
            device_ref,
        )),
        "embeddings.position_embeddings.weight" => Ok(Tensor::from_slice(
            &vec![0.01f32; 32 * 8],
            &[32, 8],
            device_ref,
        )),
        "embeddings.layer_norm.weight" => Ok(Tensor::from_slice(&[1.0f32; 8], &[8], device_ref)),
        "embeddings.layer_norm.bias" => Ok(Tensor::from_slice(&[0.0f32; 8], &[8], device_ref)),
        n if n.ends_with("query.weight")
            || n.ends_with("key.weight")
            || n.ends_with("value.weight")
            || n.ends_with("attention.output.dense.weight") =>
        {
            Ok(Tensor::from_slice(
                &vec![0.02f32; 8 * 8],
                &[8, 8],
                device_ref,
            ))
        }
        n if n.ends_with("query.bias")
            || n.ends_with("key.bias")
            || n.ends_with("value.bias")
            || n.ends_with("attention.output.dense.bias")
            || n.ends_with("output.dense.bias") =>
        {
            Ok(Tensor::from_slice(&[0.0f32; 8], &[8], device_ref))
        }
        n if n.ends_with("LayerNorm.weight") => {
            Ok(Tensor::from_slice(&[1.0f32; 8], &[8], device_ref))
        }
        n if n.ends_with("LayerNorm.bias") => {
            Ok(Tensor::from_slice(&[0.0f32; 8], &[8], device_ref))
        }
        n if n.ends_with("intermediate.dense.weight") => Ok(Tensor::from_slice(
            &vec![0.02f32; 16 * 8],
            &[16, 8],
            device_ref,
        )),
        n if n.ends_with("intermediate.dense.bias") => {
            Ok(Tensor::from_slice(&[0.0f32; 16], &[16], device_ref))
        }
        n if n.ends_with("output.dense.weight") => Ok(Tensor::from_slice(
            &vec![0.02f32; 8 * 16],
            &[8, 16],
            device_ref,
        )),
        _ => Err(Error::ModelError {
            reason: format!("unknown weight: {name}"),
        }),
    })
    .unwrap();

    let input_ids = Tensor::<CpuRuntime>::from_slice(&[0i64, 4, 7, 1, 1], &[1, 5], &device);
    let result = encoder.embed(&client, &input_ids, None);
    assert!(
        result.is_ok(),
        "xlm-roberta forward should succeed: {result:?}"
    );
    assert_eq!(result.unwrap().shape(), &[1, 8]);
}

#[test]
fn test_from_weights_quant_forward_shape() {
    use crate::nn::Weight;
    use crate::quant::format::QuantFormat;
    use crate::quant::tensor::QuantTensor;

    let (client, device) = cpu_setup();

    let hidden = 32usize;
    let inter = 64usize;
    let vocab = 64usize;
    let max_pos = 64usize;

    let config = EncoderConfig {
        vocab_size: vocab,
        hidden_size: hidden,
        num_hidden_layers: 1,
        num_attention_heads: 4,
        intermediate_size: inter,
        max_position_embeddings: max_pos,
        layer_norm_eps: 1e-12,
        hidden_act: HiddenAct::Gelu,
        type_vocab_size: 0,
        arch_family: ArchFamily::Bert,
        padding_token_id: 0,
        compute_dtype: numr::dtype::DType::F32,
        rope_freq_base: 10000.0,
        causal: false,
        ffn_variant: crate::model::encoder::config::FfnVariant::Standard,
        token_type_embed_size: 0,
        num_kv_heads: 0,
        head_dim_explicit: None,
        rms_eps: 1e-6,
        sliding_window: None,
        embed_scale: false,
        max_tokens_per_forward: None,
    };

    let make_q8_0 = |rows: usize, cols: usize| -> QuantTensor<CpuRuntime> {
        let n_blocks = (rows * cols) / 32;
        let bytes: Vec<u8> = vec![0u8; n_blocks * 34];
        QuantTensor::from_bytes(&bytes, QuantFormat::Q8_0, &[rows, cols], &device).unwrap()
    };

    let d = &device;
    let encoder = Encoder::from_weights_quant(config, Pooling::Mean, &client, |name| {
        let is_proj = name.contains("attention.self.query.weight")
            || name.contains("attention.self.key.weight")
            || name.contains("attention.self.value.weight")
            || name.contains("attention.output.dense.weight")
            || name.contains("intermediate.dense.weight")
            || (name.contains("output.dense.weight") && !name.contains("attention"));

        if is_proj {
            let qt = if name.contains("intermediate.dense.weight") {
                make_q8_0(inter, hidden)
            } else if name.contains("output.dense.weight") && !name.contains("attention") {
                make_q8_0(hidden, inter)
            } else {
                make_q8_0(hidden, hidden)
            };
            return Ok(Weight::Quantized(qt));
        }

        let t: Tensor<CpuRuntime> = match name {
            "embeddings.word_embeddings.weight" => {
                Tensor::from_slice(&vec![0.1f32; vocab * hidden], &[vocab, hidden], d)
            }
            "embeddings.position_embeddings.weight" => {
                Tensor::from_slice(&vec![0.01f32; max_pos * hidden], &[max_pos, hidden], d)
            }
            n if n.ends_with("layer_norm.weight") || n.ends_with("LayerNorm.weight") => {
                Tensor::from_slice(&vec![1.0f32; hidden], &[hidden], d)
            }
            n if n.ends_with("layer_norm.bias") || n.ends_with("LayerNorm.bias") => {
                Tensor::from_slice(&vec![0.0f32; hidden], &[hidden], d)
            }
            _ => {
                return Err(Error::ModelError {
                    reason: format!("unknown weight: {name}"),
                });
            }
        };
        Ok(Weight::Standard(t))
    })
    .unwrap();

    let input_ids = Tensor::<CpuRuntime>::from_slice(&[1i64, 2, 3], &[1, 3], &device);
    let out = encoder.encode_inference(&client, &input_ids, None).unwrap();
    assert_eq!(out.shape(), &[1, 3, hidden]);
}
