//! Qwen3-VL vision tower on synthetic weights: shapes, the position-table
//! resample path, determinism and input checks. CPU only, no model file.

use boostr::model::vision::qwen3vl::{
    PreprocessedImage, Qwen3VlVision, Qwen3VlVisionConfig, RgbImage, preprocess,
};
use boostr::nn::{VarBuilder, VarMap};
use boostr::quant::QuantFormat;
use boostr::quant::traits::QuantizeOps;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

fn tiny_config() -> Qwen3VlVisionConfig {
    Qwen3VlVisionConfig {
        image_size: 8,
        patch_size: 2,
        hidden_size: 8,
        intermediate_size: 16,
        num_layers: 2,
        num_heads: 2,
        spatial_merge: 2,
        projection_dim: 6,
        layer_norm_eps: 1e-6,
        image_mean: [0.5; 3],
        image_std: [0.5; 3],
        image_min_tokens: 1,
        image_max_tokens: 4096,
    }
}

fn synth(seed: usize, shape: &[usize], device: &CpuDevice) -> Tensor<CpuRuntime> {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| ((i + seed * 7919) as f32 * 0.37).sin() * 0.1)
        .collect();
    Tensor::from_slice(&data, shape, device).unwrap()
}

/// Every mmproj tensor of [`tiny_config`], with the merger matrices in Q8_0
/// so the quantized projection path runs too.
fn tiny_varmap(client: &CpuClient, device: &CpuDevice) -> VarMap<CpuRuntime> {
    let cfg = tiny_config();
    let (h, f, p, g) = (
        cfg.hidden_size,
        cfg.intermediate_size,
        cfg.patch_size,
        cfg.pos_grid_side(),
    );
    let mut map = VarMap::new();
    let mut seed = 1;
    let mut put = |map: &mut VarMap<CpuRuntime>, name: &str, shape: &[usize]| {
        map.insert(name.to_string(), synth(seed, shape, device));
        seed += 1;
    };
    put(&mut map, "v.patch_embd.weight", &[h, 3, p, p]);
    put(&mut map, "v.patch_embd.weight.1", &[h, 3, p, p]);
    put(&mut map, "v.patch_embd.bias", &[h]);
    put(&mut map, "v.position_embd.weight", &[g * g, h]);
    for i in 0..cfg.num_layers {
        let b = format!("v.blk.{i}");
        for ln in ["ln1", "ln2"] {
            put(&mut map, &format!("{b}.{ln}.weight"), &[h]);
            put(&mut map, &format!("{b}.{ln}.bias"), &[h]);
        }
        put(&mut map, &format!("{b}.attn_qkv.weight"), &[3 * h, h]);
        put(&mut map, &format!("{b}.attn_qkv.bias"), &[3 * h]);
        put(&mut map, &format!("{b}.attn_out.weight"), &[h, h]);
        put(&mut map, &format!("{b}.attn_out.bias"), &[h]);
        put(&mut map, &format!("{b}.ffn_up.weight"), &[f, h]);
        put(&mut map, &format!("{b}.ffn_up.bias"), &[f]);
        put(&mut map, &format!("{b}.ffn_down.weight"), &[h, f]);
        put(&mut map, &format!("{b}.ffn_down.bias"), &[h]);
    }
    put(&mut map, "v.post_ln.weight", &[h]);
    put(&mut map, "v.post_ln.bias", &[h]);
    let m = cfg.merged_width();
    put(&mut map, "mm.0.bias", &[m]);
    put(&mut map, "mm.2.bias", &[cfg.projection_dim]);
    let mm0 = client
        .quantize(&synth(101, &[m, m], device), QuantFormat::Q8_0)
        .unwrap();
    let mm2 = client
        .quantize(
            &synth(102, &[cfg.projection_dim, m], device),
            QuantFormat::Q8_0,
        )
        .unwrap();
    map.insert_quant("mm.0.weight".to_string(), mm0);
    map.insert_quant("mm.2.weight".to_string(), mm2);
    map
}

fn tiny_model(client: &CpuClient, device: &CpuDevice) -> Qwen3VlVision<CpuRuntime> {
    let mut map = tiny_varmap(client, device);
    let mut vb = VarBuilder::new(&mut map, device);
    let model = Qwen3VlVision::from_varbuilder(&mut vb, &tiny_config()).unwrap();
    assert_eq!(map.len(), 0, "every synthetic tensor must be consumed");
    model
}

fn tiny_image(width: usize, height: usize) -> PreprocessedImage {
    let cfg = tiny_config();
    let data: Vec<u8> = (0..width * height * 3)
        .map(|i| (i * 37 % 256) as u8)
        .collect();
    let img = RgbImage::new(width, height, data).unwrap();
    let pre = preprocess(&img, &cfg).unwrap();
    assert_eq!((pre.width, pre.height), (width, height));
    pre
}

#[test]
fn synthetic_encoder_shapes_on_training_grid() {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    let model = tiny_model(&client, &device);
    // 8x8 pixels: 4x4 patches, the training grid, 2x2 output tokens.
    let pre = tiny_image(8, 8);
    assert_eq!((pre.nx, pre.ny), (2, 2));
    let out = model.encode_image(&client, &pre).unwrap();
    assert_eq!(out.shape(), &[4, 6]);
    let data: Vec<f32> = out.to_vec();
    assert!(data.iter().all(|v| v.is_finite()));
    assert!(data.iter().any(|v| v.abs() > 0.0));
}

#[test]
fn synthetic_encoder_shapes_with_position_resample() {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    let model = tiny_model(&client, &device);
    // 16x8 pixels: 8x4 patches, wider than the 4x4 position grid.
    let pre = tiny_image(16, 8);
    assert_eq!((pre.nx, pre.ny), (4, 2));
    let out = model.encode_image(&client, &pre).unwrap();
    assert_eq!(out.shape(), &[8, 6]);
    let data: Vec<f32> = out.to_vec();
    assert!(data.iter().all(|v| v.is_finite()));
}

#[test]
fn synthetic_encoder_is_deterministic_and_position_aware() {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    let model = tiny_model(&client, &device);
    let pre = tiny_image(8, 8);
    let a: Vec<f32> = model.encode_image(&client, &pre).unwrap().to_vec();
    let b: Vec<f32> = model.encode_image(&client, &pre).unwrap().to_vec();
    assert_eq!(a, b);
    // A constant image still yields distinct tokens through the position
    // embedding and the 2D rope.
    let flat = tiny_image_constant(8, 8);
    let out: Vec<f32> = model.encode_image(&client, &flat).unwrap().to_vec();
    assert!(out[..6] != out[6..12], "tokens 0 and 1 must differ");
}

fn tiny_image_constant(width: usize, height: usize) -> PreprocessedImage {
    let img = RgbImage::filled(width, height, [100, 150, 200]);
    preprocess(&img, &tiny_config()).unwrap()
}

#[test]
fn encode_rejects_unaligned_image() {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    let model = tiny_model(&client, &device);
    let pixels = vec![0f32; 3 * 6 * 8];
    assert!(model.encode(&client, &pixels, 6, 8).is_err());
    assert!(model.encode(&client, &pixels[..10], 8, 8).is_err());
}
