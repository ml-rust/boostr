//! Qwen3-VL vision tower against the reference dump in
//! `tests/fixtures/vision/` (produced by `tests/tools/mtmd_embed_dump`).
//!
//! Preprocessing and token-grid tests need no model file. The embedding
//! parity tests read the mmproj from `BOOSTR_BONSAI2_DIR`
//! (`Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf`); unset or missing, they print
//! one `skip:` line and pass.
//!
//! ```bash
//! BOOSTR_BONSAI2_DIR=/path/to/dir cargo nextest run -p boostr \
//!   --test qwen3vl_vision --nocapture
//! BOOSTR_BONSAI2_DIR=/path/to/dir cargo nextest run -p boostr --features cuda \
//!   --test qwen3vl_vision --nocapture
//! ```
//!
//! # Two oracles per image
//!
//! `img_*.f32.embd` comes from the reference running an F32 copy of the
//! mmproj (`mmproj_to_f32.py`): the same dequantized weights, f32 matmuls.
//! What remains between it and the port is the reference's f16 patch
//! embedding (im2col and its weight cast to f16) and its f16 GELU lookup
//! table. The port asserts per-token cosine above 0.9998 and a max absolute
//! difference under 2% of the largest reference magnitude there.
//!
//! `img_*.embd` comes from the shipped Q8_0 mmproj as deployed. The
//! reference quantizes every matmul activation to 8 bits per 32-block
//! (relative step 1/127 of the block maximum) and casts the `ffn_down`
//! input to f16; the port keeps activations in f32. Tokens with outlier
//! activations lose the most, so this oracle only bounds the worst token at
//! cosine 0.98 and 15% of the largest magnitude, with the mean cosine above
//! 0.999. Both sets of numbers print for every token.
//!
//! The CUDA Q8_0 matmul quantizes activations to Q8_1 like the reference
//! does, so on CUDA both oracles use the wider bound.

use std::path::PathBuf;

use boostr::model::vision::qwen3vl::{
    PreprocessedImage, Qwen3VlVisionConfig, RgbImage, decode_image, load_qwen3vl_vision_gguf,
    preprocess, resize_pad_ceil, smart_resize,
};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

const ENV_DIR: &str = "BOOSTR_BONSAI2_DIR";
const MMPROJ_FILE: &str = "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf";
const FIXTURES: [&str; 2] = ["img_96x64", "img_400x300"];
/// Per-oracle bounds: `(min token cosine, mean cosine, max abs / max |ref|)`.
struct Bounds {
    min_cosine: f32,
    mean_cosine: f32,
    max_rel_abs: f32,
}

const F32_BOUNDS: Bounds = Bounds {
    min_cosine: 0.9998,
    mean_cosine: 0.9999,
    max_rel_abs: 0.02,
};

const Q8_BOUNDS: Bounds = Bounds {
    min_cosine: 0.98,
    mean_cosine: 0.999,
    max_rel_abs: 0.15,
};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/vision")
}

fn bonsai2_config() -> Qwen3VlVisionConfig {
    Qwen3VlVisionConfig {
        image_size: 768,
        patch_size: 16,
        hidden_size: 1152,
        intermediate_size: 4304,
        num_layers: 27,
        num_heads: 16,
        spatial_merge: 2,
        projection_dim: 5120,
        layer_norm_eps: 1e-6,
        image_mean: [0.5; 3],
        image_std: [0.5; 3],
        image_min_tokens: 8,
        image_max_tokens: 4096,
    }
}

/// The mmproj, or `None` after one `skip:` line.
fn require_mmproj() -> Option<PathBuf> {
    let Some(dir) = std::env::var(ENV_DIR).ok().map(PathBuf::from) else {
        println!("skip: {ENV_DIR} not set");
        return None;
    };
    let path = dir.join(MMPROJ_FILE);
    if !path.is_file() {
        println!("skip: {} not found", path.display());
        return None;
    }
    Some(path)
}

struct OracleEmbd {
    n_tokens: usize,
    n_embd: usize,
    nx: usize,
    ny: usize,
    data: Vec<f32>,
}

fn read_u32(bytes: &[u8], at: usize) -> usize {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]) as usize
}

/// `suffix` is `"embd"` for the Q8_0 run or `"f32.embd"` for the F32 run.
fn read_embd(name: &str, suffix: &str) -> OracleEmbd {
    let bytes = std::fs::read(fixture_dir().join(format!("{name}.{suffix}"))).unwrap();
    let n_tokens = read_u32(&bytes, 0);
    let n_embd = read_u32(&bytes, 4);
    let nx = read_u32(&bytes, 8);
    let ny = read_u32(&bytes, 12);
    let data: Vec<f32> = bytes[16..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    assert_eq!(data.len(), n_tokens * n_embd, "{name}.embd payload length");
    OracleEmbd {
        n_tokens,
        n_embd,
        nx,
        ny,
        data,
    }
}

/// `(width, height, planar CHW bytes)` the reference encoder consumed.
fn read_u8chw(name: &str) -> (usize, usize, Vec<u8>) {
    let bytes = std::fs::read(fixture_dir().join(format!("{name}.u8chw"))).unwrap();
    let w = read_u32(&bytes, 0);
    let h = read_u32(&bytes, 4);
    assert_eq!(bytes.len(), 8 + 3 * w * h, "{name}.u8chw payload length");
    (w, h, bytes[8..].to_vec())
}

fn load_fixture_png(name: &str) -> RgbImage {
    let bytes = std::fs::read(fixture_dir().join(format!("{name}.png"))).unwrap();
    decode_image(&bytes).unwrap()
}

// ── (a) preprocessing bytes ───────────────────────────────────────────────

#[test]
fn preprocess_bytes_match_oracle() {
    let cfg = bonsai2_config();
    for name in FIXTURES {
        let src = load_fixture_png(name);
        let (tw, th) = smart_resize(
            src.width,
            src.height,
            cfg.align(),
            cfg.min_pixels(),
            cfg.max_pixels(),
        );
        let (ow, oh, oracle) = read_u8chw(name);
        assert_eq!((tw, th), (ow, oh), "{name}: smart_resize target");
        let resized = resize_pad_ceil(&src, tw, th);
        let ours = resized.to_chw();
        let n_diff = ours.iter().zip(&oracle).filter(|(a, b)| a != b).count();
        let max_diff = ours
            .iter()
            .zip(&oracle)
            .map(|(&a, &b)| (a as i32 - b as i32).abs())
            .max()
            .unwrap_or(0);
        println!("{name}: preprocessed {tw}x{th}, bytes differing {n_diff}, max diff {max_diff}");
        assert_eq!(n_diff, 0, "{name}: resized bytes differ from the reference");
    }
}

// ── (b) token grid ────────────────────────────────────────────────────────

#[test]
fn token_grid_matches_oracle() {
    let cfg = bonsai2_config();
    for name in FIXTURES {
        let src = load_fixture_png(name);
        let pre = preprocess(&src, &cfg).unwrap();
        let oracle = read_embd(name, "embd");
        assert_eq!((pre.nx, pre.ny), (oracle.nx, oracle.ny), "{name}: nx/ny");
        assert_eq!(pre.n_tokens(), oracle.n_tokens, "{name}: n_tokens");
        assert_eq!(oracle.n_embd, cfg.projection_dim, "{name}: n_embd");
        assert_eq!(pre.pixels.len(), 3 * pre.width * pre.height);
    }
}

// ── (c) embeddings ────────────────────────────────────────────────────────

struct Parity {
    min_cosine: f32,
    mean_cosine: f32,
    max_abs: f32,
    max_ref: f32,
}

fn compare(label: &str, ours: &[f32], oracle: &OracleEmbd) -> Parity {
    assert_eq!(ours.len(), oracle.data.len(), "{label}: embedding length");
    let n = oracle.n_embd;
    let mut min_cosine = f32::INFINITY;
    let mut sum_cosine = 0f64;
    let mut max_abs = 0f32;
    let mut max_ref = 0f32;
    for t in 0..oracle.n_tokens {
        let a = &ours[t * n..(t + 1) * n];
        let b = &oracle.data[t * n..(t + 1) * n];
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (&x, &y) in a.iter().zip(b) {
            dot += x as f64 * y as f64;
            na += x as f64 * x as f64;
            nb += y as f64 * y as f64;
            max_abs = max_abs.max((x - y).abs());
            max_ref = max_ref.max(y.abs());
        }
        let cosine = dot / (na.sqrt() * nb.sqrt());
        println!("{label}: token {t:3} cosine {cosine:.6}");
        min_cosine = min_cosine.min(cosine as f32);
        sum_cosine += cosine;
    }
    let mean_cosine = (sum_cosine / oracle.n_tokens as f64) as f32;
    println!(
        "{label}: min cosine {min_cosine:.6}, mean cosine {mean_cosine:.6}, max abs diff \
         {max_abs:.5}, max |ref| {max_ref:.5}, rel {:.5}",
        max_abs / max_ref
    );
    Parity {
        min_cosine,
        mean_cosine,
        max_abs,
        max_ref,
    }
}

fn assert_parity(label: &str, p: &Parity, b: &Bounds) {
    assert!(
        p.min_cosine > b.min_cosine,
        "{label}: min cosine {} below {}",
        p.min_cosine,
        b.min_cosine
    );
    assert!(
        p.mean_cosine > b.mean_cosine,
        "{label}: mean cosine {} below {}",
        p.mean_cosine,
        b.mean_cosine
    );
    assert!(
        p.max_abs <= b.max_rel_abs * p.max_ref,
        "{label}: max abs diff {} above {} * {}",
        p.max_abs,
        b.max_rel_abs,
        p.max_ref
    );
}

/// Encode every fixture with `encode` and check both oracles.
/// `f32_bounds` applies to the F32-weight oracle; the Q8_0 oracle always
/// takes [`Q8_BOUNDS`].
fn check_all_fixtures(
    cfg: &Qwen3VlVisionConfig,
    f32_bounds: &Bounds,
    mut encode: impl FnMut(&PreprocessedImage) -> Vec<f32>,
) {
    for name in FIXTURES {
        let src = load_fixture_png(name);
        let pre = preprocess(&src, cfg).unwrap();
        let ours = encode(&pre);
        for (suffix, bounds) in [("f32.embd", f32_bounds), ("embd", &Q8_BOUNDS)] {
            let oracle = read_embd(name, suffix);
            assert_eq!(
                ours.len(),
                oracle.n_tokens * oracle.n_embd,
                "{name}: output size"
            );
            let label = format!("{name} vs {suffix}");
            let parity = compare(&label, &ours, &oracle);
            assert_parity(&label, &parity, bounds);
        }
    }
}

#[test]
fn embeddings_match_oracle_cpu() {
    let Some(path) = require_mmproj() else {
        return;
    };
    let cfg = bonsai2_config();
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    let model = load_qwen3vl_vision_gguf::<CpuRuntime, _>(&path, &device).unwrap();
    assert_eq!(model.config(), &cfg);
    check_all_fixtures(&cfg, &F32_BOUNDS, |pre| {
        let out = model.encode_image(&client, pre).unwrap();
        assert_eq!(out.shape(), &[pre.n_tokens(), cfg.projection_dim]);
        out.to_vec()
    });
}

#[cfg(feature = "cuda")]
#[test]
fn embeddings_match_oracle_cuda() {
    use numr::runtime::Runtime;
    use numr::runtime::cuda::{CudaDevice, CudaRuntime};
    if !numr::runtime::cuda::is_cuda_available() {
        println!("skip: no CUDA device");
        return;
    }
    let Some(path) = require_mmproj() else {
        return;
    };
    let cfg = bonsai2_config();
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let model = load_qwen3vl_vision_gguf::<CudaRuntime, _>(&path, &device).unwrap();
    check_all_fixtures(&cfg, &Q8_BOUNDS, |pre| {
        let out = model.encode_image(&client, pre).unwrap();
        assert_eq!(out.shape(), &[pre.n_tokens(), cfg.projection_dim]);
        out.to_vec()
    });
}
