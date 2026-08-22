//! Which RoPE convention do HuggingFace checkpoints actually want?
//!
//! This isolates the rotary op from everything else: it takes HF's OWN Q/K
//! immediately before `apply_rotary_pos_emb` and HF's own output immediately
//! after, and asks which of `boostr`'s two implementations reproduces it. No
//! model, no attention, no loader — so a failure here cannot be blamed on
//! anything downstream.
//!
//! It exists because the decoder was calling `apply_rope_interleaved` while the
//! trait's own doc says split-half is what Llama/Mistral use, and flipping the
//! call site on that reasoning alone made end-to-end logits WORSE. Guessing
//! between two conventions is cheap to get wrong and expensive to debug through
//! a 28-layer model; this test answers it directly and permanently.
//!
//! Fixtures from `dump_qwen3.py`. Skips when absent — a skip is not a pass.

use boostr::nn::RoPE;
use boostr::ops::traits::position::rope::RoPEOps;
use numr::autograd::Var;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;
use std::path::PathBuf;

const HEADS: usize = 16;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const ROPE_THETA: f32 = 1_000_000.0;

fn read_f32(path: &PathBuf) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    (worst, at)
}

fn rms(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt()
}

fn fixtures() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var("QWEN3_REF_DIR").ok()?);
    dir.join("qwen3_l0_q_pre_rope.f32").exists().then_some(dir)
}

/// Compare `boostr`'s cos/sin cache against the one HF built for the same
/// `head_dim` and `rope_theta`. If the CACHES disagree, no choice of pairing
/// convention can match, so this has to be checked before the convention is.
///
/// HF materializes `cos`/`sin` at full `[S, head_dim]` with the first half
/// duplicated into the second; `boostr` stores the unique half as
/// `[S, head_dim/2]`. So HF's first `head_dim/2` columns are the comparison.
#[test]
fn rope_cache_matches_huggingface() {
    let Some(dir) = fixtures() else {
        eprintln!("skipping: set QWEN3_REF_DIR (run dump_qwen3.py)");
        return;
    };
    let device = CpuDevice::new();

    let hf_cos = read_f32(&dir.join("qwen3_rope_cos.f32"));
    let seq = hf_cos.len() / HEAD_DIM;
    let half = HEAD_DIM / 2;

    let rope = RoPE::<CpuRuntime>::precompute_freqs(seq, HEAD_DIM, ROPE_THETA, None, &device)
        .expect("precompute_freqs");
    let got: Vec<f32> = rope
        .cos_cache()
        .tensor()
        .contiguous()
        .expect("contiguous")
        .to_vec();
    assert_eq!(got.len(), seq * half, "boostr cos cache should be [S, D/2]");

    // HF row s is [freqs(s) ; freqs(s)] — take the first half of each row.
    let want: Vec<f32> = (0..seq)
        .flat_map(|s| hf_cos[s * HEAD_DIM..s * HEAD_DIM + half].to_vec())
        .collect();

    let (d, i) = max_abs_diff(&got, &want);
    eprintln!("rope cos cache: max|d|={d:.3e} at {i}");
    assert!(
        d < 1e-5,
        "boostr's RoPE cache disagrees with HuggingFace at head_dim={HEAD_DIM}, \
         theta={ROPE_THETA}: max|d|={d} at {i}. Fix the cache before the convention."
    );

    // HF duplicates the half into the second half of each row — confirm, so the
    // "take the first half" reading above is not an accident of this dump.
    for s in 0..seq {
        let row = &hf_cos[s * HEAD_DIM..(s + 1) * HEAD_DIM];
        let (d2, _) = max_abs_diff(&row[..half], &row[half..]);
        assert!(
            d2 < 1e-6,
            "HF cos row {s} is not two identical halves (max|d|={d2}); the cache \
             layout assumption in this test is wrong"
        );
    }
}

/// The decisive one: run BOTH conventions on HF's own pre-RoPE Q and report
/// which reproduces HF's post-RoPE Q. Exactly one should match.
#[test]
fn rope_convention_matches_huggingface() {
    let Some(dir) = fixtures() else {
        eprintln!("skipping: set QWEN3_REF_DIR (run dump_qwen3.py)");
        return;
    };
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());

    let q_pre = read_f32(&dir.join("qwen3_l0_q_pre_rope.f32"));
    let q_post = read_f32(&dir.join("qwen3_l0_q_after_rope.f32"));
    let seq = q_pre.len() / (HEADS * HEAD_DIM);
    eprintln!("q is [1, {HEADS}, {seq}, {HEAD_DIM}]");

    let rope = RoPE::<CpuRuntime>::precompute_freqs(seq, HEAD_DIM, ROPE_THETA, None, &device)
        .expect("precompute_freqs");
    let (cos, sin) = (rope.cos_cache(), rope.sin_cache());

    let q = Var::new(
        Tensor::<CpuRuntime>::from_slice(&q_pre, &[1, HEADS, seq, HEAD_DIM], &device),
        false,
    );

    let split: Vec<f32> = client
        .apply_rope(&q, cos, sin)
        .expect("apply_rope")
        .tensor()
        .contiguous()
        .expect("contiguous")
        .to_vec();
    let inter: Vec<f32> = client
        .apply_rope_interleaved(&q, cos, sin)
        .expect("apply_rope_interleaved")
        .tensor()
        .contiguous()
        .expect("contiguous")
        .to_vec();

    let scale = rms(&q_post);
    let (ds, is) = max_abs_diff(&split, &q_post);
    let (di, ii) = max_abs_diff(&inter, &q_post);
    eprintln!("reference rms={scale:.3e}");
    eprintln!("  split-half  : max|d|={ds:.3e} at {is}");
    eprintln!("  interleaved : max|d|={di:.3e} at {ii}");

    let tol = 1e-3 * scale.max(1.0);
    assert!(
        ds < tol,
        "split-half does NOT reproduce HuggingFace (max|d|={ds} at {is}, rms {scale}); \
         interleaved gave {di}. HF `transformers` uses `rotate_half`, so if neither \
         matches, the cos/sin cache or the input layout is the problem, not the pairing."
    );
    assert!(
        di > tol,
        "interleaved ALSO reproduces HuggingFace (max|d|={di}) — the two conventions \
         are supposed to differ, so this test is not discriminating and something \
         upstream (cache shape, head layout) is degenerate"
    );

    // K uses the same convention but has fewer heads under GQA — a layout bug
    // that happens to be invisible at 16 heads can still show at 8.
    let k_pre = read_f32(&dir.join("qwen3_l0_k_pre_rope.f32"));
    let k_post = read_f32(&dir.join("qwen3_l0_k_after_rope.f32"));
    let k = Var::new(
        Tensor::<CpuRuntime>::from_slice(&k_pre, &[1, KV_HEADS, seq, HEAD_DIM], &device),
        false,
    );
    let k_split: Vec<f32> = client
        .apply_rope(&k, cos, sin)
        .expect("apply_rope k")
        .tensor()
        .contiguous()
        .expect("contiguous")
        .to_vec();
    let (dk, ik) = max_abs_diff(&k_split, &k_post);
    eprintln!("  split-half K: max|d|={dk:.3e} at {ik}");
    assert!(
        dk < 1e-3 * rms(&k_post).max(1.0),
        "split-half fails on K ({KV_HEADS} KV heads) while passing on Q: max|d|={dk} at {ik}"
    );
}
