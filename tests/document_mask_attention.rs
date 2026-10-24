//! Document-masked attention on the CPU backend.
//!
//! Two entries pack several documents into one batch:
//!
//! - GQA: `attention_core_flash` with `PackedSegments`, served by varlen attention.
//! - MLA: `segmented_scaled_dot_product_attention_impl` and `Mla::forward_with_segments`,
//!   served by a block-diagonal additive bias.
//!
//! Each packed run must match running every document alone.
//!
//! Tolerance: the two sides differ only in float rounding. The GQA sides use
//! different kernels (varlen against flash). The MLA sides reduce over
//! different key counts. RoPE rotates a shifted document by other absolute
//! angles. Each gap is about 1e-6 at these sizes. `ATOL = 1e-5` and
//! `RTOL = 1e-4` leave a 10x margin. A mask error moves outputs far past it.
//!
//! The leakage tests use no tolerance. A blocked key gets weight exactly 0,
//! so changing another document cannot change a single bit.

use boostr::model::{AttentionCoreSpec, AttentionKernel, attention_core_flash};
use boostr::nn::{Linear, MaybeQuantLinear, Mla, MlaConfig, MlaWeights, RoPE};
use boostr::ops::PackedSegments;
use boostr::ops::impl_generic::attention::{
    scaled_dot_product_attention_impl, segmented_scaled_dot_product_attention_impl,
};
use numr::autograd::{GradStore, Var, backward, var_mul, var_sum};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

const ATOL: f32 = 1e-5;
const RTOL: f32 = 1e-4;

/// Rows per batch and tokens per row.
const B: usize = 2;
const S: usize = 8;

/// Documents as `(row, first token, length)`. Each row is fully covered.
const DOCS: [(usize, usize, usize); 4] = [(0, 0, 3), (0, 3, 5), (1, 0, 2), (1, 2, 6)];
/// The same documents as segment lengths over the flattened `B * S` tokens.
const LENGTHS: [usize; 4] = [3, 5, 2, 6];
/// The same documents as `[B, S]` ids. Ids are labels, so their order is free.
const IDS: [i32; B * S] = [0, 0, 0, 1, 1, 1, 1, 1, 7, 7, 3, 3, 3, 3, 3, 3];

fn setup() -> (CpuClient, CpuDevice) {
    let device = CpuDevice::new();
    (CpuClient::new(device.clone()), device)
}

fn det(n: usize, phase: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.37 + phase).sin() * 0.5)
        .collect()
}

fn leaf(data: &[f32], shape: &[usize], dev: &CpuDevice) -> TestResult<Var<CpuRuntime>> {
    Ok(Var::new(
        Tensor::<CpuRuntime>::from_slice(data, shape, dev)?,
        true,
    ))
}

fn grad(grads: &GradStore<CpuRuntime>, v: &Var<CpuRuntime>, name: &str) -> TestResult<Vec<f32>> {
    let g = grads
        .get(v.tensor().id())
        .ok_or_else(|| format!("no gradient for {name}"))?;
    Ok(g.contiguous()?.to_vec::<f32>())
}

fn values(v: &Var<CpuRuntime>) -> TestResult<Vec<f32>> {
    Ok(v.tensor().contiguous()?.to_vec::<f32>())
}

fn assert_close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let tol = ATOL + RTOL * w.abs();
        assert!(
            (g - w).abs() <= tol,
            "{what}[{i}]: got {g}, want {w}, diff {} > tol {tol}",
            (g - w).abs()
        );
    }
}

fn assert_nonzero(values: &[f32], what: &str) {
    assert!(
        values.iter().any(|&x| x != 0.0) && values.iter().all(|x| x.is_finite()),
        "{what}: gradient must be finite and non-zero"
    );
}

/// Forward output plus the input gradients of `sum(out * w)`.
struct Run {
    out: Vec<f32>,
    dq: Vec<f32>,
    dk: Vec<f32>,
    dv: Vec<f32>,
}

/// Inputs for one attention call.
struct Inputs {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    w: Vec<f32>,
}

// ---------------------------------------------------------------- GQA ----

const H: usize = 4;
const H_KV: usize = 2;
const D: usize = 16;

fn gqa_inputs() -> Inputs {
    let (nq, nkv) = (B * S * H * D, B * S * H_KV * D);
    Inputs {
        q: det(nq, 0.0),
        k: det(nkv, 1.1),
        v: det(nkv, 2.3),
        w: det(nq, 4.7),
    }
}

fn gqa_spec(sliding_window: usize) -> AttentionCoreSpec<'static, CpuRuntime> {
    AttentionCoreSpec {
        num_heads: H,
        num_kv_heads: H_KV,
        head_dim: D,
        q_norm: None,
        k_norm: None,
        use_alibi: false,
        skip_rope: false,
        sliding_window,
        softcap: None,
        alibi_scale: 1.0,
        kernel: AttentionKernel::Flash,
    }
}

/// `attention_core_flash` over `[batch, seq, H * D]` inputs, forward and backward.
fn run_gqa(
    x: &Inputs,
    batch: usize,
    seq: usize,
    lengths: Option<&[usize]>,
    sliding_window: usize,
) -> TestResult<Run> {
    let (client, dev) = setup();
    let q = leaf(&x.q, &[batch, seq, H * D], &dev)?;
    let k = leaf(&x.k, &[batch, seq, H_KV * D], &dev)?;
    let v = leaf(&x.v, &[batch, seq, H_KV * D], &dev)?;
    let w = Var::new(
        Tensor::<CpuRuntime>::from_slice(&x.w, &[batch, seq, H * D], &dev)?,
        false,
    );
    let rope = RoPE::<CpuRuntime>::precompute_freqs(S, D, 10000.0, None, &dev)?;
    let segments = lengths
        .map(|l| PackedSegments::<CpuRuntime>::from_lengths(l, &dev))
        .transpose()?;
    let spec = gqa_spec(sliding_window);
    let out = attention_core_flash(
        &client,
        &q,
        &k,
        &v,
        Some(rope.cos_cache()),
        Some(rope.sin_cache()),
        &spec,
        segments.as_ref(),
    )?;
    let loss = var_sum(&var_mul(&out, &w, &client)?, &[0, 1, 2], false, &client)?;
    let grads = backward(&loss, &client)?;
    Ok(Run {
        out: out.tensor().contiguous()?.to_vec::<f32>(),
        dq: grad(&grads, &q, "q")?,
        dk: grad(&grads, &k, "k")?,
        dv: grad(&grads, &v, "v")?,
    })
}

/// Token rows `start..start + len` of a token-major `[tokens, width]` buffer.
fn token_rows(data: &[f32], width: usize, start: usize, len: usize) -> Vec<f32> {
    data[start * width..(start + len) * width].to_vec()
}

/// Every document run alone, results concatenated in token order.
fn gqa_separate(x: &Inputs, sliding_window: usize) -> TestResult<Run> {
    let mut all = Run {
        out: Vec::new(),
        dq: Vec::new(),
        dk: Vec::new(),
        dv: Vec::new(),
    };
    for (row, first, len) in DOCS {
        let start = row * S + first;
        let doc = Inputs {
            q: token_rows(&x.q, H * D, start, len),
            k: token_rows(&x.k, H_KV * D, start, len),
            v: token_rows(&x.v, H_KV * D, start, len),
            w: token_rows(&x.w, H * D, start, len),
        };
        let run = run_gqa(&doc, 1, len, None, sliding_window)?;
        all.out.extend(run.out);
        all.dq.extend(run.dq);
        all.dk.extend(run.dk);
        all.dv.extend(run.dv);
    }
    Ok(all)
}

#[test]
fn gqa_packed_documents_match_separate_runs() -> TestResult {
    let x = gqa_inputs();
    let packed = run_gqa(&x, B, S, Some(&LENGTHS[..]), 0)?;
    let separate = gqa_separate(&x, 0)?;
    assert_close(&packed.out, &separate.out, "gqa out");
    assert_close(&packed.dq, &separate.dq, "gqa dQ");
    assert_close(&packed.dk, &separate.dk, "gqa dK");
    assert_close(&packed.dv, &separate.dv, "gqa dV");
    assert_nonzero(&packed.dq, "gqa dQ");
    assert_nonzero(&packed.dk, "gqa dK");
    assert_nonzero(&packed.dv, "gqa dV");
    Ok(())
}

#[test]
fn gqa_perturbing_one_document_leaves_others_bit_identical() -> TestResult {
    let x = gqa_inputs();
    let base = run_gqa(&x, B, S, Some(&LENGTHS[..]), 0)?;
    // Document A: row 0, tokens 0..3.
    let doc_a = 3;
    let shift = |data: &[f32], width: usize| {
        let mut out = data.to_vec();
        out[..doc_a * width].iter_mut().for_each(|x| *x += 0.5);
        out
    };
    let moved = Inputs {
        q: shift(&x.q, H * D),
        k: shift(&x.k, H_KV * D),
        v: shift(&x.v, H_KV * D),
        w: x.w.clone(),
    };
    let perturbed = run_gqa(&moved, B, S, Some(&LENGTHS[..]), 0)?;
    let split = doc_a * H * D;
    assert_eq!(
        &base.out[split..],
        &perturbed.out[split..],
        "other documents changed"
    );
    assert_ne!(
        &base.out[..split],
        &perturbed.out[..split],
        "document A did not change"
    );
    Ok(())
}

#[test]
fn gqa_one_document_per_row_matches_unsegmented() -> TestResult {
    let x = gqa_inputs();
    let segmented = run_gqa(&x, B, S, Some(&[S, S][..]), 0)?;
    let plain = run_gqa(&x, B, S, None, 0)?;
    assert_close(&segmented.out, &plain.out, "single-document out");
    assert_close(&segmented.dq, &plain.dq, "single-document dQ");
    assert_close(&segmented.dk, &plain.dk, "single-document dK");
    assert_close(&segmented.dv, &plain.dv, "single-document dV");
    Ok(())
}

#[test]
fn gqa_windowed_packed_documents_match_separate_runs() -> TestResult {
    let x = gqa_inputs();
    // 2 binds in every document, 4 only in the two longest, 1 keeps the token itself.
    for window in [1, 2, 4] {
        let packed = run_gqa(&x, B, S, Some(&LENGTHS[..]), window)?;
        let separate = gqa_separate(&x, window)?;
        let what = format!("gqa window {window}");
        assert_close(&packed.out, &separate.out, &format!("{what} out"));
        assert_close(&packed.dq, &separate.dq, &format!("{what} dQ"));
        assert_close(&packed.dk, &separate.dk, &format!("{what} dK"));
        assert_close(&packed.dv, &separate.dv, &format!("{what} dV"));
        // A one-key softmax is constant, so dQ and dK are exactly zero at window 1.
        if window > 1 {
            assert_nonzero(&packed.dq, &format!("{what} dQ"));
            assert_nonzero(&packed.dk, &format!("{what} dK"));
        }
        assert_nonzero(&packed.dv, &format!("{what} dV"));
    }
    Ok(())
}

#[test]
fn gqa_window_changes_the_output() -> TestResult {
    let x = gqa_inputs();
    let open = run_gqa(&x, B, S, Some(&LENGTHS[..]), 0)?;
    let windowed = run_gqa(&x, B, S, Some(&LENGTHS[..]), 2)?;
    assert_ne!(open.out, windowed.out, "window 2 must bind");
    Ok(())
}

#[test]
fn gqa_window_wider_than_every_document_is_bit_identical_to_none() -> TestResult {
    let x = gqa_inputs();
    let open = run_gqa(&x, B, S, Some(&LENGTHS[..]), 0)?;
    let wide = run_gqa(&x, B, S, Some(&LENGTHS[..]), 64)?;
    assert_eq!(open.out, wide.out, "out");
    assert_eq!(open.dq, wide.dq, "dQ");
    assert_eq!(open.dk, wide.dk, "dK");
    assert_eq!(open.dv, wide.dv, "dV");
    Ok(())
}

#[test]
fn gqa_windowed_perturbing_one_document_leaves_others_bit_identical() -> TestResult {
    let x = gqa_inputs();
    let base = run_gqa(&x, B, S, Some(&LENGTHS[..]), 2)?;
    let doc_a = 3;
    let shift = |data: &[f32], width: usize| {
        let mut out = data.to_vec();
        out[..doc_a * width].iter_mut().for_each(|x| *x += 0.5);
        out
    };
    let moved = Inputs {
        q: shift(&x.q, H * D),
        k: shift(&x.k, H_KV * D),
        v: shift(&x.v, H_KV * D),
        w: x.w.clone(),
    };
    let perturbed = run_gqa(&moved, B, S, Some(&LENGTHS[..]), 2)?;
    let split = doc_a * H * D;
    assert_eq!(&base.out[split..], &perturbed.out[split..]);
    Ok(())
}

// ---------------------------------------------------------------- MLA ----

const MLA_H: usize = 2;
const DK: usize = 12;
const DV: usize = 8;

fn mla_inputs() -> Inputs {
    let (nk, nv) = (B * MLA_H * S * DK, B * MLA_H * S * DV);
    Inputs {
        q: det(nk, 0.4),
        k: det(nk, 1.7),
        v: det(nv, 2.9),
        w: det(nv, 5.3),
    }
}

/// SDPA over `[batch, H, seq, D]` inputs, forward and backward.
fn run_mla(x: &Inputs, batch: usize, seq: usize, ids: Option<&[i32]>) -> TestResult<Run> {
    let (client, dev) = setup();
    let q = leaf(&x.q, &[batch, MLA_H, seq, DK], &dev)?;
    let k = leaf(&x.k, &[batch, MLA_H, seq, DK], &dev)?;
    let v = leaf(&x.v, &[batch, MLA_H, seq, DV], &dev)?;
    let w = Var::new(
        Tensor::<CpuRuntime>::from_slice(&x.w, &[batch, MLA_H, seq, DV], &dev)?,
        false,
    );
    let ids = ids
        .map(|ids| Tensor::<CpuRuntime>::from_slice(ids, &[batch, seq], &dev))
        .transpose()?;
    let scale = 1.0 / (DK as f64).sqrt();
    let out = segmented_scaled_dot_product_attention_impl(
        &client,
        &q,
        &k,
        &v,
        scale,
        true,
        ids.as_ref(),
    )?;
    let loss = var_sum(&var_mul(&out, &w, &client)?, &[0, 1, 2, 3], false, &client)?;
    let grads = backward(&loss, &client)?;
    Ok(Run {
        out: out.tensor().contiguous()?.to_vec::<f32>(),
        dq: grad(&grads, &q, "q")?,
        dk: grad(&grads, &k, "k")?,
        dv: grad(&grads, &v, "v")?,
    })
}

/// Copy one document out of `[B, H, S, d]` into `[1, H, len, d]`.
fn take_doc(data: &[f32], d: usize, (row, first, len): (usize, usize, usize)) -> Vec<f32> {
    let mut out = Vec::with_capacity(MLA_H * len * d);
    for h in 0..MLA_H {
        let base = ((row * MLA_H + h) * S + first) * d;
        out.extend_from_slice(&data[base..base + len * d]);
    }
    out
}

/// Write one `[1, H, len, d]` document back into `[B, H, S, d]`.
fn put_doc(dst: &mut [f32], src: &[f32], d: usize, (row, first, len): (usize, usize, usize)) {
    for h in 0..MLA_H {
        let base = ((row * MLA_H + h) * S + first) * d;
        dst[base..base + len * d].copy_from_slice(&src[h * len * d..(h + 1) * len * d]);
    }
}

#[test]
fn mla_packed_documents_match_separate_runs() -> TestResult {
    let x = mla_inputs();
    let packed = run_mla(&x, B, S, Some(&IDS[..]))?;
    let mut separate = Run {
        out: vec![0.0; x.v.len()],
        dq: vec![0.0; x.q.len()],
        dk: vec![0.0; x.k.len()],
        dv: vec![0.0; x.v.len()],
    };
    for doc in DOCS {
        let part = Inputs {
            q: take_doc(&x.q, DK, doc),
            k: take_doc(&x.k, DK, doc),
            v: take_doc(&x.v, DV, doc),
            w: take_doc(&x.w, DV, doc),
        };
        let run = run_mla(&part, 1, doc.2, None)?;
        put_doc(&mut separate.out, &run.out, DV, doc);
        put_doc(&mut separate.dq, &run.dq, DK, doc);
        put_doc(&mut separate.dk, &run.dk, DK, doc);
        put_doc(&mut separate.dv, &run.dv, DV, doc);
    }
    assert_close(&packed.out, &separate.out, "mla out");
    assert_close(&packed.dq, &separate.dq, "mla dQ");
    assert_close(&packed.dk, &separate.dk, "mla dK");
    assert_close(&packed.dv, &separate.dv, "mla dV");
    assert_nonzero(&packed.dq, "mla dQ");
    assert_nonzero(&packed.dk, "mla dK");
    assert_nonzero(&packed.dv, "mla dV");
    Ok(())
}

#[test]
fn mla_perturbing_one_document_leaves_others_bit_identical() -> TestResult {
    let x = mla_inputs();
    let base = run_mla(&x, B, S, Some(&IDS[..]))?;
    let doc_a = DOCS[0];
    let shift = |data: &[f32], d: usize| {
        let mut out = data.to_vec();
        let moved: Vec<f32> = take_doc(data, d, doc_a).iter().map(|x| x + 0.5).collect();
        put_doc(&mut out, &moved, d, doc_a);
        out
    };
    let moved = Inputs {
        q: shift(&x.q, DK),
        k: shift(&x.k, DK),
        v: shift(&x.v, DV),
        w: x.w.clone(),
    };
    let perturbed = run_mla(&moved, B, S, Some(&IDS[..]))?;
    for &doc in &DOCS[1..] {
        assert_eq!(
            take_doc(&base.out, DV, doc),
            take_doc(&perturbed.out, DV, doc),
            "document {doc:?} changed"
        );
    }
    assert_ne!(
        take_doc(&base.out, DV, doc_a),
        take_doc(&perturbed.out, DV, doc_a),
        "document A did not change"
    );
    Ok(())
}

#[test]
fn mla_single_document_is_bit_identical_to_causal() -> TestResult {
    let x = mla_inputs();
    let segmented = run_mla(&x, B, S, Some(&[4i32; B * S][..]))?;
    let plain = run_mla(&x, B, S, None)?;
    assert_eq!(segmented.out, plain.out, "single-document out");
    assert_eq!(segmented.dq, plain.dq, "single-document dQ");
    assert_eq!(segmented.dk, plain.dk, "single-document dK");
    assert_eq!(segmented.dv, plain.dv, "single-document dV");

    // `None` ids run the unsegmented function itself.
    let (client, dev) = setup();
    let q = leaf(&x.q, &[B, MLA_H, S, DK], &dev)?;
    let k = leaf(&x.k, &[B, MLA_H, S, DK], &dev)?;
    let v = leaf(&x.v, &[B, MLA_H, S, DV], &dev)?;
    let direct = scaled_dot_product_attention_impl(&client, &q, &k, &v, 0.3, true)?;
    let via_none =
        segmented_scaled_dot_product_attention_impl(&client, &q, &k, &v, 0.3, true, None)?;
    assert_eq!(values(&direct)?, values(&via_none)?);
    Ok(())
}

// ---------------------------------------------------------- MLA layer ----

const HIDDEN: usize = 16;

fn mla_config() -> MlaConfig {
    MlaConfig {
        hidden_size: HIDDEN,
        num_heads: 2,
        head_dim: 8,
        head_dim_v: 6,
        kv_lora_rank: 8,
        q_lora_rank: 0,
        rope_head_dim: 4,
        max_seq_len: S,
        rope_theta: 10000.0,
        use_norm: false,
        norm_eps: 1e-6,
    }
}

fn linear(
    out: usize,
    inp: usize,
    phase: f32,
    dev: &CpuDevice,
) -> TestResult<MaybeQuantLinear<CpuRuntime>> {
    let weight = Tensor::<CpuRuntime>::from_slice(&det(out * inp, phase), &[out, inp], dev)?;
    Ok(MaybeQuantLinear::Standard(Linear::new(weight, None, false)))
}

fn mla_layer(dev: &CpuDevice) -> TestResult<Mla<CpuRuntime>> {
    let c = mla_config();
    let qk = c.head_dim + c.rope_head_dim;
    let weights = MlaWeights {
        q_down: None,
        q_up: linear(c.num_heads * qk, HIDDEN, 0.2, dev)?,
        q_norm: None,
        kv_compress: linear(c.kv_lora_rank + c.rope_head_dim, HIDDEN, 1.3, dev)?,
        kv_norm: None,
        kv_decompress: linear(
            c.num_heads * (c.head_dim + c.head_dim_v),
            c.kv_lora_rank,
            2.1,
            dev,
        )?,
        o_proj: linear(HIDDEN, c.num_heads * c.head_dim_v, 3.4, dev)?,
    };
    Ok(Mla::with_ids(&c, weights, dev)?)
}

#[test]
fn mla_layer_packed_documents_match_separate_runs() -> TestResult {
    let (client, dev) = setup();
    let mla = mla_layer(&dev)?;
    let hidden_data = det(B * S * HIDDEN, 0.9);
    let hidden = Var::new(
        Tensor::<CpuRuntime>::from_slice(&hidden_data, &[B, S, HIDDEN], &dev)?,
        false,
    );

    let plain = values(&mla.forward(&client, &hidden)?)?;
    let via_none = values(&mla.forward_with_segments(&client, &hidden, None)?)?;
    assert_eq!(plain, via_none, "None ids must run forward unchanged");

    let ids = Tensor::<CpuRuntime>::from_slice(&IDS, &[B, S], &dev)?;
    let packed = values(&mla.forward_with_segments(&client, &hidden, Some(&ids))?)?;
    let mut separate = Vec::with_capacity(packed.len());
    for (row, first, len) in DOCS {
        let doc = token_rows(&hidden_data, HIDDEN, row * S + first, len);
        let doc = Var::new(
            Tensor::<CpuRuntime>::from_slice(&doc, &[1, len, HIDDEN], &dev)?,
            false,
        );
        separate.extend(values(&mla.forward(&client, &doc)?)?);
    }
    assert_close(&packed, &separate, "mla layer out");
    Ok(())
}
