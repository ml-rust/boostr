//! `LabelTokenHead` through `run_decision` on a real System One decision
//! model (a Qwen3.5 text checkpoint such as Jebadiah or the Mapika
//! decider), against fixtures the family's own reference code wrote.
//!
//! # Fixtures
//!
//! - `BOOSTR_DECISION_MODEL_DIR`: the checkpoint's `config.json` and its
//!   SafeTensors files, single or sharded.
//! - `BOOSTR_DECISION_GGUF`: a Q8_0 or Q4_K_M `.gguf` build of the same
//!   checkpoint, for the `_gguf_` tests. Set one of the two model variables;
//!   both set is an error.
//! - `BOOSTR_DECISION_FIXTURE_DIR`: the `*.json` fixtures of that model,
//!   written by `ref_label_token.py` (schema
//!   `boostr.decision.label_token.v1`, documented at the top of that
//!   script): each row's prompt token ids, the shared prefix length, every
//!   scoring row's answer slot, label ids, raw label logits, probabilities
//!   and temperature, and every question's final distribution.
//!
//! Unset or missing, each test prints one `skip:` line and passes.
//!
//! ```bash
//! BOOSTR_DECISION_MODEL_DIR=/path/to/model BOOSTR_DECISION_FIXTURE_DIR=/path/to/fixtures \
//!   cargo nextest run -p boostr --test decision_label_token_real_file --no-capture
//! BOOSTR_DECISION_GGUF=/path/to/model.gguf BOOSTR_DECISION_FIXTURE_DIR=/path/to/fixtures \
//!   cargo nextest run -p boostr --test decision_label_token_real_file --no-capture gguf_cpu
//! ```
//!
//! Each fixture becomes one `DecisionPlan`: the shared prefix, each row's
//! tokens after it, and one gather per scoring row at its answer slot. Each
//! scoring row becomes one `LabelTokenItem`.
//!
//! # Tolerances
//!
//! - CPU, every weight cast to F32, against the F32 reference: raw label
//!   logits within 5e-2, probabilities within 2e-2, the same argmax.
//! - CUDA, the checkpoint in BF16: raw label logits within 0.5,
//!   probabilities within 0.1. Every weight and activation is rounded to
//!   an 8-bit mantissa (relative error up to 2^-9 per rounding) through 32
//!   layers, and label logits are large in absolute size, so a drift of
//!   this order against the F32 reference is BF16 rounding, not a readout
//!   error. A temperature `T` scales a logit drift `d` to at most `d / (2T)`
//!   in a probability. The argmax must agree wherever the reference's top
//!   two are further apart than twice the tolerance; closer than that, BF16
//!   rounding alone can swap them.
//! - GGUF, CPU and CUDA alike: see [`gguf_tolerance`].
//!
//! # Memory
//!
//! The CPU test casts every float tensor to F32: peak RSS near four bytes
//! per parameter of the text model. The CUDA test keeps two bytes per
//! parameter on the device. The GGUF tests keep block-quantized matrices
//! quantized: memory near the file size, on the host or the device.

mod common;

use std::path::Path;

use boostr::format::hf_qwen35::is_dropped;
use boostr::format::{SafeTensorsLoader, restructure_hf_checkpoint};
use boostr::inference::decision::{
    Answer, AnswerOptions, DecisionResult, RunnerConfig, run_decision, tempered_softmax,
};
use boostr::model::decision::LabelTokenHead;
use boostr::model::{LoadedModel, ModelClient, Qwen35Model, UniversalConfig, load_config_auto};
use boostr::nn::{VarBuilder, VarMap};
use common::decision_fixture::{Fixture, gguf_inputs, inputs, request};
use common::decision_gguf::{Quant, build_model as build_gguf_model};
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};

/// Where the argmax of a run must equal the reference's.
#[derive(Clone, Copy)]
enum ArgmaxGate {
    /// Everywhere.
    Always,
    /// Where the reference's top two are further apart than twice the
    /// tolerance of the compared values.
    #[cfg_attr(not(all(feature = "cuda", feature = "f16")), allow(dead_code))]
    TwiceTolerance,
    /// Where the reference's top two probabilities differ by more than this.
    ProbGap(f64),
}

/// How close the run must come to the reference.
#[derive(Clone, Copy)]
struct Tolerance {
    raw: f32,
    prob: f64,
    argmax: ArgmaxGate,
}

impl Tolerance {
    /// Whether the argmax must agree. `gap`: the reference's top-two gap in
    /// the compared values, `band`: their tolerance, `prob_gap`: the
    /// reference's top-two probability gap.
    fn gates(&self, gap: f64, band: f64, prob_gap: f64) -> bool {
        match self.argmax {
            ArgmaxGate::Always => true,
            ArgmaxGate::TwiceTolerance => gap > 2.0 * band,
            ArgmaxGate::ProbGap(min) => prob_gap > min,
        }
    }
}

const CPU_F32: Tolerance = Tolerance {
    raw: 5e-2,
    prob: 2e-2,
    argmax: ArgmaxGate::Always,
};

#[cfg(all(feature = "cuda", feature = "f16"))]
const CUDA_BF16: Tolerance = Tolerance {
    raw: 0.5,
    prob: 0.1,
    argmax: ArgmaxGate::TwiceTolerance,
};

/// The tolerance of a GGUF build against the F32 reference.
///
/// # Noise per weight
///
/// A uniform quantizer of step `s` adds noise of RMS `s / sqrt(12)`. Take a
/// block of Gaussian weights of RMS `w` with its largest magnitude near
/// `2w`:
///
/// - Q8_0, 32 weights, `s = 2w / 127`: RMS near `2^-7.8 w`.
/// - Q6_K, 16 weights, `s = 2w / 32`: RMS near `2^-5.8 w`.
/// - Q4_K, 32 weights, asymmetric, `s = 4w / 15`: RMS near `2^-3.7 w`,
///   about 16 times the Q8_0 noise. Q4_K_M keeps `token_embd.weight` (the tied
///   head) and a share of the matrices in Q6_K, the rest in Q4_K.
///
/// CUDA rounds each quantized matmul's input to Q8_1, and the CPU rounds
/// a K-quant matmul's input to Q8_K: 8 bits per block, noise near Q8_0's.
/// A Q8_0 matmul on the CPU keeps its input F32. Norms, GDN state,
/// attention and the KV cache stay F32. The bands below assume the
/// rounded input, so the CPU and CUDA runs take the same tolerance.
///
/// # Logits
///
/// Each projection's output carries the relative noise `e` of its weight
/// and input. The residual stream sums the layers' outputs and their
/// noise alike, so the final normed hidden row keeps a relative noise near
/// `e`, and a label logit of magnitude `Z` drifts near `e * Z`. Label
/// logits of these checkpoints are of order `2^4`. Combined weight and
/// input noise gives `e` near `2^-7.3` for Q8_0 and near `2^-3.7` for
/// Q4_K_M: an expected drift near `2^-3.3` and `2^0.3` logits. The raw
/// bands, 0.5 and 5.0, hold four to five times that.
///
/// # Probabilities
///
/// To first order, a probability moves at most `d / (4T)` when the largest
/// gap between two label logits drifts by `d`, at temperature `T`. A drift
/// common to every label cancels in the softmax. The decider's decision
/// temperatures are at least 1. The bands, 0.1 and 0.3, cover a gap drift
/// of 0.4 and 1.2 logits: four times the expected Q8_0 drift, and the
/// expected Q4_K_M drift.
///
/// # Argmax
///
/// The argmax must agree where the reference's top two probabilities
/// differ by more than twice the probability band. Closer than that, two
/// probabilities each inside the band can trade places, so a swap there is
/// quantization noise, not a readout error.
fn gguf_tolerance(quant: Quant) -> Tolerance {
    match quant {
        Quant::Q8_0 => Tolerance {
            raw: 0.5,
            prob: 0.1,
            argmax: ArgmaxGate::ProbGap(0.2),
        },
        Quant::Q4KM => Tolerance {
            raw: 5.0,
            prob: 0.3,
            argmax: ArgmaxGate::ProbGap(0.6),
        },
    }
}

fn config(dir: &Path) -> UniversalConfig {
    let path = dir.join("config.json");
    load_config_auto(&path).unwrap_or_else(|e| panic!("load_config_auto {}: {e}", path.display()))
}

/// The model of `dir` on `device` through the Hugging Face load path:
/// tensors loaded to the device, the layout edits made in the file
/// dtypes, then every float tensor cast to `dtype` on the device.
fn build_model<R>(
    dir: &Path,
    config: &UniversalConfig,
    device: &R::Device,
    dtype: DType,
) -> Box<Qwen35Model<R>>
where
    R: Runtime<DType = DType>,
    R::Client: ModelClient<R>,
{
    let client = R::default_client(device);
    let mut loader = SafeTensorsLoader::open(dir)
        .unwrap_or_else(|e| panic!("SafeTensorsLoader::open {}: {e}", dir.display()));
    let mut map = VarMap::<R>::new();
    for name in loader.tensor_names() {
        if is_dropped(&name) {
            continue;
        }
        let t = loader
            .load_tensor::<R>(&name, device)
            .unwrap_or_else(|e| panic!("load {name}: {e}"));
        map.insert(name, t);
    }
    restructure_hf_checkpoint(&mut map, config, device)
        .unwrap_or_else(|e| panic!("restructure_hf_checkpoint: {e}"));
    let names: Vec<String> = map.names().map(str::to_string).collect();
    for name in names {
        let t = map
            .take_tensor(&name)
            .unwrap_or_else(|e| panic!("take {name}: {e}"));
        let t = if matches!(t.dtype(), DType::F32 | DType::F16 | DType::BF16) && t.dtype() != dtype
        {
            client
                .cast(&t, dtype)
                .unwrap_or_else(|e| panic!("cast {name} to {dtype:?}: {e}"))
        } else {
            t
        };
        map.insert(name, t);
    }
    let mut vb = VarBuilder::new(&mut map, device);
    let loaded = LoadedModel::<R>::load(config, &mut vb)
        .unwrap_or_else(|e| panic!("LoadedModel::load: {e}"));
    let model = match loaded {
        LoadedModel::Qwen35(model) => model,
        _ => panic!("LoadedModel::load returned a model other than Qwen35"),
    };
    assert_eq!(
        map.len(),
        0,
        "loader left {} tensors behind: {:?}",
        map.len(),
        map.names().take(8).collect::<Vec<_>>()
    );
    model
}

/// The first index of the largest value, and the gap to the runner-up
/// (`inf` for one value).
fn top<T: Copy + Into<f64>>(values: &[T]) -> (usize, f64) {
    let v: Vec<f64> = values.iter().map(|&x| x.into()).collect();
    let best = (0..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b });
    let second = (0..v.len())
        .filter(|&i| i != best)
        .map(|i| v[i])
        .fold(f64::NEG_INFINITY, f64::max);
    (best, v[best] - second)
}

/// The argmax of `got` equals the reference's where `tol` gates it, given
/// the values' tolerance `band` and the reference's top-two probability
/// gap `prob_gap`. Returns whether it was compared.
fn same_argmax<T: Copy + Into<f64>>(
    got: &[T],
    want: &[T],
    band: f64,
    prob_gap: f64,
    tol: Tolerance,
    what: &str,
) -> bool {
    let (g, _) = top(got);
    let (w, gap) = top(want);
    let gated = tol.gates(gap, band, prob_gap);
    if gated {
        assert_eq!(
            g, w,
            "{what}: argmax {g} vs reference {w} (reference gap {gap})"
        );
    }
    gated
}

fn close_f64(got: &[f64], want: &[f64], tol: f64, what: &str) -> f64 {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mut worst = 0.0f64;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let d = (g - w).abs();
        assert!(d <= tol, "{what} entry {i}: {g} vs reference {w}");
        worst = worst.max(d);
    }
    worst
}

/// Every raw logit, probability and argmax of `out` against `fx`.
fn compare(fx: &Fixture, out: &DecisionResult, tol: Tolerance) {
    let what = fx.label();
    assert_eq!(out.raw_logits.len(), fx.scoring_rows.len(), "{what}: rows");
    let (mut raw_worst, mut prob_worst) = (0.0f32, 0.0f64);
    let (mut argmax_checked, mut argmax_total) = (0usize, 0usize);
    for (s, sr) in fx.scoring_rows.iter().enumerate() {
        let here = format!("{what} scoring row {s}");
        assert!(
            (out.temperatures[s] - sr.temperature).abs() < 1e-12,
            "{here}: temperature {} vs reference {}",
            out.temperatures[s],
            sr.temperature
        );
        let got = &out.raw_logits[s];
        assert_eq!(got.len(), sr.raw_logits.len(), "{here}: logit count");
        for (k, (g, w)) in got.iter().zip(&sr.raw_logits).enumerate() {
            let d = (g - w).abs();
            assert!(
                d <= tol.raw,
                "{here} label {k}: raw logit {g} vs reference {w}"
            );
            raw_worst = raw_worst.max(d);
        }
        argmax_total += 1;
        argmax_checked += usize::from(same_argmax(
            got.as_slice(),
            sr.raw_logits.as_slice(),
            f64::from(tol.raw),
            top(sr.probs.as_slice()).1,
            tol,
            &here,
        ));
        let probs = tempered_softmax(got, sr.temperature)
            .unwrap_or_else(|e| panic!("{here}: tempered_softmax: {e}"));
        prob_worst = prob_worst.max(close_f64(&probs, &sr.probs, tol.prob, &here));
    }
    assert_eq!(out.answers.len(), fx.questions.len(), "{what}: answers");
    for (answer, q) in out.answers.iter().zip(&fx.questions) {
        let here = format!("{what} question {}", q.id);
        argmax_total += 1;
        match answer {
            Answer::Choice { index, probs, .. } => {
                assert_eq!(q.kind, "choice", "{here}: kind");
                prob_worst = prob_worst.max(close_f64(probs, &q.probs, tol.prob, &here));
                let (w, gap) = top(q.probs.as_slice());
                if tol.gates(gap, tol.prob, gap) {
                    argmax_checked += 1;
                    assert_eq!(*index, w, "{here}: choice {index} vs reference {w}");
                }
            }
            Answer::Noul { p_yes, .. } => {
                assert_eq!(q.kind, "noul", "{here}: kind");
                let want = q.p_yes.unwrap_or_else(|| panic!("{here}: no p_yes"));
                prob_worst = prob_worst.max(close_f64(&[*p_yes], &[want], tol.prob, &here));
                // The gap between `p_yes` and `1 - p_yes`.
                let gap = (2.0 * want - 1.0).abs();
                if tol.gates(gap, tol.prob, gap) {
                    argmax_checked += 1;
                    assert_eq!(
                        *p_yes >= 0.5,
                        want >= 0.5,
                        "{here}: p_yes {p_yes} vs {want}"
                    );
                }
            }
            Answer::Score {
                probs, fit_mass, ..
            } => {
                assert_eq!(q.kind, "score", "{here}: kind");
                prob_worst = prob_worst.max(close_f64(probs, &q.probs, tol.prob, &here));
                let prob_gap = top(q.probs.as_slice()).1;
                argmax_checked += usize::from(same_argmax(
                    probs.as_slice(),
                    q.probs.as_slice(),
                    tol.prob,
                    prob_gap,
                    tol,
                    &here,
                ));
                match (fit_mass, q.fit_mass) {
                    (Some(g), Some(w)) => {
                        let d = (g - w).abs();
                        assert!(
                            d <= tol.prob * q.options as f64,
                            "{here}: fit mass {g} vs {w}"
                        );
                    }
                    (None, None) => {}
                    (g, w) => panic!("{here}: fit mass {g:?} vs reference {w:?}"),
                }
            }
            other => panic!("{here}: unexpected answer {other:?}"),
        }
    }
    println!(
        "{what}: {} rows, {} scoring rows, prefix {}: max raw diff {raw_worst}, max prob diff {prob_worst}, argmax compared {argmax_checked} of {argmax_total}",
        fx.rows.len(),
        fx.scoring_rows.len(),
        fx.shared_prefix_len
    );
}

/// Every fixture through `run_decision` on `model`.
fn check_all<R>(client: &R::Client, model: &Qwen35Model<R>, fixtures: &[Fixture], tol: Tolerance)
where
    R: Runtime<DType = DType>,
    R::Client: ModelClient<R>,
{
    let opts = AnswerOptions::default();
    let cfg = RunnerConfig::default();
    for fx in fixtures {
        let (plan, items, meta, temp) = request(fx);
        let out = run_decision(
            client,
            model,
            &LabelTokenHead,
            &plan,
            None,
            &items,
            &meta,
            &temp,
            &opts,
            &cfg,
        )
        .unwrap_or_else(|e| panic!("{}: run_decision: {e}", fx.label()));
        compare(fx, &out, tol);
    }
}

#[test]
fn label_token_matches_reference_cpu() {
    let Some((dir, fixtures)) = inputs() else {
        return;
    };
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let config = config(&dir);
    let model = build_model::<CpuRuntime>(&dir, &config, &device, DType::F32);
    check_all::<CpuRuntime>(&client, &model, &fixtures, CPU_F32);
}

/// The same fixtures on the CUDA runtime with the checkpoint in BF16.
#[cfg(all(feature = "cuda", feature = "f16"))]
#[test]
fn label_token_matches_reference_cuda() {
    use numr::runtime::cuda::{CudaDevice, CudaRuntime, is_cuda_available};
    if !is_cuda_available() {
        println!("skip: no CUDA device");
        return;
    }
    let Some((dir, fixtures)) = inputs() else {
        return;
    };
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let config = config(&dir);
    let model = build_model::<CudaRuntime>(&dir, &config, &device, DType::BF16);
    check_all::<CudaRuntime>(&client, &model, &fixtures, CUDA_BF16);
}

/// The same fixtures through the GGUF of `BOOSTR_DECISION_GGUF` on the
/// CPU: block-quantized matrices stay quantized, activations F32.
#[test]
fn label_token_matches_reference_gguf_cpu() {
    let Some((file, fixtures)) = gguf_inputs() else {
        return;
    };
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let (model, quant) = build_gguf_model::<CpuRuntime>(&file, &device);
    check_all::<CpuRuntime>(&client, &model, &fixtures, gguf_tolerance(quant));
}

/// The GGUF fixtures on the CUDA runtime, the quantized matrices on the
/// device.
#[cfg(feature = "cuda")]
#[test]
fn label_token_matches_reference_gguf_cuda() {
    use numr::runtime::cuda::{CudaDevice, CudaRuntime, is_cuda_available};
    if !is_cuda_available() {
        println!("skip: no CUDA device");
        return;
    }
    let Some((file, fixtures)) = gguf_inputs() else {
        return;
    };
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let (model, quant) = build_gguf_model::<CudaRuntime>(&file, &device);
    check_all::<CudaRuntime>(&client, &model, &fixtures, gguf_tolerance(quant));
}
