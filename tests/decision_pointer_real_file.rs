//! `PointerHead` through `run_decision` on Kev: a Qwen3.5 base with Kev's
//! LoRA adapter attached unmerged and Kev's pointer head from `head.pt`,
//! against fixtures Kev's own code wrote.
//!
//! # Fixtures
//!
//! - `BOOSTR_QWEN35_HF_DIR`: the base checkpoint's `config.json` and its
//!   SafeTensors files, single or sharded.
//! - `BOOSTR_KEV_DIR`: the Kev checkpoint: `adapter_config.json`,
//!   `adapter_model.safetensors` and `head.pt`.
//! - `BOOSTR_DECISION_FIXTURE_DIR`: the `*.json` fixtures of that
//!   checkpoint, written by `ref_kev.py` (schema
//!   `boostr.decision.pointer.v1`, documented at the top of that script):
//!   each question's row (the state, then the question's branch), its
//!   decide and option-closing positions, the raw pointer logits, the
//!   fitted temperature and the final probabilities.
//!
//! Unset or missing, each test prints one `skip:` line and passes.
//!
//! ```bash
//! BOOSTR_QWEN35_HF_DIR=/path/to/base BOOSTR_KEV_DIR=/path/to/kev \
//!   BOOSTR_DECISION_FIXTURE_DIR=/path/to/fixtures \
//!   cargo nextest run -p boostr --test decision_pointer_real_file --no-capture
//! ```
//!
//! Each fixture becomes one plan through `plan_from_rows`: one row per
//! question, read at its decide token and at each option's closing token.
//! With two or more questions the shared prefix covers the state. Each
//! question becomes one `PointerItem`, answered at
//! `TemperaturePolicy::Fixed(head.temperature())`.
//!
//! # Tolerances
//!
//! - CPU, every weight cast to F32, against the F32 reference with the
//!   adapter unmerged on both sides: raw logits within 5e-2, probabilities
//!   within 1e-2, the same argmax everywhere. Both sides round each
//!   operation to a 24-bit mantissa and differ only in summation order
//!   (matmul blocking, the GDN chunk scan): a relative drift near 2^-17 in
//!   the final hidden rows. A pointer logit `k . q / sqrt(dp)` carries the
//!   drift of both factors, so a logit of size `Z` moves near `2^-16 Z`:
//!   the band holds that for any `Z` below 3000. A logit-gap drift `d`
//!   moves a probability at most `d / (4T)`; at a fitted temperature `T`
//!   of at least 1.25 the probability band holds the whole raw band.
//! - CUDA, the base and the adapter in BF16, the head in F32: raw logits
//!   within 1.0, probabilities within 0.15. Every weight and activation of
//!   the backbone rounds to an 8-bit mantissa (relative error up to 2^-9 per
//!   rounding) through 32 layers, and both pointer factors carry that
//!   drift. The probability band holds the whole raw band at a fitted
//!   temperature of at least 1.7. The argmax must agree where the
//!   reference's top two logits are further apart than twice the raw band.
//!
//! The reference's logits with the adapter merged into the base are
//! printed against the run's for scale; they are not the compared values.
//!
//! # Memory
//!
//! The CPU test casts every float tensor to F32: peak RSS near four bytes
//! per parameter of the text model, plus the adapter. The CUDA test keeps
//! two bytes per parameter on the device.

mod common;

use boostr::format::PeftAdapter;
use boostr::inference::decision::{
    Answer, AnswerOptions, DecisionResult, RunnerConfig, TemperaturePolicy, run_decision,
    tempered_softmax,
};
use boostr::model::decision::PointerHead;
use boostr::model::{ModelClient, Qwen35Model};
use common::decision_model::{build_model, close_f64, config, top};
use common::decision_pointer_fixture::{Fixture, inputs, request};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};

/// How close the run must come to the reference.
#[derive(Clone, Copy)]
struct Tolerance {
    raw: f32,
    prob: f64,
    /// The argmax must agree everywhere; else only where the reference's
    /// top two logits are further apart than twice `raw`.
    argmax_always: bool,
}

const CPU_F32: Tolerance = Tolerance {
    raw: 5e-2,
    prob: 1e-2,
    argmax_always: true,
};

#[cfg(all(feature = "cuda", feature = "f16"))]
const CUDA_BF16: Tolerance = Tolerance {
    raw: 1.0,
    prob: 0.15,
    argmax_always: false,
};

/// Every raw logit, probability and answer of `out` against `fx`.
fn compare(fx: &Fixture, out: &DecisionResult, tol: Tolerance) {
    let what = fx.label();
    assert_eq!(out.raw_logits.len(), fx.questions.len(), "{what}: rows");
    assert_eq!(out.answers.len(), fx.questions.len(), "{what}: answers");
    let (mut raw_worst, mut merged_worst, mut prob_worst) = (0.0f32, 0.0f32, 0.0f64);
    let (mut argmax_checked, mut argmax_total) = (0usize, 0usize);
    for (q, question) in fx.questions.iter().enumerate() {
        let here = format!("{what} question {}", question.id);
        assert_eq!(out.temperatures[q], fx.temperature, "{here}: temperature");
        let got = &out.raw_logits[q];
        assert_eq!(got.len(), question.raw_logits.len(), "{here}: logit count");
        for (k, (g, w)) in got.iter().zip(&question.raw_logits).enumerate() {
            let d = (g - w).abs();
            assert!(
                d <= tol.raw,
                "{here} option {k}: raw logit {g} vs reference {w}"
            );
            raw_worst = raw_worst.max(d);
        }
        for (g, w) in got.iter().zip(&question.raw_logits_merged) {
            merged_worst = merged_worst.max((g - w).abs());
        }
        let (g, _) = top(got.as_slice());
        let (w, gap) = top(question.raw_logits.as_slice());
        argmax_total += 1;
        if tol.argmax_always || gap > 2.0 * f64::from(tol.raw) {
            argmax_checked += 1;
            assert_eq!(
                g, w,
                "{here}: argmax {g} vs reference {w} (reference gap {gap})"
            );
        }
        let probs = tempered_softmax(got, fx.temperature)
            .unwrap_or_else(|e| panic!("{here}: tempered_softmax: {e}"));
        prob_worst = prob_worst.max(close_f64(&probs, &question.probs, tol.prob, &here));

        match &out.answers[q] {
            Answer::Choice { index, probs, .. } => {
                assert_eq!(question.kind, "choice", "{here}: kind");
                close_f64(probs, &question.probs, tol.prob, &here);
                if tol.argmax_always || gap > 2.0 * f64::from(tol.raw) {
                    assert_eq!(*index, w, "{here}: choice {index} vs reference {w}");
                }
            }
            Answer::Noul { p_yes, .. } => {
                assert_eq!(question.kind, "noul", "{here}: kind");
                let want = question.p_yes.unwrap_or_else(|| panic!("{here}: no p_yes"));
                close_f64(&[*p_yes], &[want], tol.prob, &here);
            }
            Answer::Score {
                expected, probs, ..
            } => {
                assert_eq!(question.kind, "score", "{here}: kind");
                close_f64(probs, &question.probs, tol.prob, &here);
                let want: f64 = question
                    .probs
                    .iter()
                    .enumerate()
                    .map(|(i, p)| i as f64 * p)
                    .sum();
                // Each level probability within `prob` moves the expected
                // level by at most `prob * sum(i)`.
                let levels = question.options as f64;
                let band = tol.prob * levels * (levels - 1.0) / 2.0;
                assert!(
                    (expected - want).abs() <= band,
                    "{here}: expected level {expected} vs reference {want}"
                );
            }
            other => panic!("{here}: unexpected answer {other:?}"),
        }
    }
    println!(
        "{what}: {} rows, state {}: max raw diff {raw_worst} (merged reference {merged_worst}), \
         max prob diff {prob_worst}, argmax compared {argmax_checked} of {argmax_total}",
        fx.rows.len(),
        fx.state_len,
    );
}

/// Every fixture through `run_decision` on `model` with `head`.
fn check_all<R>(
    client: &R::Client,
    model: &Qwen35Model<R>,
    head: &PointerHead<R>,
    fixtures: &[Fixture],
    tol: Tolerance,
) where
    R: Runtime<DType = DType>,
    R::Client: ModelClient<R>,
{
    let opts = AnswerOptions::default();
    let cfg = RunnerConfig::default();
    let temp = TemperaturePolicy::Fixed(head.temperature());
    for fx in fixtures {
        let what = fx.label();
        assert_eq!(head.temperature(), fx.temperature, "{what}: temperature");
        assert_eq!(
            head.hidden_size(),
            fx.head.hidden_size,
            "{what}: hidden size"
        );
        assert_eq!(head.dim(), fx.head.dim, "{what}: pointer dimension");
        let (plan, items, meta) = request(fx);
        let out = run_decision(
            client, model, head, &plan, None, &items, &meta, &temp, &opts, &cfg,
        )
        .unwrap_or_else(|e| panic!("{what}: run_decision: {e}"));
        compare(fx, &out, tol);
    }
}

#[test]
fn pointer_matches_reference_cpu() {
    let Some((base, kev, fixtures)) = inputs() else {
        return;
    };
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let config = config(&base);
    let adapter = PeftAdapter::<CpuRuntime>::load(&kev, &device)
        .unwrap_or_else(|e| panic!("PeftAdapter::load {}: {e}", kev.display()));
    let model = build_model::<CpuRuntime>(&base, &config, &device, DType::F32, Some(&adapter));
    let hidden = model.config().hidden_size;
    let head = PointerHead::<CpuRuntime>::from_pt(kev.join("head.pt"), hidden, &device)
        .unwrap_or_else(|e| panic!("PointerHead::from_pt: {e}"));
    check_all::<CpuRuntime>(&client, &model, &head, &fixtures, CPU_F32);
}

/// The same fixtures on the CUDA runtime with the base and the adapter in
/// BF16 and the head in F32.
#[cfg(all(feature = "cuda", feature = "f16"))]
#[test]
fn pointer_matches_reference_cuda() {
    use numr::runtime::cuda::{CudaDevice, CudaRuntime, is_cuda_available};
    if !is_cuda_available() {
        println!("skip: no CUDA device");
        return;
    }
    let Some((base, kev, fixtures)) = inputs() else {
        return;
    };
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let config = config(&base);
    let adapter = PeftAdapter::<CudaRuntime>::load(&kev, &device)
        .unwrap_or_else(|e| panic!("PeftAdapter::load {}: {e}", kev.display()));
    let model = build_model::<CudaRuntime>(&base, &config, &device, DType::BF16, Some(&adapter));
    let hidden = model.config().hidden_size;
    let head = PointerHead::<CudaRuntime>::from_pt(kev.join("head.pt"), hidden, &device)
        .unwrap_or_else(|e| panic!("PointerHead::from_pt: {e}"));
    check_all::<CudaRuntime>(&client, &model, &head, &fixtures, CUDA_BF16);
}
