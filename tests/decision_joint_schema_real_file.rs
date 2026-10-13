//! `JointSchemaHead` on the Clef-flash release head against fixtures the
//! reference `JointSchemaHead` wrote, head only: the fixtures carry the
//! backbone's final-normed hidden rows and the `lm_head` rows of every
//! option token.
//!
//! # Fixtures
//!
//! - `BOOSTR_CLEF_HEAD_DIR`: the release directory, holding
//!   `joint_head_config.json` and `joint_head.safetensors`.
//! - `BOOSTR_CLEF_FIXTURE_DIR`: the `*.json` fixtures (schema
//!   `boostr.decision.joint.v1`) with their sibling `.npy` files: `hidden`
//!   and `lm_rows` as BF16 bits, the `choice_route` intermediates as F32.
//!
//! Unset or missing, each test prints one `skip:` line and passes.
//!
//! ```bash
//! BOOSTR_CLEF_HEAD_DIR=/path/to/clef-flash BOOSTR_CLEF_FIXTURE_DIR=/path/to/fixtures \
//!   cargo nextest run -p boostr --test decision_joint_schema_real_file --no-capture
//! ```
//!
//! # Tolerances
//!
//! - F32 head against the reference F32 head (BF16 weights upcast, the
//!   same BF16 inputs): logits within 2e-3, probabilities within 1e-3.
//!   Both sides round each op to a 24-bit mantissa and differ only in
//!   summation order and in the `exp`, `erf` and `rsqrt` kernels: a
//!   relative drift near 1e-6 per stage. The largest logit term is the
//!   prior, below 10, and the cosine term is scaled by 9.94 * 0.119, so the
//!   drift reaches the logits near 1e-4: the band holds a 20x margin.
//!   Stages: within `1e-3 + 1e-3 * |reference|`.
//! - F32 head against the BF16 release: logits within 0.05, probabilities
//!   within 0.003, the measured gap between the two reference heads
//!   (0.044 and 0.0019 on `multi`).
//! - BF16 head against the BF16 release: logits within 0.1, probabilities
//!   within 0.006. Each BF16 evaluation sits within the measured gap of
//!   the F32 head, in independent directions, so two of them sit within
//!   twice that gap.

#[path = "common/npy.rs"]
mod npy;

use boostr::inference::decision::{Answer, answer_from_probs, tempered_softmax};
use boostr::model::ModelClient;
use boostr::model::decision::joint_schema::{
    FieldKind, HeadDtype, JointOption, JointSchemaHead, JointSchemaItem, joint_answer_options,
};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;
use serde_json::Value;
use std::path::{Path, PathBuf};

const HEAD_ENV: &str = "BOOSTR_CLEF_HEAD_DIR";
const FIXTURE_ENV: &str = "BOOSTR_CLEF_FIXTURE_DIR";

const F32_LOGIT: f32 = 2e-3;
const F32_PROB: f64 = 1e-3;
const RELEASE_LOGIT: f32 = 0.05;
const RELEASE_PROB: f64 = 0.003;
#[cfg(feature = "f16")]
const BF16_LOGIT: f32 = 0.1;
#[cfg(feature = "f16")]
const BF16_PROB: f64 = 0.006;
const STAGE_ATOL: f32 = 1e-3;
const STAGE_RTOL: f32 = 1e-3;

/// The head and fixture directories, or `None` after one `skip:` line.
fn dirs() -> Option<(PathBuf, PathBuf)> {
    let get = |env: &str, file: &str| -> Option<PathBuf> {
        let Some(dir) = std::env::var(env).ok().map(PathBuf::from) else {
            println!("skip: {env} not set");
            return None;
        };
        if !dir.join(file).exists() {
            println!("skip: {} has no {file}", dir.display());
            return None;
        }
        Some(dir)
    };
    let head = get(HEAD_ENV, "joint_head.safetensors")?;
    let fixtures = get(FIXTURE_ENV, "choice_route.json")?;
    Some((head, fixtures))
}

/// One fixture: the record's fields and the head's inputs on the host.
struct Fixture {
    name: String,
    json: Value,
    dir: PathBuf,
    items: Vec<JointSchemaItem>,
    /// `[seq_len, H]` hidden rows, BF16 values in F32.
    hidden: Vec<f32>,
    /// `[ids, H]` lm_head rows of `JointSchemaHead::lexical_ids(items)`.
    lexical: Vec<f32>,
    seq_len: usize,
    hidden_size: usize,
}

fn usizes(v: &Value) -> Vec<usize> {
    v.as_array()
        .unwrap_or_else(|| panic!("expected an array, got {v}"))
        .iter()
        .map(|x| {
            x.as_u64()
                .unwrap_or_else(|| panic!("expected an integer, got {x}")) as usize
        })
        .collect()
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap_or_else(|| panic!("expected an array, got {v}"))
        .iter()
        .map(|x| {
            x.as_f64()
                .unwrap_or_else(|| panic!("expected a number, got {x}"))
        })
        .collect()
}

impl Fixture {
    fn load(path: &Path) -> Self {
        let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
        let json: Value = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{path:?}: {e}"));
        assert_eq!(json["schema"], "boostr.decision.joint.v1", "{path:?}");
        let name = json["name"].as_str().unwrap_or_default().to_string();
        let dir = path.parent().map(Path::to_path_buf).unwrap_or_default();
        let input_ids: Vec<u32> = usizes(&json["input_ids"])
            .iter()
            .map(|&t| t as u32)
            .collect();
        let seq_len = input_ids.len();

        let mut items = Vec::new();
        for q in json["questions"]
            .as_array()
            .unwrap_or_else(|| panic!("{name}: no questions"))
        {
            let kind = match q["type_id"].as_u64() {
                Some(0) => FieldKind::Noul,
                Some(1) => FieldKind::Choice,
                Some(2) => FieldKind::Score,
                other => panic!("{name}: type_id {other:?}"),
            };
            let span = usizes(&q["question_span"]);
            let spans = q["option_spans"]
                .as_array()
                .unwrap_or_else(|| panic!("{name}: spans"));
            let ids = q["option_token_ids"]
                .as_array()
                .unwrap_or_else(|| panic!("{name}: ids"));
            let options = spans
                .iter()
                .zip(ids)
                .map(|(s, i)| {
                    let s = usizes(s);
                    let ids: Vec<u32> = usizes(i).iter().map(|&t| t as u32).collect();
                    assert_eq!(
                        ids,
                        input_ids[s[0]..s[1]],
                        "{name}: option ids vs input_ids"
                    );
                    JointOption {
                        span: s[0]..s[1],
                        ids,
                    }
                })
                .collect();
            items.push(JointSchemaItem {
                kind,
                question: span[0]..span[1],
                options,
            });
        }

        let hidden = npy::read(&dir.join(json["hidden"]["file"].as_str().unwrap_or_default()));
        assert_eq!(hidden.shape[0], seq_len, "{name}: hidden rows");
        let hidden_size = hidden.shape[1];
        let rows = npy::read(&dir.join(json["lm_head"]["rows_file"].as_str().unwrap_or_default()));
        let row_ids = usizes(&json["lm_head"]["row_ids"]);
        assert_eq!(rows.shape, [row_ids.len(), hidden_size], "{name}: lm rows");
        let mut lexical = Vec::new();
        for id in JointSchemaHead::<CpuRuntime>::lexical_ids(&items) {
            let r = row_ids
                .iter()
                .position(|&x| x == id as usize)
                .unwrap_or_else(|| panic!("{name}: no lm_head row for id {id}"));
            lexical.extend_from_slice(&rows.data[r * hidden_size..(r + 1) * hidden_size]);
        }
        Self {
            name,
            json,
            dir,
            items,
            hidden: hidden.data,
            lexical,
            seq_len,
            hidden_size,
        }
    }

    fn all(dir: &Path) -> Vec<Self> {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
            .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "json"))
            .collect();
        paths.sort();
        paths.iter().map(|p| Self::load(p)).collect()
    }

    /// The head's inputs on `device`.
    fn tensors<R: Runtime<DType = DType>>(&self, device: &R::Device) -> (Tensor<R>, Tensor<R>) {
        let ids = self.lexical.len() / self.hidden_size;
        (
            Tensor::<R>::from_slice(&self.hidden, &[self.seq_len, self.hidden_size], device)
                .unwrap_or_else(|e| panic!("{}: hidden: {e}", self.name)),
            Tensor::<R>::from_slice(&self.lexical, &[ids, self.hidden_size], device)
                .unwrap_or_else(|e| panic!("{}: lexical: {e}", self.name)),
        )
    }

    fn question(&self, q: usize) -> &Value {
        &self.json["questions"][q]
    }
}

/// The head's logits for `fx`, one row per field.
fn run<R, C>(client: &C, head: &JointSchemaHead<R>, fx: &Fixture) -> Vec<Vec<f32>>
where
    R: Runtime<DType = DType>,
    C: ModelClient<R>,
{
    let (hidden, lexical) = fx.tensors::<R>(client.device());
    let out = head
        .logits(client, &hidden, &lexical, &fx.items)
        .unwrap_or_else(|e| panic!("{}: logits: {e}", fx.name));
    let flat = out.flat.to_vec::<f32>();
    let mut rows = Vec::with_capacity(out.counts.len());
    let mut first = 0;
    for &k in &out.counts {
        rows.push(flat[first..first + k].to_vec());
        first += k;
    }
    rows
}

/// Every field's logits and probabilities against the fixture's `key`
/// logits and `prob_key` probabilities; returns the worst differences.
fn compare(
    fx: &Fixture,
    got: &[Vec<f32>],
    key: &str,
    prob_key: &str,
    logit_tol: f32,
    prob_tol: f64,
) -> (f32, f64) {
    let (mut worst_logit, mut worst_prob) = (0.0f32, 0.0f64);
    for (q, row) in got.iter().enumerate() {
        let want: Vec<f32> = floats(&fx.question(q)[key])
            .iter()
            .map(|&v| v as f32)
            .collect();
        let here = format!("{} field {} ({key})", fx.name, fx.question(q)["id"]);
        assert_eq!(row.len(), want.len(), "{here}: option count");
        for (j, (g, w)) in row.iter().zip(&want).enumerate() {
            let d = (g - w).abs();
            assert!(
                d <= logit_tol,
                "{here} option {j}: logit {g} vs reference {w}"
            );
            worst_logit = worst_logit.max(d);
        }
        let probs = tempered_softmax(row, 1.0).unwrap_or_else(|e| panic!("{here}: {e}"));
        for (j, (g, w)) in probs
            .iter()
            .zip(floats(&fx.question(q)[prob_key]))
            .enumerate()
        {
            let d = (g - w).abs();
            assert!(
                d <= prob_tol,
                "{here} option {j}: probability {g} vs reference {w}"
            );
            worst_prob = worst_prob.max(d);
        }
    }
    (worst_logit, worst_prob)
}

/// Every field's answer against the fixture's `answers` map, within `tol`
/// on each rounded probability. A score `sum_i i * p_i` of `L` levels moves
/// up to `L` times the probability drift, so it gets `L * tol`.
fn compare_answers(fx: &Fixture, got: &[Vec<f32>], answers: &Value, tol: f64) {
    let opts = joint_answer_options();
    for (q, (row, item)) in got.iter().zip(&fx.items).enumerate() {
        let question = fx.question(q);
        let id = question["id"].as_str().unwrap_or_default();
        let want = &answers[id];
        let here = format!("{} field {id}", fx.name);
        let probs = tempered_softmax(row, 1.0).unwrap_or_else(|e| panic!("{here}: {e}"));
        let answer = answer_from_probs(&item.meta(), probs, &opts)
            .unwrap_or_else(|e| panic!("{here}: answer: {e}"));
        let levels = item.options.len() as f64;
        let near = |g: f64, w: &Value, what: &str| {
            let w = w
                .as_f64()
                .unwrap_or_else(|| panic!("{here}: no {what} in {want}"));
            let band = if what == "score" { levels * tol } else { tol };
            assert!((g - w).abs() <= band, "{here}: {what} {g} vs reference {w}");
        };
        let option_ids: Vec<&str> = question["option_ids"]
            .as_array()
            .unwrap_or_else(|| panic!("{here}: option_ids"))
            .iter()
            .map(|v| v.as_str().unwrap_or_default())
            .collect();
        match (&answer, item.kind) {
            (Answer::Noul { p_yes, .. }, FieldKind::Noul) => near(*p_yes, &want["noul"], "noul"),
            (Answer::Choice { index, probs, .. }, FieldKind::Choice | FieldKind::Score) => {
                assert!(item.options.len() == 1 || item.kind == FieldKind::Choice);
                if item.kind == FieldKind::Choice {
                    assert_eq!(want["choice"], option_ids[*index], "{here}: choice");
                } else {
                    near(0.0, &want["score"], "score");
                }
                near(probs[*index], &want["confidence"], "confidence");
                for (j, oid) in option_ids.iter().enumerate() {
                    near(probs[j], &want["probabilities"][*oid], "probability");
                }
            }
            (
                Answer::Score {
                    expected,
                    confidence,
                    probs,
                    ..
                },
                FieldKind::Score,
            ) => {
                near(*expected, &want["score"], "score");
                near(*confidence, &want["confidence"], "confidence");
                for (j, oid) in option_ids.iter().enumerate() {
                    near(probs[j], &want["probabilities"][*oid], "probability");
                }
            }
            (other, kind) => panic!("{here}: a {kind:?} field answered {other:?}"),
        }
    }
}

fn cpu() -> (numr::runtime::cpu::CpuClient, CpuDevice) {
    let device = CpuDevice::new();
    (CpuRuntime::default_client(&device), device)
}

/// The F32 head against the F32 reference, the release logits, and the
/// F32 reference's answers.
#[test]
fn f32_head_matches_reference() {
    let Some((head_dir, fixture_dir)) = dirs() else {
        return;
    };
    let (client, _) = cpu();
    let head = JointSchemaHead::<CpuRuntime>::from_dir(&client, &head_dir, HeadDtype::F32)
        .unwrap_or_else(|e| panic!("from_dir: {e}"));
    assert_eq!(head.dtype(), DType::F32);
    let fixtures = Fixture::all(&fixture_dir);
    assert!(
        !fixtures.is_empty(),
        "no fixtures in {}",
        fixture_dir.display()
    );
    for fx in &fixtures {
        let got = run(&client, &head, fx);
        let (l, p) = compare(fx, &got, "logits_f32", "probs_f32", F32_LOGIT, F32_PROB);
        let (rl, rp) = compare(
            fx,
            &got,
            "logits_bf16",
            "probs_bf16",
            RELEASE_LOGIT,
            RELEASE_PROB,
        );
        println!(
            "{}: vs f32 reference logit {l:.2e} prob {p:.2e}; vs release logit {rl:.2e} prob {rp:.2e}",
            fx.name
        );
        compare_answers(fx, &got, &fx.json["answers_f32"], 2e-4);
    }
}

/// Every intermediate of `choice_route` against the F32 reference, stage
/// by stage in forward order, so the first stage out of band names the
/// layer at fault.
#[test]
fn f32_stages_match_reference() {
    let Some((head_dir, fixture_dir)) = dirs() else {
        return;
    };
    let (client, device) = cpu();
    let head = JointSchemaHead::<CpuRuntime>::from_dir(&client, &head_dir, HeadDtype::F32)
        .unwrap_or_else(|e| panic!("from_dir: {e}"));
    let fx = Fixture::load(&fixture_dir.join("choice_route.json"));
    let (hidden, lexical) = fx.tensors::<CpuRuntime>(&device);
    let stages = head
        .stages(&client, &hidden, &lexical, &fx.items)
        .unwrap_or_else(|e| panic!("stages: {e}"));
    let refs = fx.json["intermediates"]
        .as_object()
        .unwrap_or_else(|| panic!("choice_route has no intermediates"));
    let mut compared = 0;
    for (name, t) in &stages {
        let Some(entry) = refs.get(name) else {
            println!("{name}: no reference");
            continue;
        };
        let want = npy::read(&fx.dir.join(entry["file"].as_str().unwrap_or_default()));
        let got = t.to_vec::<f32>();
        assert_eq!(got.len(), want.data.len(), "{name}: element count");
        let mut worst = 0.0f32;
        for (i, (g, w)) in got.iter().zip(&want.data).enumerate() {
            let d = (g - w).abs();
            assert!(
                d <= STAGE_ATOL + STAGE_RTOL * w.abs(),
                "stage {name} element {i}: {g} vs reference {w}"
            );
            worst = worst.max(d);
        }
        println!("{name}: worst {worst:.2e}");
        compared += 1;
    }
    assert_eq!(compared, refs.len(), "every reference stage has a match");
}

/// The head in the checkpoint dtype (BF16), as the release runs it.
#[cfg(feature = "f16")]
#[test]
fn bf16_head_matches_release() {
    let Some((head_dir, fixture_dir)) = dirs() else {
        return;
    };
    let (client, _) = cpu();
    let head = JointSchemaHead::<CpuRuntime>::from_dir(&client, &head_dir, HeadDtype::Checkpoint)
        .unwrap_or_else(|e| panic!("from_dir: {e}"));
    assert_eq!(head.dtype(), DType::BF16);
    for fx in &Fixture::all(&fixture_dir) {
        let got = run(&client, &head, fx);
        let (l, p) = compare(fx, &got, "logits_bf16", "probs_bf16", BF16_LOGIT, BF16_PROB);
        println!("{}: bf16 vs release logit {l:.2e} prob {p:.2e}", fx.name);
        compare_answers(fx, &got, &fx.json["answers"]["answers"], 7e-3);
    }
}

/// The F32 head on CUDA against the F32 reference, and with `f16` the
/// BF16 head against the release.
#[cfg(feature = "cuda")]
#[test]
fn cuda_heads_match_reference() {
    use numr::runtime::cuda::{CudaDevice, CudaRuntime, is_cuda_available};
    let Some((head_dir, fixture_dir)) = dirs() else {
        return;
    };
    if !is_cuda_available() {
        println!("skip: no CUDA device");
        return;
    }
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let fixtures = Fixture::all(&fixture_dir);
    let head = JointSchemaHead::<CudaRuntime>::from_dir(&client, &head_dir, HeadDtype::F32)
        .unwrap_or_else(|e| panic!("from_dir: {e}"));
    for fx in &fixtures {
        let got = run(&client, &head, fx);
        let (l, p) = compare(fx, &got, "logits_f32", "probs_f32", F32_LOGIT, F32_PROB);
        println!("{}: cuda f32 logit {l:.2e} prob {p:.2e}", fx.name);
        compare_answers(fx, &got, &fx.json["answers_f32"], 2e-4);
    }
    #[cfg(feature = "f16")]
    {
        let head =
            JointSchemaHead::<CudaRuntime>::from_dir(&client, &head_dir, HeadDtype::Checkpoint)
                .unwrap_or_else(|e| panic!("from_dir: {e}"));
        for fx in &fixtures {
            let got = run(&client, &head, fx);
            let (l, p) = compare(fx, &got, "logits_bf16", "probs_bf16", BF16_LOGIT, BF16_PROB);
            println!(
                "{}: cuda bf16 vs release logit {l:.2e} prob {p:.2e}",
                fx.name
            );
        }
    }
}
