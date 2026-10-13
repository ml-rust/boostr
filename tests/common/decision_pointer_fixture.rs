//! Kev pointer-head fixtures (schema `boostr.decision.pointer.v1`, written
//! by `ref_kev.py`) and the plan, items and question metadata each one
//! describes.

use std::path::PathBuf;

use boostr::inference::decision::{
    DecisionPlan, QuestionKind, QuestionMeta, RowSlot, plan_from_rows,
};
use boostr::model::decision::PointerItem;
use serde::Deserialize;

use super::decision_fixture::{env_path, json_fixtures};

pub const ENV_BASE: &str = "BOOSTR_QWEN35_HF_DIR";
pub const ENV_KEV: &str = "BOOSTR_KEV_DIR";
pub const SCHEMA: &str = "boostr.decision.pointer.v1";

/// The delimiter token ids of Kev's encoding.
#[derive(Deserialize)]
pub struct SpecialTokens {
    pub option_end: u32,
    pub decide: u32,
}

/// The pointer head's sizes.
#[derive(Deserialize)]
pub struct HeadSizes {
    pub hidden_size: usize,
    pub dim: usize,
}

/// One question, read in its own row.
#[derive(Deserialize)]
pub struct Question {
    pub id: String,
    pub kind: String,
    pub options: usize,
    pub yes_index: Option<usize>,
    pub row: usize,
    pub decide: usize,
    pub option_ends: Vec<usize>,
    pub raw_logits: Vec<f32>,
    pub raw_logits_merged: Vec<f32>,
    pub probs: Vec<f64>,
    pub p_yes: Option<f64>,
}

/// One reference request.
#[derive(Deserialize)]
pub struct Fixture {
    pub schema: String,
    pub name: String,
    pub special_tokens: SpecialTokens,
    pub head: HeadSizes,
    pub temperature: f64,
    pub state_len: usize,
    pub rows: Vec<Vec<u32>>,
    pub questions: Vec<Question>,
}

impl Fixture {
    /// `kev/name`, for messages.
    pub fn label(&self) -> String {
        format!("kev/{}", self.name)
    }
}

/// The base checkpoint of [`ENV_BASE`], the Kev directory of [`ENV_KEV`]
/// and every fixture of `BOOSTR_DECISION_FIXTURE_DIR`, or `None` after one
/// `skip:` line.
pub fn inputs() -> Option<(PathBuf, PathBuf, Vec<Fixture>)> {
    let base = env_path(ENV_BASE)?;
    if !base.join("config.json").is_file() {
        println!("skip: {} has no config.json", base.display());
        return None;
    }
    let kev = env_path(ENV_KEV)?;
    for file in [
        "adapter_config.json",
        "adapter_model.safetensors",
        "head.pt",
    ] {
        if !kev.join(file).is_file() {
            println!("skip: {} has no {file}", kev.display());
            return None;
        }
    }
    let fixtures = json_fixtures(SCHEMA, |fx: &Fixture| fx.schema.as_str())?;
    Some((base, kev, fixtures))
}

/// The plan, items and question metadata of `fx`.
///
/// Each question's row is one plan row, read at its decide token and at
/// each option's closing token. `plan_from_rows` takes the rows' longest
/// common prefix, capped at the earliest read position, as the shared
/// prefix: with two or more questions it covers at least the state.
pub fn request(fx: &Fixture) -> (DecisionPlan, Vec<PointerItem>, Vec<QuestionMeta>) {
    let what = fx.label();
    assert_eq!(
        fx.rows.len(),
        fx.questions.len(),
        "{what}: one row per question"
    );
    let mut slots = Vec::new();
    let mut meta = Vec::with_capacity(fx.questions.len());
    for (q, question) in fx.questions.iter().enumerate() {
        let here = format!("{what} question {}", question.id);
        assert_eq!(question.row, q, "{here}: row");
        let row = &fx.rows[q];
        assert_eq!(
            question.decide,
            row.len() - 1,
            "{here}: decide is the last token"
        );
        assert_eq!(
            row[question.decide], fx.special_tokens.decide,
            "{here}: decide token"
        );
        assert_eq!(
            question.option_ends.len(),
            question.options,
            "{here}: option ends"
        );
        assert_eq!(
            question.raw_logits.len(),
            question.options,
            "{here}: raw logits"
        );
        for &end in &question.option_ends {
            assert!(
                end >= fx.state_len,
                "{here}: option end {end} inside the state"
            );
            assert_eq!(
                row[end], fx.special_tokens.option_end,
                "{here}: option end {end}"
            );
        }
        slots.push(RowSlot {
            row: q,
            pos: question.decide,
        });
        slots.extend(
            question
                .option_ends
                .iter()
                .map(|&pos| RowSlot { row: q, pos }),
        );
        let kind = match question.kind.as_str() {
            "choice" => QuestionKind::Choice,
            "noul" => QuestionKind::Noul {
                yes: question
                    .yes_index
                    .unwrap_or_else(|| panic!("{here}: noul without yes_index")),
            },
            "score" => QuestionKind::Score {
                values: None,
                isolated: false,
            },
            other => panic!("{here}: question kind {other:?}"),
        };
        meta.push(QuestionMeta {
            kind,
            options: question.options,
        });
    }
    let built = plan_from_rows(fx.rows.clone(), &slots, 0)
        .unwrap_or_else(|e| panic!("{what}: plan_from_rows: {e}"));
    if fx.rows.len() >= 2 {
        assert!(
            built.plan.prefix_len() >= fx.state_len,
            "{what}: shared prefix {} shorter than the state {}",
            built.plan.prefix_len(),
            fx.state_len
        );
    }
    let mut gathers = built.slot_gathers.iter().copied();
    let items = fx
        .questions
        .iter()
        .map(|question| {
            let decide = gathers.next().unwrap_or_else(|| panic!("{what}: slots"));
            let options = gathers.by_ref().take(question.options).collect();
            PointerItem { decide, options }
        })
        .collect();
    (built.plan, items, meta)
}
