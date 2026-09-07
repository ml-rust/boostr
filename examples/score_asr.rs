//! Score ASR hypotheses against a reference transcript set, per group and overall.
//!
//! ```text
//! cargo run --release --features audio --example score_asr -- \
//!     --references REFERENCES.tsv --hypotheses HYPOTHESES.tsv \
//!     [--text-column text] [--group-column NAME] [--cer]
//! ```
//!
//! Neither file ships with the crate; you supply both. They are tab
//! separated, and this is their whole shape.
//!
//! `--references`: a header row, an `id` column, and a text column named by
//! `--text-column` (default `text`). Any other column may exist and is
//! ignored unless `--group-column` names it. Column order does not matter.
//!
//! ```text
//! id      group   text
//! u001    plain   the cat sat on the mat
//! u002    plain   a second sentence
//! u003    digits  there were 7 of them
//! ```
//!
//! `--hypotheses`: no header, two columns, `path<TAB>text`. The id is the
//! file stem of `path`, so `u001.wav` pairs with reference id `u001`.
//!
//! ```text
//! renders/u001.wav        the cat sat on the mat
//! renders/u002.wav        a second sentence
//! renders/u003.wav        there were seven of them
//! ```
//!
//! That is the format the `transcribe` example writes on stdout, so its
//! output redirects straight into this tool. A line with no tab is not a
//! transcript line and is skipped, which is why a log with interleaved
//! progress lines can be passed unfiltered.
//!
//! Scoring those two files with `--group-column group` reports `plain` at
//! zero and `digits` above it, because `7` and `seven` are two different
//! tokens — the reason `--group-column` exists.
//!
//! `--group-column NAME` reports one row per distinct value of that reference
//! column; omitted, everything scores as a single group named `all`. This
//! exists for splitting off any subset that should not be pooled with the
//! rest of the corpus, such as prompts that must not have their numbers
//! normalized away — see `boostr::model::audio::eval`'s module docs.
//!
//! `--cer` scores with `character_error_rate` instead of `word_error_rate`.
//!
//! # What "missing" means and why it exits non-zero
//!
//! An id in `--references` with no matching id in `--hypotheses` is a scoring
//! gap, not a quiet zero: the reference table names an id, so the row must
//! exist. That case is reported (capped list, then a count of the rest) and
//! the process exits non-zero. A hypothesis id with no matching reference is
//! reported too, since it usually means an id mismatch worth fixing.
use std::collections::HashMap;
use std::path::PathBuf;

use boostr::model::audio::{
    ErrorRate, by_group, character_error_rate, grand_total, word_error_rate,
};

const USAGE: &str = "usage: score_asr --references FILE --hypotheses FILE \
[--text-column NAME] [--group-column NAME] [--cer]";

/// How many missing ids to print by name before collapsing the rest into a count.
const MAX_LISTED_MISSING: usize = 5;

struct Args {
    references: PathBuf,
    hypotheses: PathBuf,
    text_column: String,
    group_column: Option<String>,
    cer: bool,
}

fn take_value(argv: &[String], i: &mut usize, flag: &str) -> Result<String, String> {
    *i += 1;
    argv.get(*i)
        .cloned()
        .ok_or_else(|| format!("{flag} needs a value"))
}

fn parse_args() -> Result<Args, String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut references = None;
    let mut hypotheses = None;
    let mut text_column = "text".to_string();
    let mut group_column = None;
    let mut cer = false;

    let mut i = 0usize;
    while i < argv.len() {
        let flag = argv[i].as_str();
        match flag {
            "--references" => references = Some(PathBuf::from(take_value(&argv, &mut i, flag)?)),
            "--hypotheses" => hypotheses = Some(PathBuf::from(take_value(&argv, &mut i, flag)?)),
            "--text-column" => text_column = take_value(&argv, &mut i, flag)?,
            "--group-column" => group_column = Some(take_value(&argv, &mut i, flag)?),
            "--cer" => cer = true,
            "-h" | "--help" => return Err(USAGE.to_string()),
            other => return Err(format!("unknown flag {other}\n{USAGE}")),
        }
        i += 1;
    }

    Ok(Args {
        references: references.ok_or_else(|| format!("--references is required\n{USAGE}"))?,
        hypotheses: hypotheses.ok_or_else(|| format!("--hypotheses is required\n{USAGE}"))?,
        text_column,
        group_column,
        cer,
    })
}

/// One row of the references TSV: its id, reference text, and group value.
struct RefRow {
    id: String,
    text: String,
    /// `None` when `--group-column` was not given; every row then shares the
    /// single group name `"all"` at the call site.
    group: Option<String>,
}

/// Read the references TSV and return one [`RefRow`] per data line.
///
/// `text_column` and `group_column` are column names, resolved against the
/// header row on the first line. Missing `id` or `text_column` is an error;
/// a named `group_column` that is not in the header is also an error.
fn read_references(
    path: &std::path::Path,
    text_column: &str,
    group_column: Option<&str>,
) -> Result<Vec<RefRow>, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| format!("--references {}: {e}", path.display()))?;
    let mut lines = raw.lines();
    let header = lines.next().ok_or_else(|| {
        format!(
            "--references {}: file is empty, no header row",
            path.display()
        )
    })?;
    let columns: Vec<&str> = header.split('\t').collect();

    let find = |name: &str| -> Result<usize, String> {
        columns.iter().position(|c| *c == name).ok_or_else(|| {
            format!(
                "--references {}: no `{name}` column in header",
                path.display()
            )
        })
    };
    let id_idx = find("id")?;
    let text_idx = find(text_column)?;
    let group_idx = group_column.map(find).transpose()?;

    let mut rows = Vec::new();
    for (n, line) in lines.enumerate() {
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        let line_no = n + 2; // +1 for the header, +1 for 1-based counting.
        let get = |idx: usize, name: &str| -> Result<String, String> {
            fields.get(idx).map(|s| s.to_string()).ok_or_else(|| {
                format!(
                    "--references {}:{line_no}: missing `{name}` column",
                    path.display()
                )
            })
        };
        rows.push(RefRow {
            id: get(id_idx, "id")?,
            text: get(text_idx, text_column)?,
            group: group_idx
                .map(|idx| get(idx, group_column.unwrap_or_default()))
                .transpose()?,
        });
    }
    Ok(rows)
}

/// Read the `path<TAB>text` hypotheses stream into id -> text, id being the
/// file stem of `path`. A line without a tab is skipped, per the module docs.
fn read_hypotheses(path: &std::path::Path) -> Result<HashMap<String, String>, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| format!("--hypotheses {}: {e}", path.display()))?;
    let mut out = HashMap::new();
    for line in raw.lines() {
        let Some((file, text)) = line.split_once('\t') else {
            continue;
        };
        let id = std::path::Path::new(file)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(file)
            .to_string();
        out.insert(id, text.to_string());
    }
    Ok(out)
}

/// Print one `ErrorRate` row: the counts a rate cannot be interpreted without.
fn print_row(label: &str, rate: &ErrorRate) {
    println!(
        "{label}\tsub={}\tdel={}\tins={}\tref_len={}\trate={:.4}",
        rate.substitutions,
        rate.deletions,
        rate.insertions,
        rate.reference_len(),
        rate.rate()
    );
}

/// Print `ids`, capped at [`MAX_LISTED_MISSING`] with the remainder counted.
fn print_capped(prefix: &str, ids: &[String]) {
    let shown: Vec<&String> = ids.iter().take(MAX_LISTED_MISSING).collect();
    let shown_text = shown
        .iter()
        .map(|s| s.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    let remainder = ids.len().saturating_sub(shown.len());
    if remainder == 0 {
        eprintln!("{prefix}: {shown_text}");
    } else {
        eprintln!("{prefix}: {shown_text} (and {remainder} more)");
    }
}

fn main() {
    let args = match parse_args() {
        Ok(args) => args,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    };

    let refs = match read_references(
        &args.references,
        &args.text_column,
        args.group_column.as_deref(),
    ) {
        Ok(rows) => rows,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    };
    let mut hyps = match read_hypotheses(&args.hypotheses) {
        Ok(map) => map,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    };

    let mut missing_hyp = Vec::new();
    let mut matched: Vec<(String, String, String)> = Vec::new();
    for row in &refs {
        match hyps.remove(&row.id) {
            Some(text) => matched.push((
                row.group.clone().unwrap_or_else(|| "all".to_string()),
                row.text.clone(),
                text,
            )),
            None => missing_hyp.push(row.id.clone()),
        }
    }

    let ref_ids: std::collections::HashSet<&str> = refs.iter().map(|r| r.id.as_str()).collect();
    let mut extra_hyp: Vec<String> = hyps
        .keys()
        .filter(|id| !ref_ids.contains(id.as_str()))
        .cloned()
        .collect();
    extra_hyp.sort();

    if !missing_hyp.is_empty() {
        eprintln!(
            "{} reference id(s) have no matching hypothesis",
            missing_hyp.len()
        );
        print_capped("missing hypothesis for", &missing_hyp);
    }
    if !extra_hyp.is_empty() {
        eprintln!(
            "{} hypothesis id(s) have no matching reference",
            extra_hyp.len()
        );
        print_capped("no reference for", &extra_hyp);
    }

    let metric = if args.cer {
        character_error_rate
    } else {
        word_error_rate
    };
    let items = matched.iter().map(|(group, reference, hypothesis)| {
        (group.as_str(), reference.as_str(), hypothesis.as_str())
    });
    let groups = by_group(items, metric);

    for (group, rate) in &groups {
        print_row(group, rate);
    }
    print_row("OVERALL", &grand_total(&groups));

    if !missing_hyp.is_empty() {
        std::process::exit(1);
    }
}
