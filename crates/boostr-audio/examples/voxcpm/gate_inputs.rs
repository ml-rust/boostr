//! Input rows for `voxcpm_quality_gate`: the `voxcpm_clone` render log, the
//! header-named manifest, per-row language choice, and clip decoding.

use std::path::{Path, PathBuf};

use boostr_audio::{decode_audio, extension_hint, to_mono, to_mono_at_rate};

/// Rate Whisper, the VAD and the AudioVAE encoder all consume.
pub const MODEL_RATE: u32 = 16_000;
/// Language a row without a tag is transcribed under.
pub const DEFAULT_LANGUAGE: &str = "ms";
/// Language value that means "read the row".
pub const AUTO_LANGUAGE: &str = "auto-from-row";

/// One clip to score, however it was described.
pub struct InputRow {
    pub id: String,
    pub wav: PathBuf,
    pub text: String,
    pub lang: Option<String>,
    pub axis: Option<String>,
    pub ref_wav: Option<PathBuf>,
    pub audio_seconds: Option<f64>,
    pub stop_reason: Option<String>,
    pub checks_passed: Option<bool>,
    pub ref_f0_hz: Option<f64>,
}

/// Absolute paths pass through; relative ones are tried as given, then
/// against `base`.
fn resolve(base: &Path, field: &str) -> PathBuf {
    let path = PathBuf::from(field);
    if path.is_absolute() || path.exists() {
        path
    } else {
        base.join(path)
    }
}

fn json_str(v: &serde_json::Value, key: &str) -> Option<String> {
    v.get(key).and_then(|x| x.as_str()).map(str::to_string)
}

/// Read a `voxcpm_clone` JSONL log: the run record and every render row.
pub fn load_renders(path: &Path) -> Result<(serde_json::Value, Vec<InputRow>), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    let mut run = serde_json::Value::Null;
    let mut rows = Vec::new();
    for (n, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value =
            serde_json::from_str(line).map_err(|e| format!("{}:{}: {e}", path.display(), n + 1))?;
        match v.get("record").and_then(|r| r.as_str()) {
            Some("run") => run = v,
            Some("render") => {
                let missing =
                    |key: &str| format!("{}:{}: render row lacks {key}", path.display(), n + 1);
                let id = json_str(&v, "id").ok_or_else(|| missing("id"))?;
                let text = json_str(&v, "text").ok_or_else(|| missing("text"))?;
                let out_path = json_str(&v, "out_path").ok_or_else(|| missing("out_path"))?;
                rows.push(InputRow {
                    id,
                    wav: resolve(base, &out_path),
                    text,
                    lang: json_str(&v, "lang"),
                    axis: json_str(&v, "axis"),
                    ref_wav: None,
                    audio_seconds: v.get("audio_seconds").and_then(|x| x.as_f64()),
                    stop_reason: json_str(&v, "stop_reason"),
                    checks_passed: v.get("checks_passed").and_then(|x| x.as_bool()),
                    ref_f0_hz: v.get("ref_f0_hz").and_then(|x| x.as_f64()),
                });
            }
            _ => {}
        }
    }
    if rows.is_empty() {
        return Err(format!("{}: no render rows", path.display()));
    }
    Ok((run, rows))
}

/// Read a header-named TSV: `id`, `wav`, `text` required.
pub fn load_manifest(path: &Path) -> Result<Vec<InputRow>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    let mut lines = text.lines();
    let header = lines
        .next()
        .ok_or_else(|| format!("{}: empty manifest", path.display()))?;
    let cols: Vec<&str> = header.split('\t').map(str::trim).collect();
    let col = |name: &str| cols.iter().position(|c| *c == name);
    let need = |name: &str| {
        col(name).ok_or_else(|| format!("{}: header lacks column {name:?}", path.display()))
    };
    let (id_i, wav_i, text_i) = (need("id")?, need("wav")?, need("text")?);
    let (lang_i, axis_i, ref_i) = (col("lang"), col("axis"), col("ref_wav"));
    let cell = |fields: &[&str], i: Option<usize>| {
        i.and_then(|i| fields.get(i))
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    };
    let mut rows = Vec::new();
    for (n, line) in lines.enumerate() {
        let line = line.trim_end_matches('\r');
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        let get = |i: usize| {
            fields
                .get(i)
                .map(|s| s.trim().to_string())
                .ok_or_else(|| format!("{}:{}: short row", path.display(), n + 2))
        };
        rows.push(InputRow {
            id: get(id_i)?,
            wav: resolve(base, &get(wav_i)?),
            text: get(text_i)?,
            lang: cell(&fields, lang_i),
            axis: cell(&fields, axis_i),
            ref_wav: cell(&fields, ref_i).map(|r| resolve(base, &r)),
            audio_seconds: None,
            stop_reason: None,
            checks_passed: None,
            ref_f0_hz: None,
        });
    }
    if rows.is_empty() {
        return Err(format!("{}: no data rows", path.display()));
    }
    Ok(rows)
}

/// Whisper language token for `row` under `--language`.
pub fn language_for(setting: &str, row: &InputRow) -> String {
    if setting != AUTO_LANGUAGE {
        return setting.to_string();
    }
    match row.lang.as_deref() {
        Some("mix") | None => DEFAULT_LANGUAGE.to_string(),
        Some(lang) => lang.to_string(),
    }
}

/// A clip at its native rate and at [`MODEL_RATE`].
pub struct Clip {
    pub native: Vec<f32>,
    pub native_rate: u32,
    pub at_16k: Vec<f32>,
}

pub fn load_clip(path: &Path) -> Result<Clip, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let hint = path
        .file_name()
        .and_then(|n| n.to_str())
        .and_then(extension_hint);
    let data = decode_audio(&bytes, hint).map_err(|e| format!("{}: {e}", path.display()))?;
    let native =
        to_mono(&data.samples, data.channels).map_err(|e| format!("{}: {e}", path.display()))?;
    let at_16k =
        to_mono_at_rate(&data, MODEL_RATE).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(Clip {
        native,
        native_rate: data.sample_rate,
        at_16k,
    })
}
